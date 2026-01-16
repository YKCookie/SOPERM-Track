
import argparse
import os
import os.path as osp
import time
import cv2
import torch

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from trackers.ocsort_tracker.ocsort import OCSort
from trackers.tracking_utils.timer import Timer
from trackers.ocsort_sparse_tracker.sparse_ocsort import sparse_OCSort

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

from utils.args import make_parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def draw_detections_one_color(
    img_bgr,
    outputs,
    ratio,
    color=(0, 255, 0),
    thickness=2,
    put_label=False
):
    """
    将检测框全部画成同一种颜色。
    兼容 YOLOX postprocess 的常见输出格式：
      [x1,y1,x2,y2,obj,cls_conf,cls_id] 或 [x1,y1,x2,y2,score,cls_id] 或 [x1,y1,x2,y2,score]
    """
    if outputs is None or outputs[0] is None:
        return img_bgr

    output = outputs[0].cpu()

    # 还原到原图尺度
    bboxes = output[:, 0:4] / ratio

    # 解析类别与分数（尽量兼容不同版本）
    cls_ids = None
    scores = None
    if output.shape[1] >= 7:
        # [x1,y1,x2,y2,obj,cls_conf,cls_id]
        scores = (output[:, 4] * output[:, 5]).numpy()
        cls_ids = output[:, 6].int().numpy()
    elif output.shape[1] == 6:
        # [x1,y1,x2,y2,score,cls_id]
        scores = output[:, 4].numpy()
        cls_ids = output[:, 5].int().numpy()
    elif output.shape[1] == 5:
        # [x1,y1,x2,y2,score]
        scores = output[:, 4].numpy()

    h, w = img_bgr.shape[:2]
    for i, box in enumerate(bboxes):
        x1, y1, x2, y2 = box.tolist()
        x1 = int(max(0, min(w - 1, x1)))
        y1 = int(max(0, min(h - 1, y1)))
        x2 = int(max(0, min(w - 1, x2)))
        y2 = int(max(0, min(h - 1, y2)))

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)

        if put_label and scores is not None:
            label = f"{scores[i]:.2f}"
            if cls_ids is not None:
                label = f"{int(cls_ids[i])}:{label}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img_bgr, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img_bgr


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
            if img is None:
                raise FileNotFoundError(f"Failed to read image: {img_info['file_name']}")
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            timer.toc()
        return outputs, img_info


def build_sparse_tracker(args):
    """统一构建 sparse_OCSort，方便在多处调用"""
    return sparse_OCSort(
        det_thresh=getattr(args, "track_thresh", 0.25),
        max_age=getattr(args, "max_age", 40),
        min_hits=getattr(args, "min_hits", 3),
        iou_threshold=getattr(args, "iou_thresh", 0.30),
        delta_t=getattr(args, "deltat", getattr(args, "delta_t", 3)),
        asso_func=getattr(args, "asso", "iou"),
        inertia=getattr(args, "inertia", 0.2),
        use_byte=getattr(args, "use_byte", True),

        # Soft-Depth
        depth_alpha=(0.0 if not getattr(args, "use_depth", True)
                    else getattr(args, "depth_alpha", 0.25)),
        depth_beta=getattr(args, "depth_beta", 1.6),
        depth_gate=getattr(args, "depth_gate", 0.3),
        gate_floor=getattr(args, "gate_floor", 0.35),

        # UA-HMIoU
        ua_enable=getattr(args, "ua_enable", True),
        ua_mode=getattr(args, "ua_mode", "cauchy"),  # cauchy or gauss
        ua_alpha=getattr(args, "ua_alpha", 1.8),
        ua_border_margin_frac=getattr(args, "ua_border_margin_frac", 0.02),
        ua_border_boost=getattr(args, "ua_border_boost", 0.5),
        ua_miss_boost=getattr(args, "ua_miss_boost", 0.9),
        ua_low_score=getattr(args, "ua_low_score", 0.5),
        ua_score_boost=getattr(args, "ua_score_boost", 0.35),
        ua_sigma_floor_frac=getattr(args, "ua_sigma_floor_frac", 0.08),
        ua_proc_var_frac=getattr(args, "ua_proc_var_frac", 0.03),
        ua_h_ema=getattr(args, "ua_h_ema", 0.5),
        ua_h2_ema=getattr(args, "ua_h2_ema", 0.3),

        # PC-HMIoU
        pc_enable=getattr(args, "pc_enable", True),
        pc_bins=getattr(args, "pc_bins", 16),
        pc_ema=getattr(args, "pc_ema", 0.05),
        pc_gamma=getattr(args, "pc_gamma", 0.6),
        pc_floor=getattr(args, "pc_floor", 0.8),
        pc_min_count=getattr(args, "pc_min_count", 4),
        pc_sample_score=getattr(args, "pc_sample_score", 0.7),

        # Depth EMA
        ema_obs=getattr(args, "depth_ema_obs", 0.4),
        ema_pred=getattr(args, "depth_ema_pred", 0.05),
    )


def image_demo(predictor, vis_folder, current_time, args):
    """
    修改点：
    - 强制仅处理一张图片（要求 --path 为文件而非目录）
    - 不使用 tracker，直接画检测框（统一颜色）
    - 固定保存目录为用户指定的路径
    """
    assert osp.isfile(args.path), f"--path 必须为单张图片文件，但得到：{args.path}"
    files = [args.path]  # 只处理一张

    timer = Timer()

    # 统一颜色：绿色(BGR)。如需红色改为 (0, 0, 255)
    box_color = (0, 218, 255)

    for frame_id, img_path in enumerate(files, 1):
        outputs, img_info = predictor.inference(img_path, timer)

        # 仅检测可视化（不做跟踪）
        online_im = draw_detections_one_color(
            img_info["raw_img"].copy(),
            outputs,
            ratio=img_info["ratio"],
            color=box_color,
            thickness=2,
            put_label=False
        )

        # 保存到固定目录
        if args.save_result:
            save_folder = "/public/home/cookie/anaconda3/envs/ocsort_env/OC_SORT"
            os.makedirs(save_folder, exist_ok=True)
            base = osp.splitext(osp.basename(img_path))[0]
            save_path = osp.join(save_folder, f"{base}_det.jpg")
            cv2.imwrite(save_path, online_im)
            logger.info(f"Saved visualization to: {save_path}")

        # 无需等待键盘，直接退出
        break


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo_type == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo_type == "video":
        save_path = args.out_path
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

    tracker = OCSort(det_thresh=args.track_thresh, iou_threshold=args.iou_thresh, use_byte=args.use_byte)
    # tracker = build_sparse_tracker(args)

    timer = Timer()
    frame_id = 0
    results = []
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                online_tlwhs = []
                online_ids = []
                for t in online_targets:
                    tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                    tid = t[4]
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1.0,-1,-1,-1\n"
                        )
                timer.toc()
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.toc()
                online_im = img_info['raw_img']
            if args.save_result:
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def main(exp, args):
    if not args.expn:
        args.expn = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.expn)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)
    else:
        vis_folder = output_dir

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    if args.demo_type == "image":
        image_demo(predictor, vis_folder, current_time, args)
    elif args.demo_type == "video" or args.demo_type == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)
