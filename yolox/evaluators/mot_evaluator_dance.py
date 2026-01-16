from collections import defaultdict
from loguru import logger
from tqdm import tqdm

import torch

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)
from trackers.byte_tracker.byte_tracker import BYTETracker
from trackers.ocsort_tracker.ocsort import OCSort
from trackers.deepsort_tracker.deepsort import DeepSort
from trackers.motdt_tracker.motdt_tracker import OnlineTracker

from trackers.ocsort_sparse_tracker.sparse_ocsort import sparse_OCSort

import contextlib
import io
import os
import itertools
import json
import tempfile
import time
from utils.utils import write_results, write_results_no_score


class MOTEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, args, dataloader, img_size, confthre, nmsthre, num_classes):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.args = args

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.
        NOTE: This function will change training mode to False, please save states if needed.
        Args:
            model : model to evaluate.
        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = BYTETracker(self.args)
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = BYTETracker(self.args)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start
    
            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                if tlwh[2] * tlwh[3] > self.args.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            # save results
            results.append((frame_id, online_tlwhs, online_ids, online_scores))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_ocsort(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.
        NOTE: This function will change training mode to False, please save states if needed.
        Args:
            model : model to evaluate.
        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        # tracker = OCSort(det_thresh = self.args.track_thresh, iou_threshold=self.args.iou_thresh,
        #     asso_func=self.args.asso, delta_t=self.args.deltat, inertia=self.args.inertia, use_byte=self.args.use_byte)
        
  
        
        
        ##修复PC-HMIoU
        tracker = sparse_OCSort(
            # det_thresh=getattr(self.args, "track_thresh", 0.3),
            # max_age=getattr(self.args, "max_age", 30),
            # min_hits=getattr(self.args, "min_hits", 3),
            # iou_threshold=getattr(self.args, "iou_thresh", 0.35),
            # delta_t=getattr(self.args, "deltat", getattr(self.args, "delta_t", 3)),
            # asso_func=getattr(self.args, "asso", "iou"),
            # inertia=getattr(self.args, "inertia", 0.2),
            # use_byte=getattr(self.args, "use_byte", True),

            # # Soft-Depth
            # depth_alpha=(0.0 if not getattr(self.args, "use_depth", True)
            #             else getattr(self.args, "depth_alpha", 0.5)),
            # depth_beta=getattr(self.args, "depth_beta", 2.0),
            # depth_gate=getattr(self.args, "depth_gate", 0.25),
            # gate_floor=getattr(self.args, "gate_floor", 0.2),

            # # UA-HMIoU
            # ua_enable=getattr(self.args, "ua_enable", True),
            # ua_mode=getattr(self.args, "ua_mode", "cauchy"),#cauchy or gauss
            # ua_alpha=getattr(self.args, "ua_alpha", 1.2),
            # ua_border_margin_frac=getattr(self.args, "ua_border_margin_frac", 0.02),
            # ua_border_boost=getattr(self.args, "ua_border_boost", 0.5),
            # ua_miss_boost=getattr(self.args, "ua_miss_boost", 0.8),
            # ua_low_score=getattr(self.args, "ua_low_score", 0.4),
            # ua_score_boost=getattr(self.args, "ua_score_boost", 0.4),
            # ua_sigma_floor_frac=getattr(self.args, "ua_sigma_floor_frac", 0.05),
            # ua_proc_var_frac=getattr(self.args, "ua_proc_var_frac", 0.02),
            # ua_h_ema=getattr(self.args, "ua_h_ema", 0.5),
            # ua_h2_ema=getattr(self.args, "ua_h2_ema", 0.3),

            # # PC-HMIoU
            # pc_enable=getattr(self.args, "pc_enable", True),
            # pc_bins=getattr(self.args, "pc_bins", 24),
            # pc_ema=getattr(self.args, "pc_ema", 0.08),
            # pc_gamma=getattr(self.args, "pc_gamma", 0.8),
            # pc_floor=getattr(self.args, "pc_floor", 0.7),
            # pc_min_count=getattr(self.args, "pc_min_count", 3),
            # pc_sample_score=getattr(self.args, "pc_sample_score", 0.6),

            # # Depth EMA
            # ema_obs=getattr(self.args, "depth_ema_obs", 0.5),
            # ema_pred=getattr(self.args, "depth_ema_pred", 0.1),
            
            
            
            #针对DanceTrack 调参
            det_thresh=getattr(self.args, "track_thresh", 0.25),
            max_age=getattr(self.args, "max_age", 40),
            min_hits=getattr(self.args, "min_hits", 3),
            iou_threshold=getattr(self.args, "iou_thresh", 0.30),
            delta_t=getattr(self.args, "deltat", getattr(self.args, "delta_t", 3)),
            asso_func=getattr(self.args, "asso", "iou"),
            inertia=getattr(self.args, "inertia", 0.2),
            use_byte=getattr(self.args, "use_byte", True),

            # Soft-Depth
            depth_alpha=(0.0 if not getattr(self.args, "use_depth", True)
                        else getattr(self.args, "depth_alpha", 0.25)),
            depth_beta=getattr(self.args, "depth_beta", 1.6),
            depth_gate=getattr(self.args, "depth_gate", 0.3),
            gate_floor=getattr(self.args, "gate_floor", 0.35),

            # UA-HMIoU
            ua_enable=getattr(self.args, "ua_enable", True),
            ua_mode=getattr(self.args, "ua_mode", "gauss"),#cauchy or gauss
            ua_alpha=getattr(self.args, "ua_alpha", 1.8),
            ua_border_margin_frac=getattr(self.args, "ua_border_margin_frac", 0.02),
            ua_border_boost=getattr(self.args, "ua_border_boost", 0.5),
            ua_miss_boost=getattr(self.args, "ua_miss_boost", 0.9),
            ua_low_score=getattr(self.args, "ua_low_score", 0.5),
            ua_score_boost=getattr(self.args, "ua_score_boost", 0.35),
            ua_sigma_floor_frac=getattr(self.args, "ua_sigma_floor_frac", 0.08),
            ua_proc_var_frac=getattr(self.args, "ua_proc_var_frac", 0.03),
            ua_h_ema=getattr(self.args, "ua_h_ema", 0.5),
            ua_h2_ema=getattr(self.args, "ua_h2_ema", 0.3),

            # PC-HMIoU
            pc_enable=getattr(self.args, "pc_enable", True),
            pc_bins=getattr(self.args, "pc_bins", 16),# 8,16,24,32 ，original: 16
            pc_ema=getattr(self.args, "pc_ema", 0.05),
            pc_gamma=getattr(self.args, "pc_gamma", 0.6),
            pc_floor=getattr(self.args, "pc_floor", 0.8),
            pc_min_count=getattr(self.args, "pc_min_count", 4),
            pc_sample_score=getattr(self.args, "pc_sample_score", 0.7),

            # Depth EMA
            ema_obs=getattr(self.args, "depth_ema_obs", 0.4),
            ema_pred=getattr(self.args, "depth_ema_pred", 0.05),

            
        ) 
        

        
        
        detections = dict()

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]
                
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                if video_name not in video_names:
                    video_names[video_id] = video_name

                if frame_id == 1:
                    # tracker = OCSort(det_thresh = self.args.track_thresh, iou_threshold=self.args.iou_thresh,
                    #         asso_func=self.args.asso, delta_t=self.args.deltat, inertia=self.args.inertia, use_byte=self.args.use_byte)
                    
                    
  
                

                        ###修复PC-HMIoU
                    tracker = sparse_OCSort(
                        # det_thresh=getattr(self.args, "track_thresh", 0.3),
                        # max_age=getattr(self.args, "max_age", 30),
                        # min_hits=getattr(self.args, "min_hits", 3),
                        # iou_threshold=getattr(self.args, "iou_thresh", 0.35),
                        # delta_t=getattr(self.args, "deltat", getattr(self.args, "delta_t", 3)),
                        # asso_func=getattr(self.args, "asso", "iou"),
                        # inertia=getattr(self.args, "inertia", 0.2),
                        # use_byte=getattr(self.args, "use_byte", True),

                        # # Soft-Depth
                        # depth_alpha=(0.0 if not getattr(self.args, "use_depth", True)
                        #             else getattr(self.args, "depth_alpha", 0.5)),
                        # depth_beta=getattr(self.args, "depth_beta", 2.0),
                        # depth_gate=getattr(self.args, "depth_gate", 0.25),
                        # gate_floor=getattr(self.args, "gate_floor", 0.2),

                        # # UA-HMIoU
                        # ua_enable=getattr(self.args, "ua_enable", True),
                        # ua_mode=getattr(self.args, "ua_mode", "cauchy"),#cauchy or gauss
                        # ua_alpha=getattr(self.args, "ua_alpha", 1.2),
                        # ua_border_margin_frac=getattr(self.args, "ua_border_margin_frac", 0.02),
                        # ua_border_boost=getattr(self.args, "ua_border_boost", 0.5),
                        # ua_miss_boost=getattr(self.args, "ua_miss_boost", 0.8),
                        # ua_low_score=getattr(self.args, "ua_low_score", 0.4),
                        # ua_score_boost=getattr(self.args, "ua_score_boost", 0.4),
                        # ua_sigma_floor_frac=getattr(self.args, "ua_sigma_floor_frac", 0.05),
                        # ua_proc_var_frac=getattr(self.args, "ua_proc_var_frac", 0.02),
                        # ua_h_ema=getattr(self.args, "ua_h_ema", 0.5),
                        # ua_h2_ema=getattr(self.args, "ua_h2_ema", 0.3),

                        # # PC-HMIoU
                        # pc_enable=getattr(self.args, "pc_enable", True),
                        # pc_bins=getattr(self.args, "pc_bins", 24),
                        # pc_ema=getattr(self.args, "pc_ema", 0.08),
                        # pc_gamma=getattr(self.args, "pc_gamma", 0.8),
                        # pc_floor=getattr(self.args, "pc_floor", 0.7),
                        # pc_min_count=getattr(self.args, "pc_min_count", 3),
                        # pc_sample_score=getattr(self.args, "pc_sample_score", 0.6),

                        # # Depth EMA
                        # ema_obs=getattr(self.args, "depth_ema_obs", 0.5),
                        # ema_pred=getattr(self.args, "depth_ema_pred", 0.1),
                        

                        #针对DanceTrack 调参
                        det_thresh=getattr(self.args, "track_thresh", 0.25),
                        max_age=getattr(self.args, "max_age", 40),
                        min_hits=getattr(self.args, "min_hits", 3),
                        iou_threshold=getattr(self.args, "iou_thresh", 0.30),
                        delta_t=getattr(self.args, "deltat", getattr(self.args, "delta_t", 3)),
                        asso_func=getattr(self.args, "asso", "iou"),
                        inertia=getattr(self.args, "inertia", 0.2),
                        use_byte=getattr(self.args, "use_byte", True),

                        # Soft-Depth
                        depth_alpha=(0.0 if not getattr(self.args, "use_depth", True)
                                    else getattr(self.args, "depth_alpha", 0.25)),
                        depth_beta=getattr(self.args, "depth_beta", 1.6),
                        depth_gate=getattr(self.args, "depth_gate", 0.3),
                        gate_floor=getattr(self.args, "gate_floor", 0.35),

                        # UA-HMIoU
                        ua_enable=getattr(self.args, "ua_enable", True),
                        ua_mode=getattr(self.args, "ua_mode", "gauss"),#cauchy or gauss
                        ua_alpha=getattr(self.args, "ua_alpha", 1.8),
                        ua_border_margin_frac=getattr(self.args, "ua_border_margin_frac", 0.02),
                        ua_border_boost=getattr(self.args, "ua_border_boost", 0.5),
                        ua_miss_boost=getattr(self.args, "ua_miss_boost", 0.9),
                        ua_low_score=getattr(self.args, "ua_low_score", 0.5),
                        ua_score_boost=getattr(self.args, "ua_score_boost", 0.35),
                        ua_sigma_floor_frac=getattr(self.args, "ua_sigma_floor_frac", 0.08),
                        ua_proc_var_frac=getattr(self.args, "ua_proc_var_frac", 0.03),
                        ua_h_ema=getattr(self.args, "ua_h_ema", 0.5),
                        ua_h2_ema=getattr(self.args, "ua_h2_ema", 0.3),

                        # PC-HMIoU
                        pc_enable=getattr(self.args, "pc_enable", True),
                        pc_bins=getattr(self.args, "pc_bins", 16),# 8,16,24,32，original: 16
                        pc_ema=getattr(self.args, "pc_ema", 0.05),
                        pc_gamma=getattr(self.args, "pc_gamma", 0.6),
                        pc_floor=getattr(self.args, "pc_floor", 0.8),
                        pc_min_count=getattr(self.args, "pc_min_count", 4),
                        pc_sample_score=getattr(self.args, "pc_sample_score", 0.7),

                        # Depth EMA
                        ema_obs=getattr(self.args, "depth_ema_obs", 0.4),
                        ema_pred=getattr(self.args, "depth_ema_pred", 0.05),
                    ) 
                   
                    
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results_no_score(result_filename, results)
                        results = []

                ckt_file =  "dance_detections1/{}_detetcion.pkl".format(video_name)
                if os.path.exists(ckt_file):
                    # outputs = [torch.load(ckt_file)]
                    if not video_name in detections:
                        dets = torch.load(ckt_file)
                        detections[video_name] = dets 
                
                    all_dets = detections[video_name]
                    outputs = [all_dets[all_dets[:,0] == frame_id][:, 1:]]
                else:
                    imgs = imgs.type(tensor_type)

                    # skip the the last iters since batchsize might be not enough for batch inference

                    outputs = model(imgs)
                    if decoder is not None:
                        outputs = decoder(outputs, dtype=outputs.type())

                    outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
                    # we should save the detections here ! 
                    # os.makedirs("dance_detections/{}".format(video_name), exist_ok=True)
                    # torch.save(outputs[0], ckt_file)
                
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                """
                    Here is minor issue that DanceTrack uses the same annotation
                    format as MOT17/MOT20, namely xywh to annotate the object bounding
                    boxes. But DanceTrack annotation is cropped at the image boundary, 
                    which is different from MOT17/MOT20. So, cropping the output
                    bounding boxes at the boundary may slightly fix this issue. But the 
                    influence is minor. For example, with my results on the interpolated
                    OC-SORT:
                    * without cropping: HOTA=55.731
                    * with cropping: HOTA=55.737
                """
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                if tlwh[2] * tlwh[3] > self.args.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
            # save results
            results.append((frame_id, online_tlwhs, online_ids))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results


    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list



    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        track_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_track_time = 1000 * track_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "track", "inference"],
                    [a_infer_time, a_track_time, (a_infer_time + a_track_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)
            from yolox.layers import COCOeval_opt as COCOeval
            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info