# import cv2
# import numpy as np
# from pycocotools.coco import COCO
# import os
# from ..dataloading import get_yolox_datadir
# from .datasets_wrapper import Dataset

# class Kitti(Dataset):
#     """
#     COCO dataset class for KITTI converted to COCO format.
#     """
    
#     def __init__(
#         self,
#         data_dir=None,
#         json_file="tracking_train.json",  # COCO格式的训练标注文件
#         name="train",
#         img_size=(376, 1242),  # 目标图片尺寸 (height, width)，根据您的要求修改
#         preproc=None,
#     ):
#         super().__init__(img_size)
#         if data_dir is None:
#             data_dir = os.path.join(get_yolox_datadir(), "kitti/training/image_02")
#         self.data_dir = data_dir
#         self.json_file = json_file

#         # 加载COCO数据
#         self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
#         self.ids = self.coco.getImgIds()
#         self.class_ids = sorted(self.coco.getCatIds())
#         cats = self.coco.loadCats(self.coco.getCatIds())
#         self._classes = tuple([c["name"] for c in cats])  # 获取类别名称
#         self.annotations = self._load_coco_annotations()
#         self.name = name
#         self.img_size = img_size
#         self.preproc = preproc

#     def __len__(self):
#         return len(self.ids)

#     def _load_coco_annotations(self):
#         return [self.load_anno_from_ids(_ids) for _ids in self.ids]

#     def load_anno_from_ids(self, id_):
#         im_ann = self.coco.loadImgs(id_)[0]
#         width = im_ann["width"]
#         height = im_ann["height"]
#         file_name = im_ann["file_name"]  # 通常为"{img_id:06d}.png"

#         anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
#         annotations = self.coco.loadAnns(anno_ids)
#         objs = []
#         for obj in annotations:
#             x1 = obj["bbox"][0]
#             y1 = obj["bbox"][1]
#             x2 = x1 + obj["bbox"][2]
#             y2 = y1 + obj["bbox"][3]
#             if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
#                 obj["clean_bbox"] = [x1, y1, x2, y2]
#                 objs.append(obj)

#         num_objs = len(objs)
#         res = np.zeros((num_objs, 6))

#         for ix, obj in enumerate(objs):
#             cls = self.class_ids.index(obj["category_id"])
#             res[ix, 0:4] = obj["clean_bbox"]
#             res[ix, 4] = cls
#             res[ix, 5] = obj["id"]  # 使用 obj["id"] 作为 track_id，如果有的话

#         img_info = (height, width, file_name)

#         return (res, img_info, file_name)

#     def load_anno(self, index):
#         return self.annotations[index][0]

#     def pull_item(self, index):
#         id_ = self.ids[index]
#         res, img_info, file_name = self.annotations[index]

#         img_file = os.path.join(self.data_dir, "images", file_name)  # 确认路径
#         img = cv2.imread(img_file)

#         assert img is not None, f"Image {img_file} could not be loaded."

#         # 调整图像尺寸，填充到指定尺寸
#         img = self.resize_image(img)

#         return img, res.copy(), img_info, np.array([id_])

#     def resize_image(self, img):
#         """将图像调整为目标尺寸376 x 1242并进行填充，确保一致性."""
#         target_h, target_w = 376, 1242  # 使用最大尺寸填充
#         h, w = img.shape[:2]

#         # 计算缩放比例以保持纵横比
#         scale = min(target_w / w, target_h / h)
#         new_w = int(w * scale)
#         new_h = int(h * scale)

#         # 调整图像大小
#         img_resized = cv2.resize(img, (new_w, new_h))

#         # 填充图像以适应目标尺寸
#         delta_w = target_w - new_w
#         delta_h = target_h - new_h

#         top, bottom = delta_h // 2, delta_h - (delta_h // 2)
#         left, right = delta_w // 2, delta_w - (delta_w // 2)

#         img_padded = cv2.copyMakeBorder(
#             img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
#         )

#         assert img_padded.shape[0] == target_h and img_padded.shape[1] == target_w, \
#             f"Padding failed. Expected size: ({target_h}, {target_w}), got: {img_padded.shape}"

#         print(f"Padded image size: {img_padded.shape}")  # 打印该图像的最终尺寸
#         return img_padded

#     @Dataset.resize_getitem
#     def __getitem__(self, index):
#         img, target, img_info, img_id = self.pull_item(index)

#         if self.preproc is not None:
#             img, target = self.preproc(img, target, self.input_dim)
#         return img, target, img_info, img_id

import cv2
import numpy as np
from pycocotools.coco import COCO
import os
from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset


class Kitti(Dataset):
    """
    COCO dataset class for KITTI converted to COCO format.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="tracking_train.json",  # COCO格式的训练标注文件
        name="train",
        img_size=(370, 1240),  # 目标图片尺寸 (height, width)
        preproc=None,
    ):
        super().__init__(img_size)
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "kitti/training/image_02")
        self.data_dir = data_dir
        self.json_file = json_file

        # 加载COCO数据
        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])  # 获取类别名称
        self.annotations = self._load_coco_annotations()
        self.name = name
        self.img_size = img_size
        self.preproc = preproc

    def __len__(self):
        return len(self.ids)

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        file_name = im_ann["file_name"]  # 通常为"{img_id:06d}.png"

        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = obj["bbox"][0]
            y1 = obj["bbox"][1]
            x2 = x1 + obj["bbox"][2]
            y2 = y1 + obj["bbox"][3]
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)
        res = np.zeros((num_objs, 6))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls
            res[ix, 5] = obj["id"]  # 使用 obj["id"] 作为 track_id，如果有的话

        img_info = (height, width, file_name)

        return (res, img_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def pull_item(self, index):
        id_ = self.ids[index]
        res, img_info, file_name = self.annotations[index]

        img_file = os.path.join(self.data_dir, "images", file_name)  # 确认路径
        img = cv2.imread(img_file)

        assert img is not None, f"Image {img_file} could not be loaded."

        # 调整图像尺寸，裁切到指定尺寸
        img = self.resize_image(img)

        return img, res.copy(), img_info, np.array([id_])

    def resize_image(self, img):
        """裁切图像为目标尺寸 370 x 1240."""
        target_h, target_w = 370, 1224  # 目标裁切尺寸
        h, w = img.shape[:2]

        # 裁切图像
        if h >= target_h and w >= target_w:
            img_cropped = img[:target_h, :target_w]
        else:
            # 如果图像本身太小，则保持其原始尺寸
            img_cropped = img[:h, :w]

        assert img_cropped.shape[0] == target_h and img_cropped.shape[1] == target_w, \
            f"Crop failed. Expected size: ({target_h}, {target_w}), got: {img_cropped.shape}"

        print(f"Cropped image size: {img_cropped.shape}")  # 打印最终裁切后的图像尺寸
        return img_cropped

    @Dataset.resize_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id