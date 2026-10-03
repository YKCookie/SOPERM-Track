# 

#改进后代码
import numpy as np
import cv2
import os
import glob
import sys
from collections import defaultdict

DATA_PATH = '/public/home/cookie/anaconda3/envs/ocsort_env/OC_SORT/datasets/kitti_tracking/'
IMG_PATH = DATA_PATH + 'data_tracking_image_2/testing/image_02/'
SAVE_VIDEO = True  # 设置为 True 以启用视频保存
IS_GT = False

cats = ['Pedestrian', 'Car', 'Cyclist']
cat_ids = {cat: i for i, cat in enumerate(cats)}
COLORS = [(255, 0, 255), (122, 122, 255), (255, 0, 0)]

def draw_bbox(img, bboxes):
    for bbox in bboxes:
        color = COLORS[int(bbox[5])]
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), 
                      (int(bbox[2]), int(bbox[3])), 
                      color, 2, lineType=cv2.LINE_AA)
        ct = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        txt = '{}'.format(int(bbox[4]))
        cv2.putText(img, txt, (int(ct[0]), int(ct[1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                    color, thickness=1, lineType=cv2.LINE_AA)

if __name__ == '__main__':
    seqs = os.listdir(IMG_PATH)
    if SAVE_VIDEO:
        save_path = '/public/home/cookie/anaconda3/envs/ocsort_env/OC_SORT/permatrack-main/video'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print('save_video_path', save_path)
    
    for seq in sorted(seqs):
        print('seq', seq)
        if '.DS_Store' in seq:
            continue
        if SAVE_VIDEO:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video = cv2.VideoWriter(
                '{}/{}.avi'.format(save_path, seq), fourcc, 10.0, (1024, 750)
            )
        
        preds = {}
        for K in range(1, len(sys.argv)):
            pred_path = sys.argv[K] + '/{}.txt'.format(seq)
            pred_file = open(pred_path, 'r')
            preds[K] = defaultdict(list)
            for line in pred_file:
                tmp = line[:-1].split(' ')
                frame_id = int(tmp[0])
                track_id = int(tmp[1])
                cat_id = cat_ids[tmp[2]]
                bbox = [float(tmp[6]), float(tmp[7]), float(tmp[8]), float(tmp[9])]
                score = float(tmp[17])
                preds[K][frame_id].append(bbox + [track_id, cat_id, score])

        images_path = '{}/{}/'.format(IMG_PATH, seq)
        images = os.listdir(images_path)
        num_images = len([image for image in images if 'png' in image])
        
        for i in range(num_images):
            frame_id = i
            file_path = '{}/{:06d}.png'.format(images_path, i)
            img = cv2.imread(file_path)
            for K in range(1, len(sys.argv)):
                img_pred = img.copy()
                draw_bbox(img_pred, preds[K][frame_id])
                
                if SAVE_VIDEO:
                    video.write(img_pred)  # 写入视频
                
                cv2.imshow('pred{}'.format(K), img_pred)
            cv2.waitKey(1)  # 可以设置较短的等待时间以更流畅地观看
        
        if SAVE_VIDEO:
            video.release()  # 确保释放视频资源