# Copyright (c) OpenMMLab. All rights reserved.
import mimetypes
import os
import time
from argparse import ArgumentParser
import glob

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
import pandas as pd

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def process_one_image(
                      img,
                      detector,
                      pose_estimator):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                   pred_instance.scores > 0.3)]
    bboxes = bboxes[nms(bboxes, 0.3), :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)


    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None)


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """

    keypoint_ = "../keypoint"
    data_aug = "../data_aug"
    ear_types = ["free", "attached"]
    degrees = ['15cm_0mm_0deg', '15cm_25mm_5deg', '15cm_50mm_10deg', '20cm_0mm_0deg', '20cm_25mm_5deg', '20cm_50mm_10deg']
    angles = [-15, -10, -5,5,10, 15]


    # build detector
    detector = init_detector(
        "../mmdetection/configs/rtmdet/rtmdet_nano_320-8xb32_coco-ear.py", 
        "../mmdetection/work_dirs/rtmdet_nano_320-8xb32_coco-ear/epoch_120.pth", 
        device="cuda:0")
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    
    for ear_type in ear_types:
        model_config = "./configs/body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_coco-256x192_custom_{eartype}.py".format(eartype = ear_type)
        # if ear_type == "free":
        #     acupoints_num = 21
            
        # else:
        #     acupoints_num = 14
        for angle in angles:
            names = os.listdir(os.path.join(data_aug, ear_type, str(angle)))
            for name in names:
                

                best_kpt_model = glob.glob(os.path.join(keypoint_, ear_type, "7_result", name,"best_coco*.pth"))[0]

                pose_estimator = init_pose_estimator(
                model_config,
                best_kpt_model,
                device="cuda:0",
                cfg_options=dict(
                    model=dict(test_cfg=dict(output_heatmaps=False))))



                for deg in degrees:
                    imgs = os.listdir(os.path.join(data_aug, ear_type, str(angle), name, deg, "img"))
                    
                    imgs = sorted(imgs, key = lambda s : int(os.path.splitext(os.path.basename(s))[0][5:]), reverse = False)
                    df = pd.DataFrame()
                    for img in imgs:
                        pred_instances = process_one_image(os.path.join(data_aug, ear_type, str(angle), name, deg, "img", img), detector, pose_estimator)
                        pred_instances_list = split_instances(pred_instances)
                        kpts = pred_instances_list[0]["keypoints"]
                        kpts_pd = {}
                        for i in range(0, len(kpts)):
                            x, y  = kpts[i]
                            kpts_pd[i] = str(x)+","+str(y)
                        
                        ser = pd.DataFrame(data=kpts_pd, index = [int(os.path.splitext(os.path.basename(img))[0][5:])])
                        df = pd.concat([df, ser])
                    df.to_csv(os.path.join(data_aug, ear_type, str(angle), name, deg, "pred.csv"))





if __name__ == '__main__':
    main()
