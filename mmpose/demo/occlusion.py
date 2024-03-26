# Copyright (c) OpenMMLab. All rights reserved.
import mimetypes
import os
import time
from argparse import ArgumentParser
import glob
import re

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

    # build detector
    detector = init_detector(
        "../mmdetection/configs/rtmdet/rtmdet_nano_320-8xb32_coco-ear.py", 
        "../mmdetection/work_dirs/rtmdet_nano_320-8xb32_coco-ear/epoch_120.pth", 
        device="cuda:0")
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)


    

    occlusion_image = "../keypoint/free/9_occlusion_image"
    occlusion_result = "../keypoint/free/9_occlusion_result"



        # inference

    degrees = ['15cm_0mm_0deg', '15cm_25mm_5deg', '15cm_50mm_10deg', '20cm_0mm_0deg', '20cm_25mm_5deg', '20cm_50mm_10deg']
    result = "./result"

    names = os.listdir(result)

    for name in names:
        best_kpt_model = glob.glob(os.path.join(result,name, "best_coco*.pth"))[0]

            # build pose estimator
        pose_estimator = init_pose_estimator(
        "./configs/body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_coco-256x192_custom.py",
        best_kpt_model,
        device="cuda:0",
        cfg_options=dict(
        model=dict(test_cfg=dict(output_heatmaps=False))))
        for deg in degrees:

            for frame_index in os.listdir(os.path.join(occlusion_image, name, deg)):
                
            


            # imgs = sorted(imgs, key = lambda s : (int(re.split("[_|.]",os.path.basename(s))[1]), int(re.split("[_|.]",os.path.basename(s))[2])), reverse = False)

                occlusion_result_name_deg_frame_index = os.path.join(occlusion_result, name, deg, frame_index)
                if not os.path.isdir(occlusion_result_name_deg_frame_index):
                    os.makedirs(occlusion_result_name_deg_frame_index)
                
                imgs = os.listdir(os.path.join(occlusion_image, name, deg, frame_index))
                imgs = sorted(imgs, key = lambda s : int(re.split("[_|.]",os.path.basename(s))[1]), reverse = False)
                
                df = pd.DataFrame(columns = list(range(21)), index = [0])

                acupoint_index = 0
                for img in imgs:
                    

                    pred_instances = process_one_image(os.path.join(occlusion_image, name, deg, frame_index,img), detector, pose_estimator)
                    pred_instances_list = split_instances(pred_instances)
                    kpts = pred_instances_list[0]["keypoints"]
                    x, y  = kpts[acupoint_index]
                    df.iloc[0, acupoint_index] = str(x)+str(",")+str(y)
                    acupoint_index += 1
                df.to_csv(os.path.join(occlusion_result_name_deg_frame_index, "occlusion_pred.csv"))





if __name__ == '__main__':
    main()
