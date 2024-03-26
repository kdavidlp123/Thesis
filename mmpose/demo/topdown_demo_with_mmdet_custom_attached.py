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


    


    # build detector
    detector = init_detector(
        "../mmdetection/configs/rtmdet/rtmdet_nano_320-8xb32_coco-ear.py", 
        "../mmdetection/work_dirs/rtmdet_nano_320-8xb32_coco-ear/epoch_120.pth", 
        device="cuda:0")
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    
    degrees = ['15cm_0mm_0deg', '15cm_25mm_5deg', '15cm_50mm_10deg', '20cm_0mm_0deg', '20cm_25mm_5deg', '20cm_50mm_10deg']
    
    result = "../keypoint/attached/7_result"
    after_inpainting = "../keypoint/attached/4_after_inpainting"
    
    names = os.listdir(result)

    
    for name in names:
        best_kpt_model = glob.glob(os.path.join(result,name,"best_coco*.pth"))[0]

        pose_estimator = init_pose_estimator(
        "./configs/body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_coco-256x192_custom_attached.py",
        best_kpt_model,
        device="cuda:0",
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=False))))



        result_name_pred_json = os.path.join(result, name, "pred")
        if not os.path.isdir(result_name_pred_json):
            os.makedirs(result_name_pred_json)
        

        kpt_attached_5_test_img = os.path.join(result, name, "5_test_img")

        
        imgs = os.listdir(kpt_attached_5_test_img)
        imgs = sorted(imgs, key = lambda s : int(os.path.splitext(os.path.basename(s))[0][5:]), reverse = False)

        total_count = 0
        temp_count = 0
        deg_index = 0

        df = pd.DataFrame()
        for img in imgs:
        



            pred_instances = process_one_image(os.path.join(kpt_attached_5_test_img, img), detector, pose_estimator)

            pred_instances_list = split_instances(pred_instances)
            kpts = pred_instances_list[0]["keypoints"]
            kpts_pd = {}
            for i in range(0, len(kpts)):
                x, y  = kpts[i]
                kpts_pd[i] = str(x)+str(",")+str(y)
            
            ser = pd.DataFrame(data=kpts_pd, index = [total_count])
            df = pd.concat([df, ser])

            total_count += 1


        
        count = 0
        for deg in degrees:
            after_inpainting_name_deg = os.path.join(after_inpainting, name, deg)
            frame_count = len(os.listdir(after_inpainting_name_deg))

            if not os.path.isdir(os.path.join(result, name, "pred",deg)):
                os.makedirs(os.path.join(result, name, "pred",deg))
            df.iloc[count:count+frame_count,:].to_csv(os.path.join(result, name, "pred",deg, "pred.csv"))
            count = count + frame_count






if __name__ == '__main__':
    main()
