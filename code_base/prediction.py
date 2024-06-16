import os
import json
import glob
import shutil
from math import dist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
import cv2
try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False
    
    
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline
from _info_ import ear_types, degrees, acupoints_name, cm

class pred_csv():
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.kpt_folder = "../keypoint"
        self.kpt_cfg = "../mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_coco-256x192_custom_{et}.py"
        self.det_cfg = "../mmdetection/configs/rtmdet/rtmdet_nano_320-8xb32_coco-ear.py"
        self.det_ckpt = "../mmdetection/work_dirs/rtmdet_nano_320-8xb32_coco-ear/epoch_120.pth"
    
    def read_csv(self, fpath1):
        df = pd.read_csv(fpath1, index_col = 0)
        return df

    def process_one_image(self, img, detector , pose_estimator):
        bboxes = None
        if detector is not None:
            det_result = inference_detector(detector, img)
            pred_instance = det_result.pred_instances.cpu().numpy()
            bboxes = np.concatenate(
                (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
            bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                           pred_instance.scores > 0.3)]
            bboxes = bboxes[nms(bboxes, 0.3), :4]
            
        pose_results = inference_topdown(pose_estimator, img, bboxes)
        data_samples = merge_data_samples(pose_results)
        pred_instances = data_samples.get('pred_instances', None)
        pred_instances_list = split_instances(pred_instances)
        kpts = pred_instances_list[0]["keypoints"]

        return kpts
    
    
    def generate(self, has_detector = True):
        if has_detector == True:
            detector = init_detector(self.det_cfg, self.det_ckpt, device="cuda:0")
            detector.cfg = adapt_mmdet_pipeline(detector.cfg)
        else:
            detector = None
    
        for ear_type in ear_types:
            names = os.listdir(os.path.join("..", self.data_folder, ear_type, "model_save"))
            for name in names:
                rtmpose_cfg = os.path.join(self.kpt_cfg.format(et = ear_type))
                rtmpose_ckp = glob.glob(os.path.join("..", self.data_folder, ear_type, "model_save", name,"best*.pth"))[0]

                pose_estimator = init_pose_estimator(
                rtmpose_cfg,
                rtmpose_ckp,
                device="cuda:0",
                cfg_options=dict(
                    model=dict(test_cfg=dict(output_heatmaps=False))))

                imgs = os.listdir(os.path.join("..", self.data_folder, ear_type, "result", name, "test_img"))
                imgs = sorted(imgs, key = lambda s : int(os.path.splitext(os.path.basename(s))[0][5:]), reverse = False)
                

                df = pd.DataFrame()
                total_count = 0
                for img in imgs:
                    image = os.path.join(os.path.join("..", self.data_folder, ear_type, "result", name, "test_img", img))
                    kpts = self.process_one_image(image, detector, pose_estimator)
                    kpts_pd = {}
                    for i in range(0, len(kpts)):
                        x, y  = kpts[i]
                        kpts_pd[i] = str(x)+str(",")+str(y)
                    ser = pd.DataFrame(data=kpts_pd, index = [total_count])
                    df = pd.concat([df, ser])
                    total_count += 1
                    
                count = 0
                for deg in degrees:
                    after_inpainting_name_deg = os.path.join(self.kpt_folder, ear_type, "4_after_inpainting", name, deg)
                    frame_count = len(os.listdir(after_inpainting_name_deg))
                    
                    pred = os.path.join("..", self.data_folder, ear_type, "result", name, "pred", deg)
                    if not os.path.isdir(pred):
                        os.makedirs(pred)
                    df.iloc[count:count+frame_count,:].to_csv(os.path.join(pred, "pred.csv"))
                    count = count + frame_count
                
                    