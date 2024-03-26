# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmcv.image import imread

from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
import os
import glob
import json
import shutil
# import cv2
from math import dist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# def parse_args():
#     parser = ArgumentParser()
#     parser.add_argument('img', help='Image file')
#     parser.add_argument('config', help='Config file')
#     parser.add_argument('checkpoint', help='Checkpoint file')
#     parser.add_argument('--out-file', default=None, help='Path to output file')
#     parser.add_argument(
#         '--device', default='cuda:0', help='Device used for inference')
#     parser.add_argument(
#         '--draw-heatmap',
#         action='store_true',
#         help='Visualize the predicted heatmap')
#     parser.add_argument(
#         '--show-kpt-idx',
#         action='store_true',
#         default=False,
#         help='Whether to show the index of keypoints')
#     parser.add_argument(
#         '--skeleton-style',
#         default='mmpose',
#         type=str,
#         choices=['mmpose', 'openpose'],
#         help='Skeleton style selection')
#     parser.add_argument(
#         '--kpt-thr',
#         type=float,
#         default=0.3,
#         help='Visualizing keypoint thresholds')
#     parser.add_argument(
#         '--radius',
#         type=int,
#         default=3,
#         help='Keypoint radius for visualization')
#     parser.add_argument(
#         '--thickness',
#         type=int,
#         default=1,
#         help='Link thickness for visualization')
#     parser.add_argument(
#         '--alpha', type=float, default=0.8, help='The transparency of bboxes')
#     parser.add_argument(
#         '--show',
#         action='store_true',
#         default=False,
#         help='whether to show img')
#     args = parser.parse_args()
#     return args



# answer_question = "../../answer_question/"
# k_fold = "k_fold"
# ear_types = ["free", "attached"]
# degrees = ['15cm_0mm_0deg', '15cm_25mm_5deg', '15cm_50mm_10deg', '20cm_0mm_0deg', '20cm_25mm_5deg', '20cm_50mm_10deg']
# def main():
#     for ear_type in ear_types:
#         pred_without_bbox = os.path.join(answer_question,"{ear_type}/".format(ear_type = ear_type),"pred_without_bbox")
#         names = os.listdir(os.path.join(k_fold,ear_type))
#         # for name in os.listdir(pred_without_bbox):
#         for name in names:
#             test_img = os.path.join(pred_without_bbox, name, "5_test_img")
#             best_kpt_model = glob.glob(os.path.join(k_fold,ear_type,name,"best_*.pth"))[0]
#             print(best_kpt_model)
#             model_config = "configs/body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_coco-256x192_custom_{eartype}.py".format(eartype = ear_type)
#             model = init_model(
#                     model_config,
#                     best_kpt_model,
#                     device="cuda:0",
#                     cfg_options=dict(
#                         model=dict(test_cfg=dict(output_heatmaps=False))))

#             for img_file in os.listdir(test_img):
#                 if img_file == "test_1600.png":
#                     img = os.path.join(test_img, img_file)
#                     batch_results = inference_topdown(model, img)
#                     results = merge_data_samples(batch_results)
#                     print(results)

#                     visualizer = VISUALIZERS.build(model.cfg.visualizer)
#                     visualizer.set_dataset_meta(
#                         model.dataset_meta)

            
#                     imgs = imread(img, channel_order='rgb')
#                     visualizer.add_datasample(
#                         'result',
#                         imgs,
#                         data_sample=results,
#                         draw_gt=False,
#                         draw_bbox=True,
#                         show = True)


k_fold = "k_fold"
ear_types = ["free", "attached"]
degrees = ['15cm_0mm_0deg', '15cm_25mm_5deg', '15cm_50mm_10deg', '20cm_0mm_0deg', '20cm_25mm_5deg', '20cm_50mm_10deg']

def main():
    for ear_type in ear_types:
        # pred_without_bbox = os.path.join(answer_question,"{ear_type}/".format(ear_type = ear_type),"pred_without_bbox")
        names = os.listdir(os.path.join(k_fold,ear_type))
        after_inpainting = "../keypoint/{eartype}/4_after_inpainting".format(eartype = ear_type)
        # for name in os.listdir(pred_without_bbox):
        for name in names:
            # test_img = os.path.join(pred_without_bbox, name, "5_test_img")
            best_kpt_model = glob.glob(os.path.join(k_fold, "model_save",ear_type,name,"best_*.pth"))[0]


            k_fold_result_name_pred = os.path.join(k_fold, "result", ear_type, name, "pred")

            if not os.path.isdir(k_fold_result_name_pred):
                os.makedirs(k_fold_result_name_pred)

            model_config = "configs/body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_coco-256x192_custom_{eartype}.py".format(eartype = ear_type)
            model = init_model(
                    model_config,
                    best_kpt_model,
                    device="cuda:0",
                    cfg_options=dict(
                        model=dict(test_cfg=dict(output_heatmaps=False))))
            
            kpt_attached_5_test_img = os.path.join(k_fold, "result", ear_type, name, "5_test_img")
            imgs = os.listdir(kpt_attached_5_test_img)
            imgs = sorted(imgs, key = lambda s : int(os.path.splitext(os.path.basename(s))[0][5:]), reverse = False)
            total_count = 0
            df = pd.DataFrame()
            for img in imgs:
                print(img)
                batch_results = inference_topdown(model, os.path.join(kpt_attached_5_test_img, img))
                kpts = batch_results[0].pred_instances['keypoints'][0]
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

                if not os.path.isdir(os.path.join(k_fold, "result", ear_type, name, "pred",deg)):
                    os.makedirs(os.path.join(k_fold, "result", ear_type, name, "pred",deg))
                df.iloc[count:count+frame_count,:].to_csv(os.path.join(k_fold, "result", ear_type, name, "pred",deg, "pred.csv"))
                count = count + frame_count    








if __name__ == '__main__':
    main()
