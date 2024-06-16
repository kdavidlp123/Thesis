import os
import json
import glob
import shutil
from math import dist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
from _info_ import ear_types, degrees, cm
from _common_ import split_xy_xyv

class errors_csv():
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.gt_folder = "../ground_truth"
        
    def read_csv(self, fpath1):
        df = pd.read_csv(fpath1, index_col = 0)
        return df
                
    def generate(self):
        for ear_type in ear_types:
            names = os.listdir(os.path.join("..", self.data_folder, ear_type, "result"))
            for name in names:
                for deg in degrees:
                    gt = os.path.join(self.gt_folder, ear_type, name , deg, "gt.csv")
                    pred = os.path.join("..", self.data_folder, ear_type, "result", name, "pred", deg, "pred.csv")

                    gt_df = self.read_csv(gt)
                    pred_df = self.read_csv(pred)

                    error_df = pd.DataFrame(columns=gt_df.columns, index = gt_df.index)

                    assert len(gt_df) == len(pred_df), "different frames"
                    for i in range(0, len(gt_df)):
                        
                        scale0_x, scale0_y, scale0_v= split_xy_xyv(gt_df.iloc[i,0])
                        scale11_x, scale11_y, scale11_v = split_xy_xyv(gt_df.iloc[i,11])

                        scale = cm[name] / dist((scale0_x, scale0_y), (scale11_x, scale11_y))

                        for j in range(0, len(gt_df.columns)):
                            gt_x, gt_y, gt_v = split_xy_xyv(gt_df.iloc[i,j])

                            if gt_v == "2":
                                pred_x, pred_y = split_xy_xyv(pred_df.iloc[i,j])
                                error_df.iloc[i,j] = scale * dist((gt_x, gt_y), (pred_x, pred_y))

                    error_path = os.path.join("..", self.data_folder, ear_type, "result", name, "error",deg)
                    if not os.path.isdir(error_path):
                        os.makedirs(error_path)
                    error_df.to_csv(os.path.join(error_path, "error.csv"))
                        
