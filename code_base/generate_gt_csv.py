import os
import json
import glob
import numpy as np
import pandas as pd
from _info_ import ear_types, degrees

class ground_truth():
    def __init__(self, src):
        self.src = src
        self.dst = "../ground_truth"
    
    def check_gt_exist(self):
        if os.path.isdir(os.path.join(self.dst)):
            return True
        return False
    
    def read_json(self, fpath1):
        with open(fpath1, "r") as file:
            f = json.load(file)
        return f
     
    def generate_gt_csv(self):   #generate gt.csv
        
        for ear_type in ear_types:
            names = os.listdir(os.path.join("..", self.src, ear_type, "2_json"))
            for name in names:
                for deg in degrees:
                    kpt_json = os.path.join("..", self.src, ear_type, "2_json", name, deg, "keypoint_location.json")
                    visible_json = os.path.join("..", self.src, ear_type, "2_json", name, deg, "visible.json")
                    
                    kpts_coordinates = self.read_json(kpt_json)
                    kpts_visibles = self.read_json(visible_json)
                    
                    gt_df = pd.DataFrame()
                    for i in range(0 ,len(kpts_coordinates)):
                        kpt_c = kpts_coordinates["frame_"+str(i)]
                        kpt_v = kpts_visibles["frame_"+str(i)]
                        kpts_pd = {}
                        for j in range(0,len(kpt_c)):
                            x, y = kpt_c[str(j)]["keypoint_location"]
                            v = kpt_v[str(j)]["visible"]
                            kpts_pd[j] = str(x)+","+str(y)+","+str(v)
                        ser = pd.DataFrame(data=kpts_pd, index = [i])
                        gt_df = pd.concat([gt_df, ser])
                        
                    if not os.path.isdir(os.path.join(self.dst, ear_type, name, deg)):
                        os.makedirs(os.path.join(self.dst, ear_type, name, deg))
                    gt_df.to_csv(os.path.join(self.dst, ear_type, name, deg, "gt.csv"))
        
        
        
        
                    