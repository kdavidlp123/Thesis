# Thesis
# Introduction
LENS-iAcu is an mobile application which can localize auricular acupoint in real-time scenario. This project is implemented based on OpenMMLab.
Here is the demo video of LENS-iAcu:
# Environment Settings
* Windows 10
* Anaconda
* cuda version: 11.1
# Installation
```
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
git clone https://github.com/kdavidlp123/Thesis.git
cd Thesis/mmdetection
pip install -v -e .
cd ../mmpose
pip install -r requirements.txt
pip install -v -e .
```
# Keypoint Detection Dataset Preparation
Please contact us through the e-mail: P76091519@gs.ncku.edu.tw to acquire the data.
## 1. Data collection 
* Camera: Logitech StreamCam
## 2. Keypoint coordinates labeling
1. We used CSRT tracking algorithm from OPENCV to track the manually selected ROIs around markers in the first frame, then the trakcers started to track these ROIs in every frame.<br>
2. We recored the tracking results in every frame and retrieved the center points of these ROIs as keypoint coordinates.
## 3. Remove the markers using inpainting
1. Based on the ROIs saved in the previous step, we performed adaptive binarization on these regions to obtain the masks converted from the markers.
2. We used the inpainting algorithm from OPENCV to reconstruct the images. After that, all the markers were removed!!!







