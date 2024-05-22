# Thesis
# Introduction
LENS-iAcu is an mobile application which can localize auricular acupoint in real-time scenario. This project is implemented based on OpenMMLab.
Here is the demo video of LENS-iAcu:
# Environment Settings
* python: 3.8.16
* torch: 1.9.1+cu111
* torchvision: 0.10.1+cu111
* cuda version: 11.1
* Camera: Logitech StreamCam
# Installation
* MMDetection : Please refer to the official website of MMDetection for more detailed installation
* MMPose: Please refer to the official website of MMPose for more detailed installation
# Dataset Preparation
We provide the original MP4 videos, allowing users to obtain training clips according to the program sequence. Additionally, we offer pre-processed data for direct use.
## 1. Data collection 
## 2. Keypoint coordinates labeling
We used CSRT tracking algorithm from OPENCV to track the manually selected ROIs around markers in the first frame, then the trakcers started to track these ROIs in every frame.<br>
We recored the tracking results in every frame and retrieved the center points of these ROIs as keypoint coordinates.
## 3. Remove the markers using inpainting
1. Based on the ROIs saved in the previous step, we performed adaptive binarization on these regions to obtain the masks converted from the markers.
2. We used the inpainting algorithm from OPENCV to reconstruct the images. After that, all the markers were removed!!!
## 4. Annotation file preparation and dataset registry
