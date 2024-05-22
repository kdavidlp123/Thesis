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
1. We used CSRT tracking algorithm from OPENCV to track the manually selected ROIs around markers in the first frame, then the trakcers started to track these ROIs in every frame.<br>
2. We recored the tracking results in every frame and retrieved the center points of these ROIs as keypoint coordinates.
## 3. Remove the markers using inpainting
1. Based on the ROIs saved in the previous step, we performed adaptive binarization on these regions to obtain the masks converted from the markers.
2. We used the inpainting algorithm from OPENCV to reconstruct the images. After that, all the markers were removed!!!
## 4. Annotation file preparation and dataset registry
To train our own data, we had to create the annotation file and register custom dataset.
1. Follow the COCO dataset format, we had to create our annotation file like below:<br>
  ![image](https://github.com/kdavidlp123/Thesis/assets/69571884/16ea3e82-6a62-4be4-b26d-224552425c7d)
2. Add a “custom.py” python file under folder “configs/_base_/datasets”, the content of config file is shown as below:<br>
  ![image](https://github.com/kdavidlp123/Thesis/assets/69571884/ad78ce1a-1206-4649-af93-45aa8f6e65d5)
3. Add a “custom_dataset.py” python file under folder “mmpose/datasets/datasets/body” and referenced the code from the “coco_dataset.py” to modify our own code in “custom_dataset.py”, as presented below:<br>
  ![image](https://github.com/kdavidlp123/Thesis/assets/69571884/5a3bf913-c47a-4b13-9a75-8d07b4fee4f8)
4. After all the above steps were finished, import the class “customdataset” and add the string “customdataset” to the list “__all__” in the “__init__.py” python file under folder “mmpose/datasets/datasets/body” as shown below:<br>
  ![image](https://github.com/kdavidlp123/Thesis/assets/69571884/773e6899-5bb9-41d3-9bee-e549004a081a)
# Train RTMPose Model
1. Copy a RTMPose config file under the folder “configs/body_2d_keypoint/rtmpose/coco” and modify the code as below:<br>
   ![image](https://github.com/kdavidlp123/Thesis/assets/69571884/47c2d224-0726-4087-85a3-2800c11d17c7)<br>
   ※ The “data_root”, “ann_file”, “data_prefix” can be changed to what you arrage.

2. Run the cmd:
```python tools\train.py configs\body_2d_keypoint\rtmpose\coco\**your_own_rtmpose_cfg_file_name** ```





