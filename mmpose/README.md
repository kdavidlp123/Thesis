# Train RTMPose Model
## 1. Annotation file preparation and dataset registry
**To train our own data, we had to create the annotation file and register custom dataset.** <br>
**In this part, we only demonstrate free earlobes type of our data. We have two types of data which are free and attached earlobes type.** <br>
1. Follow the COCO dataset format, we had to create our annotation file like below:<br>
  <br>![image](https://github.com/kdavidlp123/Thesis/assets/69571884/16ea3e82-6a62-4be4-b26d-224552425c7d)<br><br>
2. Add a “custom.py” python file under folder “configs/_base_/datasets”, the content of config file is shown as below:<br>
  <br>![image](https://github.com/kdavidlp123/Thesis/assets/69571884/ad78ce1a-1206-4649-af93-45aa8f6e65d5)<br><br>
3. Add a “custom_dataset.py” python file under folder “mmpose/datasets/datasets/body” and referenced the code from the “coco_dataset.py” to modify our own code in “custom_dataset.py”, as presented below:<br>
  <br>![image](https://github.com/kdavidlp123/Thesis/assets/69571884/5a3bf913-c47a-4b13-9a75-8d07b4fee4f8)<br><br>
4. After all the above steps were finished, import the class “customdataset” and add the string “customdataset” to the list “__all__” in the “__init__.py” python file under folder “mmpose/datasets/datasets/body” as shown below:<br>
  <br>![image](https://github.com/kdavidlp123/Thesis/assets/69571884/773e6899-5bb9-41d3-9bee-e549004a081a)<br><br>
## 2. Model config and training command line
**Modify the contents of config file to match our data structure.**
1. Copy a RTMPose config file under the folder “configs/body_2d_keypoint/rtmpose/coco” and modify the code as below:<br>
   <br>![image](https://github.com/kdavidlp123/Thesis/assets/69571884/47c2d224-0726-4087-85a3-2800c11d17c7)<br>
   **※ The “data_root”, “ann_file”, “data_prefix” can be changed to what you arrage.**
2. Run the cmd:<br>
   ```python tools\train_custom.py configs\body_2d_keypoint\rtmpose\coco\rtmpose-s_8xb256-420e_coco-256x192_custom ```<br>
   **※ We modified the original train.py to do the k-fold validation** <br>
# Inference
   ```python demo\topdown_demo_with_mmdet_custom.py```<br>
