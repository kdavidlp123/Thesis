# Train RTMPose Model
1. Copy a RTMPose config file under the folder “configs/body_2d_keypoint/rtmpose/coco” and modify the code as below:<br>
   ![image](https://github.com/kdavidlp123/Thesis/assets/69571884/47c2d224-0726-4087-85a3-2800c11d17c7)<br>
   ※ The “data_root”, “ann_file”, “data_prefix” can be changed to what you arrage.
2. Run the cmd:<br>
```python tools\train.py configs\body_2d_keypoint\rtmpose\coco\**your_own_rtmpose_cfg_file_name** ```
