# Keypoint Detection Dataset Preparation
In this research, we treated auricular acupoints as keypoints.
* Device: 
  1. Camera: Logitech StreamCam<br>
  2. Wewow camera dolly
* Software:<br>
  1. [Logitech Capture](<https://www.logitech.com/zh-tw/software/capture.html>)
* Experimental environment:<br>
  <img src="https://github.com/kdavidlp123/Thesis/assets/69571884/8e9804b5-c8d7-4c88-9798-6c77293539e0" width="100%" height="100%" />

## 1. Data collection 
We collected the data <br>
<img src="https://github.com/kdavidlp123/Thesis/assets/69571884/c769aa7e-3bed-4675-8494-4fb322d4dab2" width="60%" height="60%" />
<img src="https://github.com/kdavidlp123/Thesis/assets/69571884/4f2a3703-390f-47c0-96bf-88b9f02b13c8" width="60%" height="60%" />




## 2. Auriclur acupoint labeling
1. We used CSRT tracking algorithm from OPENCV to track the manually selected ROIs around markers in the first frame, then the trakcers started to track these ROIs in every frame.<br>
2. We recored the tracking results in every frame and retrieved the center points of these ROIs as keypoint coordinates.
## 3. Remove the markers using inpainting
1. Based on the ROIs saved in the previous step, we performed adaptive binarization on these regions to obtain the masks converted from the markers.
2. We used the inpainting algorithm from OPENCV to reconstruct the images. After that, all the markers were removed!!!
