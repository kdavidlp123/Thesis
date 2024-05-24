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
We set the camera at different heights, angles and distances to collect the data.
* Distance: The horizontal distance from lens to ear
* Height – The height of table from desktop to ground
* High Angle – The angle of camera overlooking the ear
* Rotating Angle – The angle of camera rotating around the ears which is adjusted to match the Height to prevent the ears from being out of the frame."



<img src="https://github.com/kdavidlp123/Thesis/assets/69571884/c769aa7e-3bed-4675-8494-4fb322d4dab2" width="60%" height="60%" />
<img src="https://github.com/kdavidlp123/Thesis/assets/69571884/0df6f3bd-eead-4265-9d62-cb45ab1549ea" width="60%" height="60%" />







## 2. Auriclur acupoint labeling
1. We used CSRT tracking algorithm from OPENCV to track the manually selected ROIs around markers in the first frame, then the trakcers started to track these ROIs in every frame.<br>
2. We recored the tracking results in every frame and retrieved the center points of these ROIs as keypoint coordinates.
## 3. Remove the markers using inpainting
1. Based on the ROIs saved in the previous step, we performed adaptive binarization on these regions to obtain the masks converted from the markers.
2. We used the inpainting algorithm from OPENCV to reconstruct the images. After that, all the markers were removed!!!
