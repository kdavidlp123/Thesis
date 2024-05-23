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
