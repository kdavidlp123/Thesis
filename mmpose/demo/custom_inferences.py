from mmpose.apis import MMPoseInferencer

img_path = 'tests/data/coco/test_80.png'   # replace this with your own image path

# create the inferencer using the model alias
inferencer = MMPoseInferencer(
    pose2d=r"D:\mmpose\configs\body_2d_keypoint\topdown_heatmap\coco\td-hm_res152_8xb32-210e_coco-384x288.py",
    pose2d_weights=r"D:\mmpose\training_log\epoch_210.pth",
    det_model=r"D:\mmdetection\configs\faster_rcnn\custom_model.py",
    det_weights=r"D:\mmdetection\work_dirs\custom_model\epoch_12.pth",
)

# The MMPoseInferencer API employs a lazy inference approach,
# creating a prediction generator when given input
result_generator = inferencer(img_path, show=True)
result = next(result_generator)