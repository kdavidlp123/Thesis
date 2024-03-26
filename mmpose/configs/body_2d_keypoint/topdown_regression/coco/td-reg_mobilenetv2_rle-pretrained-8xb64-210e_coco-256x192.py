_base_ = ['../../../_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=100, val_interval=10)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=1e-3,
))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=train_cfg['max_epochs'],
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# codec settings
codec = dict(type='RegressionLabel', input_size=(192, 256))

channel_cfg = dict(
    num_output_channels=21,
    dataset_joints=21,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
#        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
#        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
    ])


# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='MobileNetV2',
        widen_factor=1.,
        out_indices=(7, ),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmpose/top_down/'
            'mobilenetv2/mobilenetv2_coco_256x192-d1e58e7b_20200727.pth')),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='RLEHead',
        in_channels=1280,
        num_joints=21,
        loss=dict(type='RLELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        shift_coords=True,
    ),
)

# base dataset settings
# dataset_type = 'CocoDataset'
# data_mode = 'topdown'
# data_root = 'data/coco/'

dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = r'D:\keypoint\free'

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# # data loaders
# train_dataloader = dict(
#     batch_size=64,
#     num_workers=2,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_mode=data_mode,
#         ann_file='annotations/person_keypoints_train2017.json',
#         data_prefix=dict(img='train2017/'),
#         pipeline=train_pipeline,
#     ))
# val_dataloader = dict(
#     batch_size=32,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_mode=data_mode,
#         ann_file='annotations/person_keypoints_val2017.json',
#         bbox_file=f'{data_root}person_detection_results/'
#         'COCO_val2017_detections_AP_H_56_person.json',
#         data_prefix=dict(img='val2017/'),
#         test_mode=True,
#         pipeline=val_pipeline,
#     ))
# test_dataloader = val_dataloader

# # hooks
# default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))

# # evaluators
# val_evaluator = dict(
#     type='CocoMetric',
#     ann_file=f'{data_root}annotations/person_keypoints_val2017.json',
#     score_mode='bbox_rle')
# test_evaluator = val_evaluator

# data loaders
train_dataloader = dict(
    batch_size=24,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='5_temp_training_json/training.json',
        data_prefix=dict(img='5_temp_training_img/'),
        pipeline=train_pipeline,
        metainfo=dict(from_file='configs/_base_/datasets/custom.py')
    ))
val_dataloader = dict(
    batch_size=24,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='5_test_json/test.json',
        # bbox_file='data/coco/person_detection_results/'
        # 'COCO_val2017_detections_AP_H_56_person.json',
        data_prefix=dict(img='5_test_img/'),
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=dict(from_file='configs/_base_/datasets/custom.py')
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/5_test_json/test.json',
    use_area=False
)
test_evaluator = val_evaluator