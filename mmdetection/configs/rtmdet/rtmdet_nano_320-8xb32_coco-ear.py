from calendar import EPOCH


_base_ = './rtmdet_l_8xb32-300e_coco.py'

input_shape = 320
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=120, val_interval=5)
model = dict(
    backbone=dict(
        deepen_factor=0.33,
        widen_factor=0.25,
        use_depthwise=True,
    ),
    neck=dict(
        in_channels=[64, 128, 256],
        out_channels=64,
        num_csp_blocks=1,
        use_depthwise=True,
    ),
    bbox_head=dict(
        in_channels=64,
        feat_channels=64,
        share_conv=False,
        exp_on_reg=False,
        use_depthwise=True,
        num_classes=1),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='CachedMosaic',
        img_scale=(input_shape, input_shape),
        pad_val=114.0,
        max_cached_images=20,
        random_pop=False),
    dict(
        type='RandomResize',
        scale=(input_shape * 2, input_shape * 2),
        ratio_range=(0.5, 1.5),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(input_shape, input_shape)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Pad',
        size=(input_shape, input_shape),
        pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=(input_shape, input_shape),
        ratio_range=(0.5, 1.5),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(input_shape, input_shape)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Pad',
        size=(input_shape, input_shape),
        pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(input_shape, input_shape), keep_ratio=True),
    dict(
        type='Pad',
        size=(input_shape, input_shape),
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]


dataset_type = 'EarDataset'
data_root = 'data/'


default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
)
# log_processor = dict(window_size=10, by_epoch=True, custom_cfg=[
#         dict(data_src='loss',
#              method_name='mean',
#              window_size='epoch')], num_digits=4)

train_dataloader = dict(
    batch_size=24,
    num_workers=4,
    dataset=dict(type='EarDataset',
        data_root='data/',
        ann_file='training_set_json/training.json',
        data_prefix=dict(img='training_set/'),
        pipeline=train_pipeline,
        metainfo=dict(from_file='configs/_base_/datasets/ear_detection.py')))

val_dataloader = dict(
    batch_size=24,
    num_workers=4,
    dataset=dict(
        type='EarDataset',
        data_root='data/',
        ann_file='test_set_json/test.json',
        data_prefix=dict(img='test_set/'),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=dict(from_file='configs/_base_/datasets/ear_detection.py')))
test_dataloader = val_dataloader

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=280,
        switch_pipeline=train_pipeline_stage2)
]
