# Copyright (c) OpenMMLab. All rights reserved.
from .aic_dataset import AicDataset
from .coco_dataset import CocoDataset
from .crowdpose_dataset import CrowdPoseDataset
from .jhmdb_dataset import JhmdbDataset
from .mhp_dataset import MhpDataset
from .mpii_dataset import MpiiDataset
from .mpii_trb_dataset import MpiiTrbDataset
from .ochuman_dataset import OCHumanDataset
from .posetrack18_dataset import PoseTrack18Dataset
from .posetrack18_video_dataset import PoseTrack18VideoDataset
from .custom_dataset import customdataset
from .custom_attached_dataset import custom_attached_dataset

__all__ = [
    'CocoDataset', 'MpiiDataset', 'MpiiTrbDataset', 'AicDataset',
    'CrowdPoseDataset', 'OCHumanDataset', 'MhpDataset', 'PoseTrack18Dataset',
    'JhmdbDataset', 'PoseTrack18VideoDataset', 'customdataset','custom_attached_dataset'
]