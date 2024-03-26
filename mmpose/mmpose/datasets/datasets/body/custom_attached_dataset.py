# Copyright (c) OpenMMLab. All rights reserved.
from mmpose.registry import DATASETS
# from mmengine.dataset import BaseDataset
from ..base import BaseCocoStyleDataset


@DATASETS.register_module(name='custom_attached_dataset')
class custom_attached_dataset(BaseCocoStyleDataset):

    METAINFO: dict = dict(from_file='configs/_base_/datasets/custom_attached.py')
