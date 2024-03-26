# Copyright (c) OpenMMLab. All rights reserved.
from mmpose.registry import DATASETS
# from mmengine.dataset import BaseDataset
from ..base import BaseCocoStyleDataset


@DATASETS.register_module(name='customdataset')
class customdataset(BaseCocoStyleDataset):

    METAINFO: dict = dict(from_file='configs/_base_/datasets/custom.py')
