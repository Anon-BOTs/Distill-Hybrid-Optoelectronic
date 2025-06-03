import numpy as np
import mmcv
import logging
import os
import torch
import warnings
import contextlib
import io
import copy
import random
import math
import itertools
import numpy as np

from mmdet.datasets import DATASETS
from torch.utils.data import Dataset
from mmdet3d.datasets.pipelines import Compose

@DATASETS.register_module()
class VectorDataset(Dataset):
    def __init__(self, ann_file, pipeline, test_mode=False, use_mask=False, **kwargs):
        self.ann_file = ann_file
        self.data_infos = mmcv.load(self.ann_file, file_format='pkl')['infos']
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode
        self.use_mask = use_mask
        self.CLASSES = ['test']
        self.flag = np.zeros(len(self.data_infos), dtype=np.uint8)
    
    def __len__(self):
        return len(self.data_infos)
    
    def prepare_train_data(self, idx):
        input_dict = copy.deepcopy(self.data_infos[idx])
        if self.use_mask:
            new_filename = input_dict['img_filename']
            new_filename = os.path.join(os.path.dirname(new_filename), 'mask_' + os.path.basename(new_filename))
            input_dict['img_filename'] = new_filename

        results = self.pipeline(input_dict)
        return results
    
    def prepare_test_data(self, idx):
        input_dict = copy.deepcopy(self.data_infos[idx])

        results = self.pipeline(input_dict)
        return results
    
    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)