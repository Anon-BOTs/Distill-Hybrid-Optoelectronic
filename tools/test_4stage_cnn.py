
from mmcv import Config, DictAction
from mmdet3d.models import build_model
from tqdm import trange
from torch.utils.data import Dataset

import os
import numpy as np
import torch
import importlib

class PseudoDataset(Dataset):
    def __init(self, img_size):
        self.img_size = img_size
    
    def __len__(self):
        return 100000

def import_dir(plugin_dir):
    _module_dir = os.path.dirname(plugin_dir)
    _module_dir = _module_dir.split('/')
    _module_path = _module_dir[0]

    for m in _module_dir[1:]:
        _module_path = _module_path + '.' + m
    plg_lib = importlib.import_module(_module_path)   

def main(cfg_file):
    cfg = Config.fromfile(cfg_file)

    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                if isinstance(plugin_dir, (list, tuple)):
                    for pd in plugin_dir:
                        import_dir(pd)
                else:
                    import_dir(plugin_dir)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)


    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

    model = model.cuda()

    for i in trange(10000):
        img = torch.rand(1, 3, 416, 608).cuda()
        img_metas = [
                        {
                            'filename': '/high_perf_store/surround-view/datasets/1/val2017/000000397133.jpg', 
                            'ori_filename': '000000397133.jpg', 
                            'ori_shape': (427, 640, 3), 
                            'img_shape': (400, 600, 3), 
                            'pad_shape': (416, 608, 3), 
                            'scale_factor': np.array([0.9375   , 0.9367682, 0.9375   , 0.9367682], dtype=np.float32), 
                            'flip': False, 'flip_direction': None, 
                            'img_norm_cfg': {'mean': np.array([123.675, 116.28 , 103.53 ], dtype=np.float32), 
                            'std': np.array([58.395, 57.12 , 57.375], dtype=np.float32), 'to_rgb': True}, 
                            'batch_input_shape': (416, 608)
                        }
                    ]

        result = model([img], [img_metas], return_loss=False)


if __name__ == '__main__':
    ## 1cnn stage
    stage_num = 1
    cfg_file = f'projects/configs/test_speed/{stage_num}stage.py'
    main(cfg_file)