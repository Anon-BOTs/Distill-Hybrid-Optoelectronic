
import numpy as np
import mmcv
from mmdet.datasets.builder import PIPELINES
import torch
import cv2
from PIL import Image
from torchvision import transforms


@PIPELINES.register_module()
class LoadVectorImage(object):
    def __init__(self):
        self.transform = transforms.Compose(
                    [
                        transforms.Resize([256, 448], interpolation=transforms.InterpolationMode.BILINEAR),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                    ]
                )

    def __call__(self, results):
        img = Image.open(results['img_filename'])
        img = self.transform(img)
        img = np.array(img).transpose(1, 2, 0)
        results['img'] = img
        results['ego_states'] = np.array([results['speed'], results['yaw']])
        results['ego_action'] = np.array([results['action_speed'], results['action_yaw']])
        return results