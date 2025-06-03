# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
from .vovnet import VoVNet
from .vovnetcp import VoVNetCP
from .eva_vit import *
from .resnet_onn import ONNResNet
from .regnet_onn import ONNRegNet
from .regnet import RegNet
from .test_regnet import TestRegNet
from .test_mm_regnet import TestMMRegNet
from .mask_convnext import MaskConvNeXt