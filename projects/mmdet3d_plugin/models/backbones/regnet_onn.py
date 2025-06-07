import abc
import copy
from typing import List, Union

import numpy as np
import torch
from torch import nn
from .onn_module import ONNLayer
from .regnet import Backbone, SE, make_divisible
from mmdet.models.builder import BACKBONES

class Stage(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        stride,
        bottleneck_ratio,
        group_width,
        reduction_ratio=0,
        norm=nn.BatchNorm2d,
        quantization_mode=False,
        onn_cfg=None,
        stage_index=0,
        onn_stage=[3],
        use_bias=False,
        use_abs=False,
        use_square=False,
    ):
        super().__init__()
        width = make_divisible(out_c * bottleneck_ratio)
        groups = width // group_width

        if stage_index not in onn_stage:
            self.block = nn.Sequential(
                nn.Conv2d(in_c, width, kernel_size=1, bias=False),
                norm(width),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    width,
                    width,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=groups,
                    bias=False,
                ),
                norm(width),
                nn.ReLU(inplace=True),
                SE(width, reduction_ratio) if reduction_ratio != 0 else nn.Identity(),
                nn.Conv2d(width, out_c, kernel_size=1, bias=False),
                norm(out_c),
            )
        else:
            if onn_cfg is not None:
                config = copy.deepcopy(onn_cfg)
                config['N'] = config['N'][stage_index]
                config['nd'] = config['ndMap'][stage_index]
                config['layersCount'] = config['layersCountMap'][stage_index]
            else:
                num_map = {0 : 200, 1 : 100, 2 : 50, 3 : 50}
                layers_count_map = {0 : 4, 1 : 4, 2 : 2, 3 : 2}
                config = {"conv1_out_channels": 2,
                        "conv2_out_channels": 4, 
                        "lmbda": 640e-9,  ## 1050e-9
                        "dx": 1.75e-6,
                        "nd": 50, # 50 ## ~2 * N
                        "d1": 100e-6,
                        "d2": 100e-6,
                        "N": 50,  # 50
                        "layersCount": layers_count_map[stage_index]
                }
            nd = config['nd']
            dx = config['dx']
            L = nd * dx
            self.block = nn.Sequential(
                            ONNLayer(in_c, 
                                    out_c,
                                    config['nd'], 
                                    L, 
                                    config['lmbda'], 
                                    config['d1'], 
                                    config['d2'], 
                                    config['N'], 
                                    28, 
                                    config['layersCount'],
                                    use_bias,
                                    use_abs,
                                    use_square),
                            norm(out_c),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(
                                out_c,
                                out_c,
                                kernel_size=3,
                                stride=stride,
                                padding=1,
                                groups=groups,
                                bias=False,
                            ),
                            norm(out_c)
                        )

        if in_c != out_c or stride != 1:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                norm(out_c),
                nn.Identity(),
            )
        else:
            self.skip_connection = (
                nn.Identity()
            )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        skip = self.skip_connection(x)
        x = self.block(x)
        x = self.act(x + skip)
        return x

class Stem(nn.Module):
    def __init__(self, in_c, out_c, norm=nn.BatchNorm2d, fast_downsample=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1, bias=False)
            if not fast_downsample
            else nn.Conv2d(in_c, out_c, kernel_size=4, stride=4, bias=False),
            norm(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class Regnet(Backbone):
    def __init__(
        self,
        d,
        w,
        g,
        num_classes=1000,
        b=1,
        se=False,
        out_strides=None,
        norm=nn.BatchNorm2d,
        fast_downsample: bool = False,
        quantization_mode: bool = False,
        onn_cfg=None,
        onn_stage = [],
        delete_extra = False,
        use_bias = False,
        use_abs = False,
        use_v2=True,
        use_square = False,
    ):
        strides = [2, 4, 8, 16, 32]
        channels = [32] + w
        super(Regnet, self).__init__(out_strides, channels, strides)
        self.reduction_ratio = 0.25 if se else 0
        self.bottleneck_ratio = b
        self.group_width = g
        stem_c = 32
        self.quantization_mode = quantization_mode
        self.onn_stage = onn_stage
        self.delete_extra = delete_extra
        self.use_bias = use_bias
        self.use_abs = use_abs
        self.use_square = use_square
        self.use_v2 = use_v2

        stem = Stem(3, stem_c, norm=norm, fast_downsample=fast_downsample)
        stage1 = self._make_layer(stem_c, w[0], d[0], 2, norm, onn_cfg, stage_index=0)
        stage2 = self._make_layer(w[0], w[1], d[1], 2, norm, onn_cfg, stage_index=1)
        stage3 = self._make_layer(w[1], w[2], d[2], 2, norm, onn_cfg, stage_index=2)
        stage4 = self._make_layer(w[2], w[3], d[3], 2, norm, onn_cfg, stage_index=3)
        self.features = nn.Sequential(stem, stage1, stage2, stage3, stage4)


    def _make_layer(self, in_c, out_c, blocks, stride=2, norm=nn.BatchNorm2d, onn_cfg=None, stage_index=0):
        layers = []
        layers.append(
            Stage(
                in_c,
                out_c,
                stride,
                self.bottleneck_ratio,
                self.group_width,
                self.reduction_ratio,
                norm,
                quantization_mode=self.quantization_mode,
                onn_cfg=onn_cfg,
                stage_index=stage_index,
                onn_stage=self.onn_stage,
                use_bias=self.use_bias,
                use_abs=self.use_abs,
                use_square=self.use_square,
            )
        )
        if not self.delete_extra or (self.use_v2 and stage_index not in self.onn_stage):
            for _ in range(1, blocks):
                layers.append(
                    Stage(
                        out_c,
                        out_c,
                        1,
                        self.bottleneck_ratio,
                        self.group_width,
                        self.reduction_ratio,
                        norm,
                        quantization_mode=self.quantization_mode,
                        onn_cfg=onn_cfg,
                        stage_index=stage_index,
                        onn_stage=self.onn_stage,
                        use_bias=self.use_bias,
                        use_abs=self.use_abs,
                        use_square=self.use_square,
                    )
                )
        return nn.Sequential(*layers)

    def coll_forward(self, x):
        coll = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_feature_index:
                coll.append(x)
        return coll

    def forward(self, x):
        return self.coll_forward(x)


_regnetx_config = {
    "200MF": {"d": [1, 1, 4, 7], "w": [24, 56, 152, 368], "g": 8},
    "200MFONN": {"d": [1, 1, 4, 7], "w": [24, 56, 152, 368], "g": 8},
    "400MF": {"d": [1, 2, 7, 12], "w": [32, 64, 160, 384], "g": 16},
    "600MF": {"d": [1, 3, 5, 7], "w": [48, 96, 240, 528], "g": 24},
    "800MF": {"d": [1, 3, 7, 5], "w": [64, 128, 288, 672], "g": 16},
    "800MFONNV1": {"d": [1, 3, 7, 5], "w": [64, 128, 288, 256], "g": 16},
    "800MFONNV1.1": {"d": [1, 3, 7, 5], "w": [64, 128, 288, 128], "g": 16},
    "800MFONNV1.2": {"d": [1, 3, 7, 5], "w": [64, 128, 288, 32], "g": 16},
    "800MFONNV1.3": {"d": [1, 3, 7, 5], "w": [64, 128, 288, 16], "g": 16},
    "800MFONNV2": {"d": [1, 3, 7, 5], "w": [64, 128, 160, 256], "g": 16},
    "800MFONNV3": {"d": [1, 3, 7, 5], "w": [64, 96, 96, 96], "g": 16},
    "1.6GF": {"d": [2, 4, 10, 2], "w": [72, 168, 408, 912], "g": 24},
    "3.2GF": {"d": [2, 6, 15, 2], "w": [96, 192, 432, 1008], "g": 48},
    "4.0GF": {"d": [2, 5, 14, 2], "w": [80, 240, 560, 1360], "g": 40},
    "6.4GF": {"d": [2, 4, 10, 1], "w": [168, 392, 784, 1624], "g": 56},
    "8.0GF": {"d": [2, 5, 15, 1], "w": [80, 240, 720, 1920], "g": 120},
    "12GF": {"d": [2, 5, 11, 1], "w": [224, 448, 896, 2240], "g": 112},
    "16GF": {"d": [2, 6, 13, 1], "w": [256, 512, 896, 2048], "g": 128},
    "32GF": {"d": [2, 7, 13, 1], "w": [336, 672, 1344, 2520], "g": 168},
}

_regnety_config = {
    "200MF": {"d": [1, 1, 4, 7], "w": [24, 56, 152, 368], "g": 8},
    "400MF": {"d": [1, 3, 6, 6], "w": [48, 104, 208, 440], "g": 8},
    "600MF": {"d": [1, 3, 7, 4], "w": [48, 112, 256, 608], "g": 16},
    "800MF": {"d": [1, 3, 8, 2], "w": [64, 128, 320, 768], "g": 16},
    "1.6GF": {"d": [2, 6, 17, 2], "w": [48, 120, 336, 888], "g": 24},
    "3.2GF": {"d": [2, 5, 13, 1], "w": [72, 216, 576, 1512], "g": 24},
    "4.0GF": {"d": [2, 6, 12, 2], "w": [128, 192, 512, 1088], "g": 64},
    "6.4GF": {"d": [2, 7, 14, 2], "w": [144, 288, 576, 1296], "g": 72},
    "8.0GF": {"d": [2, 4, 10, 1], "w": [168, 448, 896, 2016], "g": 56},
    "12GF": {"d": [2, 5, 11, 1], "w": [224, 448, 896, 2240], "g": 112},
    "16GF": {"d": [2, 4, 11, 1], "w": [224, 448, 1232, 3024], "g": 112},
    "32GF": {"d": [2, 5, 12, 1], "w": [232, 696, 1392, 3712], "g": 232},
}

def _regnet(name, b=1, se=False, **kwargs):
    config = _regnetx_config[name] if not se else _regnety_config[name]

    d, w, g = config["d"], config["w"], config["g"]
    return Regnet(d, w, g, b=b, se=se, **kwargs)

@BACKBONES.register_module()
class ONNRegNet(nn.Module):
    def __init__(self, choice, pretrained_path=None, freeze=False, ignore_keys=[], **kwargs):
        super(ONNRegNet, self).__init__()
        self.pretrained_path = pretrained_path
        self.regnet = _regnet(choice, **kwargs)
        self.freeze = freeze

        if self.pretrained_path is not None:
            print(
                "load {} regnet pretrained model from file: {}".format(
                    choice, pretrained_path
                )
            )
            pretrained_dict = torch.load(self.pretrained_path)
            if "state_dict" in pretrained_dict:
                pretrained_dict = pretrained_dict["state_dict"]
            elif "model" in pretrained_dict:
                pretrained_dict = pretrained_dict["model"]

            if len(ignore_keys) > 0:
                new_pretrained_dict = {}
                for key, v in pretrained_dict.items():
                    flag = True
                    for k in ignore_keys:
                        if k in key:
                            flag = False
                    if flag:
                        new_pretrained_dict[key] = v

                pretrained_dict = new_pretrained_dict

            model_dict = self.regnet.state_dict()
            load_model_dict = {}
            for key in model_dict:
                find_key = "module." + key
                if find_key in pretrained_dict:
                    load_model_dict[key] = pretrained_dict[find_key]
                elif key in pretrained_dict:
                    load_model_dict[key] = pretrained_dict[key]
                else:
                    print(
                        "=> can't load regnet key {} from pretrained model".format(key)
                    )
            model_dict.update(load_model_dict)
            self.regnet.load_state_dict(model_dict, strict=False)

    def forward(self, x):
        out = self.regnet(x)
        # todo: @hanbing, temp writing
        # if len(out) == 4:
        #     return out[2:]
        return out

    def train(self, mode=True):
        super(ONNRegNet, self).train(mode)

        if self.freeze:
            for param in self.regnet.parameters():
                param.requires_grad = False

            # freeze bn
            for module in self.regnet.modules():
                if isinstance(module, nn.BatchNorm2d):
                    if hasattr(module, "weight"):
                        module.weight.requires_grad_(False)
                    if hasattr(module, "bias"):
                        module.bias.requires_grad_(False)
                    module.eval()

