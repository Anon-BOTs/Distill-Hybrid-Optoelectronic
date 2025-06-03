import abc
from typing import List, Union

import numpy as np
import torch
from torch import nn
# from torchtoolbox.network.modules.backbones.regnet import _regnet

from mmdet.models.builder import BACKBONES

class Backbone(nn.Module, abc.ABC):
    def __init__(
        self, out_features_stride: Union[None, list], channels: List, strides: List
    ):
        """Basic backbone impl for cls and det.

        Args:
            out_features_stride: needed out strides.
            features: main backbone(need to impl).
            strides: features strides.
            channels: features channels.
        """

        super(Backbone, self).__init__()
        if out_features_stride is not None:
            out_features_stride = sorted(list(set(out_features_stride)))

        self._out_features_stride = out_features_stride
        self._out_index = None
        self._out_channels = None

        self.features = None
        self.channels = channels
        self.strides = strides

        self._set_out_index()
        self._set_out_channels()

    def _set_out_channels(self):
        if self._out_channels is not None:
            return
        elif self._out_features_stride is not None:
            self._out_channels = []
            for idx in self._out_index:
                self._out_channels.append(self.channels[idx])
        else:
            return

    def _set_out_index(self):
        if self._out_index is not None:
            return
        elif self._out_features_stride is not None:
            self._out_index = []
            for i in range(len(self.strides)):
                if self.strides[i] in self._out_features_stride and (
                    i == len(self.strides) - 1 or self.strides[i + 1] != self.strides[i]
                ):
                    self._out_index.append(i)
        else:
            return

    # call this when finish build model(init) if you want.
    def check_structures(self):
        assert (
            self.features is not None
            and self.channels is not None
            and self.strides is not None
        ), "features, channels and strides must be defined."

        assert (
            len(self.channels) == len(self.strides) == len(self.features)
        ), "The length of channels, strides and features should be same."
        if self._out_features_stride is not None:
            assert (
                len(self.out_feature_index)
                == len(self.out_features_stride)
                == len(self.out_channels)
            ), (
                "length out_index: {len(self.out_feature_index)}, length out_strides: {len(self.out_features_stride)},"
                "length out_channels: {self.out_channels} should be same."
            )

    @property
    def out_features_stride(self):
        return self._out_features_stride

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def out_feature_index(self):
        return self._out_index

    def coll_forward(self, x):
        coll = []
        print(len(self.features))
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_feature_index:
                coll.append(x)
        return coll

    @abc.abstractmethod
    def forward(self, x):
        pass


# from torchtoolbox.tools import make_divisible
def make_divisible(v, divisible_by=8, min_value=None):
    """
    This function is taken from the original tf repo.
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisible_by
    new_v = max(min_value, int(v + divisible_by / 2) // divisible_by * divisible_by)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisible_by
    return new_v


class Stem(nn.Module):
    def __init__(self, in_c, out_c, norm=nn.BatchNorm2d, fast_downsample=False):
        super(Stem, self).__init__()
        self.block = nn.Sequential(
            (
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1, bias=False)
                if not fast_downsample
                else nn.Conv2d(in_c, out_c, kernel_size=4, stride=4, bias=False)
            ),
            norm(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SE(nn.Module):
    def __init__(self, in_c, reduction_ratio=0.25):
        super(SE, self).__init__()
        reducation_c = int(in_c * reduction_ratio)
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, reducation_c, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reducation_c, in_c, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)


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
    ):
        super(Stage, self).__init__()
        width = make_divisible(out_c * bottleneck_ratio)
        groups = width // group_width

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

        if in_c != out_c or stride != 1:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                norm(out_c),
            )
        else:
            self.skip_connection = nn.Identity()

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        skip = self.skip_connection(x)
        x = self.block(x)
        x = self.act(x + skip)
        return x


class Head(nn.Module):
    def __init__(self, in_c, out_c):
        super(Head, self).__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_c, out_c, bias=True)
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
        valid_layers=[0, 1, 2, 3],
        **kwargs,
    ):
        strides = [2, 4, 8, 16, 32]
        channels = [32] + w
        super(Regnet, self).__init__(out_strides, channels, strides)
        self.reduction_ratio = 0.25 if se else 0
        self.bottleneck_ratio = b
        self.group_width = g
        stem_c = 32

        print(valid_layers)
        layers = []
        stem = Stem(3, stem_c, norm=norm, fast_downsample=fast_downsample)
        layers.append(stem)
        if 0 in valid_layers:
            stage1 = self._make_layer(stem_c, w[0], d[0], 2, norm)
            layers.append(stage1)
        if 1 in valid_layers:
            stage2 = self._make_layer(w[0], w[1], d[1], 2, norm)
            layers.append(stage2)
        if 2 in valid_layers:
            stage3 = self._make_layer(w[1], w[2], d[2], 2, norm)
            layers.append(stage3)
        if 3 in valid_layers:
            stage4 = self._make_layer(w[2], w[3], d[3], 2, norm)
            layers.append(stage4)
        self.features = nn.Sequential(*layers)

        if self.out_features_stride is None:
            self.head = Head(w[3], num_classes)

    def _make_layer(self, in_c, out_c, blocks, stride=2, norm=nn.BatchNorm2d):
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
            )
        )
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
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.out_features_stride is None:
            x = self.features(x)
            x = self.head(x)
            return x
        else:
            return self.coll_forward(x)


_regnetx_config = {
    "200MF": {"d": [1, 1, 4, 7], "w": [24, 56, 152, 368], "g": 8},
    "400MF": {"d": [1, 2, 7, 12], "w": [32, 64, 160, 384], "g": 16},
    "600MF": {"d": [1, 3, 5, 7], "w": [48, 96, 240, 528], "g": 24},
    "800MF": {"d": [1, 3, 7, 5], "w": [64, 128, 288, 672], "g": 16},
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

@BACKBONES.register_module(force=True)
class TestRegNet(nn.Module):
    def __init__(self, choice, pretrained_path=None, freeze=False, **kwargs):
        super(TestRegNet, self).__init__()
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
            self.regnet.load_state_dict(model_dict)

    def forward(self, x):
        out = self.regnet(x)
        # todo: @hanbing, temp writing
        # if len(out) == 4:
        #     return out[2:]
        return out

    def train(self, mode=True):
        super(TestRegNet, self).train(mode)

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

