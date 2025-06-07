# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import math
import torch
import numpy as np
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmcv.runner import BaseModule, Sequential
from torch import nn as nn

from mmdet.models.builder import BACKBONES
from mmdet.models.backbones.resnet import ResNet
from mmdet.models.backbones.resnext import Bottleneck

from .onn_module import ONNLayer

@BACKBONES.register_module()
class ONNMMRegNet(ResNet):
    """RegNet backbone.

    More details can be found in `paper <https://arxiv.org/abs/2003.13678>`_ .

    Args:
        arch (dict): The parameter of RegNets.

            - w0 (int): initial width
            - wa (float): slope of width
            - wm (float): quantization parameter to quantize the width
            - depth (int): depth of the backbone
            - group_w (int): width of group
            - bot_mul (float): bottleneck ratio, i.e. expansion of bottleneck.
        strides (Sequence[int]): Strides of the first block of each stage.
        base_channels (int): Base channels after stem layer.
        in_channels (int): Number of input image channels. Default: 3.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    Example:
        >>> from mmdet.models import RegNet
        >>> import torch
        >>> self = RegNet(
                arch=dict(
                    w0=88,
                    wa=26.31,
                    wm=2.25,
                    group_w=48,
                    depth=25,
                    bot_mul=1.0))
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 96, 8, 8)
        (1, 192, 4, 4)
        (1, 432, 2, 2)
        (1, 1008, 1, 1)
    """
    arch_settings = {
        'regnetx_400mf':
        dict(w0=24, wa=24.48, wm=2.54, group_w=16, depth=22, bot_mul=1.0),
        'regnetx_800mf':
        dict(w0=56, wa=35.73, wm=2.28, group_w=16, depth=16, bot_mul=1.0), ##[64, 128, 288, 672] 16 [1, 3, 7, 5]
        'regnetx_1.6gf':
        dict(w0=80, wa=34.01, wm=2.25, group_w=24, depth=18, bot_mul=1.0),
        'regnetx_3.2gf':
        dict(w0=88, wa=26.31, wm=2.25, group_w=48, depth=25, bot_mul=1.0),
        'regnetx_4.0gf':
        dict(w0=96, wa=38.65, wm=2.43, group_w=40, depth=23, bot_mul=1.0),
        'regnetx_6.4gf':
        dict(w0=184, wa=60.83, wm=2.07, group_w=56, depth=17, bot_mul=1.0),
        'regnetx_8.0gf':
        dict(w0=80, wa=49.56, wm=2.88, group_w=120, depth=23, bot_mul=1.0),
        'regnetx_12gf':
        dict(w0=168, wa=73.36, wm=2.37, group_w=112, depth=19, bot_mul=1.0),
    }

    def __init__(self,
                 arch,
                 in_channels=3,
                 stem_channels=32,
                 base_channels=32,
                 strides=(2, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True,

                 ############## ONN ############
                 delete_extra=False,
                 add_noise=False,
                 use_abs=False,
                 use_bias=False,
                 use_v2=True,
                 use_square=False,
                 onn_cfg=None,
                 onn_stage=[3],
                 up_channels=False,
                 before_downsample=True,
                 onn_channels=None,

                 pretrained=None,
                 init_cfg=None):
        super(ResNet, self).__init__(init_cfg)

        self.onn_cfg = onn_cfg
        self.use_abs = use_abs
        self.use_bias = use_bias
        self.use_square = use_square
        self.onn_stage = onn_stage
        self.use_v2 = use_v2
        self.delete_extra = delete_extra

        # Generate RegNet parameters first
        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'"arch": "{arch}" is not one of the' \
                ' arch_settings'
            arch = self.arch_settings[arch]
        elif not isinstance(arch, dict):
            raise ValueError('Expect "arch" to be either a string '
                             f'or a dict, got {type(arch)}')

        widths, num_stages = self.generate_regnet(
            arch['w0'],
            arch['wa'],
            arch['wm'],
            arch['depth'],
        )
        # Convert to per stage format
        stage_widths, stage_blocks = self.get_stages_from_blocks(widths)

        # Generate group widths and bot muls
        group_widths = [arch['group_w'] for _ in range(num_stages)]
        self.bottleneck_ratio = [arch['bot_mul'] for _ in range(num_stages)]
        # Adjust the compatibility of stage_widths and group_widths
        stage_widths, group_widths = self.adjust_width_group(
            stage_widths, self.bottleneck_ratio, group_widths)

        # Group params by stage
        self.ori_channels = stage_widths
        if onn_channels is not None:
            stage_widths = onn_channels
        self.stage_widths = stage_widths
        self.group_widths = group_widths
        self.depth = sum(stage_blocks)
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.zero_init_residual = zero_init_residual
        self.block = ONNBottleneck
        expansion_bak = self.block.expansion
        self.block.expansion = 1
        self.stage_blocks = stage_blocks[:num_stages]

        self._make_stem_layer(in_channels, stem_channels)

        block_init_cfg = None
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
                if self.zero_init_residual:
                    block_init_cfg = dict(
                        type='Constant', val=0, override=dict(name='norm3'))
        else:
            raise TypeError('pretrained must be a str or None')

        self.add_noise = add_noise
        self.onn_stage = onn_stage
        self.inplanes = stem_channels
        self.res_layers = []
        self.up_modules = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = self.strides[i]
            dilation = self.dilations[i]
            group_width = self.group_widths[i]
            width = int(round(self.stage_widths[i] * self.bottleneck_ratio[i]))
            stage_groups = width // group_width

            dcn = self.dcn if self.stage_with_dcn[i] else None
            if self.plugins is not None:
                stage_plugins = self.make_stage_plugins(self.plugins, i)
            else:
                stage_plugins = None

            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=self.stage_widths[i],
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=self.with_cp,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                dcn=dcn,
                plugins=stage_plugins,
                groups=stage_groups,
                base_width=group_width,
                base_channels=self.stage_widths[i],
                init_cfg=block_init_cfg,

                ### for onn ###
                stage_id=i,
                use_abs=use_abs,
                use_bias=use_bias,
                use_square=use_square,
                delete_extra=delete_extra,
                onn_cfg=onn_cfg,
                onn_stage=onn_stage,
                before_downsample=before_downsample,
                )
            self.inplanes = self.stage_widths[i]
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

            if up_channels and self.ori_channels[i] != self.stage_widths[i]:
                self.up_modules.append(nn.Conv2d(self.stage_widths[i], self.ori_channels[i], bias=True, kernel_size=1))
            else:
                self.up_modules.append(nn.Identity())
            self.up_modules = nn.ModuleList(self.up_modules)

        self._freeze_stages()

        self.feat_dim = stage_widths[-1]
        self.block.expansion = expansion_bak

    def _make_stem_layer(self, in_channels, base_channels):
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            base_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, base_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)

    def generate_regnet(self,
                        initial_width,
                        width_slope,
                        width_parameter,
                        depth,
                        divisor=8):
        """Generates per block width from RegNet parameters.

        Args:
            initial_width ([int]): Initial width of the backbone
            width_slope ([float]): Slope of the quantized linear function
            width_parameter ([int]): Parameter used to quantize the width.
            depth ([int]): Depth of the backbone.
            divisor (int, optional): The divisor of channels. Defaults to 8.

        Returns:
            list, int: return a list of widths of each stage and the number \
                of stages
        """
        assert width_slope >= 0
        assert initial_width > 0
        assert width_parameter > 1
        assert initial_width % divisor == 0
        widths_cont = np.arange(depth) * width_slope + initial_width
        ks = np.round(
            np.log(widths_cont / initial_width) / np.log(width_parameter))
        widths = initial_width * np.power(width_parameter, ks)
        widths = np.round(np.divide(widths, divisor)) * divisor
        num_stages = len(np.unique(widths))
        widths, widths_cont = widths.astype(int).tolist(), widths_cont.tolist()
        return widths, num_stages

    @staticmethod
    def quantize_float(number, divisor):
        """Converts a float to closest non-zero int divisible by divisor.

        Args:
            number (int): Original number to be quantized.
            divisor (int): Divisor used to quantize the number.

        Returns:
            int: quantized number that is divisible by devisor.
        """
        return int(round(number / divisor) * divisor)

    def adjust_width_group(self, widths, bottleneck_ratio, groups):
        """Adjusts the compatibility of widths and groups.

        Args:
            widths (list[int]): Width of each stage.
            bottleneck_ratio (float): Bottleneck ratio.
            groups (int): number of groups in each stage

        Returns:
            tuple(list): The adjusted widths and groups of each stage.
        """
        bottleneck_width = [
            int(w * b) for w, b in zip(widths, bottleneck_ratio)
        ]
        groups = [min(g, w_bot) for g, w_bot in zip(groups, bottleneck_width)]
        bottleneck_width = [
            self.quantize_float(w_bot, g)
            for w_bot, g in zip(bottleneck_width, groups)
        ]
        widths = [
            int(w_bot / b)
            for w_bot, b in zip(bottleneck_width, bottleneck_ratio)
        ]
        return widths, groups

    def get_stages_from_blocks(self, widths):
        """Gets widths/stage_blocks of network at each stage.

        Args:
            widths (list[int]): Width in each stage.

        Returns:
            tuple(list): width and depth of each stage
        """
        width_diff = [
            width != width_prev
            for width, width_prev in zip(widths + [0], [0] + widths)
        ]
        stage_widths = [
            width for width, diff in zip(widths, width_diff[:-1]) if diff
        ]
        stage_blocks = np.diff([
            depth for depth, diff in zip(range(len(width_diff)), width_diff)
            if diff
        ]).tolist()
        return stage_widths, stage_blocks

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        if self.use_v2:
            return ResLayerV2(**kwargs)
        else:
            return ResLayer(**kwargs)

    def forward(self, x):
        """Forward function."""
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if self.up_modules[i] != None:
                x = self.up_modules[i](x)
            if self.add_noise and i in self.onn_stage:
                shape = x.shape
                noise = torch.rand(*shape).to(x.device)
                x = x + noise

            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


class ResLayerV2(Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Default: True
    """

    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 num_blocks,
                 stage_id,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 downsample_first=True,

                 ## for onn
                #  use_abs=False,
                #  use_bias=False,
                #  use_square=False,
                #  delete_extra=True,
                #  onn_cfg=None,
                #  onn_stage=[3],
                 **kwargs):
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []

        if kwargs.get('onn_cfg') is not None:
            onn_cfg = kwargs.get('onn_cfg')
            onn_cfg['N'] = onn_cfg['NMap'][stage_id]
            onn_cfg['nd'] = onn_cfg['ndMap'][stage_id]
            onn_cfg['layersCount'] = onn_cfg['layersCountMap'][stage_id]
            kwargs['onn_cfg'] = onn_cfg

        if downsample_first:
            if stage_id in kwargs.get('onn_stage'):
                use_onn = True
            else:
                use_onn = False

            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    
                    use_onn=use_onn,
                    **kwargs))
            inplanes = planes * block.expansion
            if not kwargs.get('delete_extra') or not use_onn:
                for _ in range(1, num_blocks):
                    layers.append(
                        block(
                            inplanes=inplanes,
                            planes=planes,
                            stride=1,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            use_onn=use_onn,
                            **kwargs))
        else:  # downsample_first=False is for HourglassModule
            for _ in range(num_blocks - 1):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=inplanes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
        super(ResLayerV2, self).__init__(*layers)



class ResLayer(Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Default: True
    """

    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 num_blocks,
                 stage_id,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 downsample_first=True,

                 ## for onn
                #  use_abs=False,
                #  use_bias=False,
                #  use_square=False,
                #  delete_extra=True,
                #  onn_cfg=None,
                #  onn_stage=[3],
                 **kwargs):
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []

        if kwargs.get('onn_cfg') is not None:
            onn_cfg = kwargs.get('onn_cfg')
            onn_cfg['N'] = onn_cfg['NMap'][stage_id]
            onn_cfg['nd'] = onn_cfg['ndMap'][stage_id]
            onn_cfg['layersCount'] = onn_cfg['layersCountMap'][stage_id]
            kwargs['onn_cfg'] = onn_cfg

        if downsample_first:
            if stage_id in kwargs.get('onn_stage'):
                use_onn = True
            else:
                use_onn = False

            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    
                    use_onn=use_onn,
                    **kwargs))
            inplanes = planes * block.expansion
            if not kwargs.get('delete_extra'):
                for _ in range(1, num_blocks):
                    layers.append(
                        block(
                            inplanes=inplanes,
                            planes=planes,
                            stride=1,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            use_onn=use_onn,
                            **kwargs))
        else:  # downsample_first=False is for HourglassModule
            for _ in range(num_blocks - 1):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=inplanes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
        super(ResLayer, self).__init__(*layers)


class ONNBottleneck(BaseModule):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 groups=1,
                 base_width=4,
                 base_channels=64,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,

                 ### for onn
                 use_onn=False,
                 use_abs=False,
                 use_bias=False,
                 use_square=False,
                 onn_cfg=None,
                 before_downsample=True,

                 init_cfg=None,
                 **kwargs):
        super(ONNBottleneck, self).__init__(init_cfg)
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_conv2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            self.after_conv3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv3'
            ]

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if groups == 1:
            width = self.planes
        else:
            width = math.floor(self.planes *
                               (base_width / base_channels)) * groups
        
        self.use_onn = use_onn
        if not use_onn:
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, width, postfix=1)
            self.norm2_name, norm2 = build_norm_layer(
                self.norm_cfg, width, postfix=2)
            self.norm3_name, norm3 = build_norm_layer(
                self.norm_cfg, self.planes * self.expansion, postfix=3)
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                self.inplanes,
                width,
                kernel_size=1,
                stride=self.conv1_stride,
                bias=False)
            self.add_module(self.norm1_name, norm1)
            fallback_on_stride = False
            self.with_modulated_dcn = False
            if self.with_dcn:
                fallback_on_stride = self.dcn.pop('fallback_on_stride', False)
            if not self.with_dcn or fallback_on_stride:
                self.conv2 = build_conv_layer(
                    self.conv_cfg,
                    width,
                    width,
                    kernel_size=3,
                    stride=self.conv2_stride,
                    padding=self.dilation,
                    dilation=self.dilation,
                    groups=groups,
                    bias=False)
            else:
                assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
                self.conv2 = build_conv_layer(
                    self.dcn,
                    width,
                    width,
                    kernel_size=3,
                    stride=self.conv2_stride,
                    padding=self.dilation,
                    dilation=self.dilation,
                    groups=groups,
                    bias=False)

            self.add_module(self.norm2_name, norm2)
            self.conv3 = build_conv_layer(
                self.conv_cfg,
                width,
                self.planes * self.expansion,
                kernel_size=1,
                bias=False)
            self.add_module(self.norm3_name, norm3)
        else:
            print(onn_cfg)
            nd = onn_cfg['nd']
            dx = onn_cfg['dx']
            L = nd * dx
            onn_layer = ONNLayer(inplanes, 
                        planes,
                        onn_cfg['nd'], 
                        L, 
                        onn_cfg['lmbda'], 
                        onn_cfg['d1'], 
                        onn_cfg['d2'], 
                        onn_cfg['N'], 
                        28, 
                        onn_cfg['layersCount'],
                        use_bias,
                        use_abs,
                        use_square)
            if before_downsample:
                self.norm1_name, norm1 = build_norm_layer(
                    self.norm_cfg, planes, postfix=1)
                self.norm2_name, norm2 = build_norm_layer(
                    self.norm_cfg, planes, postfix=2)
                print(groups)
                self.onn_module = nn.Sequential(
                                    onn_layer,
                                    norm1,
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(
                                        planes,
                                        planes,
                                        kernel_size=3,
                                        stride=stride,
                                        padding=1,
                                        groups=groups,
                                        bias=False,
                                    ),
                                    norm2
                                )
            else:
                self.norm1_name, norm1 = build_norm_layer(
                    self.norm_cfg, inplanes, postfix=1)
                self.norm2_name, norm2 = build_norm_layer(
                    self.norm_cfg, planes, postfix=2)
                self.onn_module = nn.Sequential(
                                    nn.Conv2d(
                                        inplanes,
                                        inplanes,
                                        kernel_size=3,
                                        stride=stride,
                                        padding=1,
                                        groups=1,
                                        bias=False,
                                    ),
                                    norm1,
                                    nn.ReLU(inplace=True),
                                    onn_layer,
                                    norm2,
                                )

        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                width, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                width, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                self.planes * self.expansion, self.after_conv3_plugins)

    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(out)
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""
        def _inner_forward(x):
            identity = x
            if not self.use_onn:
                ### 1x1
                out = self.conv1(x)
                out = self.norm1(out)
                out = self.relu(out)

                if self.with_plugins:
                    out = self.forward_plugin(out, self.after_conv1_plugin_names)

                ### 3x3
                out = self.conv2(out)
                out = self.norm2(out)
                out = self.relu(out)

                if self.with_plugins:
                    out = self.forward_plugin(out, self.after_conv2_plugin_names)

                ### 1x1
                out = self.conv3(out)
                out = self.norm3(out)

                if self.with_plugins:
                    out = self.forward_plugin(out, self.after_conv3_plugin_names)
            else:
                out = self.onn_module(x)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out
