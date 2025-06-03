# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from mmcv.cnn import Conv2d, Linear, build_activation_layer
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmcv.runner import force_fp32
from torch.nn.init import normal_
from mmdet.models.utils.transformer import inverse_sigmoid
from projects.mmdet3d_plugin.models.utils.misc import locations

from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean, bbox_overlaps)
from mmdet.models.utils import build_transformer
from mmdet.models.builder import HEADS, build_loss
from mmcv.cnn import (build_activation_layer, 
                      build_conv_layer,
                      build_norm_layer, xavier_init)

from mmcv.cnn.bricks.transformer import build_feedforward_network
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.utils import NormedLinear
from projects.mmdet3d_plugin.models.utils.positional_encoding import pos2posemb3d, pos2posemb1d
from projects.mmdet3d_plugin.models.utils.misc import topk_gather, transform_reference_points
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead


@HEADS.register_module()
class CustomDecoderHead(AnchorFreeHead):
    """Implements the DETR transformer head.

    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    _version = 2

    def __init__(self,
                 num_classes,
                 in_channels,
                 use_3d_pe=False,
                 depth_num=64,
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                 stride=[4, 8, 16, 32, 64],
                 depth_start=1,
                 LID=False,
                 num_query=100,
                 num_reg_fcs=2,
                 with_3d_pe=False,
                 use_cls_embed=False,
                 use_cam_groups=False,
                 use_topk=False,
                 use_deformable=False,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(
                             type='IoUCost', iou_mode='giou', weight=2.0))),
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        # if class_weight is not None and (self.__class__ is CustomDecoderHead):
        if class_weight is not None:
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            # assert loss_cls['loss_weight'] == assigner['cls_cost']['weight'], \
            #     'The classification weight for loss and matcher should be' \
            #     'exactly the same.'
            # assert loss_bbox['loss_weight'] == assigner['reg_cost'][
            #     'weight'], 'The regression L1 weight for loss and matcher ' \
            #     'should be exactly the same.'
            # assert loss_iou['loss_weight'] == assigner['iou_cost']['weight'], \
            #     'The regression iou weight for loss and matcher should be' \
            #     'exactly the same.'
            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        if loss_cls.get('type') == 'QualityFocalLoss':
            self.use_quality = True
        else:
            self.use_quality = False
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)
        self.use_cls_embed = use_cls_embed
        self.use_topk = use_topk
        self.use_cam_groups = use_cam_groups
        self.use_deformable = use_deformable

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.activate = build_activation_layer(self.act_cfg)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
            f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
            f' and {num_feats}.'

        if self.use_cls_embed:
            self.cls_embed = nn.Parameter(torch.Tensor(
            self.cls_out_channels, self.embed_dims))


        self.use_3d_pe = use_3d_pe
        if self.use_3d_pe:
            self.pc_range = nn.Parameter(torch.tensor(
                pc_range), requires_grad=False)
            self.position_range = nn.Parameter(torch.tensor(
                position_range), requires_grad=False)
            self.position_dim = depth_num * 3
            self.depth_num = depth_num
            self.depth_start = depth_start
            self.stride = stride

            norm_cfg=dict(type='LN')
            ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=256,
                    feedforward_channels=1024,
                    num_fcs=2,
                    ffn_drop=0.,
                    act_cfg=dict(type='ReLU', inplace=True),
                )
            self.extra_norms = ModuleList()
            self.extra_ffns = ModuleList()
            extra_num_norms = 2
            for _ in range(extra_num_norms):
                self.extra_norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])
            
            if 'embed_dims' not in ffn_cfgs:
                ffn_cfgs['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs['embed_dims'] == self.embed_dims
            self.extra_ffns.append(
                build_feedforward_network(ffn_cfgs, dict(type='FFN')))

            if LID:
                index  = torch.arange(start=0, end=self.depth_num, step=1).float()
                index_1 = index + 1
                bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
                coords_d = self.depth_start + bin_size * index * index_1
            else:
                index  = torch.arange(start=0, end=self.depth_num, step=1).float()
                bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
                coords_d = self.depth_start + bin_size * index

            self.coords_d = nn.Parameter(coords_d, requires_grad=False)

            self.position_encoder = nn.Sequential(
                    nn.Linear(self.position_dim, self.embed_dims*4),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims*4, self.embed_dims),
                )

        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        if not self.use_deformable:
            self.input_proj = Conv2d(
                self.in_channels, self.embed_dims, kernel_size=1)
            self.fc_cls = Linear(self.embed_dims, self.cls_out_channels)
            self.reg_ffn = FFN(
                self.embed_dims,
                self.embed_dims,
                self.num_reg_fcs,
                self.act_cfg,
                dropout=0.0,
                add_residual=False)
            self.fc_reg = Linear(self.embed_dims, 4)
            self.query_embedding = nn.Embedding(self.num_query, self.embed_dims)
        else:
            num_pred = self.transformer.decoder.num_layers
            fc_cls = Linear(self.embed_dims, self.cls_out_channels)
            reg_branch = []
            for _ in range(self.num_reg_fcs):
                reg_branch.append(Linear(self.embed_dims, self.embed_dims))
                reg_branch.append(nn.ReLU())
            reg_branch.append(Linear(self.embed_dims, 4))
            reg_branch = nn.Sequential(*reg_branch)
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

            self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.transformer.init_weights()
        if self.use_cls_embed:
            normal_(self.cls_embed)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.

        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        # if (version is None or version < 2) and self.__class__ is CustomDecoderHead:
        if (version is None or version < 2):
            convert_dict = {
                '.self_attn.': '.attentions.0.',
                '.ffn.': '.ffns.0.',
                '.multihead_attn.': '.attentions.1.',
                '.decoder.norm.': '.decoder.post_norm.'
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k and 'roi_head' in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]

        super(AnchorFreeHead,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)

    def position_embeding(self, data, memory_centers, topk_indexes, img_metas):
        eps = 1e-5
        BN, H, W, _ = memory_centers.shape
        B = data['intrinsics'].size(0)

        intrinsic = torch.stack([data['intrinsics'][..., 0, 0], data['intrinsics'][..., 1, 1]], dim=-1)
        intrinsic = torch.abs(intrinsic) / 1e3
        intrinsic = intrinsic.repeat(1, H*W, 1).view(BN, -1, 2)
        LEN = intrinsic.size(1)

        num_sample_tokens = topk_indexes.size(1) if topk_indexes is not None else LEN

        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        memory_centers[..., 0] = memory_centers[..., 0] * pad_w
        memory_centers[..., 1] = memory_centers[..., 1] * pad_h

        D = self.coords_d.shape[0]

        memory_centers = memory_centers.detach().view(BN, LEN, 1, 2)
        topk_centers = topk_gather(memory_centers, topk_indexes).repeat(1, 1, D, 1)
        coords_d = self.coords_d.view(1, 1, D, 1).repeat(BN, num_sample_tokens, 1 , 1)
        coords = torch.cat([topk_centers, coords_d], dim=-1)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        coords = coords.unsqueeze(-1)

        img2lidars = data['lidar2img'].inverse()
        img2lidars = img2lidars.view(BN, 1, 1, 4, 4).repeat(1, H*W, D, 1, 1).view(BN, LEN, D, 4, 4)
        img2lidars = topk_gather(img2lidars, topk_indexes)

        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        # if self.trans_to_global:
        #     N = BN // B
        #     ego_pose = data['ego_pose'].repeat_interleave(N, 0)
        #     coords3d = coords3d.view(BN, -1, 3)
        #     coords3d = transform_reference_points(coords3d, ego_pose)
        #     coords3d = coords3d.view(BN, LEN, D, 3)
        #     position_range = torch.stack([self.position_range[0:3], self.position_range[3:6]]).unsqueeze(0).repeat_interleave(BN, 0)
        #     position_range = transform_reference_points(position_range, ego_pose).reshape(BN, 1, 1, -1)
        #     coords3d[..., 0:3] = (coords3d[..., 0:3] - position_range[..., 0:3]) / (position_range[..., 3:6] - position_range[..., 0:3])
        # else:
        coords3d[..., 0:3] = (coords3d[..., 0:3] - self.position_range[0:3]) / (self.position_range[3:6] - self.position_range[0:3])
        coords3d = coords3d.reshape(BN, -1, D*3)
      
        pos_embed  = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(pos_embed)
        intrinsic = topk_gather(intrinsic, topk_indexes)

        # for spatial alignment in focal petr
        cone = torch.cat([intrinsic, coords3d[..., -3:], coords3d[..., -90:-87]], dim=-1)

        return coords_position_embeding, cone

    def prepare_location(self, stride, img_feats):
        ## hard code when use ori imgs, pad shape should be input TODO@lifang
        bn, _, pad_h, pad_w = img_feats.shape
        # x = img_feats.flatten(0, 1)
        x = img_feats
        location = locations(x, stride, pad_h, pad_w)[None].repeat(bn, 1, 1, 1)
        return location

    def forward(self, locations, img_metas, **data):
        """Forward function.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[list[Tensor], list[Tensor]]: Outputs for all scale levels.

                - all_cls_scores_list (list[Tensor]): Classification scores \
                    for each scale level. Each is a 4D-tensor with shape \
                    [nb_dec, bs, num_query, cls_out_channels]. Note \
                    `cls_out_channels` should includes background.
                - all_bbox_preds_list (list[Tensor]): Sigmoid regression \
                    outputs for each scale level. Each is a 4D-tensor with \
                    normalized coordinate format (cx, cy, w, h) and shape \
                    [nb_dec, bs, num_query, 4].
        """
        feats = data['img_feats']
        # num_levels = len(feats)
        # img_metas_list = [img_metas for _ in range(num_levels)]
        return self.forward_single(feats, img_metas, data)

    def forward_single(self, x, img_metas, data):
        """"Forward function for a single feature level.

        Args:
            x (Tensor): Input feature from backbone's single stage, shape
                [bs, c, h, w].
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format (cx, cy, w, h).
                Shape [nb_dec, bs, num_query, 4].
        """
        # construct binary masks which used for the transformer.
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.
        if isinstance(x, list):
            B, N, C, H, W = x[0].shape
        else:
            B, N, C, H, W = x.shape
        query_embed = self.query_embedding.weight
        new_img_metas = []
        for i in range(B):
            img_metas[i]['img_shape'] = img_metas[i]['img_shape'][0]
            for _ in range(6):
                new_img_metas.append(img_metas[i])

        input_img_h, input_img_w, _ = new_img_metas[0]['img_shape']
        if not self.use_deformable:
            B, N, C, H, W = x.shape
            num_tokens = H * W
            masks = x.new_ones((B * N, input_img_h, input_img_w))
            for img_id in range(B * N):
                img_h, img_w, _ = new_img_metas[img_id]['img_shape']
                masks[img_id, :img_h, :img_w] = 0
            ## hard code TODO
            if self.use_cam_groups:
                assert self.num_query // 6
                query_embed = query_embed.reshape(-1, 6, self.embed_dims)
                query_embed = query_embed.repeat(1, B, 1)
            else:
                query_embed = query_embed.unsqueeze(1).repeat(
                    1, B * N, 1)  # [num_query, dim] -> [num_query, bs, dim]

            x = x.reshape(-1, C, H, W)
            x = self.input_proj(x)
            _, C, H, W = x.shape
            # interpolate masks to have the same spatial shape with x
            masks = F.interpolate(
                masks.unsqueeze(1), size=x.shape[-2:]).to(torch.bool).squeeze(1)
            # position encoding
            pos_embed = self.positional_encoding(masks)  # [bs, embed_dim, h, w]
            if not self.use_3d_pe:
                x = x.permute(2, 3, 0, 1).reshape(num_tokens, B * N, C)
                # outs_dec: [nb_dec, bs, num_query, embed_dim]
                outs_dec = self.transformer(x, masks, query_embed,
                                            pos_embed)
            else:
                assert len(self.stride) == 1
                locations = self.prepare_location(self.stride[0], x)
                x = x.permute(2, 3, 0, 1).reshape(num_tokens, B * N, C)
                pos_embed_3d, cone = self.position_embeding(data, locations, None, img_metas)
                pos_embed_3d = pos_embed_3d.permute(1, 0, 2)
                outs_dec = self.transformer(x, masks, query_embed,
                                            pos_embed, pos_embed_3d)

            cls_scores = self.fc_cls(outs_dec)
            bbox_preds = self.fc_reg(self.activate(
                self.reg_ffn(outs_dec))).sigmoid()

            # if self.use_cls_embed:
            #     NL, BN, NQ, _ = cls_scores.shape
            #     labels = torch.argmax(cls_scores.sigmoid(), dim=-1)
            #     class_embeding = self.cls_embed[labels.reshape(-1)].reshape(NL, BN, NQ, self.embed_dims)
            #     outs_dec = outs_dec + class_embeding

        else:
            mlvl_feats = x
            mlvl_feats = [feat.flatten(0, 1) for feat in mlvl_feats]
            batch_size = mlvl_feats[0].size(0)
            img_masks = mlvl_feats[0].new_ones(
                (batch_size, input_img_h, input_img_w))
            for img_id in range(batch_size):
                img_h, img_w, _ = new_img_metas[img_id]['img_shape']
                img_masks[img_id, :img_h, :img_w] = 0

            if self.use_3d_pe:
                mlvl_pe3d = []
                assert len(self.stride) == len(mlvl_feats)
                for stride, img_feat in zip(self.stride, mlvl_feats):
                    locations = self.prepare_location(stride, img_feat)
                    mlvl_pe3d.append(self.position_embeding(data, locations, None, img_metas)[0])
            else:
                mlvl_pe3d = None

            mlvl_masks = []
            mlvl_positional_encodings = []
            for feat in mlvl_feats:
                mlvl_masks.append(
                    F.interpolate(img_masks[None],
                                size=feat.shape[-2:]).to(torch.bool).squeeze(0))
                mlvl_positional_encodings.append(
                    self.positional_encoding(mlvl_masks[-1]))

            outs_dec, init_reference, inter_references, \
                enc_outputs_class, enc_outputs_coord = self.transformer(
                        mlvl_feats,
                        mlvl_masks,
                        query_embed,
                        mlvl_positional_encodings,
                        reg_branches=None,  # noqa:E501
                        cls_branches=None,  # noqa:E501
                        pos_embed_3d=mlvl_pe3d,
                )
            outs_dec = outs_dec.permute(0, 2, 1, 3)
            outputs_classes = []
            outputs_coords = []

            for lvl in range(outs_dec.shape[0]):
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
                outputs_class = self.cls_branches[lvl](outs_dec[lvl])
                tmp = self.reg_branches[lvl](outs_dec[lvl])
                if reference.shape[-1] == 4:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 2
                    tmp[..., :2] += reference
                outputs_coord = tmp.sigmoid()
                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)

            cls_scores = torch.stack(outputs_classes)
            bbox_preds = torch.stack(outputs_coords)

        if self.use_3d_pe:
            query_with_3d_pe = self.transformer.query_with_3d_pe
            query_with_3d_pe = self.extra_norms[0](query_with_3d_pe)
            query_with_3d_pe = self.extra_ffns[0](
                    query_with_3d_pe, None)
            query_with_3d_pe = self.extra_norms[1](query_with_3d_pe)
            memory_2d = query_with_3d_pe
        else:
            memory_2d = outs_dec

        outs = {
                'enc_cls_scores': cls_scores,
                'enc_bbox_preds': bbox_preds,
                'memory': memory_2d,
                'topk_indexes' : None,
            }

        return outs

    def get_bboxes_topk(self,
                        all_cls_scores_list,
                        all_bbox_preds_list,
                        outs_dec_list,
                        img_metas):

        if not self.use_topk:
            outs = {
                'enc_cls_scores': all_cls_scores_list,
                'enc_bbox_preds': all_bbox_preds_list,
                'memory': outs_dec_list,
                'topk_indexes' : None,
            }
            return outs

        assert len(all_cls_scores_list) == len(all_bbox_preds_list)
        cls_scores = all_cls_scores_list[-1]
        bbox_preds = all_bbox_preds_list[-1]
        outs_decs = outs_dec_list[-1]
        img_metas = [img_metas[0] for _ in range(len(cls_scores))]

        bbox_list = []
        labels_list = []
        outs_dec_list = []
        scores_list = []
        for img_id in range(len(cls_scores)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            outs_dec = outs_decs[img_id]
            max_per_img = self.train_cfg.get('max_per_img', self.num_query)
            # exclude background
            if self.loss_cls.use_sigmoid:
                cls_score = cls_score.sigmoid()
                scores, indexes = cls_score.view(-1).topk(max_per_img)
                det_labels = indexes % self.num_classes
                bbox_index = indexes // self.num_classes
                bbox_pred = bbox_pred[bbox_index]
                outs_dec = outs_decs[bbox_index]
            else:
                scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
                scores, bbox_index = scores.topk(max_per_img)
                bbox_pred = bbox_pred[bbox_index]
                det_labels = det_labels[bbox_index]
                outs_dec = outs_dec[bbox_index]
            bbox_list.append(bbox_pred)
            labels_list.append(det_labels)
            outs_dec_list.append(outs_dec)
            scores_list.append(scores)

        bboxes = torch.stack(bbox_list)
        labels = torch.stack(labels_list)
        outs_dec = torch.stack(outs_dec_list)
        scores = torch.stack(scores_list)

        outs = {
                'enc_cls_scores': scores.unsqueeze(0),
                'enc_bbox_preds': bboxes.unsqueeze(0),
                'memory': outs_dec.unsqueeze(0),
                'topk_indexes' : None,
            }
        return outs
            

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             centers,
             depth,
             outs_roi,
             img_metas,
             gt_bboxes_ignore=None):
        """"Loss function.

        Only outputs from the last feature level are used for computing
        losses by default.

        Args:
            all_cls_scores_list (list[Tensor]): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds_list (list[Tensor]): Sigmoid regression
                outputs for each feature level. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # NOTE defaultly only the outputs from the last feature scale is used.
        all_cls_scores = outs_roi['enc_cls_scores']
        all_bbox_preds = outs_roi['enc_bbox_preds']
        assert gt_bboxes_ignore is None, \
            'Only supports for gt_bboxes_ignore setting to None.'

        gt_bboxes_list = [bboxes2d for i in gt_bboxes_list for bboxes2d in i]
        gt_labels_list = [labels2d for i in gt_labels_list for labels2d in i]

        if all_cls_scores.shape[1] % 6 == 0:
            bs = all_cls_scores.shape[1] // 6
        else:
            bs = all_cls_scores.shape[1]
        new_img_metas = []
        for i in range(bs):
            for _ in range(6):
                new_img_metas.append(img_metas[i])


        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        img_metas_list = [new_img_metas for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, img_metas_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['2d_loss_cls'] = losses_cls[-1]
        loss_dict['2d_loss_bbox'] = losses_bbox[-1]
        loss_dict['2d_loss_iou'] = losses_iou[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls[:-1],
                                                       losses_bbox[:-1],
                                                       losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.2d_loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.2d_loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.2d_loss_iou'] = loss_iou_i
            num_dec_layer += 1
        return loss_dict

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           img_metas, gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if self.use_quality:
            bg_class_ind = self.num_classes
            pos_inds = ((labels >= 0)
                        & (labels < bg_class_ind)).nonzero().squeeze(1)
            scores = label_weights.new_zeros(labels.shape)
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
            pos_bbox_pred = bbox_preds.reshape(-1, 4)[pos_inds]
            pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
            scores[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)
            loss_cls = self.loss_cls(
                cls_scores, (labels, scores),
                weight=label_weights,
                avg_factor=cls_avg_factor)
        else:
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            tuple: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_bboxes,
                           gt_labels,
                           img_meta,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            img_meta (dict): Meta information for one image.
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, img_meta,
                                             gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    # over-write because img_metas are needed as inputs for bbox_head.
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        outs = self(x, img_metas)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def get_bboxes(self,
                   all_cls_scores_list,
                   all_bbox_preds_list,
                   img_metas,
                   rescale=False):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            all_cls_scores_list (list[Tensor]): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds_list (list[Tensor]): Sigmoid regression
                outputs for each feature level. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Default False.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple. \
                The first item is an (n, 5) tensor, where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1. The second item is a \
                (n,) tensor where each item is the predicted class label of \
                the corresponding box.
        """
        # NOTE defaultly only using outputs from the last feature level,
        # and only the outputs from the last decoder layer is used.
        assert len(img_metas) == 1
        cls_scores = all_cls_scores_list[-1]
        bbox_preds = all_bbox_preds_list[-1]
        img_metas = [img_metas[0] for _ in range(len(cls_scores))]

        result_list = []
        for img_id in range(len(cls_scores)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            image_id = img_metas[img_id]['image_id'][img_id % 6]
            crop = img_metas[img_id].get('crop', (0, 0, 0, 0))
            proposals = self._get_bboxes_single(cls_score, bbox_pred,
                                                img_shape, scale_factor,
                                                crop,
                                                rescale)
            result_list.append((proposals, image_id))
        return result_list

    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           img_shape,
                           scale_factor,
                           crop,
                           rescale=False):
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_query, 4].
            img_shape (tuple[int]): Shape of input image, (height, width, 3).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool, optional): If True, return boxes in original image
                space. Default False.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels.

                - det_bboxes: Predicted bboxes with shape [num_query, 5], \
                    where the first 4 columns are bounding box positions \
                    (tl_x, tl_y, br_x, br_y) and the 5-th column are scores \
                    between 0 and 1.
                - det_labels: Predicted labels of the corresponding box with \
                    shape [num_query].
        """
        assert len(cls_score) == len(bbox_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] + crop[1]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0] + crop[1])
        if rescale:
            ### hard code TODO
            scale_factor = [704 / 1600, 396 / 900, 704 / 1600, 396 / 900]
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)

        return det_bboxes, det_labels

    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """
        # forward of this head requires img_metas
        outs = self.forward(feats, img_metas)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        return results_list

    def forward_onnx(self, feats, img_metas):
        """Forward function for exporting to ONNX.

        Over-write `forward` because: `masks` is directly created with
        zero (valid position tag) and has the same spatial size as `x`.
        Thus the construction of `masks` is different from that in `forward`.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[list[Tensor], list[Tensor]]: Outputs for all scale levels.

                - all_cls_scores_list (list[Tensor]): Classification scores \
                    for each scale level. Each is a 4D-tensor with shape \
                    [nb_dec, bs, num_query, cls_out_channels]. Note \
                    `cls_out_channels` should includes background.
                - all_bbox_preds_list (list[Tensor]): Sigmoid regression \
                    outputs for each scale level. Each is a 4D-tensor with \
                    normalized coordinate format (cx, cy, w, h) and shape \
                    [nb_dec, bs, num_query, 4].
        """
        num_levels = len(feats)
        img_metas_list = [img_metas for _ in range(num_levels)]
        return multi_apply(self.forward_single_onnx, feats, img_metas_list)

    def forward_single_onnx(self, x, img_metas):
        """"Forward function for a single feature level with ONNX exportation.

        Args:
            x (Tensor): Input feature from backbone's single stage, shape
                [bs, c, h, w].
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format (cx, cy, w, h).
                Shape [nb_dec, bs, num_query, 4].
        """
        # Note `img_shape` is not dynamically traceable to ONNX,
        # since the related augmentation was done with numpy under
        # CPU. Thus `masks` is directly created with zeros (valid tag)
        # and the same spatial shape as `x`.
        # The difference between torch and exported ONNX model may be
        # ignored, since the same performance is achieved (e.g.
        # 40.1 vs 40.1 for DETR)
        batch_size = x.size(0)
        h, w = x.size()[-2:]
        masks = x.new_zeros((batch_size, h, w))  # [B,h,w]

        x = self.input_proj(x)
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks.unsqueeze(1), size=x.shape[-2:]).to(torch.bool).squeeze(1)
        pos_embed = self.positional_encoding(masks)
        outs_dec, _ = self.transformer(x, masks, self.query_embedding.weight,
                                       pos_embed)

        all_cls_scores = self.fc_cls(outs_dec)
        all_bbox_preds = self.fc_reg(self.activate(
            self.reg_ffn(outs_dec))).sigmoid()
        return all_cls_scores, all_bbox_preds

    def onnx_export(self, all_cls_scores_list, all_bbox_preds_list, img_metas):
        """Transform network outputs into bbox predictions, with ONNX
        exportation.

        Args:
            all_cls_scores_list (list[Tensor]): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds_list (list[Tensor]): Sigmoid regression
                outputs for each feature level. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            img_metas (list[dict]): Meta information of each image.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        assert len(img_metas) == 1, \
            'Only support one input image while in exporting to ONNX'

        cls_scores = all_cls_scores_list[-1][-1]
        bbox_preds = all_bbox_preds_list[-1][-1]

        # Note `img_shape` is not dynamically traceable to ONNX,
        # here `img_shape_for_onnx` (padded shape of image tensor)
        # is used.
        img_shape = img_metas[0]['img_shape_for_onnx']
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        batch_size = cls_scores.size(0)
        # `batch_index_offset` is used for the gather of concatenated tensor
        batch_index_offset = torch.arange(batch_size).to(
            cls_scores.device) * max_per_img
        batch_index_offset = batch_index_offset.unsqueeze(1).expand(
            batch_size, max_per_img)

        # supports dynamical batch inference
        if self.loss_cls.use_sigmoid:
            cls_scores = cls_scores.sigmoid()
            scores, indexes = cls_scores.view(batch_size, -1).topk(
                max_per_img, dim=1)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
            bbox_index = (bbox_index + batch_index_offset).view(-1)
            bbox_preds = bbox_preds.view(-1, 4)[bbox_index]
            bbox_preds = bbox_preds.view(batch_size, -1, 4)
        else:
            scores, det_labels = F.softmax(
                cls_scores, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img, dim=1)
            bbox_index = (bbox_index + batch_index_offset).view(-1)
            bbox_preds = bbox_preds.view(-1, 4)[bbox_index]
            det_labels = det_labels.view(-1)[bbox_index]
            bbox_preds = bbox_preds.view(batch_size, -1, 4)
            det_labels = det_labels.view(batch_size, -1)

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_preds)
        # use `img_shape_tensor` for dynamically exporting to ONNX
        img_shape_tensor = img_shape.flip(0).repeat(2)  # [w,h,w,h]
        img_shape_tensor = img_shape_tensor.unsqueeze(0).unsqueeze(0).expand(
            batch_size, det_bboxes.size(1), 4)
        det_bboxes = det_bboxes * img_shape_tensor
        # dynamically clip bboxes
        x1, y1, x2, y2 = det_bboxes.split((1, 1, 1, 1), dim=-1)
        from mmdet.core.export import dynamic_clip_for_onnx
        x1, y1, x2, y2 = dynamic_clip_for_onnx(x1, y1, x2, y2, img_shape)
        det_bboxes = torch.cat([x1, y1, x2, y2], dim=-1)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(-1)), -1)

        return det_bboxes, det_labels
