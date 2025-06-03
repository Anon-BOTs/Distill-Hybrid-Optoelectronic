# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from mmcv.cnn import Conv2d, Linear, build_activation_layer
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmcv.runner import force_fp32
from torch.nn.init import normal_

from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils import build_transformer
from mmdet.models.builder import HEADS, build_loss
from .custom_decoder_2d_head import CustomDecoderHead
from mmcv.cnn import (build_activation_layer, 
                      build_conv_layer,
                      build_norm_layer, xavier_init)

from mmcv.cnn.bricks.transformer import build_feedforward_network
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.utils import NormedLinear
from projects.mmdet3d_plugin.models.utils.positional_encoding import pos2posemb3d, pos2posemb1d
from projects.mmdet3d_plugin.models.utils.misc import topk_gather, transform_reference_points
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from .custom_decoder_2d_head import CustomDecoderHead

@HEADS.register_module()
class CustomDecoder3DHead(CustomDecoderHead):
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
                 use_3d_pe=True,
                 depth_num=64,
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                 depth_start=1,
                 LID=False,
                 trans_to_global=False,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        super(CustomDecoder3DHead, self).__init__(**kwargs)

        self.pc_range = nn.Parameter(torch.tensor(
            pc_range), requires_grad=False)

        self.position_range = nn.Parameter(torch.tensor(
            position_range), requires_grad=False)
        self.position_dim = depth_num * 3
        self.depth_num = depth_num
        self.depth_start = depth_start
        self.trans_to_global = trans_to_global
        self.use_3d_pe = use_3d_pe
        if self.use_3d_pe:
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
        # self.input_proj = Conv2d(
        #     in_channels, self.embed_dims, kernel_size=1)
        
        # self.positional_encoding = build_positional_encoding(
        #     positional_encoding)

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
        if self.trans_to_global:
            N = BN // B
            ego_pose = data['ego_pose'].repeat_interleave(N, 0)
            coords3d = coords3d.view(BN, -1, 3)
            coords3d = transform_reference_points(coords3d, ego_pose)
            coords3d = coords3d.view(BN, LEN, D, 3)
            position_range = torch.stack([self.position_range[0:3], self.position_range[3:6]]).unsqueeze(0).repeat_interleave(BN, 0)
            position_range = transform_reference_points(position_range, ego_pose).reshape(BN, 1, 1, -1)
            coords3d[..., 0:3] = (coords3d[..., 0:3] - position_range[..., 0:3]) / (position_range[..., 3:6] - position_range[..., 0:3])
        else:
            coords3d[..., 0:3] = (coords3d[..., 0:3] - self.position_range[0:3]) / (self.position_range[3:6] - self.position_range[0:3])
        coords3d = coords3d.reshape(BN, -1, D*3)
      
        pos_embed  = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(pos_embed)
        intrinsic = topk_gather(intrinsic, topk_indexes)

        # for spatial alignment in focal petr
        cone = torch.cat([intrinsic, coords3d[..., -3:], coords3d[..., -90:-87]], dim=-1)

        return coords_position_embeding, cone


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
        feats = data.pop('img_feats')
        # num_levels = len(feats)
        # img_metas_list = [img_metas for _ in range(num_levels)]
        return self.forward_single(feats, locations, data, img_metas)

    def forward_single(self, x, locations, data, img_metas):
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
        batch_size = x.size(0)
        B, N, C, H, W = x.shape
        num_tokens = H * W
        query_embed = self.query_embedding.weight
        query_embed = query_embed.unsqueeze(1).repeat(
            1, B * N, 1)  # [num_query, dim] -> [num_query, bs, dim]

        pos_embed_3d, cone = self.position_embeding(data, locations, None, img_metas)
        pos_embed_3d = pos_embed_3d.permute(1, 0, 2)

        new_img_metas = []
        for i in range(B):
            img_metas[i]['img_shape'] = img_metas[i]['img_shape'][0]
            for _ in range(6):
                new_img_metas.append(img_metas[i])

        input_img_h, input_img_w, _ = new_img_metas[0]['img_shape']
        masks = x.new_ones((B * N, input_img_h, input_img_w))
        for img_id in range(B * N):
            img_h, img_w, _ = new_img_metas[img_id]['img_shape']
            masks[img_id, :img_h, :img_w] = 0

        x = x.reshape(-1, C, H, W)
        x = self.input_proj(x)
        _, C, H, W = x.shape
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks.unsqueeze(1), size=x.shape[-2:]).to(torch.bool).squeeze(1)
        x = x.permute(2, 3, 0, 1).reshape(num_tokens, B * N, C)
        # position encoding
        pos_embed = self.positional_encoding(masks)  # [bs, embed_dim, h, w]
        # outs_dec: [nb_dec, bs, num_query, embed_dim]
        outs_dec = self.transformer(x, masks, query_embed,
                                       pos_embed, pos_embed_3d)

        cls_scores = self.fc_cls(outs_dec)
        bbox_preds = self.fc_reg(self.activate(
            self.reg_ffn(outs_dec))).sigmoid()

        if self.use_3d_pe:
            query_with_3d_pe = self.transformer.query_with_3d_pe
            query_with_3d_pe = self.extra_norms[0](query_with_3d_pe)
            query_with_3d_pe = self.extra_ffns[0](
                    query_with_3d_pe, None)
            query_with_3d_pe = self.extra_norms[1](query_with_3d_pe)
            memory_2d = query_with_3d_pe
        else:
            memory_2d = outs_dec

        # if self.use_cls_embed:
        #     NL, BN, NQ, _ = cls_scores.shape
        #     labels = torch.argmax(cls_scores.sigmoid(), dim=-1)
        #     class_embeding = self.cls_embed[labels.reshape(-1)].reshape(NL, BN, NQ, self.embed_dims)
        #     outs_dec = outs_dec + class_embeding

        outs = {
                'enc_cls_scores': cls_scores,
                'enc_bbox_preds': bbox_preds,
                'memory': memory_2d,
                'topk_indexes' : None,
                'use_2d_box' : False
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
                'use_2d_box' : False,
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
            