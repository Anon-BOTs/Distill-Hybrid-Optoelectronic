import warnings

import torch
import torch.nn as nn
import numpy as np

from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models.detectors.base import BaseDetector
from projects.mmdet_plugin.models.co_detr import CoDETR
from mmcv.runner import force_fp32, auto_fp16

@DETECTORS.register_module()
class CustomCoDETR(CoDETR):
    def extract_feat(self, img, img_metas=None):
        """Directly extract features from the backbone+neck."""
        if img.dim() == 5:
            img = img.flatten(0, 1)
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x
    
    @force_fp32(apply_to=('img'))
    def forward(self, return_loss=True, **data):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            for key in ['gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'img_metas', 'img']:
                data[key] = data[key][0]
            return self.forward_train(**data)
        else:
            return self.forward_test(**data)

    def forward_test(self, img, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(img, 'img'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(img)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(img)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        new_img_metas = []
        for cur_img, img_meta in zip(img, img_metas):
            img_metas_copy = []
            for meta in img_meta:
                meta['batch_input_shape'] = cur_img.shape[-2:]
                meta['img_shape'] = meta['img_shape'][0]
                meta['pad_shape'] = meta['pad_shape'][0]
                ## hard code TODO@lifang
                for _ in range(6):
                    img_metas_copy.append(meta)
            new_img_metas.append(img_metas_copy)
        img_metas = new_img_metas

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(img[0], img_metas[0], **kwargs)
        else:
            assert img[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{img[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(img, img_metas, **kwargs)


    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        gt_bboxes = [bboxes2d for i in gt_bboxes for bboxes2d in i]
        gt_labels = [labels2d for i in gt_labels for labels2d in i]
        batch_input_shape = tuple(img[0].size()[-2:])
        img_metas_copy = []
        for meta in img_metas:
            meta['batch_input_shape'] = batch_input_shape
            meta['img_shape'] = meta['img_shape'][0]
            meta['pad_shape'] = meta['pad_shape'][0]
            ## hard code TODO@lifang
            for _ in range(6):
                img_metas_copy.append(meta)
        img_metas = img_metas_copy

        if not self.with_attn_mask: # remove attn mask for LSJ
            for i in range(len(img_metas)):
                input_img_h, input_img_w = img_metas[i]['batch_input_shape']
                img_metas[i]['img_shape'] = [input_img_h, input_img_w, 3]

        x = self.extract_feat(img, img_metas)

        losses = dict()
        def upd_loss(losses, idx, weight=1):
            new_losses = dict()
            for k,v in losses.items():
                new_k = '{}{}'.format(k,idx)
                if isinstance(v,list) or isinstance(v,tuple):
                    new_losses[new_k] = [i*weight for i in v]
                else:new_losses[new_k] = v*weight
            return new_losses

        # DETR encoder and decoder forward
        if self.with_query_head:
            bbox_losses, x = self.query_head.forward_train(x, img_metas, gt_bboxes,
                                                          gt_labels, gt_bboxes_ignore)
            losses.update(bbox_losses)
            

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg[self.head_idx].get('rpn_proposal',
                                              self.test_cfg[self.head_idx].rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        positive_coords = []
        for i in range(len(self.roi_head)):
            roi_losses = self.roi_head[i].forward_train(x, img_metas, proposal_list,
                                                    gt_bboxes, gt_labels,
                                                    gt_bboxes_ignore, gt_masks,
                                                    **kwargs)
            if self.with_pos_coord:
                positive_coords.append(roi_losses.pop('pos_coords'))
            else: 
                if 'pos_coords' in roi_losses.keys():
                    tmp = roi_losses.pop('pos_coords')     
            roi_losses = upd_loss(roi_losses, idx=i)
            losses.update(roi_losses)
            
        for i in range(len(self.bbox_head)):
            bbox_losses = self.bbox_head[i].forward_train(x, img_metas, gt_bboxes,
                                                        gt_labels, gt_bboxes_ignore)
            if self.with_pos_coord:
                pos_coords = bbox_losses.pop('pos_coords')
                positive_coords.append(pos_coords)
            else:
                if 'pos_coords' in bbox_losses.keys():
                    tmp = bbox_losses.pop('pos_coords')          
            bbox_losses = upd_loss(bbox_losses, idx=i+len(self.roi_head))
            losses.update(bbox_losses)

        if self.with_pos_coord and len(positive_coords)>0:
            for i in range(len(positive_coords)):
                bbox_losses = self.query_head.forward_train_aux(x, img_metas, gt_bboxes,
                                                            gt_labels, gt_bboxes_ignore, positive_coords[i], i)
                bbox_losses = upd_loss(bbox_losses, idx=i)
                losses.update(bbox_losses)                    

        return losses