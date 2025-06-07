import torch
import cv2
import os
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models import DETECTORS, build_backbone, build_detector
from mmdet.models.detectors.fcos import FCOS
from mmdet.core import bbox2result
from mmcv import Config

from mmdet.core.visualization import imshow_det_bboxes

PALETTE = [(0, 228, 0), (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
            (106, 0, 228), (0, 60, 100), (220, 20, 60), (0, 80, 100),
            (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
            (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
            (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
            (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
            (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
            (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
            (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
            (134, 134, 103), (145, 148, 174), (255, 208, 186),
            (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
            (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
            (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
            (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
            (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
            (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
            (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
            (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
            (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
            (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
            (191, 162, 208)]

@DETECTORS.register_module(force=True)
class CustomFCOS(FCOS):
    def __init__(self, ori_backbone=None, distill_layer=[0, 1, 2, 3], 
                 distill_weight=1, add_noise=False, fpn_distill=None,
                 ori_channels=[64, 128, 288, 672],
                 teacher_cfg=None, teacher_ckpt=None, **kwargs):
        super(FCOS, self).__init__(**kwargs)
        if ori_backbone is not None:
            out_channels = self.neck.in_channels
            self.ori_backbone = build_backbone(ori_backbone)
            self.distill_layer = distill_layer
            layers = []
            for i in range(len(ori_channels)):
                if i in distill_layer:
                    layers.append(nn.Conv2d(out_channels[i], ori_channels[i], kernel_size=1, padding=0))
            self.adapter = nn.ModuleList(layers)
        
            for name, param in self.named_parameters():
                if 'ori_backbone' in name:
                    param.requires_grad = False
        else:
            self.ori_backbone = None
        self.distill_weight = distill_weight
        self.add_noise = add_noise
        self.fpn_distill = fpn_distill
        
        if teacher_cfg is not None:
            teacher_cfg = Config.fromfile(teacher_cfg)
            self.teacher = build_detector(teacher_cfg.model)
            for name, param in self.named_parameters():
                if 'teacher' in name:
                    param.requires_grad=False
        else:
            self.teacher = None


    def extract_feat(self, img):
        x = self.backbone(img)

        if self.add_noise:
            shape = x[-1].shape
            noise = torch.rand(*shape).to(x[-1].device)
            x = list(x)
            x[-1] = x[-1] + noise

        if self.ori_backbone is not None and self.fpn_distill is None:
            distill_x = self.ori_backbone(img)
            loss_distill = 0
            for i, j in enumerate(self.distill_layer):
                embed_feat = self.adapter[i](x[j])
                loss_distill += F.mse_loss(distill_x[j], embed_feat, reduction='mean')
        else:
            loss_distill = None

        if self.with_neck:
            x = self.neck(x)
        return x, loss_distill

    def extract_teacher_feat(self, img):
        x = self.teacher.backbone(img)
        x = self.teacher.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):

        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

        x, loss_distill = self.extract_feat(img)

        fpn_distill_loss = None
        teacher_outs = None
        if self.teacher is not None:
            if self.fpn_distill is None:
                with torch.no_grad():
                    teacher_x = self.extract_teacher_feat(img)
                    teacher_outs, bbox_preds, centerness = self.teacher.bbox_head(teacher_x)
                    head_outs = [teacher_outs, bbox_preds, centerness]
                    result_list = self.teacher.bbox_head.get_bboxes(**head_outs)

            else:
                if self.fpn_distill == 'cwd':
                    with torch.no_grad():
                        teacher_x = self.extract_teacher_feat(img)
                    loss_module = ChannelWiseDivergence(tau=1, loss_weight=10)
                    fpn_distill_loss = 0
                    for student_x_, teacher_x_ in zip(x, teacher_x):
                        fpn_distill_loss += loss_module(student_x_, teacher_x_)


        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore, teacher_outs=teacher_outs)
        
        if loss_distill is not None:
            losses['loss_distill'] = loss_distill * self.distill_weight
        
        if fpn_distill_loss is not None:
            losses['fpn_distill_loss'] = fpn_distill_loss

        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat, _ = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            (bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes), metas['image_id'][0])
            for (det_bboxes, det_labels), metas in zip(results_list, img_metas)
        ]
        # bbox_pts = [
        #     (self.bbox2result2d(det_bboxes, det_labels, self.img_roi_head.num_classes), image_id)
        #     for (det_bboxes, det_labels), image_id in bbox_pts
        #     ]
        self.show_result(img, bbox_results[0], img_metas=img_metas)
        return bbox_results

    def bbox2result2d(self, bboxes, labels, num_classes):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor | np.ndarray): shape (n, 5)
            labels (torch.Tensor | np.ndarray): shape (n, )
            num_classes (int): class number, including background class

        Returns:
            list(ndarray): bbox results of each class
        """
        if bboxes.shape[0] == 0:
            return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
        else:
            if isinstance(bboxes, torch.Tensor):
                bboxes = bboxes.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
            return [bboxes[labels == i, :] for i in range(num_classes)]

    def show_result(self, data, result, **kwargs):
        """
        data,
        result,
        out_dir=out_dir,
        show=show,
        score_thr=show_score_thr
        """
        cats = [{'id': 0, 'name': 'car'}, {'id': 1, 'name': 'truck'}, {'id': 2, 'name': 'trailer'}, 
               {'id': 3, 'name': 'bus'}, {'id': 4, 'name': 'construction_vehicle'}, {'id': 5, 'name': 'bicycle'}, 
               {'id': 6, 'name': 'motorcycle'}, {'id': 7, 'name': 'pedestrian'}, 
               {'id': 8, 'name': 'traffic_cone'}, {'id': 9, 'name': 'barrier'}]
        class_names = [_['name'] for _ in cats]
        img_metas = kwargs['img_metas']
        token = result[1]
        bbox = result[0]
        img = cv2.imread(img_metas[0]['filename'])
        all_boxes = []
        all_labels = []
        for label, box in enumerate(bbox):
            all_labels.extend([label] * len(box))
            all_boxes.append(box)
        bboxes = np.concatenate(all_boxes)
        all_labels =np.array(all_labels)
        img = imshow_det_bboxes(img, 
                                bboxes, 
                                all_labels, 
                                class_names=class_names, 
                                score_thr=0.2,
                                thickness=2,
                                font_size=13,
                                bbox_color=PALETTE,
                                text_color=PALETTE,
                                )
        
        cv2.imwrite(f'{kwargs.get("show_dir")}/{token}.jpg',img)
