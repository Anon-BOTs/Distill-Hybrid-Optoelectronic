from mmdet.models import DETECTORS, build_backbone
from mmdet.models.detectors.fcos import FCOS
from mmdet.core import bbox2result
import torch.nn as nn
import torch.nn.functional as F

@DETECTORS.register_module(force=True)
class DistillONN(FCOS):
    def __init__(self, ori_backbone=None, **kwargs):
        super(DistillONN, self).__init__(**kwargs)
        if ori_backbone is not None:
            ori_channels = [64, 128, 288, 672]
            out_channels = self.neck.in_channels
            self.ori_backbone = build_backbone(ori_backbone)
            layers = []
            for i in range(len(ori_channels)):
                layers.append(nn.Conv2d(out_channels[i], ori_channels[i], kernel_size=1, padding=0))
            self.adapter = nn.ModuleList(layers)
        
            for name, param in self.named_parameters():
                if 'ori_backbone' in name:
                    param.requires_grad = False
                if 'neck' in name or 'bbox_head' in name:
                    param.requires_grad = False
        else:
            self.ori_backbone = None
    
    def extract_feat(self, img):
        x = self.backbone(img)

        if self.ori_backbone is not None:
            distill_x = self.ori_backbone(img)
            loss_distill = 0
            for i in range(len(self.adapter)):
                embed_feat = self.adapter[i](x[i])
                loss_distill += F.mse_loss(distill_x[i], embed_feat, reduction='mean')
        else:
            loss_distill = None
        if self.with_neck:
            x = self.neck(x)
        return x, loss_distill
    
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
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        losses = {}
        if loss_distill is not None:
            losses['loss_distill'] = loss_distill
        return losses