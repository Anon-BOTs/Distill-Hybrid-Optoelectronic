# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import torch
import cv2
import os
import numpy as np
import random
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.utils.misc import locations
from tools.vis_util import show_multicam_bboxes

@DETECTORS.register_module()
class RepDetr3D(MVXTwoStageDetector):
    """RepDetr3D."""

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 use_2d_loss=True,
                 num_frame_head_grads=2,
                 num_frame_backbone_grads=2,
                 num_frame_losses=2,
                 stride=[16],
                 position_level=[0],
                 aux_2d_only=True,
                 single_test=False,
                 pretrained=None):
        super(RepDetr3D, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.prev_scene_token = None
        self.num_frame_head_grads = num_frame_head_grads
        self.num_frame_backbone_grads = num_frame_backbone_grads
        self.num_frame_losses = num_frame_losses
        self.single_test = single_test
        self.stride = stride
        self.position_level = position_level
        self.aux_2d_only = aux_2d_only
        self.test_flag = False
        self.use_2d_loss = use_2d_loss

    def extract_img_feat(self, img, len_queue=1, training_mode=False):
        """Extract features of images."""
        B = img.size(0)

        if img is not None:
            if img.dim() == 6:
                img = img.flatten(1, 2)
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        
        img_feats_reshaped = []

        if self.training or training_mode:
            for i in self.position_level:
                BN, C, H, W = img_feats[i].size()
                img_feat_reshaped = img_feats[i].view(B, len_queue, int(BN/B / len_queue), C, H, W)
                img_feats_reshaped.append(img_feat_reshaped)
        else:
            for i in self.position_level:
                BN, C, H, W = img_feats[i].size()
                img_feat_reshaped = img_feats[i].view(B, int(BN/B/len_queue), C, H, W)
                img_feats_reshaped.append(img_feat_reshaped)

        return img_feats_reshaped


    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, T, training_mode=False):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, T, training_mode)
        return img_feats

    def obtain_history_memory(self,
                            gt_bboxes_3d=None,
                            gt_labels_3d=None,
                            gt_bboxes=None,
                            gt_labels=None,
                            img_metas=None,
                            centers2d=None,
                            depths=None,
                            gt_bboxes_ignore=None,
                            **data):
        losses = dict()
        T = data['img'].size(1)
        num_nograd_frames = T - self.num_frame_head_grads
        num_grad_losses = T - self.num_frame_losses
        for i in range(T):
            requires_grad = False
            return_losses = False
            data_t = dict()
            for key in data:
                if key == 'img_feats':
                    data_t[key] = [feat[:, i] for feat in data[key]]
                else:
                    data_t[key] = data[key][:, i] 

            data_t['img_feats'] = data_t['img_feats']
            if i >= num_nograd_frames:
                requires_grad = True
            if i >= num_grad_losses:
                return_losses = True
            loss = self.forward_pts_train(gt_bboxes_3d[i],
                                        gt_labels_3d[i], gt_bboxes[i],
                                        gt_labels[i], img_metas[i], centers2d[i], depths[i], requires_grad=requires_grad,return_losses=return_losses,**data_t)
            if loss is not None:
                for key, value in loss.items():
                    losses['frame_'+str(i)+"_"+key] = value
        return losses


    def prepare_location(self, img_metas, **data):
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        bs, n = data['img_feats'].shape[:2]
        x = data['img_feats'].flatten(0, 1)
        location = locations(x, self.stride, pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)
        return location

    def forward_roi_head(self, location, img_metas, **data):
        if self.with_img_roi_head:
            outs_roi = self.img_roi_head(location, img_metas, **data)
        else:
            return {'topk_indexes':None}
        return outs_roi


    def forward_pts_train(self,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_bboxes,
                          gt_labels,
                          img_metas,
                          centers2d,
                          depths,
                          requires_grad=True,
                          return_losses=False,
                          **data):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        # location = self.prepare_location(img_metas, **data)
        ## hard code 
        location = None

        if not requires_grad:
            self.eval()
            with torch.no_grad():
                outs = self.pts_bbox_head(location, img_metas, None, **data)
            self.train()

        else:
            data['gt_bboxes'] = gt_bboxes
            data['gt_labels'] = gt_labels
            outs_roi = self.forward_roi_head(location, img_metas, **data)
            # outs_roi_copy = self.img_roi_head.get_bboxes_topk(outs_roi['enc_cls_scores'], 
            #                                                 outs_roi['enc_bbox_preds'], 
            #                                                 outs_roi['memory'], 
            #                                                 img_metas)
            topk_indexes = outs_roi['topk_indexes']
            data['outs_roi'] = outs_roi
            if self.with_pts_bbox:
                outs = self.pts_bbox_head(location, img_metas, topk_indexes, **data)

        if return_losses:
            if self.with_pts_bbox:
                loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
                losses = self.pts_bbox_head.loss(*loss_inputs)
            else:
                losses = dict()

            if self.with_img_roi_head and self.use_2d_loss:
                loss2d_inputs = [gt_bboxes, gt_labels, centers2d, depths, outs_roi, img_metas]
                losses2d = self.img_roi_head.loss(*loss2d_inputs)
                losses.update(losses2d) 

            return losses
        else:
            return None

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
            for key in ['gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'img_metas']:
                data[key] = list(zip(*data[key]))
            return self.forward_train(**data)
        else:
            return self.forward_test(**data)

    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      depths=None,
                      centers2d=None,
                      **data):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.extract_feat
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        if self.test_flag: #for interval evaluation
            self.pts_bbox_head.reset_memory()
            self.test_flag = False
        T = data['img'].size(1)
        prev_img = data['img'][:, :-self.num_frame_backbone_grads]
        rec_img = data['img'][:, -self.num_frame_backbone_grads:]
        rec_img_feats = self.extract_feat(rec_img, self.num_frame_backbone_grads)

        if T-self.num_frame_backbone_grads > 0:
            self.eval()
            with torch.no_grad():
                prev_img_feats = self.extract_feat(prev_img, T-self.num_frame_backbone_grads, True)
            self.train()
            data['img_feats'] = [torch.cat([prev_img_feats[i], rec_img_feats[i]], dim=1) for i in range(len(self.position_level))]
        else:
            data['img_feats'] = rec_img_feats

        losses = self.obtain_history_memory(gt_bboxes_3d,
                        gt_labels_3d, gt_bboxes,
                        gt_labels, img_metas, centers2d, depths, gt_bboxes_ignore, **data)

        return losses
  
  
    def forward_test(self, img_metas, rescale, **data):
        self.test_flag = True
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        for key in data:
            if key != 'img':
                data[key] = data[key][0][0].unsqueeze(0)
            else:
                data[key] = data[key][0]
        return self.simple_test(img_metas[0], **data)

    def simple_test_pts(self, img_metas, **data):
        """Test function of point cloud branch."""
        # location = self.prepare_location(img_metas, **data)
        location = None
        outs_roi = self.forward_roi_head(location, img_metas, **data)
        topk_indexes = outs_roi['topk_indexes']
        data['outs_roi'] = outs_roi

        if self.with_pts_bbox:
            if img_metas[0]['scene_token'] != self.prev_scene_token:
                self.prev_scene_token = img_metas[0]['scene_token']
                data['prev_exists'] = data['img'].new_zeros(1)
                self.pts_bbox_head.reset_memory()
            else:
                data['prev_exists'] = data['img'].new_ones(1)

            outs = self.pts_bbox_head(location, img_metas, topk_indexes, **data)
            bbox_list = self.pts_bbox_head.get_bboxes(
                outs, img_metas)
            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in bbox_list
            ]
            if self.with_img_roi_head:
                bbox_results_2d = self.img_roi_head.get_bboxes(outs_roi['enc_cls_scores'], outs_roi['enc_bbox_preds'], img_metas, rescale=True)
                for bs in range(len(bbox_results)):
                    ## hard code TODO only support for nusc
                    bbox_results[bs]['res_2d'] = bbox_results_2d[bs * 6 : (bs + 1) * 6]
        else:
            bbox_results = self.img_roi_head.get_bboxes(outs_roi['enc_cls_scores'], outs_roi['enc_bbox_preds'], img_metas, rescale=True)
        return bbox_results
    
    def simple_test(self, img_metas, **data):
        """Test function without augmentaiton."""
        data['img_feats'] = self.extract_img_feat(data['img'], 1)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            img_metas, **data)
        if self.with_pts_bbox:
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
            if self.with_img_roi_head:
                for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                    bbox_pts_2d = [
                        (self.bbox2result2d(det_bboxes, det_labels, self.img_roi_head.num_classes), image_id)
                        for (det_bboxes, det_labels), image_id in pts_bbox['res_2d']
                        ]
                    result_dict['pts_bbox']['res_2d'] = bbox_pts_2d
        else:
            bbox_list = [
                (self.bbox2result2d(det_bboxes, det_labels, self.img_roi_head.num_classes), image_id)
                for (det_bboxes, det_labels), image_id in bbox_pts
                ]
        return bbox_list

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

    
    def show_results(self, data, results, **kwargs):
        """
        data,
        result,
        out_dir=out_dir,
        show=show,
        score_thr=show_score_thr
        """
        tasks = [  # 第一个必须是class
            dict(
                task_name = 'class',
                num_out = 10,
                level = 'box',
                names = [
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
                    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
                ]
            ),
        ]

        grid_config = {
            'xbound': [-40, 40, 0.8],
            'ybound': [-40, 40, 0.8],
            'zbound': [-3.0, 5.0, 0.8],
            'dbound': [1.0, 64.0, 1.0]}

        cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK',  'CAM_BACK_RIGHT']
        
        show_dir = '/root/tmp/nusc_vis_3d'
        for bs in range(len(results)):
            result = results[bs]
            filenames = data.get('img_metas')[bs]._data[0][0]['filename']
            filenames = {name.split('/')[4] : name for name in filenames}
            ## for vis aug imgs
            # lidar2cam = data['extrinsics'][bs]._data[0][0].cpu().numpy()
            # intrinsics = data['intrinsics'][bs]._data[0][0][:, :3, :3].cpu().numpy()

            ## for vis ori imgs
            lidar2cam = np.array(data.get('img_metas')[bs]._data[0][0]['ori_extrinsics'])
            intrinsics = np.array(data.get('img_metas')[bs]._data[0][0]['ori_intrinsics'])[:, :3, :3]
            cam2lidar = np.linalg.inv(lidar2cam)
            all_rots = cam2lidar[:, :3, :3]
            all_trans = cam2lidar[:, :3, 3]

            img_shape = data.get('img_metas')[bs]._data[0][0]['img_shape'][0]
            sample_idx = data.get('img_metas')[bs]._data[0][0]['sample_idx']
            # h, w, _ = img_shape
            h, w = 900, 1600

            if 'pts_bbox' in result:
                result = result['pts_bbox']
            ## vis 2d res
            if 'res_2d' in result:
                img_2d_show = np.zeros((h * 2, w * 3, 3))
                for img_i, cam_name in enumerate(cams):
                    index = list(filenames.keys()).index(cam_name)
                    res = result['res_2d'][index]
                    # (res, image_id)
                    res = res[0]
                    filename = filenames[cam_name]
                    img = cv2.imread(filename)
                    # img = np.array(data.get('img_metas')[bs]._data[0][0]['ori_img'][img_i])
                    for label, bbox in enumerate(res):
                        mask = bbox[:, -1] > 0.2
                        bbox = bbox[mask]
                        labels = [label] * len(bbox)

                        for box, label in zip(bbox, labels):
                            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)
                    
                    if img_i < 3:
                        img_2d_show[:h, img_i * w : (img_i + 1) * w, :] = img
                    else:
                        img_2d_show[h:, (img_i - 3) * w : (img_i - 2) * w, :] = img

                # basename = os.path.basename(filename)
                # cv2.imwrite(f'/root/tmp/vis_2d/{sample_idx}',img)
        
            ### vis 3d res
            if 'boxes_3d' in result:
                all_imgs = []
                rots = []
                trans = []
                intrins = []
                for img_i, cam_name in enumerate(cams):
                    index = list(filenames.keys()).index(cam_name)
                    filename = filenames[cam_name]
                    img = cv2.imread(filename)
                    # img = np.array(data.get('img_metas')[0]._data[0][0]['ori_img'][index])
                    all_imgs.append(img)
                    rots.append(all_rots[index])
                    trans.append(all_trans[index])
                    intrins.append(intrinsics[index])

                mask = result['scores_3d'].numpy() > 0.1
                bbox = result['boxes_3d'].tensor.numpy()[mask]
                bbox[:, 2] += bbox[:, 5] / 2
                pred_dict = {
                    'box' : bbox,
                    'class' : result['labels_3d'].numpy()[mask]
                } 

                img_to_show, img_bev = show_multicam_bboxes(all_imgs, intrins, rots, trans, cams,
                                    grid_config, 1, tasks, show_dir, pred_dict,
                                    nuscenes=True)
                img_3d_show = np.concatenate([img_to_show, img_bev], axis=1)
        
                # cv2.imwrite(os.path.join(show_dir, timestamp), img)
        if "img_2d_show" in locals() and "img_3d_show" in locals():
            img_2d_show = cv2.resize(img_2d_show, (1600, 600))
            img_2d_show = np.concatenate([img_2d_show, np.zeros((600, 600, 3))], axis=1)
            all_img_show = np.concatenate([img_3d_show, img_2d_show], axis=0)
            cv2.imwrite(os.path.join(show_dir, sample_idx +'_all.jpg'), all_img_show)
        elif "img_2d_show" in locals():
            cv2.imwrite(os.path.join(show_dir, sample_idx +'_2d.jpg'), img_2d_show)
        elif "img_3d_show" in locals():
            cv2.imwrite(os.path.join(show_dir, sample_idx +'_3d.jpg'), img_3d_show)
