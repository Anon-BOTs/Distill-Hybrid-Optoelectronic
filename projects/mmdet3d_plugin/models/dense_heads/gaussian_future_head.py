import torch
import copy
from torch import nn
import torch.nn.functional as F
import numpy as np
from mmcv.runner.base_module import BaseModule
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import HEADS
from mmcv.cnn import xavier_init
import pickle
from projects.mmdet3d_plugin.models import utils
from einops import rearrange
from torch.autograd import Variable
from math import exp
from mmdet3d.models.builder import build_neck, NECKS, build_head
from mmdet.models.losses.utils import weight_reduce_loss

from .. import modules
from .utils import get_bev_grids, bev_coords_to_grids, bev_grids_to_coordinates
from scipy.spatial import KDTree
from projects.mmdet3d_plugin.models.utils import sparse_utils

def get_multi_opa_label(pts, sampled_pts, pc_range, voxel_size):
    sampled_pts = sampled_pts.reshape(-1, 3)
    pc_range = torch.tensor(pc_range).to(pts)
    voxel_size = torch.tensor(voxel_size).to(pts)
    occ_shape = (pc_range[3:] - pc_range[:3]) / voxel_size
    occ = torch.zeros((int(occ_shape[0]), int(occ_shape[1]), int(occ_shape[2]))).to(pts)
    index = ((pts[:, :3] - pc_range[:-3]) // voxel_size).long()
    occ[index[:, 0], index[:, 1], index[:, 2]] = 1

    opa_label = torch.zeros((len(sampled_pts))).to(pts)

    mask_x = torch.logical_and(sampled_pts[:, 0] > pc_range[0], sampled_pts[:, 0] < pc_range[3])
    mask_y = torch.logical_and(sampled_pts[:, 1] > pc_range[1], sampled_pts[:, 1] < pc_range[4])
    mask_z = torch.logical_and(sampled_pts[:, 2] > pc_range[2] + 0.1, sampled_pts[:, 2] < pc_range[5] - 0.1)

    mask = torch.logical_and(mask_x, mask_y)
    mask = torch.logical_and(mask, mask_z)

    sampled_pts = sampled_pts[mask]

    index = ((sampled_pts[:, :3] - pc_range[:-3]) // voxel_size).long()
    label = occ[index[:, 0], index[:, 1], index[:, 2]]
    opa_label[mask] = label
    return opa_label

def get_one_opa_label(sampled_pts, anchor_gaussian_interval=100):
    sampled_opacity_gt = ((sampled_pts[:,2] - 1)/(60.0 / anchor_gaussian_interval)).long()
    sampled_opacity_gt = F.one_hot(sampled_opacity_gt, num_classes=anchor_gaussian_interval)
    return sampled_opacity_gt

def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # pred_sigmoid = pred.sigmoid()
    pred_sigmoid = pred         # 我修改了

    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

class LiftChannel(BaseModule):
    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        lift_z=5,
        bev_h=None,
        bev_w=None,
    ):
        super().__init__()
        # self.lift_cond = nn.Sequential(
        #                 nn.conv2
        #                 BasicBlock(in_channels, mid_channels),
        #                 BasicBlock(mid_channels, out_channels*lift_z)
        #                 )
        self.lift_z = lift_z
        self.bev_h = bev_h
        self.bev_w = bev_w

        self.lift_cond = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels*lift_z, 1),
                nn.BatchNorm2d(mid_channels*lift_z),
                nn.ReLU(True),
                nn.Conv2d(mid_channels*lift_z, mid_channels*lift_z, 3, padding=1),
                nn.BatchNorm2d(mid_channels*lift_z),
                nn.ReLU(True),
                nn.Conv2d(mid_channels*lift_z, out_channels*lift_z, 1))

        # self.final_conv1 = ConvModule(
        #     in_channels,
        #     out_channels * lift_z // 2,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1,
        #     bias=True,
        #     conv_cfg=dict(type='Conv2d')
        # )
        # self.final_conv2 = ConvModule(
        #     out_channels * lift_z // 2,
        #     out_channels * lift_z,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1,
        #     bias=True,
        #     conv_cfg=dict(type='Conv2d')
        # )

        # self.final_conv3 = nn.Conv2d(out_channels * lift_z, out_channels * lift_z, 1)

    def forward(self, bev_feats):
        if self.bev_h is not None and bev_feats.dim()==3:
            b, n, c = bev_feats.shape
            assert self.bev_h * self.bev_w == n

            bev_feats = bev_feats.reshape(b, self.bev_h, self.bev_w, c)
            bev_feats = bev_feats.permute(0, 3, 1, 2).contiguous()

        lift_feats = self.lift_cond(bev_feats)
        # lift_feats = self.final_conv1(bev_feats)
        # lift_feats = self.final_conv2(lift_feats)
        # lift_feats = self.final_conv3(lift_feats)
        bs, c, h, w = lift_feats.shape
        lift_feats = lift_feats.reshape(bs, c // self.lift_z, self.lift_z, h, w)
        return lift_feats

@HEADS.register_module()
class GaussianFutureHead(BaseModule):
    def __init__(
        self,
        in_channels,
        unified_voxel_size,
        unified_voxel_shape,
        pc_range,
        render_conv_cfg,
        view_cfg,
        gs_cfg,
        loss_cfg=None,
        cam_nums=6,
        ray_sampler_cfg=None,
        all_depth=False,
        use_semantic_guide=False,
        opa_one_hot=True,
        fusion_layer=None,
        future_pred_num=1,
        future_pred_head=None,
        frame_loss_weight=[1.0],
        liftfeats_cfg = None,
        **kwargs
    ):
        super().__init__()
        if kwargs.get("fp16_enabled", False):
            self.fp16_enabled = fp16_enabled
        self.all_depth = all_depth
        self.in_channels = in_channels
        self.pc_range = np.array(pc_range, dtype=np.float32)
        self.unified_voxel_shape = np.array(unified_voxel_shape, dtype=np.int32)
        self.unified_voxel_size = np.array(unified_voxel_size, dtype=np.float32)
        self.ray_sampler_cfg = ray_sampler_cfg
        if view_cfg is not None:
            vtrans_type = view_cfg.pop("type", "Uni3DViewTrans")
            self.view_trans = getattr(utils, vtrans_type)(
                pc_range=self.pc_range,
                voxel_size=self.unified_voxel_size,
                voxel_shape=self.unified_voxel_shape,
                **view_cfg
            )  # max pooling, deformable detr, bilinear

        self.gs_param_regresser = getattr(modules, gs_cfg['type'])(**gs_cfg)

        self.render_conv = nn.Sequential(
            nn.Conv3d(
                in_channels,
                render_conv_cfg["out_channels"],
                kernel_size=render_conv_cfg["kernel_size"],
                padding=render_conv_cfg["padding"],
                stride=1,
            ),
            nn.BatchNorm3d(render_conv_cfg["out_channels"]),
            nn.ReLU(inplace=True),
        )

        # self.voxel_grid = self.create_voxel_grid()
        self.create_voxel_grid()
        self.pts_mask = torch.rand_like(self.voxel_grid) > 0.95
        
        self.loss_cfg = loss_cfg
        self.cam_nums = cam_nums
        self.use_semantic_guide = use_semantic_guide
        self.opa_one_hot = opa_one_hot

        if fusion_layer is not None:
            self.fusion_layer = build_neck(fusion_layer)
        else:
            self.fusion_layer = None

        self.future_pred_num = future_pred_num
        if future_pred_head is not None:
            self.future_pred_head = build_head(future_pred_head)
            self.bev_w = self.future_pred_head.bev_w
            self.bev_h = self.future_pred_head.bev_h
            self.point_cloud_range = self.future_pred_head.pc_range
            assert len(frame_loss_weight) == self.future_pred_num + 1
            self.frame_loss_weight = frame_loss_weight
        else:
            self.future_pred_head = None

        self.lift_feats = LiftChannel(**liftfeats_cfg) if liftfeats_cfg is not None else None
    

    def create_voxel_grid(self,):
        # 计算体素网格的范围
        x_range = np.linspace(self.pc_range[0], self.pc_range[3], self.unified_voxel_shape[0], endpoint=False) + self.unified_voxel_size[0] / 2
        y_range = np.linspace(self.pc_range[1], self.pc_range[4], self.unified_voxel_shape[1], endpoint=False) + self.unified_voxel_size[1] / 2
        z_range = np.linspace(self.pc_range[2], self.pc_range[5], self.unified_voxel_shape[2], endpoint=False) + self.unified_voxel_size[2] / 2

        # 生成所有体素的中心点坐标
        xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing='ij')
        voxel_centers_coors = np.concatenate((xx[:, :, :, None], yy[:, :, :, None], zz[:, :, :, None]), axis=-1)   # 注意这里是三个冒号
        voxel_centers_coors = torch.tensor(voxel_centers_coors, dtype=torch.float32)
        
        # self.register_buffer('voxel_grid', voxel_centers_coors.view(-1, 3))   # 之前的有问题；feats.shape [Z, H, W]
        self.register_buffer('voxel_grid', voxel_centers_coors.permute(2, 0, 1, 3).contiguous().view(-1, 3))

        xx_2d, yy_2d = torch.meshgrid([ torch.arange(0, 1600, 5), torch.arange(0, 900, 5)] )
        frustum = torch.stack([xx_2d, yy_2d, torch.ones_like(yy_2d), torch.ones_like(yy_2d)], dim=-1).view(-1, 4).to(torch.float32)  # (W, H, D, 3)
        self.register_buffer('frustum', frustum)


    @force_fp32(apply_to=("preds_dict", "targets"))
    def loss(self, preds_dicts, targets):
        loss_dict = {}
        for i in range(self.future_pred_num + 1):
            preds_dict = preds_dicts[i]
            target = targets[i]
            depth_pred = preds_dict["img_depth"].permute(0, 1, 3, 4, 2)
            # depth_gt = targets["img_depth"]
            depth_gt = torch.stack( [ t["img_depth"] for t in target], dim=0)

            rgb_pred = preds_dict["img_rgb"].permute(0, 1, 3, 4, 2)
            rgb_gt = torch.stack( [ t["img_rgb"] for t in target], dim=0)
            valid_gt_mask = torch.stack( [ t["rgb_mask"] for t in target], dim=0).squeeze(2)
            valid_depth_gt_mask = torch.stack( [ t["depth_mask"] for t in target], dim=0).squeeze(2)[..., :1]

            bs, n, h, w, c = rgb_pred.shape

            rgb_gt = rgb_gt[:, :, :h, :w, ...]
            valid_gt_mask = valid_gt_mask[:, :, :h, :w, ...]
            depth_gt = depth_gt[:, :, :h, :w, ...]

            loss_weights = self.loss_cfg.weights

            if loss_weights.get("rgb_loss", 0.0) > 0:
                rgb_loss = torch.sum(
                    valid_gt_mask * torch.abs(rgb_pred - rgb_gt)
                ) / torch.clamp(valid_gt_mask.sum(), min=1.0)
                loss_dict[f"frame.{i}.`rgb_loss`"] = rgb_loss * loss_weights.rgb_loss * self.frame_loss_weight[i]


            if loss_weights.get("depth_loss", 0.0) > 0:
                depth_loss = torch.sum(
                    valid_depth_gt_mask * torch.abs(depth_gt - depth_pred)
                ) / torch.clamp(valid_depth_gt_mask.sum(), min=1.0)
                loss_dict[f"frame.{i}.depth_loss"] = depth_loss * loss_weights.depth_loss * self.frame_loss_weight[i]

            if loss_weights.get("opacity_loss", 0.0) > 0:
                gt_opacity = torch.stack([ t["opacity_sampled_gt"] for t in target], dim=0)
                gt_opacity_mask = torch.stack([ t["opacity_mask"] for t in target], dim=0)
                bs, ray_num, sample_num = gt_opacity.shape
                pred_oci = preds_dict["voxel_opacity"].reshape(bs, ray_num, sample_num)
                loss_dict[f"frame.{i}.opacity_loss"] = loss_weights.opacity_loss * F.l1_loss(pred_oci * gt_opacity_mask,
                                                                                gt_opacity * gt_opacity_mask) * self.frame_loss_weight[i]

            if loss_weights.get("opacity_focal_loss", 0.0) > 0:
                gt_opacity = torch.stack([ t["opacity_sampled_gt"] for t in target], dim=0)
                bs, ray_num, sample_num = gt_opacity.shape
                gt_opacity_mask = torch.stack([ t["opacity_mask"] for t in target], dim=0).unsqueeze(-1).repeat(1, 1, sample_num)
                pred_oci = preds_dict["voxel_opacity"].reshape(bs, ray_num, sample_num)
                loss_dict[f"frame.{i}.opacity_focal_loss"] = loss_weights.opacity_focal_loss * self.frame_loss_weight[i] * \
                                py_sigmoid_focal_loss(pred_oci * gt_opacity_mask, 
                                                    gt_opacity * gt_opacity_mask, 
                                                    alpha=loss_weights.get('alpha', 0.25), gamma=loss_weights.get('gamma', 2.0))
                
        return loss_dict

    @force_fp32(apply_to=("viewmatrix", "projmatrix", "rgb_i", 'rot_i', 'scale_i', 'opacity_i', 'offset'))
    def render(self, pts, viewmatrix, projmatrix, cam_pos, rgb_i, rot_i, scale_i, opacity_i, offset=None, height=None, width=None, fovx=None, fovy=None):
        render_img_rgb, _, render_img_depth= modules.render(
            viewmatrix, 
            projmatrix=projmatrix, 
            cam_pos=cam_pos, 
            pts_xyz=pts if offset is None else pts + offset, 
            pts_rgb=rgb_i, 
            rotations=rot_i, 
            scales=scale_i, 
            opacity=opacity_i,
            height=height, 
            width=width, 
            fovx=fovx, 
            fovy=fovy, 
            )

        return render_img_rgb, render_img_depth
    
    def forward_render(self, uni_feats, batch_rays, img_metas, frame_index):
        uni_feats = self.render_conv(uni_feats)     # [1, 32, 5, 128, 128]
        batch_ret = []
        render_img_batch = {'img_rgb':[], 'img_depth':[], 'img_mask':[], 'voxel_opacity':None, 'pts_sampled':[]}
        for bs_idx in range(len(img_metas)):
            gs_param = self.gs_param_regresser(uni_feats[bs_idx], batch_rays[bs_idx])

            pts = batch_rays[bs_idx]['pts_sampled'].view(-1, 3)
            rot_i = gs_param['rot']
            scale_i = gs_param['scale']
            opacity_i = gs_param['opacity']
            rgb_i = gs_param['rgb']
            offset_i = None
            img_rgb = []
            img_depth = []
            for cam_idx in range(self.cam_nums):
                fovx = img_metas[bs_idx]['cam_params'][frame_index][cam_idx]['fovx']
                fovy = img_metas[bs_idx]['cam_params'][frame_index][cam_idx]['fovy']
                viewmatrix = img_metas[bs_idx]['cam_params'][frame_index][cam_idx]['viewmatrix'].to(uni_feats.device)
                projmatrix = img_metas[bs_idx]['cam_params'][frame_index][cam_idx]['projmatrix'].to(uni_feats.device)
                cam_pos = img_metas[bs_idx]['cam_params'][frame_index][cam_idx]['cam_pos'].to(uni_feats.device)


                render_img_rgb, render_img_depth = self.render(pts, 
                                                            viewmatrix, 
                                                            projmatrix, 
                                                            cam_pos, 
                                                            rgb_i, 
                                                            rot_i, 
                                                            scale_i,
                                                            opacity_i, 
                                                            offset_i, 
                                                            img_metas[bs_idx]['cam_params'][frame_index][cam_idx]['height'], 
                                                            img_metas[bs_idx]['cam_params'][frame_index][cam_idx]['width'], 
                                                            fovx, fovy)

                img_rgb.append(render_img_rgb)
                img_depth.append(render_img_depth)

            render_img_batch['img_rgb'].append(torch.stack(img_rgb, dim=0))
            render_img_batch['img_depth'].append(torch.stack(img_depth, dim=0))
            render_img_batch['pts_sampled'].append(pts if offset_i is None else pts + offset_i)

        render_img_batch['img_rgb'] = torch.clamp_max(torch.stack(render_img_batch['img_rgb'], dim=0), 1.0)
        render_img_batch['img_depth'] = torch.stack(render_img_batch['img_depth'], dim=0)
        render_img_batch['voxel_opacity'] = gs_param['opacity']
        render_img_batch['pts_sampled'] = torch.stack(render_img_batch['pts_sampled'], dim=0)
        
        return render_img_batch

    ############# Align coordinates between reference (current frame) to other frames. #############
    def _get_history_ref_to_previous_transform(self, tensor, num_frames, img_metas_list):
        """Get transformation matrix from reference frame to all previous frames.

        Args:
            tensor: to convert {ref_to_prev_transform} to device and dtype.
            num_frames: total num of available history frames.
            img_metas_list: a list of batch_size items.
                In each item, there is {num_prev_frames} img_meta for transformation alignment.

        Return:
            ref_to_history_list (torch.Tensor): with shape as [bs, num_prev_frames, 4, 4]
        """
        ref_to_history_list = []
        for img_metas in img_metas_list:
            cur_ref_to_prev = [img_metas[i]['ref_lidar_to_cur_lidar'] for i in range(num_frames)]
            ref_to_history_list.append(cur_ref_to_prev)
        ref_to_history_list = tensor.new_tensor(np.array(ref_to_history_list))
        return ref_to_history_list

    def _align_bev_coordnates(self, frame_idx, ref_to_history_list, img_metas):
        """Align the bev_coordinates of frame_idx to each of history_frames.

        Args:
            frame_idx: the index of target frame.
            ref_to_history_list (torch.Tensor): a tensor with shape as [bs, num_prev_frames, 4, 4]
                indicating the transformation metric from reference to each history frames.
            img_metas: a list of batch_size items.
                In each item, there is one img_meta (reference frame)
                whose {future2ref_lidar_transform} & {ref2future_lidar_transform} are for
                transformation alignment.
        """
        bs, num_frame = ref_to_history_list.shape[:2]
        # 1. get future2ref and ref2future_matrix of frame_idx.
        future2ref = [img_meta['future2ref_lidar_transform'][frame_idx] for img_meta in img_metas]  # b, 4, 4
        future2ref = ref_to_history_list.new_tensor(np.array(future2ref))  # bs, 4, 4

        ref2future = [img_meta['ref2future_lidar_transform'][frame_idx] for img_meta in img_metas]  # b, 4, 4
        ref2future = ref_to_history_list.new_tensor(np.array(ref2future))  # bs, 4, 4

        # 2. compute the transformation matrix from current frame to all previous frames.
        future2ref = future2ref.unsqueeze(1).repeat(1, num_frame, 1, 1).contiguous()
        future_to_history_list = torch.matmul(future2ref, ref_to_history_list)

        # 3. compute coordinates of future frame.
        bev_grids = get_bev_grids(
            self.bev_h, self.bev_w, bs * num_frame)  # bs * num_frame, bev_h, bev_w, 2 (x, y)
        bev_grids = bev_grids.view(bs, num_frame, -1, 2)
        bev_coords = bev_grids_to_coordinates(
            bev_grids, self.point_cloud_range)

        # 4. align target coordinates of future frame to each of previous frames.
        aligned_bev_coords = torch.cat([
            bev_coords, torch.ones_like(bev_coords[..., :2])], -1)  # b, num_frame, h*w, 4
        aligned_bev_coords = torch.matmul(aligned_bev_coords, future_to_history_list)
        aligned_bev_coords = aligned_bev_coords[..., :2]  # b, num_frame, h*w, 2
        aligned_bev_grids, _ = bev_coords_to_grids(
            aligned_bev_coords, self.bev_h, self.bev_w, self.point_cloud_range)
        aligned_bev_grids = (aligned_bev_grids + 1) / 2.  # range of [0, 1]
        # b, h*w, num_frame, 2
        aligned_bev_grids = aligned_bev_grids.permute(0, 2, 1, 3).contiguous()

        # 5. get target bev_grids at target future frame.
        tgt_grids = bev_grids[:, -1].contiguous()
        return tgt_grids, aligned_bev_grids, ref2future

    @auto_fp16(apply_to=("pts_feats", "img_feats", "img_depth"))
    def forward(self, pts_feats, img_feats, img_metas, img_depth, batch_rays=None):
        """
        Args:
            Currently only support single-frame, no 3D data augmentation, no 2D data augmentation
            ray_o: [(N*C*K, 3), ...]
            ray_d: [(N*C*K, 3), ...]
            img_feats: [(B, N*C, C', H, W), ...]
            img_depth: [(B*N*C, 64, H, W), ...]
        Returns:

        """
        bs = len(img_metas)
        uni_feats = []
        if img_feats is not None:
            uni_feats.append(
                self.view_trans(img_feats, img_metas=img_metas, img_depth=img_depth)
            )
        if pts_feats is not None:
            uni_feats.append(pts_feats)

        if self.fusion_layer is not None:
            uni_feats = self.fusion_layer(uni_feats)
        else:
            uni_feats = sum(uni_feats)

        if uni_feats.shape[2] == 1:
            uni_feats = uni_feats.squeeze(2)
            uni_feats_3d = self.lift_feats(uni_feats)
        else:
            uni_feats_3d = uni_feats


        render_res = []
        ### for cur frame
        render_img_batch = self.forward_render(uni_feats_3d, batch_rays[0], img_metas, frame_index=0)
        render_res.append(render_img_batch)
        ############# render future attrs
        if self.future_pred_head is not None:
            prev_bev_input = uni_feats.flatten(2, 3).unsqueeze(1).permute(0, 1, 3, 2)
            prev_img_metas = copy.deepcopy(img_metas)
            # ref_to_history_list = self._get_history_ref_to_previous_transform(
            #             prev_bev_input, prev_bev_input.shape[1], prev_img_metas)

            ## only has one previous frame 
            ref_to_history_list = prev_bev_input.new_tensor(np.array([img_metas[i]['ref_lidar_to_cur_lidar'] for i in range(len(img_metas))]))
            ref_to_history_list = ref_to_history_list.unsqueeze(1)
            
            for future_frame_index in range(1, self.future_pred_num + 1):
                tgt_grids, aligned_prev_grids, ref2future = self._align_bev_coordnates(
                        future_frame_index, ref_to_history_list, img_metas)
                pred_feat = self.future_pred_head(prev_bev_input,
                                                    prev_img_metas,
                                                    future_frame_index,
                                                    tgt_grids,  # tgt_points config for self-attention.
                                                    aligned_prev_grids,  # ref_points config for cross-attention.
                                                    self.bev_h, self.bev_w)
                future_bev = pred_feat[-1].permute(0, 2, 1).reshape(bs, -1, self.bev_w, self.bev_h)
                future_bev = self.lift_feats(future_bev)
                render_img_batch = self.forward_render(future_bev, batch_rays[future_frame_index], img_metas, frame_index=future_frame_index)
                render_res.append(render_img_batch)

                # 3. update pred_feat to prev_bev_input and update ref_to_history_list.
                prev_bev_input = torch.cat([prev_bev_input, pred_feat[-1].unsqueeze(1)], 1)
                prev_bev_input = prev_bev_input[:, 1:, ...].contiguous()
                # update ref2future to ref_to_history_list.
                ref_to_history_list = torch.cat([ref_to_history_list, ref2future.unsqueeze(1)], 1)
                ref_to_history_list = ref_to_history_list[:, 1:].contiguous()

        return render_res
    
    def points2depthmap(self, points, img_depth):
        height, width = img_depth.shape[0], img_depth.shape[1]
        depth = points[:, 2]
        # coor = torch.round(points[:, :2])
        coor = points[:, :2].long().float()

        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.0).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept], depth[kept]
        coor = coor.long()
        img_depth[coor[:, 1], coor[:, 0], 0] = depth
        return img_depth, coor
        
    def sample_rays(self, pts, imgs, img_metas):
        all_pts = copy.deepcopy(pts)
        frame_batch_ret = []
        ##TODO
        assert imgs.shape[1] == 1 + self.future_pred_num
        all_imgs = copy.deepcopy(imgs)
        for frame_index in range(1 + self.future_pred_num):
            imgs = all_imgs[:, frame_index]
            lidar2img, lidar2cam = [], []
            for img_meta in img_metas:
                lidar2img.append(img_meta["all_lidar2img"][frame_index])
                lidar2cam.append(img_meta["all_lidar2cam"][frame_index])
            lidar2img = np.asarray(lidar2img)
            lidar2cam = np.asarray(lidar2cam)

            new_pts = []
            for bs_idx, i_pts in enumerate(all_pts):
                cur_frame_pts = i_pts[i_pts[:, -1] == frame_index][:, :5]
                dis = cur_frame_pts[:, :2].norm(dim=-1)
                dis_mask = (dis > self.ray_sampler_cfg.close_radius) & (
                    dis < self.ray_sampler_cfg.get("far_radius", 100.0)
                )
                new_pts.append(cur_frame_pts[dis_mask])     # 过滤掉太近和太远的
            pts = new_pts
            if (
                sparse_utils._cur_active_voxel is not None
                and self.ray_sampler_cfg.only_point_mask
            ):
                pc_range = torch.from_numpy(self.pc_range).to(pts[0])
                mask_voxel_size = (
                    torch.from_numpy(self.unified_voxel_size).to(pts[0])
                    / sparse_utils._cur_voxel_scale
                )
                mask_voxel_shape = (
                    torch.from_numpy(self.unified_voxel_shape).to(pts[0].device)
                    * sparse_utils._cur_voxel_scale
                )
                nonactive_voxel_mask = torch.zeros(
                    (len(pts), *mask_voxel_shape.flip(dims=[0])),
                    dtype=torch.bool,
                    device=pts[0].device,
                )
                nonactive_voxel_mask[
                    sparse_utils._cur_voxel_coords[~sparse_utils._cur_active_voxel]
                    .long()
                    .unbind(dim=1)
                ] = True
                new_pts = []
                for bs_idx in range(len(pts)):
                    p_pts = pts[bs_idx]
                    p_coords = (p_pts[:, :3] - pc_range[:3]) / mask_voxel_size
                    kept = torch.all(
                        (p_coords >= torch.zeros_like(mask_voxel_shape))
                        & (p_coords < mask_voxel_shape),
                        dim=-1,
                    )
                    p_coords = F.pad(
                        p_coords[:, [2, 1, 0]].long(), (1, 0), mode="constant", value=bs_idx
                    )
                    p_coords, p_pts = p_coords[kept], p_pts[kept]
                    p_nonactive_pts_mask = nonactive_voxel_mask[p_coords.unbind(dim=1)]
                    new_pts.append(p_pts[p_nonactive_pts_mask])
                pts = new_pts       # 过滤掉非活跃的点, 现在我还不知道 这个蒙板是怎么来的
            
            # sparse_utils._cur_active: [6, 1, 29, 50]
            # self.ray_sampler_cfg.only_img_mask: False
            if sparse_utils._cur_active is not None and self.ray_sampler_cfg.only_img_mask:
                active_mask = sparse_utils._get_active_ex_or_ii(imgs.shape[-2])
                assert (
                    active_mask.shape[-2] == imgs.shape[-2]
                    and active_mask.shape[-1] == imgs.shape[-1]
                )
                active_mask = active_mask.view(
                    imgs.shape[0], -1, imgs.shape[-2], imgs.shape[-1]
                )

            batch_ret = []
            for bs_idx in range(len(pts)):
                i_imgs = imgs[bs_idx]
                i_pts = pts[bs_idx]
                ############ 过屡历史帧点云
                i_pts = i_pts[i_pts[:, -1] < 0.06]
                i_lidar2img = i_pts.new_tensor(lidar2img[bs_idx]).flatten(0, 1)
                i_img2lidar = torch.inverse(
                    i_lidar2img
                )  # TODO: Are img2lidar and img2cam consistent after image data augmentation?
                i_cam2lidar = torch.inverse(
                    i_pts.new_tensor(lidar2cam[bs_idx]).flatten(0, 1)
                )
                i_pts = torch.cat([i_pts[..., :3], torch.ones_like(i_pts[..., :1])], -1)
                i_pts_cam = torch.matmul(
                    i_lidar2img.unsqueeze(1), i_pts.view(1, -1, 4, 1)
                ).squeeze(-1)

                eps = 1e-5
                i_pts_mask = i_pts_cam[..., 2] > eps
                i_pts_cam[..., :2] = i_pts_cam[..., :2] / torch.maximum(
                    i_pts_cam[..., 2:3], torch.ones_like(i_pts_cam[..., 2:3]) * eps
                )

                # (N*C, 3) [(H, W, 3), ...]
                pad_before_shape = torch.tensor(
                    img_metas[bs_idx]["pad_before_shape"], device=i_pts_cam.device
                )
                Hs, Ws = pad_before_shape[:, 0:1], pad_before_shape[:, 1:2]

                # (N*C, M)
                i_pts_mask = (
                    i_pts_mask
                    & (i_pts_cam[..., 0] > 0)
                    & (i_pts_cam[..., 0] < Ws - 1)
                    & (i_pts_cam[..., 1] > 0)
                    & (i_pts_cam[..., 1] < Hs - 1)
                )

                i_imgs = i_imgs.permute(0, 2, 3, 1)
                i_imgs = i_imgs * i_imgs.new_tensor(
                    img_metas[0]["img_norm_cfg"]["std"]
                ) + i_imgs.new_tensor(img_metas[0]["img_norm_cfg"]["mean"])
                if not img_metas[0]["img_norm_cfg"]["to_rgb"]:
                    i_imgs[..., [0, 1, 2]] = i_imgs[..., [2, 1, 0]]  # bgr->rgb
                i_imgs = i_imgs[:, :img_metas[0]['pad_before_shape'][0][0], :img_metas[0]['pad_before_shape'][0][1]]
        
                i_sampled_rgb_gt, i_sampled_rgb_mask, i_sampled_pts, i_sampled_depth_gt, i_sampled_pts_gt, i_sampled_opacity_gt, i_sampled_depth_mask  = ([], [], [], [], [], [], [])
                i_opt_mask = []
                for c_idx in range(len(i_pts_mask)):
                    j_sampled_all_pts, j_sampled_all_pts_cam, j_sampled_all_depth_mask,  j_sampled_all_lidar= (
                        [],
                        [],
                        [],
                        [],
                    )

                    """ sample points """
                    j_sampled_pts_idx = i_pts_mask[c_idx].nonzero(as_tuple=True)[0]
                    j_sampled_pts_cam = i_pts_cam[c_idx][j_sampled_pts_idx]
                    j_sampled_pts_cam_all = copy.deepcopy(j_sampled_pts_cam)
                    if self.ray_sampler_cfg.only_img_mask:
                        j_sampled_pts_mask = ~active_mask[
                            bs_idx,
                            c_idx,
                            j_sampled_pts_cam[:, 1].long(),
                            j_sampled_pts_cam[:, 0].long(),
                        ]
                        j_sampled_pts_idx = j_sampled_pts_mask.nonzero(as_tuple=True)[0]
                    else:
                        j_sampled_pts_idx = torch.arange(
                            len(j_sampled_pts_cam),
                            dtype=torch.long,
                            device=j_sampled_pts_cam.device,
                        )

                    # -------------- 采样点云中的点
                    point_nsample = min(        # 采样点的个数,     我怀疑这个会影响最终的效果
                        len(j_sampled_pts_idx),
                        int(len(j_sampled_pts_idx) * self.ray_sampler_cfg.point_ratio)
                        if self.ray_sampler_cfg.point_nsample == -1
                        else self.ray_sampler_cfg.point_nsample,
                    )

                    if point_nsample > 0:
                        replace_sample = (
                            True
                            if point_nsample > len(j_sampled_pts_idx)
                            else self.ray_sampler_cfg.replace_sample
                        )
                        j_sampled_pts_idx = j_sampled_pts_idx[
                            torch.from_numpy(
                                np.random.choice(
                                    len(j_sampled_pts_idx),
                                    point_nsample,
                                    replace=replace_sample,
                                )
                            )
                            .long()
                            .to(j_sampled_pts_idx.device)
                        ]
                        j_sampled_pts_cam = j_sampled_pts_cam[j_sampled_pts_idx]

                        # -------------- 从图像-lidar中采样到点,然后化再投影回lidar坐标系z
                        depth_bin = torch.linspace(1, 60, self.ray_sampler_cfg.get('anchor_gaussian_interval', 100)).to(j_sampled_pts_cam.device)
                        pixel_points_repeated=j_sampled_pts_cam[...,:2].repeat_interleave(self.ray_sampler_cfg.get('anchor_gaussian_interval', 100), dim=0).long().float()    # 否则会存在错位的情况

                        depths_repeated = depth_bin.repeat(j_sampled_pts_cam.shape[0])
                        pixel_3d = torch.cat((pixel_points_repeated, depths_repeated.unsqueeze(1)), dim=1)

                        lidar_3d_sampled = torch.matmul(
                                            i_img2lidar[c_idx : c_idx + 1],
                                            torch.cat([
                                                pixel_3d[..., :2] * pixel_3d[..., 2:3],
                                                pixel_3d[..., 2:3],
                                                torch.ones_like(pixel_3d[..., 2:3]),
                                            ], dim=-1).unsqueeze(-1)
                                            ).squeeze(-1)[..., :3].view(-1, self.ray_sampler_cfg.get('anchor_gaussian_interval', 100), 3)
                    
                        i_sampled_pts.append(lidar_3d_sampled)
                        
                        j_sampled_pts = torch.matmul(
                            i_img2lidar[c_idx : c_idx + 1],
                            torch.cat(
                                [
                                    j_sampled_pts_cam[..., :2]
                                    * j_sampled_pts_cam[..., 2:3],
                                    j_sampled_pts_cam[..., 2:],
                                ],
                                dim=-1,
                            ).unsqueeze(-1),
                        ).squeeze(-1)[..., :3]
                        i_sampled_pts_gt.append(j_sampled_pts)
                        j_sampled_all_pts.append(j_sampled_pts)
                        j_sampled_all_pts_cam.append(j_sampled_pts_cam[..., :3])

                        ### get opa label 
                        if self.opa_one_hot:
                            sampled_opacity_gt = get_one_opa_label(j_sampled_pts_cam, self.ray_sampler_cfg.get('anchor_gaussian_interval', 100))
                        else:
                            sampled_opacity_gt = get_multi_opa_label(i_pts, lidar_3d_sampled, self.gs_param_regresser.pc_range, self.gs_param_regresser.voxel_size)
                            sampled_opacity_gt = sampled_opacity_gt.reshape(-1, 100)
                        i_sampled_opacity_gt.append(sampled_opacity_gt)
                        ori_len = len(i_sampled_opacity_gt[-1])
                        opt_mask = torch.zeros((len(i_sampled_opacity_gt[-1]))).to(sampled_opacity_gt)
                        opt_mask[:ori_len] = 1

                        img_depth = torch.zeros_like(i_imgs[c_idx:c_idx+1][..., 0:1])
                        
                        if self.all_depth:
                            img_depth, _ = self.points2depthmap(j_sampled_pts_cam_all, img_depth[0])
                        else:
                            img_depth, _ = self.points2depthmap(j_sampled_pts_cam, img_depth[0])

                        i_sampled_depth_gt.append(img_depth)
                        depth_mask = torch.zeros_like(i_imgs[c_idx:c_idx+1])
            
                        depth_mask[
                                0,
                                j_sampled_pts_cam[:, 1].long(),
                                j_sampled_pts_cam[:, 0].long(), 
                            ] = 1
                        i_sampled_depth_mask.append(depth_mask)

                        img_mask = torch.zeros_like(i_imgs[c_idx:c_idx+1])
            
                        img_mask[
                                0,
                                j_sampled_pts_cam[:, 1].long(),
                                j_sampled_pts_cam[:, 0].long(), 
                            ] = 1


                        if self.use_semantic_guide:
                            semantic_map = torch.from_numpy(img_metas[bs_idx]['semantic_maps'][c_idx])
                            select_index = []
                            for instance_id in [3, 5, 7, 1, 0]:
                                ind0, ind1 = torch.where(semantic_map == instance_id)
                                select_index.append(torch.cat([ind1[:, None], ind0[:, None]], axis=-1))
                            j_sampled_pixels_cam = torch.cat(select_index).to(j_sampled_pts)

                            j_sampled_pixels_idx = torch.arange(
                                len(j_sampled_pixels_cam),
                                dtype=torch.long,
                                device=j_sampled_pixels_cam.device,
                            )
                            pixel_nsample = min(
                                len(j_sampled_pixels_idx),
                                self.ray_sampler_cfg.merged_nsample - point_nsample,
                            )
                            j_sampled_pixels_idx = j_sampled_pixels_idx[
                                torch.from_numpy(
                                    np.random.choice(
                                        len(j_sampled_pixels_idx),
                                        pixel_nsample,
                                        replace=self.ray_sampler_cfg.replace_sample,
                                    )
                                )
                                .long()
                                .to(j_sampled_pixels_idx.device)
                            ]
                            j_sampled_pixels_cam = j_sampled_pixels_cam[j_sampled_pixels_idx]

                            pixel_points_repeated_cam=j_sampled_pixels_cam[...,:2].repeat_interleave(self.ray_sampler_cfg.get('anchor_gaussian_interval', 100), dim=0).long().float()    # 否则会存在错位的情况
                            depths_repeated_cam = depth_bin.repeat(j_sampled_pixels_cam.shape[0])
                            pixel_3d_cam = torch.cat((pixel_points_repeated_cam, depths_repeated_cam.unsqueeze(1)), dim=1)

                            j_sampled_pixels = torch.matmul(
                                    i_img2lidar[c_idx : c_idx + 1],
                                    torch.cat([
                                        pixel_3d_cam[..., :2] * pixel_3d_cam[..., 2:3],
                                        pixel_3d_cam[..., 2:3],
                                        torch.ones_like(pixel_3d_cam[..., 2:3]),
                                    ], dim=-1).unsqueeze(-1)
                                    ).squeeze(-1)[..., :3].view(-1, self.ray_sampler_cfg.get('anchor_gaussian_interval', 100), 3)

                            i_sampled_pts[-1] = torch.cat([i_sampled_pts[-1], j_sampled_pixels], axis=0)
                            i_sampled_opacity_gt[-1] = torch.cat([i_sampled_opacity_gt[-1], 
                                                                torch.ones((pixel_nsample, i_sampled_opacity_gt[-1].shape[1])).to(i_sampled_opacity_gt[-1])], axis=0)
                            opt_mask = torch.zeros((len(i_sampled_opacity_gt[-1]))).to(sampled_opacity_gt)
                            opt_mask[:ori_len] = 1
                            img_mask[
                                0,
                                j_sampled_pixels_cam[:, 1].long(),
                                j_sampled_pixels_cam[:, 0].long()
                            ] = 1
                        i_opt_mask.append(opt_mask)
                        i_sampled_rgb_mask.append(img_mask)
                        i_sampled_rgb_gt.append(i_imgs[c_idx] / 255.0)


                    else:
                        depth_mask = torch.zeros_like(i_imgs[c_idx:c_idx+1])
                        i_sampled_depth_mask.append(depth_mask)
                        img_depth = torch.zeros_like(i_imgs[c_idx:c_idx+1][..., 0:1])[0]
                        i_sampled_depth_gt.append(img_depth)
                        i_sampled_rgb_mask.append(torch.zeros_like(i_imgs[c_idx:c_idx+1]))
                        i_sampled_rgb_gt.append(i_imgs[c_idx] / 255.0)

                batch_ret.append(
                    {
                        'rgb_mask':torch.stack(i_sampled_rgb_mask, dim=0),
                        'pts_sampled':torch.cat(i_sampled_pts, dim=0),
                        'img_rgb':torch.stack(i_sampled_rgb_gt, dim=0),
                        'img_depth':torch.stack(i_sampled_depth_gt, dim=0),
                        'pts_sampled_gt':torch.cat(i_sampled_pts_gt, dim=0),
                        'opacity_sampled_gt':torch.cat(i_sampled_opacity_gt, dim=0),
                        'depth_mask':torch.stack(i_sampled_depth_mask, dim=0),
                        'opacity_mask':torch.cat(i_opt_mask, dim=0),
                    }
                )
            frame_batch_ret.append(batch_ret)
        return frame_batch_ret

    def sample_rays_match(self, pts, imgs, img_metas, sampled_pts_first):
        assert False
        batch_ret = []
        for bs_idx in range(len(pts)):
            i_imgs = imgs[bs_idx]
            i_pts = pts[bs_idx]
            ############ 过屡历史帧点云
            i_pts = i_pts[i_pts[:, -1] < 0.06]
            i_lidar2img = i_pts.new_tensor(lidar2img[bs_idx]).flatten(0, 1)
            i_img2lidar = torch.inverse(
                i_lidar2img
            )
            i_cam2lidar = torch.inverse(
                i_pts.new_tensor(lidar2cam[bs_idx]).flatten(0, 1)
            )
            i_pts = torch.cat([i_pts[..., :3], torch.ones_like(i_pts[..., :1])], -1)
            i_pts_cam = torch.matmul(
                i_lidar2img.unsqueeze(1), i_pts.view(1, -1, 4, 1)
            ).squeeze(-1)

            eps = 1e-5
            i_pts_mask = i_pts_cam[..., 2] > eps
            i_pts_cam[..., :2] = i_pts_cam[..., :2] / torch.maximum(
                i_pts_cam[..., 2:3], torch.ones_like(i_pts_cam[..., 2:3]) * eps
            )

            # (N*C, 3) [(H, W, 3), ...]
            pad_before_shape = torch.tensor(
                img_metas[bs_idx]["pad_before_shape"], device=i_pts_cam.device
            )
            Hs, Ws = pad_before_shape[:, 0:1], pad_before_shape[:, 1:2]

            # (N*C, M)
            i_pts_mask = (
                i_pts_mask
                & (i_pts_cam[..., 0] > 0)
                & (i_pts_cam[..., 0] < Ws - 1)
                & (i_pts_cam[..., 1] > 0)
                & (i_pts_cam[..., 1] < Hs - 1)
            )

            i_imgs = i_imgs.permute(0, 2, 3, 1)
            i_imgs = i_imgs * i_imgs.new_tensor(
                img_metas[0]["img_norm_cfg"]["std"]
            ) + i_imgs.new_tensor(img_metas[0]["img_norm_cfg"]["mean"])
            if not img_metas[0]["img_norm_cfg"]["to_rgb"]:
                i_imgs[..., [0, 1, 2]] = i_imgs[..., [2, 1, 0]]  # bgr->rgb
            i_imgs = i_imgs[:, :img_metas[0]['pad_before_shape'][0][0], :img_metas[0]['pad_before_shape'][0][1]]
    
            i_sampled_rgb_gt, i_sampled_rgb_mask, i_sampled_pts, i_sampled_depth_gt, i_sampled_pts_gt, i_sampled_opacity_gt, i_sampled_depth_mask  = ([], [], [], [], [], [], [])
            i_opt_mask = []
            for c_idx in range(len(i_pts_mask)):
                j_sampled_all_pts, j_sampled_all_pts_cam, j_sampled_all_depth_mask,  j_sampled_all_lidar= (
                    [],
                    [],
                    [],
                    [],
                )

                """ sample points """
                j_sampled_pts_idx = i_pts_mask[c_idx].nonzero(as_tuple=True)[0]
                j_sampled_pts_cam = i_pts_cam[c_idx][j_sampled_pts_idx]
                j_sampled_pts_cam_all = copy.deepcopy(j_sampled_pts_cam)
                if self.ray_sampler_cfg.only_img_mask:
                    j_sampled_pts_mask = ~active_mask[
                        bs_idx,
                        c_idx,
                        j_sampled_pts_cam[:, 1].long(),
                        j_sampled_pts_cam[:, 0].long(),
                    ]
                    j_sampled_pts_idx = j_sampled_pts_mask.nonzero(as_tuple=True)[0]
                else:
                    j_sampled_pts_idx = torch.arange(
                        len(j_sampled_pts_cam),
                        dtype=torch.long,
                        device=j_sampled_pts_cam.device,
                    )

                # -------------- 采样点云中的点
                point_nsample = min(        # 采样点的个数,     我怀疑这个会影响最终的效果
                    len(j_sampled_pts_idx),
                    int(len(j_sampled_pts_idx) * self.ray_sampler_cfg.point_ratio)
                    if self.ray_sampler_cfg.point_nsample == -1
                    else self.ray_sampler_cfg.point_nsample,
                )

                if point_nsample > 0:
                    replace_sample = (
                        True
                        if point_nsample > len(j_sampled_pts_idx)
                        else self.ray_sampler_cfg.replace_sample
                    )
                    j_sampled_pts_idx = j_sampled_pts_idx[
                        torch.from_numpy(
                            np.random.choice(
                                len(j_sampled_pts_idx),
                                point_nsample,
                                replace=replace_sample,
                            )
                        )
                        .long()
                        .to(j_sampled_pts_idx.device)
                    ]
                    j_sampled_pts_cam = j_sampled_pts_cam[j_sampled_pts_idx]

                    # -------------- 从图像-lidar中采样到点,然后化再投影回lidar坐标系z
                    depth_bin = torch.linspace(1, 60, self.ray_sampler_cfg.get('anchor_gaussian_interval', 100)).to(j_sampled_pts_cam.device)
                    pixel_points_repeated=j_sampled_pts_cam[...,:2].repeat_interleave(self.ray_sampler_cfg.get('anchor_gaussian_interval', 100), dim=0).long().float()    # 否则会存在错位的情况

                    depths_repeated = depth_bin.repeat(j_sampled_pts_cam.shape[0])
                    pixel_3d = torch.cat((pixel_points_repeated, depths_repeated.unsqueeze(1)), dim=1)

                    lidar_3d_sampled = torch.matmul(
                                        i_img2lidar[c_idx : c_idx + 1],
                                        torch.cat([
                                            pixel_3d[..., :2] * pixel_3d[..., 2:3],
                                            pixel_3d[..., 2:3],
                                            torch.ones_like(pixel_3d[..., 2:3]),
                                        ], dim=-1).unsqueeze(-1)
                                        ).squeeze(-1)[..., :3].view(-1, self.ray_sampler_cfg.get('anchor_gaussian_interval', 100), 3)
                
                    i_sampled_pts.append(lidar_3d_sampled)
                    
                    j_sampled_pts = torch.matmul(
                        i_img2lidar[c_idx : c_idx + 1],
                        torch.cat(
                            [
                                j_sampled_pts_cam[..., :2]
                                * j_sampled_pts_cam[..., 2:3],
                                j_sampled_pts_cam[..., 2:],
                            ],
                            dim=-1,
                        ).unsqueeze(-1),
                    ).squeeze(-1)[..., :3]
                    i_sampled_pts_gt.append(j_sampled_pts)
                    j_sampled_all_pts.append(j_sampled_pts)
                    j_sampled_all_pts_cam.append(j_sampled_pts_cam[..., :3])

                    ### get opa label 
                    if self.opa_one_hot:
                        sampled_opacity_gt = get_one_opa_label(j_sampled_pts_cam, self.ray_sampler_cfg.get('anchor_gaussian_interval', 100))
                    else:
                        sampled_opacity_gt = get_multi_opa_label(i_pts, lidar_3d_sampled, self.gs_param_regresser.pc_range, self.gs_param_regresser.voxel_size)
                        sampled_opacity_gt = sampled_opacity_gt.reshape(-1, 100)
                    i_sampled_opacity_gt.append(sampled_opacity_gt)
                    ori_len = len(i_sampled_opacity_gt[-1])
                    opt_mask = torch.zeros((len(i_sampled_opacity_gt[-1]))).to(sampled_opacity_gt)
                    opt_mask[:ori_len] = 1

                    img_depth = torch.zeros_like(i_imgs[c_idx:c_idx+1][..., 0:1])
                    
                    if self.all_depth:
                        img_depth, _ = self.points2depthmap(j_sampled_pts_cam_all, img_depth[0])
                    else:
                        img_depth, _ = self.points2depthmap(j_sampled_pts_cam, img_depth[0])

                    i_sampled_depth_gt.append(img_depth)
                    depth_mask = torch.zeros_like(i_imgs[c_idx:c_idx+1])
        
                    depth_mask[
                            0,
                            j_sampled_pts_cam[:, 1].long(),
                            j_sampled_pts_cam[:, 0].long(), 
                        ] = 1
                    i_sampled_depth_mask.append(depth_mask)

                    img_mask = torch.zeros_like(i_imgs[c_idx:c_idx+1])
        
                    img_mask[
                            0,
                            j_sampled_pts_cam[:, 1].long(),
                            j_sampled_pts_cam[:, 0].long(), 
                        ] = 1


                    if self.use_semantic_guide:
                        semantic_map = torch.from_numpy(img_metas[bs_idx]['semantic_maps'][c_idx])
                        select_index = []
                        for instance_id in [3, 5, 7, 1, 0]:
                            ind0, ind1 = torch.where(semantic_map == instance_id)
                            select_index.append(torch.cat([ind1[:, None], ind0[:, None]], axis=-1))
                        j_sampled_pixels_cam = torch.cat(select_index).to(j_sampled_pts)

                        j_sampled_pixels_idx = torch.arange(
                            len(j_sampled_pixels_cam),
                            dtype=torch.long,
                            device=j_sampled_pixels_cam.device,
                        )
                        pixel_nsample = min(
                            len(j_sampled_pixels_idx),
                            self.ray_sampler_cfg.merged_nsample - point_nsample,
                        )
                        j_sampled_pixels_idx = j_sampled_pixels_idx[
                            torch.from_numpy(
                                np.random.choice(
                                    len(j_sampled_pixels_idx),
                                    pixel_nsample,
                                    replace=self.ray_sampler_cfg.replace_sample,
                                )
                            )
                            .long()
                            .to(j_sampled_pixels_idx.device)
                        ]
                        j_sampled_pixels_cam = j_sampled_pixels_cam[j_sampled_pixels_idx]

                        pixel_points_repeated_cam=j_sampled_pixels_cam[...,:2].repeat_interleave(self.ray_sampler_cfg.get('anchor_gaussian_interval', 100), dim=0).long().float()    # 否则会存在错位的情况
                        depths_repeated_cam = depth_bin.repeat(j_sampled_pixels_cam.shape[0])
                        pixel_3d_cam = torch.cat((pixel_points_repeated_cam, depths_repeated_cam.unsqueeze(1)), dim=1)

                        j_sampled_pixels = torch.matmul(
                                i_img2lidar[c_idx : c_idx + 1],
                                torch.cat([
                                    pixel_3d_cam[..., :2] * pixel_3d_cam[..., 2:3],
                                    pixel_3d_cam[..., 2:3],
                                    torch.ones_like(pixel_3d_cam[..., 2:3]),
                                ], dim=-1).unsqueeze(-1)
                                ).squeeze(-1)[..., :3].view(-1, self.ray_sampler_cfg.get('anchor_gaussian_interval', 100), 3)

                        i_sampled_pts[-1] = torch.cat([i_sampled_pts[-1], j_sampled_pixels], axis=0)
                        i_sampled_opacity_gt[-1] = torch.cat([i_sampled_opacity_gt[-1], 
                                                            torch.ones((pixel_nsample, i_sampled_opacity_gt[-1].shape[1])).to(i_sampled_opacity_gt[-1])], axis=0)
                        opt_mask = torch.zeros((len(i_sampled_opacity_gt[-1]))).to(sampled_opacity_gt)
                        opt_mask[:ori_len] = 1
                        img_mask[
                            0,
                            j_sampled_pixels_cam[:, 1].long(),
                            j_sampled_pixels_cam[:, 0].long()
                        ] = 1
                    i_opt_mask.append(opt_mask)
                    i_sampled_rgb_mask.append(img_mask)
                    i_sampled_rgb_gt.append(i_imgs[c_idx] / 255.0)


                else:
                    depth_mask = torch.zeros_like(i_imgs[c_idx:c_idx+1])
                    i_sampled_depth_mask.append(depth_mask)
                    img_depth = torch.zeros_like(i_imgs[c_idx:c_idx+1][..., 0:1])[0]
                    i_sampled_depth_gt.append(img_depth)
                    i_sampled_rgb_mask.append(torch.zeros_like(i_imgs[c_idx:c_idx+1]))
                    i_sampled_rgb_gt.append(i_imgs[c_idx] / 255.0)

            batch_ret.append(
                {
                    'rgb_mask':torch.stack(i_sampled_rgb_mask, dim=0),
                    'pts_sampled':torch.cat(i_sampled_pts, dim=0),
                    'img_rgb':torch.stack(i_sampled_rgb_gt, dim=0),
                    'img_depth':torch.stack(i_sampled_depth_gt, dim=0),
                    'pts_sampled_gt':torch.cat(i_sampled_pts_gt, dim=0),
                    'opacity_sampled_gt':torch.cat(i_sampled_opacity_gt, dim=0),
                    'depth_mask':torch.stack(i_sampled_depth_mask, dim=0),
                    'opacity_mask':torch.cat(i_opt_mask, dim=0),
                }
            )

        return batch_ret

    def sample_rays_test(self, pts, imgs, img_metas):
        lidar2img, lidar2cam = [], []
        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])
            lidar2cam.append(img_meta["lidar2cam"])
        lidar2img = np.asarray(lidar2img)
        lidar2cam = np.asarray(lidar2cam)

        new_pts = []
        for bs_idx, i_pts in enumerate(pts):
            dis = i_pts[:, :2].norm(dim=-1)
            dis_mask = (dis > self.ray_sampler_cfg.close_radius) & (
                dis < self.ray_sampler_cfg.get("far_radius", 100.0)
            )
            new_pts.append(i_pts[dis_mask])     # 过滤掉太近和太远的
        pts = new_pts
        batch_ret = []
        for bs_idx in range(len(pts)):
            i_imgs = imgs[bs_idx]
            i_pts = pts[bs_idx]
            i_lidar2img = i_pts.new_tensor(lidar2img[bs_idx]).flatten(0, 1)
            i_img2lidar = torch.inverse(
                i_lidar2img
            )  # TODO: Are img2lidar and img2cam consistent after image data augmentation?
            i_cam2lidar = torch.inverse(
                i_pts.new_tensor(lidar2cam[bs_idx]).flatten(0, 1)
            )
            i_pts = torch.cat([i_pts[..., :3], torch.ones_like(i_pts[..., :1])], -1)
            i_pts_cam = torch.matmul(
                i_lidar2img.unsqueeze(1), i_pts.view(1, -1, 4, 1)
            ).squeeze(-1)

            eps = 1e-5
            i_pts_mask = i_pts_cam[..., 2] > eps
            i_pts_cam[..., :2] = i_pts_cam[..., :2] / torch.maximum(
                i_pts_cam[..., 2:3], torch.ones_like(i_pts_cam[..., 2:3]) * eps
            )

            # (N*C, 3) [(H, W, 3), ...]
            pad_before_shape = torch.tensor(
                img_metas[bs_idx]["pad_before_shape"], device=i_pts_cam.device
            )
            Hs, Ws = pad_before_shape[:, 0:1], pad_before_shape[:, 1:2]

            # (N*C, M)
            i_pts_mask = (
                i_pts_mask
                & (i_pts_cam[..., 0] > 0)
                & (i_pts_cam[..., 0] < Ws - 1)
                & (i_pts_cam[..., 1] > 0)
                & (i_pts_cam[..., 1] < Hs - 1)
            )

            i_imgs = i_imgs.permute(0, 2, 3, 1)
            i_imgs = i_imgs * i_imgs.new_tensor(
                img_metas[0]["img_norm_cfg"]["std"]
            ) + i_imgs.new_tensor(img_metas[0]["img_norm_cfg"]["mean"])
            if not img_metas[0]["img_norm_cfg"]["to_rgb"]:
                i_imgs[..., [0, 1, 2]] = i_imgs[..., [2, 1, 0]]  # bgr->rgb
            i_imgs = i_imgs[:, :900, :1600]
            # import pdb; pdb.set_trace()
            i_sampled_ray_o, i_sampled_ray_d, i_sampled_rgb, i_sampled_depth = (
                [],
                [],
                [],
                [],
            )
            i_sampled_rgb_gt, i_sampled_rgb_mask, i_sampled_pts, i_sampled_dpeth = ([], [], [], [])
            for c_idx in range(len(i_pts_mask)):
                j_sampled_all_pts, j_sampled_all_pts_cam, j_sampled_all_depth_mask = (
                    [],
                    [],
                    [],
                )

                """ sample points """
                j_sampled_pts_idx = i_pts_mask[c_idx].nonzero(as_tuple=True)[0]
                j_sampled_pts_cam = i_pts_cam[c_idx][j_sampled_pts_idx]

                if self.ray_sampler_cfg.only_img_mask:
                    j_sampled_pts_mask = ~active_mask[
                        bs_idx,
                        c_idx,
                        j_sampled_pts_cam[:, 1].long(),
                        j_sampled_pts_cam[:, 0].long(),
                    ]
                    j_sampled_pts_idx = j_sampled_pts_mask.nonzero(as_tuple=True)[0]
                else:
                    j_sampled_pts_idx = torch.arange(
                        len(j_sampled_pts_cam),
                        dtype=torch.long,
                        device=j_sampled_pts_cam.device,
                    )

                # -------------- 采样点云中的点
                point_nsample = min(        # 采样点的个数,     我怀疑这个会影响最终的效果
                    len(j_sampled_pts_idx),
                    int(len(j_sampled_pts_idx) * self.ray_sampler_cfg.point_ratio)
                    if self.ray_sampler_cfg.point_nsample == -1
                    else self.ray_sampler_cfg.point_nsample,
                )

                if point_nsample > 0:
                    replace_sample = (
                        True
                        if point_nsample > len(j_sampled_pts_idx)
                        else self.ray_sampler_cfg.replace_sample
                    )
                    j_sampled_pts_idx = j_sampled_pts_idx[
                        torch.from_numpy(
                            np.random.choice(
                                len(j_sampled_pts_idx),
                                point_nsample,
                                replace=replace_sample,
                            )
                        )
                        .long()
                        .to(j_sampled_pts_idx.device)
                    ]
                    j_sampled_pts_cam = j_sampled_pts_cam[j_sampled_pts_idx]

                    j_sampled_pts_cam = self.frustum            # inference 

                    # -------------- 从图像-lidar中采样到点,然后化再投影回lidar坐标系z
                    depth_bin = torch.linspace(1, 60, 100).to(j_sampled_pts_cam.device)
                    pixel_points_repeated=j_sampled_pts_cam[...,:2].repeat_interleave(100, dim=0).long().float()    # 否则会存在错位的情况
                    
                    depths_repeated = depth_bin.repeat(j_sampled_pts_cam.shape[0])
                    pixel_3d = torch.cat((pixel_points_repeated, depths_repeated.unsqueeze(1)), dim=1)

                    lidar_3d_sampled = torch.matmul(
                                        i_img2lidar[c_idx : c_idx + 1],
                                        torch.cat([
                                            pixel_3d[..., :2] * pixel_3d[..., 2:3],
                                            pixel_3d[..., 2:3],
                                            torch.ones_like(pixel_3d[..., 2:3]),
                                        ], dim=-1).unsqueeze(-1)
                                         ).squeeze(-1)[..., :3].view(-1, 100, 3)
                    # i_sampled_rgb_jsy, 
                    # import pdb; pdb.set_trace()
                    i_sampled_pts.append(lidar_3d_sampled)
                    # i_sampled_pts_jsy
                    j_sampled_all_depth_mask
                    j_sampled_pts = torch.matmul(
                        i_img2lidar[c_idx : c_idx + 1],
                        torch.cat(
                            [
                                j_sampled_pts_cam[..., :2]
                                * j_sampled_pts_cam[..., 2:3],
                                j_sampled_pts_cam[..., 2:],
                            ],
                            dim=-1,
                        ).unsqueeze(-1),
                    ).squeeze(-1)[..., :3]
                    j_sampled_all_pts.append(j_sampled_pts)
                    j_sampled_all_pts_cam.append(j_sampled_pts_cam[..., :2])
                    j_sampled_all_depth_mask.append(
                        torch.ones_like(j_sampled_pts_cam[:, 0])
                    )
                """ sample pixels """
                if self.ray_sampler_cfg.merged_nsample - point_nsample > 0:
                    pixel_interval = self.ray_sampler_cfg.pixel_interval
                    sky_region = self.ray_sampler_cfg.sky_region
                    tx = torch.arange(
                        0,
                        Ws[c_idx, 0],
                        pixel_interval,
                        device=i_imgs.device,
                        dtype=i_imgs.dtype,
                    )
                    ty = torch.arange(
                        int(sky_region * Hs[c_idx, 0]),     # 排除掉上空部分
                        Hs[c_idx, 0],
                        pixel_interval,
                        device=i_imgs.device,
                        dtype=i_imgs.dtype,
                    )
                    pixels_y, pixels_x = torch.meshgrid(ty, tx)
                    i_pixels_cam = torch.stack([pixels_x, pixels_y], dim=-1)

                    j_sampled_pixels_cam = i_pixels_cam.flatten(0, 1)
                    if self.ray_sampler_cfg.only_img_mask:
                        j_sampled_pixels_mask = ~active_mask[   # 获取图像上的活跃区域
                            bs_idx,
                            c_idx,
                            j_sampled_pixels_cam[:, 1].long(),
                            j_sampled_pixels_cam[:, 0].long(),
                        ]  # (Q,)
                        j_sampled_pixels_idx = j_sampled_pixels_mask.nonzero(
                            as_tuple=True
                        )[0]
                    else:
                        j_sampled_pixels_idx = torch.arange(
                            len(j_sampled_pixels_cam),
                            dtype=torch.long,
                            device=j_sampled_pixels_cam.device,
                        )

                    pixel_nsample = min(
                        len(j_sampled_pixels_idx),
                        self.ray_sampler_cfg.merged_nsample - point_nsample,
                    )
                    j_sampled_pixels_idx = j_sampled_pixels_idx[
                        torch.from_numpy(
                            np.random.choice(
                                len(j_sampled_pixels_idx),
                                pixel_nsample,
                                replace=self.ray_sampler_cfg.replace_sample,
                            )
                        )
                        .long()
                        .to(j_sampled_pixels_idx.device)
                    ]
                    j_sampled_pixels_cam = j_sampled_pixels_cam[j_sampled_pixels_idx]
                    j_sampled_pixels = torch.matmul(
                        i_img2lidar[c_idx : c_idx + 1],
                        torch.cat(
                            [
                                j_sampled_pixels_cam,
                                torch.ones_like(j_sampled_pixels_cam),
                            ],
                            dim=-1,
                        ).unsqueeze(-1),
                    ).squeeze(-1)[..., :3]
                    j_sampled_all_pts.append(j_sampled_pixels)
                    j_sampled_all_pts_cam.append(j_sampled_pixels_cam)
                    j_sampled_all_depth_mask.append(
                        torch.zeros_like(j_sampled_pixels_cam[:, 0])
                    )

                if len(j_sampled_all_pts) > 0:
                    """merge"""
                    j_sampled_all_pts = torch.cat(j_sampled_all_pts, dim=0)
                    j_sampled_all_pts_cam = torch.cat(j_sampled_all_pts_cam, dim=0)
                    j_sampled_all_depth_mask = torch.cat(
                        j_sampled_all_depth_mask, dim=0
                    )

                    unscaled_ray_o = i_cam2lidar[c_idx : c_idx + 1, :3, 3].repeat(
                        j_sampled_all_pts.shape[0], 1
                    )
                    i_sampled_ray_o.append(
                        unscaled_ray_o 
                    )
                    i_sampled_ray_d.append(
                        F.normalize(j_sampled_all_pts - unscaled_ray_o, dim=-1)
                    )
                    sampled_depth = (
                        torch.norm(
                            j_sampled_all_pts - unscaled_ray_o, dim=-1, keepdim=True
                        )
                        
                    )
                    sampled_depth[j_sampled_all_depth_mask == 0] = -1.0
                    i_sampled_depth.append(sampled_depth)
                    i_sampled_rgb.append(
                        i_imgs[
                            c_idx,
                            j_sampled_all_pts_cam[:, 1].long(),
                            j_sampled_all_pts_cam[:, 0].long(),
                        ]
                        / 255.0
                    )
                    img_mask = torch.zeros_like(i_imgs[c_idx:c_idx+1])
                    img_mask[
                            0,
                            j_sampled_all_pts_cam[:, 1].long(),
                            j_sampled_all_pts_cam[:, 0].long(), 
                        ] = 1

                    img_depth = torch.zeros_like(i_imgs[c_idx:c_idx+1])
                    img_depth, _ = self.points2depthmap(j_sampled_pts_cam, img_depth[0])
                    # img_depth[
                    #         0,
                    #         j_sampled_all_pts_cam[:, 1].long(),
                    #         j_sampled_all_pts_cam[:, 0].long(), 
                    #     ] = j_sampled_all_pts_cam[:, 2]

                    i_sampled_rgb_mask.append(
                        img_mask
                    )
                    i_sampled_rgb_gt.append(i_imgs[c_idx] / 255.0)
                    i_sampled_dpeth.append(img_depth)

                    # with open('outputs/{}_{}.pkl'.format(img_metas[bs_idx]['sample_idx'], c_idx), 'wb') as f:
                    #     pickle.dump({
                    #         'pts': i_pts[:, :3].cpu().numpy(),
                    #         'img':  i_imgs[c_idx].cpu().numpy(),
                    #         'pts_cam': i_pts_cam[c_idx][i_pts_mask[c_idx].nonzero(as_tuple=True)[0]][:, :3].cpu().numpy(),
                    #         'sampled_pts': j_sampled_all_pts.cpu().numpy(),
                    #         'ray_o': unscaled_ray_o.cpu().numpy()
                    #     }, f)
                    #     print('save to outputs/{}_{}.pkl'.format(img_metas[bs_idx]['sample_idx'], c_idx))
            batch_ret.append(
                {
                    "ray_o": torch.cat(i_sampled_ray_o, dim=0),     # [3072, 3]
                    "ray_d": torch.cat(i_sampled_ray_d, dim=0),     # [3072, 3]
                    "rgb": torch.cat(i_sampled_rgb, dim=0),         # [3072, 3]
                    "depth": torch.cat(i_sampled_depth, dim=0),         # [3072, 1]    
                    "scaled_points": pts[bs_idx][:, :3] ,        # [25682, 3]
                    'rgb_mask':torch.stack(i_sampled_rgb_mask, dim=0),
                    'pts_sampled':torch.cat(i_sampled_pts, dim=0),
                    'img_rgb':torch.stack(i_sampled_rgb_gt, dim=0),
                    'img_depth':torch.stack(i_sampled_dpeth, dim=0),
                    
                }
            
            )
        return batch_ret

