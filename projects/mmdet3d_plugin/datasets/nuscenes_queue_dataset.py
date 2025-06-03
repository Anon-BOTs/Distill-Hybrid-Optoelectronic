# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import pyquaternion
import tempfile
import copy
import torch

from nuscenes.utils.data_classes import Box as NuScenesBox
from os import path as osp
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from mmcv.parallel import DataContainer as DC

from mmdet.datasets import DATASETS
from mmdet3d.core import show_result
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet3d.datasets.pipelines import Compose
from .nuscenes_sweep_dataset import NuScenesSweepDataset


@DATASETS.register_module()
class NuscenesQueueDataset(NuScenesSweepDataset):
    def __init__(self, future_length=1, queue_length = 0, **kwargs):
        super(NuscenesQueueDataset, self).__init__(**kwargs)
        self.queue_length = queue_length
        self.future_length = future_length
    
    def union2one(self, previous_queue, future_queue):
        # 1. get transformation from all frames to current (reference) frame
        ref_meta = previous_queue[-1]['img_metas'].data
        valid_scene_token = ref_meta['scene_token']
        # compute reference e2g_transform and g2e_transform.
        ref_e2g_translation = ref_meta['ego2global_translation']
        ref_e2g_rotation = ref_meta['ego2global_rotation']
        ref_e2g_transform = transform_matrix(
            ref_e2g_translation, Quaternion(ref_e2g_rotation), inverse=False)
        ref_g2e_transform = transform_matrix(
            ref_e2g_translation, Quaternion(ref_e2g_rotation), inverse=True)
        # compute reference l2e_transform and e2l_transform
        ref_l2e_translation = ref_meta['lidar2ego_translation']
        ref_l2e_rotation = ref_meta['lidar2ego_rotation']
        ref_l2e_transform = transform_matrix(
            ref_l2e_translation, Quaternion(ref_l2e_rotation), inverse=False)
        ref_e2l_transform = transform_matrix(
            ref_l2e_translation, Quaternion(ref_l2e_rotation), inverse=True)

        queue = previous_queue[:-1] + future_queue
        pts_list = [each['points'].data for each in queue]
        imgs_list = [each['img'].data for each in queue]
        total_cur2ref_lidar_transform = []
        total_ref2cur_lidar_transform = []
        total_pts_list = []
        lidar2img = []
        lidar2cam = []
        cam_params = []
        for i, each in enumerate(queue):
            meta = each['img_metas'].data

            # store points in the current frame.
            cur_pts = pts_list[i].cpu().numpy().copy()
            # cur_pts[:, -1] = i
            cur_pts = np.concatenate([cur_pts, np.ones((len(cur_pts), 1)) * i], axis=-1)
            total_pts_list.append(cur_pts)
            lidar2img.append(meta['lidar2img'])
            lidar2cam.append(meta['lidar2cam'])
            cam_params.append(meta['cam_params'])

            # store the transformation from current frame to reference frame.
            curr_e2g_translation = meta['ego2global_translation']
            curr_e2g_rotation = meta['ego2global_rotation']
            curr_e2g_transform = transform_matrix(
                curr_e2g_translation, Quaternion(curr_e2g_rotation), inverse=False)
            curr_g2e_transform = transform_matrix(
                curr_e2g_translation, Quaternion(curr_e2g_rotation), inverse=True)

            curr_l2e_translation = meta['lidar2ego_translation']
            curr_l2e_rotation = meta['lidar2ego_rotation']
            curr_l2e_transform = transform_matrix(
                curr_l2e_translation, Quaternion(curr_l2e_rotation), inverse=False)
            curr_e2l_transform = transform_matrix(
                curr_l2e_translation, Quaternion(curr_l2e_rotation), inverse=True)

            # compute future to reference matrix.
            cur_lidar_to_ref_lidar = (curr_l2e_transform.T @
                                      curr_e2g_transform.T @
                                      ref_g2e_transform.T @
                                      ref_e2l_transform.T)
            total_cur2ref_lidar_transform.append(cur_lidar_to_ref_lidar)

            # compute reference to future matrix.
            ref_lidar_to_cur_lidar = (ref_l2e_transform.T @
                                      ref_e2g_transform.T @
                                      curr_g2e_transform.T @
                                      curr_e2l_transform.T)
            total_ref2cur_lidar_transform.append(ref_lidar_to_cur_lidar)

        # 2. Parse previous and future can_bus information.
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        ref_meta = previous_queue[-1]['img_metas'].data

        # 2.2. Previous
        for i, each in enumerate(previous_queue):
            metas_map[i] = each['img_metas'].data

            if 'aug_param' in each:
                metas_map[i]['aug_param'] = each['aug_param']

            if metas_map[i]['scene_token'] != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['scene_token']
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                # Set the original point of this motion.
                new_can_bus = copy.deepcopy(metas_map[i]['can_bus'])
                new_can_bus[:3] = 0
                new_can_bus[-1] = 0
                metas_map[i]['can_bus'] = new_can_bus
            else:
                metas_map[i]['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                # Compute the later waypoint.
                # To align the shift and rotate difference due to the BEV.
                new_can_bus = copy.deepcopy(metas_map[i]['can_bus'])
                new_can_bus[:3] = tmp_pos - prev_pos
                new_can_bus[-1] = tmp_angle - prev_angle
                metas_map[i]['can_bus'] = new_can_bus
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)

            # compute cur_lidar_to_ref_lidar transformation matrix for quickly align generated
            #  bev features to the reference frame.
            metas_map[i]['ref_lidar_to_cur_lidar'] = total_ref2cur_lidar_transform[i]

        # 2.3. Future
        current_scene_token = ref_meta['scene_token']
        ref_can_bus = None
        future_can_bus = []
        future2ref_lidar_transform = []
        ref2future_lidar_transform = []
        for i, each in enumerate(future_queue):
            future_meta = each['img_metas'].data
            if future_meta['scene_token'] != current_scene_token:
                break
            # store the transformation:
            future2ref_lidar_transform.append(
                total_cur2ref_lidar_transform[i + len(previous_queue) - 1]
            )  # current -> reference.
            ref2future_lidar_transform.append(
                total_ref2cur_lidar_transform[i + len(previous_queue) - 1]
            )  # reference -> current.

            # can_bus information.
            if i == 0:
                new_can_bus = copy.deepcopy(future_meta['can_bus'])
                new_can_bus[:3] = 0
                new_can_bus[-1] = 0
                future_can_bus.append(new_can_bus)
                ref_can_bus = copy.deepcopy(future_meta['can_bus'])
            else:
                new_can_bus = copy.deepcopy(future_meta['can_bus'])

                new_can_bus_pos = np.array([0, 0, 0, 1]).reshape(1, 4)
                ref2prev_lidar_transform = ref2future_lidar_transform[-2]
                cur2ref_lidar_transform = future2ref_lidar_transform[-1]
                new_can_bus_pos = new_can_bus_pos @ cur2ref_lidar_transform @ ref2prev_lidar_transform

                new_can_bus_angle = new_can_bus[-1] - ref_can_bus[-1]
                new_can_bus[:3] = new_can_bus_pos[:, :3]
                new_can_bus[-1] = new_can_bus_angle
                future_can_bus.append(new_can_bus)
                ref_can_bus = copy.deepcopy(future_meta['can_bus'])

        ret_queue = previous_queue[-1]
        ret_queue['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        ret_queue.pop('aug_param', None)

        metas_map[len(previous_queue) - 1]['future_can_bus'] = np.array(future_can_bus)
        metas_map[len(previous_queue) - 1]['future2ref_lidar_transform'] = (
            np.array(future2ref_lidar_transform))
        metas_map[len(previous_queue) - 1]['ref2future_lidar_transform'] = (
            np.array(ref2future_lidar_transform))
        metas_map[len(previous_queue) - 1]['total_cur2ref_lidar_transform'] = (
            np.array(total_cur2ref_lidar_transform))
        metas_map[len(previous_queue) - 1]['total_ref2cur_lidar_transform'] = (
            np.array(total_ref2cur_lidar_transform))
        metas_map[len(previous_queue) - 1]['all_lidar2img'] = np.stack(lidar2img)
        metas_map[len(previous_queue) - 1]['all_lidar2cam'] = np.stack(lidar2cam)
        metas_map[len(previous_queue) - 1]['lidar2img'] = np.stack(lidar2img)[0]
        metas_map[len(previous_queue) - 1]['lidar2cam'] = np.stack(lidar2cam)[0]     
        metas_map[len(previous_queue) - 1]['cam_params'] = cam_params

        ### TODO
        ret_queue['img_metas'] = DC(metas_map[0], cpu_only=True)
        # ret_queue.pop('points')
        ret_queue['points'] = DC(
            torch.from_numpy(np.concatenate(total_pts_list, 0)), cpu_only=False)
        if len(future_can_bus) < 1 + self.future_length:
            return None
        return ret_queue

    def _prepare_data_info_single(self, index, aug_param=None):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = input_dict
        for transform, transform_type in zip(self.pipeline.transforms, self.pipeline_types):
            if self._skip_type_keys is not None and transform_type in self._skip_type_keys:
                continue
            example = transform(example)

        if self.filter_empty_gt and \
                (example is None or
                    ~(example['gt_labels_3d']._data != -1).any()):
            return None
        return example
        
    def prepare_train_data(self, index, rand_interval=None):
        """
        Modified from BEVFormer:CustomNuScenesDataset,
            BEVFormer logits: randomly select (queue_length-1) previous images.
            Modified logits: directly select (queue_length) previous images.
        """
        #@TODO
        # rand_interval = None
        # rand_interval = (
        #     rand_interval if rand_interval is not None else
        #     np.random.choice(self.rand_frame_interval, 1)[0]
        # )
        rand_interval = 1
        # 1. get previous camera information.
        previous_queue = []
        previous_index_list = list(range(
            index - self.queue_length * rand_interval, index, rand_interval))
        previous_index_list = sorted(previous_index_list)
        if rand_interval < 0:  # the inverse chain.
            previous_index_list = previous_index_list[::-1]
        previous_index_list.append(index)
        aug_param = None
        for i in previous_index_list:
            i = min(max(0, i), len(self.data_infos) - 1)
            example = self._prepare_data_info_single(i, aug_param=aug_param)

            aug_param = copy.deepcopy(example['aug_param']) if 'aug_param' in example else None
            if example is None:
                return None
            previous_queue.append(example)

        # 2. get future lidar information.
        future_queue = []
        # Future: from current to future frames.
        # use current frame as the 0-th future.
        future_index_list = list(range(
            index, index + (self.future_length + 1) * rand_interval, rand_interval))
        future_index_list = sorted(future_index_list)
        if rand_interval < 0:  # the inverse chain.
            future_index_list = future_index_list[::-1]
        has_future = False
        for i in future_index_list:
            i = min(max(0, i), len(self.data_infos) - 1)
            example = self._prepare_data_info_single(i)
            if example is None and not has_future:
                return None
            future_queue.append(example)
            has_future = True
        return self.union2one(previous_queue, future_queue)
