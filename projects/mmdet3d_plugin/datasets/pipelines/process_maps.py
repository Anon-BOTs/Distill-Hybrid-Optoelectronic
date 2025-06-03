import pickle
import cv2
import numpy as np
import torch
import os
from shapely import affinity, ops
from tqdm import tqdm
from pyquaternion import Quaternion
from PIL import Image
from numba import jit
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.nuscenes import NuScenes
from projects.mmdet3d_plugin.datasets.pipelines.utils import *

map_names = [
            'boston-seaport',
            'singapore-onenorth',
            'singapore-hollandvillage',
            'singapore-queenstown',
        ]

class2label = {
    'road_divider': 0,
    'lane_divider': 0,
    'ped_crossing': 1,
    'contours': 2,
    'others': -1,
}

nuscenes_version = 'v1.0-trainval'
dataroot = '/high_perf_store/surround-view/datasets/'
save_dir = "/high_perf_store3/data-lm/lf/datasets/bev_map"

nusc = NuScenes(version=nuscenes_version, dataroot=dataroot)
nusc_maps = dict()
map_explorer_all = dict()
for map_name in map_names:
    nusc_maps[map_name] = NuScenesMap(dataroot=dataroot, map_name=map_name)
    map_explorer_all[map_name] = NuScenesMapExplorer(nusc_maps[map_name])


def get_bev_hdmap_v2(ego_pose, boxes, scene_token, patch_size, line_vector_dict, ped_vector_list, poly_bound_list, union_polygon, delta_speed = None, delta_yaw = None, bev_size=(1024, 1024), max_channel=3, thickness=4):
    """
    根据ego pose 获取 BEV HD地图图像（语义向量栅格图）
    :param ego_pose: [x, y, yaw]，ego在global坐标系下的位置+朝向
    :param scene_token: 场景token
    :param patch_size: 地图 patch 尺寸（单位：米）
    :param bev_size: 输出 BEV 图像的像素尺寸
    :return: BEV 语义地图图像（PIL.Image）
    """
    # 1. 加载地图和矢量图
    location = nusc.get('log', nusc.get('scene', scene_token)['log_token'])['location']
    nusc_map = nusc_maps[location]
    map_explorer = map_explorer_all[location]
    patch_angle = np.degrees(ego_pose[2])  # yaw angle in degrees
    # 2. 提取地图矢量信息（line, polygon）
    # line_vector_dict = line_geoms_to_vectors(line_geom)
    # ped_vector_list = line_geoms_to_vectors(ped_geom)['ped_crossing']
    # poly_bound_list = poly_geoms_to_vectors(polygon_geom)
    # 3. 聚合并赋予语义类别
    vectors = []
    for line_type, vects in line_vector_dict.items():
        for line, length in vects:
            vectors.append((line.astype(float), length, class2label.get(line_type, -1)))
    for ped_line, length in ped_vector_list:
        vectors.append((ped_line.astype(float), length, class2label.get('ped_crossing', -1)))
    for contour, length in poly_bound_list:
        vectors.append((contour.astype(float), length, class2label.get('contours', -1)))
    # 4. global → ego → 图像坐标
    filtered_vectors = []
    yaw = ego_pose[2]
    R = np.array([[np.cos(yaw), -np.sin(yaw)],
                    [np.sin(yaw),  np.cos(yaw)]])
    t = np.array([ego_pose[0], ego_pose[1]])
    map_h, map_w = bev_size
    scale = map_w / patch_size
    for pts, length, type_id in vectors:
        if type_id == -1:
            continue
        pts_2d = pts[:, :2]
        pts_2d = pts_2d   # ego 旋转对齐
        pts_2d = pts_2d * scale  # 米 → 像素
        # ego → 图像坐标
        pts_2d[:, 0] = map_w / 2 + pts_2d[:, 0]
        pts_2d[:, 1] = map_h - (map_h / 2 + pts_2d[:, 1])
        filtered_vectors.append({'pts': pts_2d, 'type': type_id})
    # 5. 创建空图像并绘制线条
    color_map = np.ones((map_h, map_w, 3), dtype=np.uint8) * 255
    if isinstance(union_polygon, MultiPolygon):
        for poly in union_polygon.geoms:
            # 将每个 Polygon 转换为全局坐标系坐标
            union_polygon_coords_ego = np.array(list(poly.exterior.coords))
            # 进行 global → ego 坐标系转换
            # union_polygon_coords_ego = (union_polygon_coords - t) @ R.T  # 先平移，再旋转
            # 将 ego 坐标系的 polygon 转换为图像坐标系
            union_polygon_coords_ego[:, 0] = map_w / 2 + union_polygon_coords_ego[:, 0] * scale
            union_polygon_coords_ego[:, 1] = map_h - (map_h / 2 + union_polygon_coords_ego[:, 1] * scale)
            # 绘制阴影（浅蓝色，位于底层）
            cv2.fillPoly(color_map, [union_polygon_coords_ego.astype(np.int32)], color=(211, 211, 211))  # 浅蓝色
    # 如果是 Polygon，直接处理
    elif isinstance(union_polygon, Polygon):
        # 将 Shapely Polygon 转换为全局坐标系坐标
        union_polygon_coords_ego = np.array(list(union_polygon.exterior.coords))
        # 进行 global → ego 坐标系转换
        # union_polygon_coords_ego = (union_polygon_coords - t) @ R.T  # 先平移，再旋转
        # 将 ego 坐标系的 polygon 转换为图像坐标系
        union_polygon_coords_ego[:, 0] = map_w / 2 + union_polygon_coords_ego[:, 0] * scale
        union_polygon_coords_ego[:, 1] = map_h - (map_h / 2 + union_polygon_coords_ego[:, 1] * scale)
        # 绘制阴影（浅蓝色，位于底层）
        cv2.fillPoly(color_map, [union_polygon_coords_ego.astype(np.int32)], color=(211, 211, 211))  # 浅蓝色
    color_dict = {
        class2label.get('road_divider', -1): (0, 0, 255),      # 红
        class2label.get('lane_divider', -1): (0, 255, 0),      # 绿
        class2label.get('ped_crossing', -1): (255, 0, 0),      # 蓝
        class2label.get('contours', -1): (0, 255, 255),        # 黄
    }
    for vec in filtered_vectors:
        pts = vec['pts'].astype(np.int32)
        color = color_dict.get(vec['type'], (128, 128, 128))  # 默认为灰色
        if len(pts) >= 2:
            cv2.polylines(color_map, [pts], isClosed=False, color=color, thickness=thickness)
    for box in boxes:
        # 获取 box 的 3D 角点，并只保留 x, y 平面投影
        corners = box.corners()[:2].T  # shape: (8, 2)，取前两个维度
        pts = corners[[0, 1, 5, 4]]  # BEV 上 box 的前后左右顶角
        # global → ego
        pts = pts - t
        pts = pts @ R
        # ego → 图像坐标
        pts *= scale
        pts[:, 0] = map_w / 2 + pts[:, 0]
        pts[:, 1] = map_h - (map_h / 2 + pts[:, 1])
        pts = pts.astype(np.int32)
        # cv2.polylines(color_map, [pts], isClosed=True, color=(255, 0, 255), thickness=thickness)
        cv2.fillPoly(color_map, [pts.reshape(-1, 1, 2)], color=(255, 0, 255))  # 黑色填充
        # cv2.circle(color_map, tuple(pts[0]), radius=3, color=(255, 0, 255), thickness=-1)  # 标记 front-left
    # 5.6 绘制 ego vehicle（用矩形表示）
    ego_length = 4.0  # meter，假设自车长度为4米
    ego_width = 2.0   # meter，假设自车宽度为2米
    # 自车在 ego 坐标系下的轮廓矩形（朝向由旋转矩阵控制）
    ego_box = np.array([
        [ ego_length/2,  ego_width/2],   # front-right
        [ ego_length/2, -ego_width/2],   # front-left
        [-ego_length/2, -ego_width/2],   # back-left
        [-ego_length/2,  ego_width/2]    # back-right
    ])
    # ego → 图像坐标
    ego_box *= scale
    ego_box[:, 0] = map_w / 2 + ego_box[:, 0]
    ego_box[:, 1] = map_h - (map_h / 2 + ego_box[:, 1])
    ego_box = ego_box.astype(np.int32)
    # 绘制矩形框表示 ego
    # cv2.polylines(color_map, [ego_box], isClosed=True, color=(0, 0, 0), thickness=thickness)  # 黑色矩形框表示 ego
    cv2.fillPoly(color_map, [ego_box.reshape(-1, 1, 2)], color=(0, 0, 0))  # 黑色填充
    # if delta_speed and delta_yaw:
    #     cv2.putText(color_map, f'delta_speed: {delta_speed:.2f}, delta_yaw: {delta_yaw:.2f}', (50, 800), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255))
    # 6. 转换为 PIL Image 并返回
    pil_image = Image.fromarray(color_map)
    # 如果需要保存图像，可以在这里保存
    # pil_image.save(os.path.join(self.save_dir, f'{self.frame_index}_bev_test.png'))
    return pil_image


def get_bev_hdmap(ego_pose, boxes, scene_token, patch_size, line_vector_dict, ped_vector_list, poly_bound_list, bev_size=(1024, 1024), max_channel=3, thickness=4):
    """
    根据ego pose 获取 BEV HD地图图像（语义向量栅格图）
    :param ego_pose: [x, y, yaw]，ego在global坐标系下的位置+朝向
    :param scene_token: 场景token
    :param patch_size: 地图 patch 尺寸（单位：米）
    :param bev_size: 输出 BEV 图像的像素尺寸
    :return: BEV 语义地图图像（PIL.Image）
    """
    # 1. 加载地图和矢量图
    location = nusc.get('log', nusc.get('scene', scene_token)['log_token'])['location']
    nusc_map = nusc_maps[location]
    map_explorer = map_explorer_all[location]

    patch_angle = np.degrees(ego_pose[2])  # yaw angle in degrees

    # 2. 提取地图矢量信息（line, polygon）
    # line_vector_dict = line_geoms_to_vectors(line_geom)
    # ped_vector_list = line_geoms_to_vectors(ped_geom)['ped_crossing']
    # poly_bound_list = poly_geoms_to_vectors(polygon_geom)

    # 3. 聚合并赋予语义类别
    vectors = []
    for line_type, vects in line_vector_dict.items():
        for line, length in vects:
            vectors.append((line.astype(float), length, class2label.get(line_type, -1)))
    for ped_line, length in ped_vector_list:
        vectors.append((ped_line.astype(float), length, class2label.get('ped_crossing', -1)))
    for contour, length in poly_bound_list:
        vectors.append((contour.astype(float), length, class2label.get('contours', -1)))

    # 4. global → ego → 图像坐标
    filtered_vectors = []
    yaw = ego_pose[2]
    R = np.array([[np.cos(yaw), -np.sin(yaw)],
                    [np.sin(yaw),  np.cos(yaw)]])
    t = np.array([ego_pose[0], ego_pose[1]])

    map_h, map_w = bev_size
    scale = map_w / patch_size

    for pts, length, type_id in vectors:
        if type_id == -1:
            continue
        pts_2d = pts[:, :2]
        pts_2d = pts_2d   # ego 旋转对齐
        pts_2d = pts_2d * scale  # 米 → 像素
        # ego → 图像坐标
        pts_2d[:, 0] = map_w / 2 + pts_2d[:, 0]
        pts_2d[:, 1] = map_h - (map_h / 2 + pts_2d[:, 1])
        filtered_vectors.append({'pts': pts_2d, 'type': type_id})

    # 5. 创建空图像并绘制线条
    color_map = np.ones((map_h, map_w, 3), dtype=np.uint8) * 255

    color_dict = {
        class2label.get('road_divider', -1): (0, 0, 255),      # 红
        class2label.get('lane_divider', -1): (0, 255, 0),      # 绿
        class2label.get('ped_crossing', -1): (255, 0, 0),      # 蓝
        class2label.get('contours', -1): (0, 255, 255),        # 黄
    }

    for vec in filtered_vectors:
        pts = vec['pts'].astype(np.int32)
        color = color_dict.get(vec['type'], (128, 128, 128))  # 默认为灰色
        if len(pts) >= 2:
            cv2.polylines(color_map, [pts], isClosed=False, color=color, thickness=thickness)

    for box in boxes:
        # 获取 box 的 3D 角点，并只保留 x, y 平面投影
        corners = box.corners()[:2].T  # shape: (8, 2)，取前两个维度
        pts = corners[[0, 1, 5, 4]]  # BEV 上 box 的前后左右顶角

        # global → ego
        pts = pts - t
        pts = pts @ R

        # ego → 图像坐标
        pts *= scale
        pts[:, 0] = map_w / 2 + pts[:, 0]
        pts[:, 1] = map_h - (map_h / 2 + pts[:, 1])
        pts = pts.astype(np.int32)

        # cv2.polylines(color_map, [pts], isClosed=True, color=(255, 0, 255), thickness=thickness)
        cv2.fillPoly(color_map, [pts.reshape(-1, 1, 2)], color=(255, 0, 255))  # 黑色填充
        # cv2.circle(color_map, tuple(pts[0]), radius=3, color=(255, 0, 255), thickness=-1)  # 标记 front-left

    # 5.6 绘制 ego vehicle（用矩形表示）
    ego_length = 4.0  # meter，假设自车长度为4米
    ego_width = 2.0   # meter，假设自车宽度为2米

    # 自车在 ego 坐标系下的轮廓矩形（朝向由旋转矩阵控制）
    ego_box = np.array([
        [ ego_length/2,  ego_width/2],   # front-right
        [ ego_length/2, -ego_width/2],   # front-left
        [-ego_length/2, -ego_width/2],   # back-left
        [-ego_length/2,  ego_width/2]    # back-right
    ])

    # ego → 图像坐标
    ego_box *= scale
    ego_box[:, 0] = map_w / 2 + ego_box[:, 0]
    ego_box[:, 1] = map_h - (map_h / 2 + ego_box[:, 1])
    ego_box = ego_box.astype(np.int32)

    # 绘制矩形框表示 ego
    # cv2.polylines(color_map, [ego_box], isClosed=True, color=(0, 0, 0), thickness=thickness)  # 黑色矩形框表示 ego
    cv2.fillPoly(color_map, [ego_box.reshape(-1, 1, 2)], color=(0, 0, 0))  # 黑色填充

    # 6. 转换为 PIL Image 并返回
    pil_image = Image.fromarray(color_map)

    # 如果需要保存图像，可以在这里保存
    # pil_image.save(os.path.join(self.save_dir, f'{self.frame_index}_bev_test.png'))

    return pil_image

def get_ego_info(cam_token, sample):
    cam_record = nusc.get('sample_data', cam_token)
    cs_record = nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', cam_record['ego_pose_token'])
    rotation = Quaternion(pose_record['rotation'])
    # 使用 quaternion_yaw 函数提取 yaw 角
    ego_yaw = quaternion_yaw(rotation)
    ego_pose_token_1 = sample['data']['LIDAR_TOP']
    ego_pose_1 = nusc.get('ego_pose', ego_pose_token_1)

    # 获取下一个 sample 的 token
    next_sample_token = sample['next']
    if next_sample_token != '':  # 确保有下一个 sample
        next_sample = nusc.get('sample', next_sample_token)
        ego_pose_token_2 = next_sample['data']['LIDAR_TOP']
        ego_pose_2 = nusc.get('ego_pose', ego_pose_token_2)

        # 计算时间差
        timestamp_1 = ego_pose_1['timestamp']
        timestamp_2 = ego_pose_2['timestamp']
        delta_t = (timestamp_2 - timestamp_1) / 1e6  # 转换为秒

        # 提取位置信息
        x1, y1 = ego_pose_1['translation'][0], ego_pose_1['translation'][1]
        x2, y2 = ego_pose_2['translation'][0], ego_pose_2['translation'][1]

        # 计算位移大小
        displacement = np.linalg.norm([x2 - x1, y2 - y1])

        # 计算速度大小
        speed = displacement / delta_t

    else:
        # 如果没有下一个 sample，返回 NaN
        speed = np.nan

    return speed, ego_yaw

def main():
    infos = {'metadata' : {}, "infos" : []}
    for current_scene in tqdm(nusc.sample):
        try:
            # current_sample = nusc.get('sample', current_scene['first_sample_token'])
            # first_sample_record = current_sample
            first_sample_record = current_scene

            cam_token = first_sample_record['data']['CAM_FRONT']
            cam_boxes = nusc.get_boxes(cam_token)
            cam_record = nusc.get('sample_data', cam_token)
            pose_record = nusc.get('ego_pose', cam_record['ego_pose_token'])        
            imsize = (cam_record['width'], cam_record['height'])
            ego2global_translation = np.array(pose_record['translation'])
            rotation = Quaternion(pose_record['rotation'])
            map_pose = ego2global_translation[:2]
            patch_box = (map_pose[0], map_pose[1], 102.4, 102.4)  # TODO 102.4 is hardcode
            patch_angle = quaternion_yaw(rotation) / np.pi * 180
            ego_pose = np.concatenate([ego2global_translation[:2], np.array([quaternion_yaw(rotation)])])
            # location = nusc.get('log', nusc.get('scene', current_scene['token'])['log_token'])['location']

            location = nusc.get('log', nusc.get('scene', current_scene['scene_token'])['log_token'])['location']
            nusc_map = nusc_maps[location]
            map_explorer = map_explorer_all[location]
            line_geom = get_map_geom(patch_box, patch_angle, ['road_divider', 'lane_divider'], nusc_map, map_explorer)
            line_vector_dict = line_geoms_to_vectors(line_geom)  # line to points
            ped_geom = get_map_geom(patch_box, patch_angle, ['ped_crossing'], nusc_map, map_explorer)
            ped_vector_list = line_geoms_to_vectors(ped_geom)['ped_crossing']
            polygon_geom = get_map_geom(patch_box, patch_angle, ['road_segment', 'lane'], nusc_map, map_explorer)
            poly_bound_list = poly_geoms_to_vectors(polygon_geom)
            all_velocities = [nusc.box_velocity(token)[:2] for token in first_sample_record['anns']]
            # ego_velocity = self.get_ego_velocity(first_sample_record)
            scene_token = first_sample_record['scene_token']

            all_polygons = []
            for _, polygons in polygon_geom:
                for polygon in polygons:
                    all_polygons.extend(list(polygon.geoms))
            union_polygon = ops.unary_union(all_polygons)

            bev_map = get_bev_hdmap_v2(ego_pose, cam_boxes, scene_token, 102.4, line_vector_dict, ped_vector_list, poly_bound_list, union_polygon)

            bev_map.save(os.path.join(save_dir, f"mask_{current_scene['token']}.png"))

            cam_token = first_sample_record['data']['CAM_FRONT']
            speed, yaw = get_ego_info(cam_token, first_sample_record)
            infos['infos'].append({
                'token' : current_scene['token'],
                'speed' : speed,
                'yaw' : yaw,
                'img_filename' : os.path.join(save_dir, f"{current_scene['token']}.png") 
            })
        except Exception as e:
            print(e)
    # with open('vector_trainval_infos.pkl', 'wb') as f:
    #     pickle.dump(infos, f)

if __name__ == '__main__':
    main()