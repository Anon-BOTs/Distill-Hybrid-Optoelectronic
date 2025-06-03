import numpy as np
import torch
import torch.nn.functional as F
import cv2
import os
import matplotlib.cm as cm
from PIL import Image, ImageDraw, ImageFont

_RED = (0, 0, 255)  # direction

_GREEN = (0, 255, 0)
_YELLOW = (0, 255, 255)
_BLUE = (255, 0, 0)
_ORANGE = (0, 165, 255)

_BLUE2 = (158, 168, 3)

_GRAY = (218, 227, 218)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_PURPLE = (214, 112, 218)

_static = (229, 35, 98)

nuscenes_set_colors = {
    ("car",): _BLUE,
    ("truck", "construction_vehicle", "bus", "trailer"): _BLUE2,
    ("motorcycle", "bicycle"): _GREEN,
    ("pedestrian",): _YELLOW,
    ("traffic_cone",): _ORANGE
}
nuscenes_set_colors = {cls: color for clses, color in nuscenes_set_colors.items() for cls in clses}

bhx_set_colors = {
    ('car', 'sport_utility_vehicle', 'van', 'Car', 'dummy_car', 'unknown'): _BLUE,
    ('bus', 'tanker', 'truck', 'trailer', 'other_vehicle', 'Truck',): _BLUE2,
    ('cyclist', 'motorcyclist', 'tricyclist', 'parked_cycle', 'Cyclist', "handcart", 'dummy_cyclist',
     'bicycle', 'motorcycle',): _GREEN,
    ('pedestrian', 'Pedestrian', 'people', 'dummy',): _YELLOW,
    ("traffic_cone", "barrier", "crash_pile",): _ORANGE
}
bhx_set_colors = {cls: color for clses, color in bhx_set_colors.items() for cls in clses}

trk_colors = np.random.randint(256, size=(30, 3)).tolist()

merge_h = 600  # 2 x 3
merge_w = 1600


def vis_depth_on_img(img, depth_map, save_path, max_depth=80):
    depth_map[depth_map > 80] = 80
    color_map = cm.get_cmap(name='jet')
    h, w = depth_map.shape
    for col in range(h):
        for row in range(w):
            if depth_map[col, row] != 0:
                pt_color_value = min(255, int(min(depth_map[col, row], max_depth) / max_depth * 255))
                color = color_map(pt_color_value)
                color = [c * 255 for c in color[:3]]
                cv2.circle(img, (row, col), 1, color)
    cv2.imwrite(save_path, img)


def get_rotz_matrix_array(rz):
    """
    :param rz: [N, ]
    :return: [N, 3, 3]
    """
    bbox_num = rz.shape[0]
    temp_zeros = np.zeros_like(rz)
    temp_ones = np.ones_like(rz)
    mat = [np.cos(rz), -np.sin(rz), temp_zeros.copy(),
           np.sin(rz), np.cos(rz), temp_zeros.copy(),
           temp_zeros.copy(), temp_zeros.copy(), temp_ones.copy()]
    return np.stack(mat, axis=1).reshape(bbox_num, 3, 3)


def gen_dx_bx(xbound, ybound, zbound, **kwargs):
    dx = np.array([row[2] for row in [xbound, ybound, zbound]])
    bx = np.array([row[0] for row in [xbound, ybound, zbound]])
    nx = np.array([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]], dtype=np.long)

    return dx, bx, nx

def gen_3d_object_corners_array(location, dimension, yaw):
    """
    :param location: [N, 3]
    :param dimension: [N, 3]
    :param yaw: [N,]
    :return: [N, 8, 3]
    """
    l, w, h = dimension[:, 0:1], dimension[:, 1:2], dimension[:, 2:3]
    x_corners = np.concatenate([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], axis=1)
    y_corners = np.concatenate([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], axis=1)
    z_corners = np.concatenate([-h / 2., -h / 2., -h / 2., -h / 2., h / 2., h / 2., h / 2., h / 2.], axis=1)
    box_points_coords = np.stack((x_corners, y_corners, z_corners), axis=1)

    mat = get_rotz_matrix_array(yaw)
    corners_3d = np.matmul(mat, box_points_coords)
    corners_3d = corners_3d + location[:, :, np.newaxis]
    corners_3d = corners_3d.transpose(0, 2, 1)
    return corners_3d


def get_rotz_matrix(rz):
    # rz: float
    mat = [np.cos(rz), -np.sin(rz), 0,
           np.sin(rz), np.cos(rz), 0,
           0, 0, 1]
    return mat


def get_rotz_matrix_tensor(rz):
    """
    :param rz: [N, ]
    :return: [N, 3, 3]
    """
    bbox_num = rz.shape[0]
    temp_zeros = torch.zeros_like(rz)
    temp_ones = torch.ones_like(rz)
    mat = [torch.cos(rz), -torch.sin(rz), temp_zeros.clone(),
           torch.sin(rz), torch.cos(rz), temp_zeros.clone(),
           temp_zeros.clone(), temp_zeros.clone(), temp_ones.clone()]
    return torch.stack(mat, dim=1).reshape(bbox_num, 3, 3)


def gen_3d_object_corners_tensor(location, dimension, yaw):
    """
    :param location: [N, 3]
    :param dimension: [N, 3]
    :param yaw: [N,]
    :return: [N, 8, 3]
    """
    l, w, h = dimension[:, 0:1], dimension[:, 1:2], dimension[:, 2:3]
    x_corners = torch.cat([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], dim=1)
    y_corners = torch.cat([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], dim=1)
    z_corners = torch.cat([-h / 2., -h / 2., -h / 2., -h / 2., h / 2., h / 2., h / 2., h / 2.], dim=1)
    box_points_coords = torch.stack((x_corners, y_corners, z_corners), dim=1)

    mat = get_rotz_matrix_tensor(yaw)
    corners_3d = torch.matmul(mat, box_points_coords)

    corners_3d = corners_3d + location.unsqueeze(-1)
    corners_3d = corners_3d.permute(0, 2, 1)
    return corners_3d


def gen_3d_object_corners(location, dimension, yaw):
    """
    gen 3d 8 corners
    :param location: xyz, [3,]
    :param dimension: lwh, [3,]
    :param yaw: int
    :return: [8, 3]
    """

    l, w, h = dimension[0], dimension[1], dimension[2]
    x_corners = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y_corners = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
    z_corners = np.array([-h / 2., -h / 2., -h / 2., -h / 2., h / 2., h / 2., h / 2., h / 2.])
    box_points_coords = np.vstack((x_corners, y_corners, z_corners))

    mat = get_rotz_matrix(yaw)
    mat = np.array(mat).reshape(3, 3)

    corners_3d = np.matmul(mat, box_points_coords)
    corners_3d = corners_3d + np.array(location[0:3]).reshape(3, 1)
    corners_3d = corners_3d.transpose()
    return corners_3d


def proj_3d_object_to_2d(location, dimension, yaw, intrin, rot, tran, img_shape):
    """
    gen 3d bbox eight 2d corners coor
    :param location: xyz, [3,], lidar coor
    :param dimension: lwh, [3,], lwh
    :param yaw: int
    :param intrin: [3, 3]
    :param rot: cam2lidar_rot [3, 3]
    :param tran: cam2lidar_tran [3,]
    :param img_shape: [w, h]
    :return: [8, 2]
    """
    cam2lidar = np.hstack([rot, tran.reshape(3, 1)])
    cam2lidar = np.vstack([cam2lidar, np.array([0, 0, 0, 1]).reshape(1, 4)])

    corners_3d_lidar = gen_3d_object_corners(location, dimension, yaw)
    corners_3d_lidar = np.hstack([corners_3d_lidar, np.ones((corners_3d_lidar.shape[0], 1))])
    corners_3d_cam = np.matmul(np.linalg.inv(cam2lidar), corners_3d_lidar.T)

    corners_2d = np.matmul(intrin, corners_3d_cam[:3, :]).T
    corners_2d[:, :2] = corners_2d[:, :2] / corners_2d[:, 2:3]

    img_w, img_h = img_shape
    inds = (corners_2d[:, 0] > 0) & (corners_2d[:, 0] < img_w) & \
           (corners_2d[:, 1] > 0) & (corners_2d[:, 1] < img_h) & \
           (corners_2d[:, 2] > 0)
    if inds.sum() < 4:
        return None
    return corners_2d


def lidarcorners_to_2dboxs_tensor(corners_3d_lidar, lidar2cam, intrin, remap_shape):
    """
    :param corners_3d_lidar: [N, 8, 4]
    :param lidar2cam: [4, 4]
    :param intrin: [3, 3]
    :param remap_shape: [w, h]
    :return:
    """
    if intrin.shape[0] == 4:
        intrin = intrin[:3, :3]

    img_w, img_h = remap_shape

    corners_3d_cam = lidar2cam[None, ...] @ corners_3d_lidar.permute(0, 2, 1)
    corners_2d = intrin[None, ...] @ corners_3d_cam[:, :3, :]
    corners_2d = corners_2d.permute(0, 2, 1)
    corners_2d[:, :, :2] = corners_2d[:, :, :2] / corners_2d[:, :, 2:3]

    inds = (corners_2d[:, :, 0] > 0) & (corners_2d[:, :, 0] < img_w) & \
           (corners_2d[:, :, 1] > 0) & (corners_2d[:, :, 1] < img_h) & \
           (corners_2d[:, :, 2] > 0)
    valid_mask = inds.sum(1) >= 2
    corners_2d = corners_2d[:, :, :2]  # [N, 8, 2]

    assert corners_3d_lidar.shape[0] == corners_2d.shape[0]
    bbox_num = corners_3d_lidar.shape[0]

    bbox_2ds = []
    for bbox_i in range(bbox_num):
        corner_2d = corners_2d[bbox_i, ...]
        x_min = max(0, corner_2d[:, 0].min())
        x_max = min(img_w, corner_2d[:, 0].max())
        y_min = max(0, corner_2d[:, 1].min())
        y_max = min(img_h, corner_2d[:, 1].max())
        bbox_2ds.append([x_min, y_min, x_max, y_max])
    bbox_2ds = torch.tensor(bbox_2ds, dtype=torch.float, device=corners_3d_lidar.device)
    return bbox_2ds, valid_mask


def lidarcorners_to_2dboxs(corners_3d_lidar, lidar2cam, intrin, remap_shape):
    """
    :param corners_3d_lidar: [N, 8, 4]
    :param lidar2cam: [4, 4]
    :param intrin: [4, 4]
    :param remap_shape: [w, h]
    :return:
    """
    img_w, img_h = remap_shape
    intrin = intrin[:3, :3]

    corners_3d_cam = np.matmul(lidar2cam[np.newaxis, ...],
                               corners_3d_lidar.transpose((0, 2, 1)))

    corners_2d = np.matmul(intrin[np.newaxis, ...],
                           corners_3d_cam[:, :3, :]).transpose((0, 2, 1))
    corners_2d[:, :, :2] = corners_2d[:, :, :2] / corners_2d[:, :, 2:3]

    inds = (corners_2d[:, :, 0] > 0) & (corners_2d[:, :, 0] < img_w) & \
           (corners_2d[:, :, 1] > 0) & (corners_2d[:, :, 1] < img_h) & \
           (corners_2d[:, :, 2] > 0)
    valid_mask = inds.sum(1) >= 2

    # corners_2d = corners_2d[valid_mask, ...]
    corners_2d = corners_2d[:, :, :2]  # [N, 8, 2]

    assert corners_3d_lidar.shape[0] == corners_2d.shape[0]
    bbox_num = corners_3d_lidar.shape[0]

    bbox_2ds = []
    for bbox_i in range(bbox_num):
        corner_2d = corners_2d[bbox_i, ...]
        x_min = max(0, corner_2d[:, 0].min())
        x_max = min(img_w, corner_2d[:, 0].max())
        y_min = max(0, corner_2d[:, 1].min())
        y_max = min(img_h, corner_2d[:, 1].max())
        bbox_2ds.append([x_min, y_min, x_max, y_max])
    return bbox_2ds, valid_mask


def lidar3dboxs_to_2dboxs(bboxes_3d,
                          calib_infos,
                          cams,
                          remap_shape=[960, 540]):
    """
    bboxes_3d: [N, 9], [x, y, z, l, w, h, yaw, vx, vy]
    labels_3d: [N]
    scores_3d: [N]
    """
    all_cams_boxes_2d = {}
    corners_3d_lidar = gen_3d_object_corners_array(location=bboxes_3d[:, :3],
                                                   dimension=bboxes_3d[:, 3:6],
                                                   yaw=bboxes_3d[:, 6])
    corners_3d_lidar = np.concatenate(
        [corners_3d_lidar,
         np.ones((corners_3d_lidar.shape[0], 8, 1))], axis=2)

    # calib_infos = self.dates_calib_infos[(sample_info["car_type"], sample_info["sample_date"])]
    for cam in cams:
        bbox_2ds, bbox_2ds_valid_mask = lidarcorners_to_2dboxs(
            corners_3d_lidar,
            calib_infos[cam]["lidar2cam"],
            calib_infos[cam]["intrinsics"],
            remap_shape=remap_shape)
        bbox_2ds = np.array(bbox_2ds)  # [N, 4], [x1, y1, x2, y2]

        w = bbox_2ds[:, 2] - bbox_2ds[:, 0]
        h = bbox_2ds[:, 3] - bbox_2ds[:, 1]

        bbox_2ds_valid_mask = bbox_2ds_valid_mask & (w > 5) & (h > 5)
        all_cams_boxes_2d[cam] = dict(bbox_2ds=bbox_2ds,
                                      bbox_2ds_valid_mask=bbox_2ds_valid_mask)
    return all_cams_boxes_2d


def lidar3dboxs_to_2dboxs_tensor(bboxes_3d,
                                 rots,
                                 trans,
                                 intrins,
                                 remap_shape=[960, 540]):
    """
    :param bboxes_3d: [N, 9]
    :param rot: [cam_num, 3, 3]
    :param tran: [cam_num, 3]
    :param intrin: [cam_num, 3, 3]
    :param remap_shape:
    :return:
    """
    cam_num = rots.size(0)
    device = bboxes_3d.device

    corners_3d_lidar = gen_3d_object_corners_tensor(location=bboxes_3d[:, :3],
                                                    dimension=bboxes_3d[:, 3:6],
                                                    yaw=bboxes_3d[:, 6])
    corners_3d_lidar = torch.cat([corners_3d_lidar,
                                  torch.ones((corners_3d_lidar.size(0), 8, 1), device=device)], dim=2)

    all_bbox_2ds = []
    all_bbox_2ds_valid_mask = []
    for cam_idx in range(cam_num):
        cam2lidar = torch.eye(4, dtype=torch.float, device=device)
        cam2lidar[:3, :3] = rots[cam_idx]
        cam2lidar[:3, 3] = trans[cam_idx]
        lidar2cam = torch.inverse(cam2lidar)

        bbox_2ds, bbox_2ds_valid_mask = lidarcorners_to_2dboxs_tensor(
            corners_3d_lidar,
            lidar2cam,
            intrins[cam_idx],
            remap_shape=remap_shape)
        all_bbox_2ds.append(bbox_2ds)
        all_bbox_2ds_valid_mask.append(bbox_2ds_valid_mask)

    all_bbox_2ds = torch.stack(all_bbox_2ds, dim=0)
    all_bbox_2ds_valid_mask = torch.stack(all_bbox_2ds_valid_mask, dim=0)
    return all_bbox_2ds, all_bbox_2ds_valid_mask


def draw_box_3d(image, corners, pred_dict,tasks, c=(0, 0, 255), direction_c=_RED, cam=None):
    """
    :param image:  [256, 512, 3]
    :param corners: [8, 2], 2d corners
    :param c: colors
    :return:
    """
    extra_txt = ""
    if 'cipo' in pred_dict and pred_dict['cipo']!=-1:
        extra_txt += f"{str(pred_dict['cipo']+1)} "
    if 'subclass' in pred_dict:
        subclass_names = [k['names'] for k in tasks if k['task_name'] == 'subclass'][0]
        extra_txt += f"{subclass_names[pred_dict['subclass']]} "
    else:
        subclass_names = [k['names'] for k in tasks if k['task_name'] == 'class'][0]
        extra_txt += f"{subclass_names[pred_dict['class']]} "
    if cam in ['mid_center_top_wide','mid_center_top_tele','crop_f30_2']:
        if pred_dict.get('brake_light',0) == 1:
            c = (0,0,255)
        if pred_dict.get('high_brake_light',0) == 1:
            c = (0,0,255)
            extra_txt += "H "
        if pred_dict.get('side_brake_light',0) == 1:
            c = (0,0,255)
            extra_txt += f"S "
    if pred_dict.get('cross_lane',0) == 1:
        c  = (255,255,255)
    if extra_txt:
        cv2.putText(image,extra_txt.replace("sport_utility_vehicle","SUV"),(int(min(corners[:,0])),int(min(corners[:,1]-10))),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,255),2)

    face_idx = [[0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [3, 0, 4, 7]]
    im_h, im_w = image.shape[:2]
    valid = (corners[:, 0] > 0) * (corners[:, 0] < im_w) * \
            (corners[:, 1] > 0) * (corners[:, 1] < im_h)
    for ind_f in range(3, -1, -1):
        f = face_idx[ind_f]
        for j in range(4):
            if valid[f[j]] and valid[f[(j + 1) % 4]]:
                cv2.line(image, (int(corners[f[j], 0]), int(corners[f[j], 1])),
                         (int(corners[f[(j + 1) % 4], 0]), int(corners[f[(j + 1) % 4], 1])), c, 1, lineType=cv2.LINE_AA)
        if ind_f == 0:
            if valid[f[0]] and valid[f[2]]:
                cv2.line(image, (int(corners[f[0], 0]), int(corners[f[0], 1])),
                         (int(corners[f[2], 0]), int(corners[f[2], 1])), direction_c, 1, lineType=cv2.LINE_AA)
            if valid[f[1]] and valid[f[3]]:
                cv2.line(image, (int(corners[f[1], 0]), int(corners[f[1], 1])),
                         (int(corners[f[3], 0]), int(corners[f[3], 1])), direction_c, 1, lineType=cv2.LINE_AA)

    return image



def draw_camvas(img_bev, dx, bx, nx, ratio, bev_painter_h, bev_painter_w):
    for x in range(int(-50), int(51), 10):
        canvas_bev_x = int((x - bx[0]) / dx[0] * ratio)
        canvas_bev_y = int(bev_painter_h)
        cv2.line(img_bev, (canvas_bev_x, canvas_bev_y),
                 (canvas_bev_x, canvas_bev_y - 10), _WHITE, 2)
        cv2.putText(img_bev, str(x) + "m", (canvas_bev_x - 10, canvas_bev_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, _WHITE, 1)

    for y in range(int(-50), int(51), 10):
        canvas_bev_x = bev_painter_w
        canvas_bev_y = int((y - bx[1]) / dx[1] * ratio)
        cv2.line(img_bev, (canvas_bev_x, canvas_bev_y),
                 (canvas_bev_x - 10, canvas_bev_y), _WHITE, 2)
        cv2.putText(img_bev, str(-y) + "m", (canvas_bev_x - 40, canvas_bev_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, _WHITE, 1)
    return img_bev


def gen_bev_painter(bev_painter_h, bev_painter_w, grid_config, ratio, background=128):
    img_bev = np.full((bev_painter_h, bev_painter_w, 3), background, dtype=np.uint8)

    x_min, x_max, x_interval = grid_config["xbound"]
    y_min, y_max, y_interval = grid_config["ybound"]

    for x in range(int(x_min) // 10 * 10, int(x_max) // 10 * 10 + 1, 10):
        canvas_bev_x = int((x - x_min) / x_interval * ratio)
        canvas_bev_y = int(bev_painter_h)
        cv2.line(img_bev, (canvas_bev_x, canvas_bev_y),
                 (canvas_bev_x, canvas_bev_y - 10), _WHITE, 2)
        cv2.putText(img_bev, str(x) + "m", (canvas_bev_x - 10, canvas_bev_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, _WHITE, 1)

    for y in range(int(y_min) // 10 * 10, int(y_max) // 10 * 10 + 1, 10):
        canvas_bev_x = bev_painter_w
        canvas_bev_y = bev_painter_h - int((y - y_min) / y_interval * ratio)
        cv2.line(img_bev, (canvas_bev_x, canvas_bev_y),
                 (canvas_bev_x - 10, canvas_bev_y), _WHITE, 2)
        cv2.putText(img_bev, str(y) + "m", (canvas_bev_x - 40, canvas_bev_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, _WHITE, 1)
    return img_bev


def draw_bev_bboxes(grid_config, corners_3d_lidar, tasks,
                    pred_dict, set_colors, filter_cids, vis_dir_cids,
                    rot_uncerts=None, bev_h=None, trk_check=False,
                    show_extra_txt=None):
    """
    :param grid_config:
    :param corners_3d_lidar: [N, 8, 3]
    :return:
    """
    
    dx, bx, nx = gen_dx_bx(**grid_config)

    if bev_h is None:
        bev_h = merge_h

    ratio = bev_h / int(nx[1])
    bev_painter_h = bev_h

    bev_painter_w = int(nx[0] * ratio)
    # img_bev = np.full((bev_painter_h, bev_painter_w, 3), 128, dtype=np.uint8)
    img_bev = gen_bev_painter(bev_painter_h, bev_painter_w, grid_config, ratio)

    # draw selfcar
    selfcar_position = np.array([[-0.75, 1.5], [0.75, -1.5]])
    selfcar_bev_x = ((selfcar_position[:, 0] - bx[0]) / dx[0] * ratio).astype(np.int)
    selfcar_bev_y = bev_painter_h - ((selfcar_position[:, 1] - bx[1]) / dx[1] * ratio).astype(np.int)
    cv2.rectangle(img_bev, (selfcar_bev_x[0], selfcar_bev_y[0]),
                  (selfcar_bev_x[1], selfcar_bev_y[1]), (255, 255, 255), -1)

    if rot_uncerts is not None:
        assert rot_uncerts.shape[0] == corners_3d_lidar.shape[0]

    # draw bev
    corners_bev_lidar = corners_3d_lidar[:, :4, :2]
    for bbox_i in range(corners_bev_lidar.shape[0]):
        pts_bev = corners_bev_lidar[bbox_i, ...]
        pts_bev[:, 0] = (pts_bev[:, 0] - bx[0]) / dx[0]
        pts_bev[:, 1] = (pts_bev[:, 1] - bx[1]) / dx[1]

        pts_center = pts_bev.mean(0)
        velo = pred_dict['box'][bbox_i, 7:9]
        next_pts_center = pts_center + velo

        pts_bev = (pts_bev * ratio).astype(np.int)
        pts_bev[:, 1] = bev_painter_h - pts_bev[:, 1]

        pts_center = (pts_center * ratio).astype(np.int)
        pts_center[1] = bev_painter_h - pts_center[1]

        next_pts_center = (next_pts_center * ratio).astype(np.int)
        next_pts_center[1] = bev_painter_h - next_pts_center[1]

        box_task_names = [k['task_name'] for k in tasks if k['level'] == 'box']
        cur_pred_dict = {}

        for t in box_task_names:
            cur_pred_dict[t] = pred_dict[t][bbox_i].copy()
        # label = labels_3d[bbox_i]
        # static = statics_3d[bbox_i]
        if cur_pred_dict['class'] in filter_cids:
            continue
        class_name = tasks[0]['names'][cur_pred_dict['class']]
        assert class_name in set_colors
        
        bboxes_3d = pred_dict['box']
        # if bboxes_3d.shape[1] == 10:
        #     color = trk_colors[int(bboxes_3d[bbox_i, 9]) % 30]  if cur_pred_dict.get('static',1) else (0,0,0)
        # else:
        #     color = set_colors[class_name] if cur_pred_dict.get('static',1) else (0,0,0)

        color = set_colors[class_name] if cur_pred_dict.get('static', 1) else (0,0,0)
        cv2.polylines(img_bev, [pts_bev], True, color, 2)

        if cur_pred_dict['class'] in vis_dir_cids:
            cv2.line(img_bev, pts_bev[0], pts_bev[1], color=_RED, thickness=2)

        # draw vxvy

        # speed = bboxes_3d[bbox_i, 7:9]
        # speed_norm = np.sqrt(speed[0] * speed[0] + speed[1] * speed[1])
        # if speed_norm < 0.3:
        #     cv2.polylines(img_bev, [pts_bev], True, _static, 2)

        # cv2.arrowedLine(img_bev, pts_center, next_pts_center,
        #                 color=_RED, thickness=2)

        # eval_online show(velo > 500 must crop_model pred bbox)
        velo = bboxes_3d[bbox_i, 7:9]
        if velo.max() > 1000:
            cv2.polylines(img_bev, [pts_bev], True, _PURPLE, 2)

        # temp zoom velo vis
        temp_velo_zoom_ratio = 2
        draw_next_pts_center = (next_pts_center + pts_center) // temp_velo_zoom_ratio
        if velo.max() > 1000:
            draw_next_pts_center = pts_center       # eval_online_show bbox
        cv2.arrowedLine(img_bev, pts_center, draw_next_pts_center,
                        color=_RED, thickness=2)

        # draw uncert
        if rot_uncerts is not None:
            rot_uncert = rot_uncerts[bbox_i]
            yaw = bboxes_3d[bbox_i, 6]
            if yaw < 0:
                yaw += 2 * np.pi
            # img_bev = cv2.circle(img_bev, pts_center, int(rot_uncert * 10), (0, 0, 0), 1)
            yaw_angle = -yaw * 180 / np.pi + 90
            cv2.ellipse(img_bev, pts_center, (int(24 * rot_uncert), 12),
                        angle=yaw_angle, startAngle=0, endAngle=360, color=(0, 0, 0), thickness=1, lineType=1)

        # cv2.putText(img_bev, str(["%.2f"%(scores_3d[bbox_i]), "%.2f"%(pts_center_cp[1])]),
        #             (int(pts_bev[0][0]), int(pts_bev[0][1])),
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #             fontScale=0.5,
        #             color=(0, 255, 0),
        #             thickness=1)
    if 'scene' in pred_dict:
        scene_map = {0: 'Pilot', 1: 'Parking'}
        cv2.putText(img_bev, scene_map[int(pred_dict['scene'])],(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1)

    if show_extra_txt is not None:
        cv2.putText(img_bev, str(show_extra_txt), (bev_painter_w//2-10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return img_bev


def show_road_seg(bev):
    bev = bev[0][0]["seg"]
    bev = torch.argmax(bev, dim=1)[0]
    bev = bev.data.cpu().numpy()

    im_bev = np.zeros((bev.shape[1], bev.shape[0], 3))

    bev_color_0 = bev * 80
    bev_color_1 = bev * 160
    bev_color_2 = bev * 30

    im_bev[:, :, 0] = bev_color_0
    im_bev[:, :, 1] = bev_color_1
    im_bev[:, :, 2] = bev_color_2
    return im_bev


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds // width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind // K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _nms(heat, kernel=2):
    pad = (kernel - 1) // 2

    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()

    return heat * keep


def decode_roadmap_det(heatmap):
    heatmap = _nms(heatmap, kernel=3)
    scores, inds, clses, ys, xs = _topk(heatmap, K=20)
    return scores[0], inds[0], clses[0], ys[0], xs[0]


def draw_dets(img_to_show, scores, clses, ys, xs, thres=0.2):
    # print(scores)
    points = []
    for sc, c, y, x in zip(scores, clses, ys, xs):
        if sc > thres:
            cv2.circle(img_to_show, (int(x), int(y)), radius=2, color=_YELLOW, thickness=2)
            cv2.putText(
                img_to_show,
                str(int(c)),
                # (int(x + 2), int(y)),
                (int(x - 2), int(y + 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                _WHITE,
                1,
            )
            points.append([int(x), int(y)])
    return img_to_show, points


def show_line(lane):
    lbl_color_map_2 = [
        (255, 0, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 0, 0),
        (255, 128, 0),
        (255, 128, 128),
        (123, 4, 30),
        (32, 45, 12),
        (98, 156, 123),
        (230, 23, 129),
        (229, 35, 98),
        (120, 34, 56),
        (23, 4, 65),
        (154, 34, 87),
        (57, 87, 90),
        (23, 56, 32),
        (57, 87, 90),
        (23, 56, 32),
        (57, 87, 90),
        (23, 56, 32),
        (57, 87, 90),
        (23, 56, 32),
    ]

    bev = lane["lane_line_hm"]
    # direction = lane['lane_direction']
    # ins_mtc = lane['lane_line_instance']
    lane_semantic = lane["lane_semantic"]
    road_edge = lane["road_edge_hm"]

    road_edge = F.interpolate(
        road_edge, (road_edge.size()[2], road_edge.size()[3]), mode="bilinear"
    )
    road_edge = F.sigmoid(road_edge)

    # road_edge = _nms(road_edge)

    road_edge = road_edge.data.cpu().numpy()[0][0]
    v_road_edge, u_road_edge = np.where(road_edge > 0.01)

    bev = F.interpolate(bev, (bev.size()[2], bev.size()[3]), mode="bilinear")

    # bev = _nms(bev)

    lane_semantic = F.interpolate(
        lane_semantic, (lane_semantic.size()[2], lane_semantic.size()[3]), mode="bilinear"
    )
    lane_semantic = F.log_softmax(lane_semantic, dim=1).data.cpu().numpy()
    lane_semantic = lane_semantic.argmax(axis=1)[0]

    dst = np.zeros((lane_semantic.shape[0], lane_semantic.shape[1], 3), dtype=np.uint8)
    for i in range(7):
        dst[lane_semantic == i] = lbl_color_map_2[i]

    bev = bev.sigmoid()
    bev = bev.data.cpu().numpy()[0][0]

    # im_bev = np.zeros((bev.shape[1], bev.shape[0], 3))
    im_bev = np.zeros((bev.shape[0], bev.shape[1], 3))

    # bev = np.where(bev > 0.65, 1, 0)
    # bev[bev<0.5] = 0
    vs, us = np.where(bev[:, :] > 0.5)

    im_bev[:, :, 0] = bev * dst[:, :, 0]
    im_bev[:, :, 1] = bev * dst[:, :, 1]
    im_bev[:, :, 2] = bev * dst[:, :, 2]

    im_bev = np.array(im_bev, np.uint8)
    road_edge_points = []
    road_lines_points = []
    for i in range(len(vs)):
        road_lines_points.append([us[i], vs[i]])

    return im_bev, road_edge_points, road_lines_points


def get_bev_hlane_img(lane):
    lbl_color_map_2 = [
        (255, 0, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 0, 0),
        (255, 128, 0),
    ]
    bev = lane["lane_line_hm"]
    lane_semantic = lane["lane_semantic"]
    bev = F.interpolate(bev, (bev.size()[2], bev.size()[3]), mode="bilinear")
    lane_semantic = F.interpolate(
        lane_semantic, (lane_semantic.size()[2], lane_semantic.size()[3]), mode="bilinear"
    )
    lane_semantic = F.log_softmax(lane_semantic, dim=1).data.cpu().numpy()
    lane_semantic = lane_semantic.argmax(axis=1)[0]

    dst = np.zeros((lane_semantic.shape[0], lane_semantic.shape[1], 3), dtype=np.uint8)
    for i in range(7):
        dst[lane_semantic == i] = lbl_color_map_2[i]

    bev = bev.sigmoid()
    bev = bev.data.cpu().numpy()[0][0]

    im_bev = np.zeros((bev.shape[0], bev.shape[1], 3))
    im_bev[:, :, 0] = bev * dst[:, :, 0]
    im_bev[:, :, 1] = bev * dst[:, :, 1]
    im_bev[:, :, 2] = bev * dst[:, :, 2]

    im_bev = np.array(im_bev, np.uint8)

    return im_bev


def get_lines_points_cams(road_edge_points, dx, bx, cam2lidar, intrins, upsample=0.2):
    road_edge_points = np.array(road_edge_points, dtype=np.float32).copy()

    road_edge_points[:, 0] = road_edge_points[:, 0] * dx[0] * upsample + bx[0]
    road_edge_points[:, 1] = road_edge_points[:, 1] * dx[1] * upsample + bx[1]

    road_edge_points = np.concatenate(
        [road_edge_points, np.zeros((road_edge_points.shape[0], 1))], axis=1
    )
    road_edge_points = np.concatenate(
        [road_edge_points, np.ones((road_edge_points.shape[0], 1))], axis=1
    )
    num_cams = cam2lidar.shape[0]
    road_edge_points = np.tile(road_edge_points[np.newaxis], (num_cams, 1, 1))
    cam_roadseg_points = np.linalg.inv(cam2lidar) @ road_edge_points.transpose((0, 2, 1))
    ims_roadseg_points = intrins @ cam_roadseg_points[:, :3, :]
    ims_roadseg_points = ims_roadseg_points.transpose((0, 2, 1))  # [6, N, 3]
    ims_roadseg_points[:, :, :2] = ims_roadseg_points[:, :, :2] / ims_roadseg_points[:, :, 2:3]
    return ims_roadseg_points


def multicam_show_imgs(cams, draw_imgs):
    resize_h = merge_h // 2
    resize_w = merge_w // 3
    img_to_show = np.zeros((merge_h, merge_w, 3))

    if len(cams) > 1 and len(cams) <= 6:
        for cam_i, cam in enumerate(cams[:6]):
            col_idx = cam_i // 3
            row_idx = cam_i % 3
            img = cv2.resize(draw_imgs[cam_i].copy(), (resize_w, resize_h))
            img_to_show[int(col_idx * resize_h): int((col_idx + 1) * resize_h),
            int(row_idx * resize_w): int((row_idx + 1) * resize_w), :] = img
    elif len(cams) > 6:
        for cam_i, cam in enumerate(cams[:6]):
            col_idx = cam_i // 3
            row_idx = cam_i % 3
            img = cv2.resize(draw_imgs[cam_i].copy(), (resize_w, resize_h))
            img_to_show[int(col_idx * resize_h): int((col_idx + 1) * resize_h),
            int(row_idx * resize_w): int((row_idx + 1) * resize_w), :] = img

        cropf30_extra_cams_paints = np.zeros((resize_h, merge_w, 3))
        cropr60_extra_cams_paints = np.zeros((resize_h, merge_w, 3))
        for cam_i, cam in enumerate(cams[6:]):      # temp only f30
            img = cv2.resize(draw_imgs[cam_i + 6].copy(), (resize_w, resize_h))
            if cam == "mid_center_top_tele" or cam == "crop_f30_2":
                cropf30_extra_cams_paints[0: int(resize_h), int(resize_w): int(2 * resize_w)] = img
            elif cam == "crop_r60":
                cropr60_extra_cams_paints[0: int(resize_h), int(resize_w): int(2 * resize_w)] = img

        img_to_show = np.concatenate([cropf30_extra_cams_paints, img_to_show, cropr60_extra_cams_paints], axis=0)

    elif len(cams) == 1:
        img_to_show = cv2.resize(draw_imgs[0].copy(), (merge_w, merge_h))
    return img_to_show


def project_fisheye_omni(points, alpha, ck, cdist):
    x = points[:, 0, :]
    y = points[:, 1, :]
    z = points[:, 2, :]
    d = np.sqrt(x * x + y * y + z * z)
    rz = z + alpha * d
    x = x / rz
    y = y / rz
    xx = x * x
    yy = y * y
    xy = x * y
    theta2 = xx + yy
    distortion = cdist[0] * theta2 + cdist[1] * theta2 * theta2
    x = x + x * distortion + 2.0 * cdist[2] * xy + cdist[3] * (theta2 + 2.0 * xx)
    y = y + y * distortion + 2.0 * cdist[3] * xy + cdist[2] * (theta2 + 2.0 * yy)
    x = ck[0] * x + ck[2]
    y = ck[1] * y + ck[3]
    return x, y, z


def draw_bbox_imgs(imgs,
                   intrins,
                   rots,
                   trans,
                   cams,
                   pred_dict,
                   tasks,
                   corners_3d_lidar,
                   filter_cids,
                   set_colors,
                   post_rots=None,
                   post_trans=None):
    draw_imgs = []

    for i, cam in enumerate(cams):

        img = imgs[i]
        img_h, img_w = img.shape[:2]
        intrin = intrins[i]
        rot = rots[i]
        tran = trans[i]

        cam2lidar = np.eye(4)
        cam2lidar[:3, :3] = rot
        cam2lidar[:3, 3] = tran
        lidar2cam = np.linalg.inv(cam2lidar)
        corners_3d_cam = np.matmul(lidar2cam[np.newaxis, ...],
                                   corners_3d_lidar.transpose((0, 2, 1)))

        corners_2d = np.matmul(intrin[np.newaxis, ...],
                               corners_3d_cam[:, :3, :]).transpose((0, 2, 1))

        corners_2d[:, :, :2] = corners_2d[:, :, :2] / corners_2d[:, :, 2:3]

        if post_rots is not None and post_trans is not None:
            post_rot = post_rots[i][:2, :2][None, ...]  # [1, 2, 2]
            post_tran = post_trans[i][:2][None, ...]  # [1, 2]

            corners_2d_xy = corners_2d[:, :, :2]
            corners_2d_z = corners_2d[:, :, 2:3]
            corners_2d_xy = np.matmul(post_rot, np.transpose(corners_2d_xy, (0, 2, 1))).transpose((0, 2, 1))
            corners_2d_xy = corners_2d_xy + post_tran[:, None, :]  # [N, 8, 2]
            corners_2d = np.concatenate([corners_2d_xy, corners_2d_z], axis=-1)

        inds = (corners_2d[:, :, 0] > 0) & (corners_2d[:, :, 0] < img_w) & \
               (corners_2d[:, :, 1] > 0) & (corners_2d[:, :, 1] < img_h) & \
               (corners_2d[:, :, 2] > 0)
        valid_mask = inds.sum(1) >= 4

        corners_2d = corners_2d[valid_mask, ...]
        corners_2d = corners_2d[:, :, :2]

        box_task_names = [k['task_name'] for k in tasks if k['level'] == 'box']

        cur_pred_dict = pred_dict.copy()
        for t in box_task_names:
            cur_pred_dict[t] = cur_pred_dict[t][valid_mask]

        for bbox_i in range(corners_2d.shape[0]):
            # score = scores[bbox_i]
            tmp_pred_dict = cur_pred_dict.copy()
            for t in box_task_names:
                tmp_pred_dict[t] = cur_pred_dict[t][bbox_i]

            if tmp_pred_dict['class'] in filter_cids:
                continue
            # if tmp_pred_dict['box'].shape[1] == 10:
            #     color = trk_colors[int(tmp_pred_dict['box'][valid_mask][bbox_i, 9]) % 30]
            # else:
            #     color = set_colors[tasks[0]['names'][tmp_pred_dict['class']]]

            color = set_colors[tasks[0]['names'][tmp_pred_dict['class']]]

            draw_box_3d(img, corners_2d[bbox_i], tmp_pred_dict, tasks, c=color, cam=cam)
        draw_imgs.append(img)

    img_to_show = multicam_show_imgs(cams, draw_imgs)
    return img_to_show


def show_multicam_bboxes(imgs, intrins, rots, trans, cams,
                         grid_config, timestamp, tasks, show_dir, pred_dict,
                         filter_cids=[5], nuscenes=True, rot_uncerts=None,
                         trk_check=False, bev_paint_extra_info=None,
                         draw_bbox_img=True, draw_aug_img=False,
                         aug_imgs=None, post_rots=None, post_trans=None):
    """
    :param imgs:  [len(cams), H, W]
    :param intrins:  [len(cams), 3, 3]
    :param rots:  [len(cams), 3, 3]
    :param trans:  [len(cams), 3]
    :param bboxes_3d:  [N, 9](without tracking) or [N, 10](with tracking)
    :param scores_3d: [N, ]
    :param labels_3d: [N, ]
    :param cams:
    :param grid_config: {'xbound': [-51.2, 51.2, 0.8],
                         'ybound': [-51.2, 51.2, 0.8],
                         'zbound': [-10.0, 10.0, 20.0],
                         'dbound': [1.0, 60.0, 1.0],}
    :param timestamp:
    :param show_dir:
    :return:
    """
    if nuscenes:
        set_colors = nuscenes_set_colors
        vis_dir_cids = [0, 1, 2, 3, 4, 6, 7]
    else:
        set_colors = bhx_set_colors
        vis_dir_cids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12]

    os.makedirs(show_dir, exist_ok=True)
    corners_3d_lidar = gen_3d_object_corners_array(location=pred_dict['box'][:, :3],
                                                   dimension=pred_dict['box'][:, 3:6],
                                                   yaw=pred_dict['box'][:, 6])
    corners_3d_lidar = np.concatenate([corners_3d_lidar,
                                       np.ones((corners_3d_lidar.shape[0], 8, 1))], axis=2)

    if len(cams) > 6:
        bev_h = merge_h // 2 * 4
    else:
        bev_h = merge_h

    img_bev = draw_bev_bboxes(grid_config,
                              corners_3d_lidar.copy(),
                              tasks,
                              pred_dict,
                              filter_cids=filter_cids,
                              set_colors=set_colors,
                              vis_dir_cids=vis_dir_cids,
                              rot_uncerts=rot_uncerts,
                              bev_h=bev_h,
                              trk_check=trk_check,
                              show_extra_txt=bev_paint_extra_info)

    img_to_show = None
    if draw_bbox_img:
        img_to_show = draw_bbox_imgs(imgs=imgs,
                                     intrins=intrins,
                                     rots=rots,
                                     trans=trans,
                                     cams=cams,
                                     pred_dict=pred_dict,
                                     tasks=tasks,
                                     corners_3d_lidar=corners_3d_lidar,
                                     filter_cids=filter_cids,
                                     set_colors=set_colors)

    if draw_aug_img:
        assert aug_imgs is not None
        aug_img_to_show = draw_bbox_imgs(imgs=aug_imgs,
                                         intrins=intrins,
                                         rots=rots,
                                         trans=trans,
                                         cams=cams,
                                         pred_dict=pred_dict,
                                         tasks=tasks,
                                         corners_3d_lidar=corners_3d_lidar,
                                         filter_cids=filter_cids,
                                         set_colors=set_colors,
                                         post_rots=post_rots,
                                         post_trans=post_trans)

        img_to_show = np.concatenate([aug_img_to_show, img_to_show], axis=1)

    # all_img_show = np.concatenate([im_to_show, img_bev], axis=1)
    # save_path = os.path.join(show_dir, str(timestamp) + ".jpg")
    # cv2.imwrite(save_path, all_img_show)

    # return all_img_show
    return img_to_show, img_bev


def show_radar_bev_obj_feature(img_bev,
                               ratio,
                               grid_config,
                               radar_bev_obj_feature):
    """

    Args:
        show_bev_shape: (bev_h, bev_w)
        radar_bev_obj_feature: [bev_h, bev_w, 17]

    Returns:

    """
    
    dx, bx, nx = gen_dx_bx(**grid_config)
    show_bev_h, show_bev_w = img_bev.shape[:2]
    use_bev_h, use_bev_w = radar_bev_obj_feature.shape[:2]

    assert show_bev_h / use_bev_h == ratio and show_bev_w / use_bev_w == ratio

    show_img_bev = np.full((use_bev_h, use_bev_w, 3), 128, dtype=np.uint8)

    #valid_mask = (radar_bev_obj_feature[:, :, 2] >= 0.1) & (radar_bev_obj_feature[:, :, 3] >= 0.1)
    # valid_cls_masks = np.zeros_like(radar_bev_obj_feature[:, :, 0]).astype(np.bool)
    # for cls_id in radar_valid_cls_ids:
    #     valid_cls_masks = valid_cls_masks | (radar_bev_obj_feature[:, :, 15] == cls_id)
    # valid_mask = valid_mask & valid_cls_masks
    # print("valid_mask_sum: ", valid_mask.sum())

    valid_mask = (radar_bev_obj_feature[:, :, 2] > 0) & (radar_bev_obj_feature[:, :, 3] > 0)
    show_img_bev[valid_mask] = (0, 0, 0)

    show_img_bev = cv2.resize(show_img_bev, (show_bev_w, show_bev_h))
    show_img_bev = np.flip(show_img_bev, 0)
    show_img_bev = show_img_bev.astype(np.uint8)

    cv2.putText(show_img_bev, str("radar_obj_bev_feat"), (show_bev_w // 2 - 60, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 0, 0), 1)

    return show_img_bev


def show_radar_ego_obj_feature(img_bev,
                               ratio,
                               grid_config,
                               radar_class_names,
                               radar_valid_cls_ids,
                               ego_obj_feature):
    """

    Args:
        img_bev: [show_bev_h, show_bev_w]
        ego_obj_feature: [N_obj, 16]

    Returns:

    """
    
    dx, bx, nx = gen_dx_bx(**grid_config)
    bev_painter_h, bev_painter_w = img_bev.shape[:2]

    # 先可视化ego_obj_feature，直接画到img_bev上
    # obj_feature_dic = {
    #     'center_x': 0, 'center_y': 1, 'length': 2, 'width': 3, 'heading': 4,
    #     'raw_x': 5, 'raw_y': 6, 'doppler_x': 7, 'doppler_y': 8, 'velocity_x': 9, 'velocity_y': 10,
    #     'rcs': 11, 'exist_time': 12, 'exist_prob': 13, 'dynamic_prob': 14, 'type': 15,
    # }
    # fill z=0
    bbox_locations = ego_obj_feature[:, :2]  # (x, y)
    bbox_locations = np.concatenate((bbox_locations,
                                     np.zeros_like(bbox_locations[:, :1])), axis=-1)  # [N_obj, 3]
    # fill h=1
    bbox_dimensions = ego_obj_feature[:, 2:4]  # (l, w)
    bbox_dimensions = np.concatenate((bbox_dimensions,
                                      np.ones_like(bbox_dimensions[:, :1])), axis=-1)  # [N_obj, 3]

    bbox_yaws = ego_obj_feature[:, 4]
    corners_3d_lidar = gen_3d_object_corners_array(location=bbox_locations,
                                                   dimension=bbox_dimensions,
                                                   yaw=bbox_yaws)
    corners_3d_lidar = np.concatenate(
        [corners_3d_lidar,
         np.ones((corners_3d_lidar.shape[0], 8, 1))], axis=2)

    bbox_velos = ego_obj_feature[:, 9:11]  # (vx, vy)
    bbox_cls_ids = ego_obj_feature[:, -1].astype(int)  # 注：这是radar的类别id，和od不是对应的
    bbox_motion_states = ego_obj_feature[:, 14]  # 0为不确定，1为静态，2为动态

    # draw bev
    corners_bev_lidar = corners_3d_lidar[:, :4, :2]
    for bbox_i in range(corners_bev_lidar.shape[0]):
        velo = bbox_velos[bbox_i]
        dim = bbox_dimensions[bbox_i]
        # 0为不确定，1为点目标，2为car，3为truck，4为pedestrian，5为motorcycle，6为bicycle
        radar_cls_id = bbox_cls_ids[bbox_i]
        bbox_motion_state = bbox_motion_states[bbox_i]
        if dim.min() < 0.1:
            continue

        if radar_cls_id not in radar_valid_cls_ids:
            continue

        pts_bev = corners_bev_lidar[bbox_i, ...]
        pts_bev[:, 0] = (pts_bev[:, 0] - bx[0]) / dx[0]
        pts_bev[:, 1] = (pts_bev[:, 1] - bx[1]) / dx[1]

        pts_center = pts_bev.mean(0)
        next_pts_center = pts_center + velo

        pts_bev = (pts_bev * ratio).astype(np.int)
        pts_bev[:, 1] = bev_painter_h - pts_bev[:, 1]

        pts_center = (pts_center * ratio).astype(np.int)
        pts_center[1] = bev_painter_h - pts_center[1]
        next_pts_center = (next_pts_center * ratio).astype(np.int)
        next_pts_center[1] = bev_painter_h - next_pts_center[1]

        class_name = radar_class_names[radar_cls_id]
        bbox_color = bhx_set_colors[class_name]
        if bbox_motion_state == 1:  # static
            bbox_color = (0, 0, 0)

        cv2.polylines(img_bev, [pts_bev], True, bbox_color, 2)

        cv2.line(img_bev, pts_bev[0], pts_bev[1], color=_RED, thickness=2)

        # temp zoom velo vis
        temp_velo_zoom_ratio = 2
        draw_next_pts_center = (next_pts_center + pts_center) // temp_velo_zoom_ratio
        cv2.arrowedLine(img_bev, pts_center, draw_next_pts_center,
                        color=_RED, thickness=2)

    cv2.putText(img_bev, str("radar_obj_ego_feat"), (bev_painter_w // 2 - 60, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 0, 0), 1)

    return img_bev


def show_radar_ego_pcl_feature(img_bev,
                               ratio,
                               grid_config,
                               ego_pcl_feature):
    """

    Args:
        img_bev:
        ratio:
        grid_config:
        ego_pcl_feature: [N_clu, 8]

    Returns:

    """
    
    dx, bx, nx = gen_dx_bx(**grid_config)
    bev_painter_h, bev_painter_w = img_bev.shape[:2]

    cluster_locations = ego_pcl_feature[:, :2]
    cluster_locations[:, 0] = (cluster_locations[:, 0] - bx[0]) / dx[0]
    cluster_locations[:, 1] = (cluster_locations[:, 1] - bx[1]) / dx[1]

    cluster_locations = (cluster_locations * ratio).astype(np.int)
    cluster_locations[:, 1] = bev_painter_h - cluster_locations[:, 1]

    for point_i in range(cluster_locations.shape[0]):
        point_loc = cluster_locations[point_i]
        cv2.circle(img_bev,
                   (int(point_loc[0]), int(point_loc[1])),
                   radius=2,
                   color=_BLACK,
                   thickness=-1)

    cv2.putText(img_bev, str("radar_pcl_ego_feat"), (bev_painter_w // 2 - 60, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 0, 0), 1)
    return img_bev


def show_radar_bev_pcl_feature(img_bev,
                               ratio,
                               grid_config,
                               radar_bev_pcl_feature):
    """
    Args:
        img_bev: (bev_h, bev_w)
        ratio:
        grid_config:
        radar_bev_pcl_feature: [bev_h, bev_w, 10]

    Returns:
    """
    
    dx, bx, nx = gen_dx_bx(**grid_config)
    show_bev_h, show_bev_w = img_bev.shape[:2]
    use_bev_h, use_bev_w = radar_bev_pcl_feature.shape[:2]

    assert show_bev_h / use_bev_h == ratio and show_bev_w / use_bev_w == ratio
    show_img_bev = np.full((use_bev_h, use_bev_w, 3), 128, dtype=np.uint8)

    valid_mask = radar_bev_pcl_feature[:, :, -1] > 0
    show_img_bev[valid_mask] = (0, 0, 0)

    show_img_bev = cv2.resize(show_img_bev, (show_bev_w, show_bev_h))
    show_img_bev = np.flip(show_img_bev, 0)
    show_img_bev = show_img_bev.astype(np.uint8)

    cv2.putText(show_img_bev, str("radar_pcl_bev_feat"), (show_bev_w // 2 - 60, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 0, 0), 1)

    return show_img_bev


def show_bev_radar_bboxes(grid_config,
                          radar_infos,
                          radar_class_names,
                          radar_valid_cls_ids,
                          bev_h=None):
    
    dx, bx, nx = gen_dx_bx(**grid_config)

    if bev_h is None:
        bev_h = merge_h
    ratio = bev_h / int(nx[1])
    bev_painter_h = bev_h
    bev_painter_w = int(nx[0] * ratio)
    img_bev = gen_bev_painter(bev_painter_h, bev_painter_w, grid_config, ratio)

    ego_obj_feature = radar_infos["radar_ego_obj_feature"]                # [N_obj, 16]
    ego_pcl_feature = radar_infos["radar_ego_pcl_feature"]                # [N_clu, 8]
    radar_bev_obj_feature = radar_infos["radar_bev_obj_feature"]    # [bev_h, bev_w, 17]
    radar_bev_pcl_feature = radar_infos["radar_bev_pcl_feature"]    # [bev_h, bev_w, 10]

    obj_ego_feat_drawbev = show_radar_ego_obj_feature(img_bev=img_bev.copy(),
                                                      ratio=ratio,
                                                      grid_config=grid_config,
                                                      radar_class_names=radar_class_names,
                                                      radar_valid_cls_ids=radar_valid_cls_ids,
                                                      ego_obj_feature=ego_obj_feature)
    obj_bev_feat_drawbev = show_radar_bev_obj_feature(img_bev=img_bev.copy(),
                                                      ratio=ratio,
                                                      grid_config=grid_config,
                                                      radar_bev_obj_feature=radar_bev_obj_feature)
    pcl_ego_feat_drawbev = show_radar_ego_pcl_feature(img_bev=img_bev.copy(),
                                                      ratio=ratio,
                                                      grid_config=grid_config,
                                                      ego_pcl_feature=ego_pcl_feature)
    pcl_bev_feat_drawbev = show_radar_bev_pcl_feature(img_bev=img_bev,
                                                      ratio=ratio,
                                                      grid_config=grid_config,
                                                      radar_bev_pcl_feature=radar_bev_pcl_feature)

    radar_show_bev = np.concatenate((obj_ego_feat_drawbev, obj_bev_feat_drawbev,
                                     pcl_ego_feat_drawbev, pcl_bev_feat_drawbev),
                                    axis=1)

    return radar_show_bev


def draw_bev_points(img_bev, grid_config, roadseg_points):
    """
    :param merge_h:
    :param merge_w:
    :param grid_config:
    :param roadseg_points: [N, 3]
    :return:
    """
    
    dx, bx, nx = gen_dx_bx(**grid_config)
    ratio = merge_h / int(nx[1])
    bev_painter_h, bev_painter_w = img_bev.shape[:2]

    # bev_painter_h = merge_h
    # bev_painter_w = int(nx[0] * ratio)
    # img_bev = gen_bev_painter(bev_painter_h, bev_painter_w, dx, bx, nx, ratio)

    roadseg_points[:, 0] = (roadseg_points[:, 0] - bx[0]) / dx[0] * ratio
    roadseg_points[:, 1] = bev_painter_h - (roadseg_points[:, 1] - bx[1]) / dx[1] * ratio

    roadseg_points = roadseg_points.astype(int)

    valid_mask = (roadseg_points[:, 0] > 0) & (roadseg_points[:, 0] < bev_painter_w) & \
                 (roadseg_points[:, 1] > 0) & (roadseg_points[:, 1] < bev_painter_h)
    roadseg_points = roadseg_points[valid_mask]

    img_bev[roadseg_points[:, 1], roadseg_points[:, 0]] = _YELLOW
    return img_bev


def show_multicam_roadsegpoints(
        imgs,
        intrins,
        rots,
        trans,
        roadseg_points,
        cams,
        grid_config,
        timestamp,
        show_dir,
        img_bev=None,
):
    os.makedirs(show_dir, exist_ok=True)

    if img_bev is None:
        

        dx, bx, nx = gen_dx_bx(**grid_config)
        ratio = merge_h / int(nx[1])
        bev_painter_h = merge_h
        bev_painter_w = int(nx[0] * ratio)
        # img_bev = np.full((bev_painter_h, bev_painter_w, 3), 128, dtype=np.uint8)
        img_bev = gen_bev_painter(bev_painter_h, bev_painter_w, grid_config, ratio)

    img_bev = draw_bev_points(img_bev, grid_config, roadseg_points.copy())

    cam2lidar = np.eye(4)
    cam2lidar = np.tile(cam2lidar[np.newaxis], (6, 1, 1))
    # cam2lidar = np.tile(cam2lidar[np.newaxis], (12, 1, 1))
    cam2lidar[:, :3, :3] = rots
    cam2lidar[:, :3, 3] = trans

    roadseg_points = np.concatenate(
        [roadseg_points, np.ones((roadseg_points.shape[0], 1))], axis=1
    )
    roadseg_points = np.tile(roadseg_points[np.newaxis], (6, 1, 1))
    # roadseg_points = np.tile(roadseg_points[np.newaxis], (12, 1, 1))

    cam_roadseg_points = np.linalg.inv(cam2lidar) @ roadseg_points.transpose((0, 2, 1))
    ims_roadseg_points = intrins @ cam_roadseg_points[:, :3, :]
    ims_roadseg_points = ims_roadseg_points.transpose((0, 2, 1))  # [6, N, 3]
    ims_roadseg_points[:, :, :2] = ims_roadseg_points[:, :, :2] / ims_roadseg_points[:, :, 2:3]

    draw_imgs = []
    for cam_i, cam in enumerate(cams):
        img = imgs[cam_i]
        img_h, img_w = img.shape[:2]
        im_roadseg_points = ims_roadseg_points[cam_i]

        valid_mask = (
                (im_roadseg_points[:, 0] > 0)
                & (im_roadseg_points[:, 0] < img_w)
                & (im_roadseg_points[:, 1] > 0)
                & (im_roadseg_points[:, 1] < img_h)
                & (im_roadseg_points[:, 2] > 0)
        )
        im_roadseg_points = im_roadseg_points[valid_mask, ...]

        us = im_roadseg_points[:, 1].astype(np.int)
        vs = im_roadseg_points[:, 0].astype(np.int)
        # img[us, vs] = img[us, vs] + [[255, 0, 0]]
        img[us, vs] = [_YELLOW]
        # import cv2
        # for u, v in zip(us, vs):
        #     cv2.circle(img, (int(v), int(u)), 3, _RED, -1)
        draw_imgs.append(img)

    img_to_show = multicam_show_imgs(cams, draw_imgs)

    # all_img_show = np.concatenate([img_to_show, img_bev], axis=1)
    #
    # save_path = os.path.join(show_dir, str(timestamp) + ".jpg")
    # cv2.imwrite(save_path, all_img_show)
    return img_to_show, img_bev


def get_road_points(pred_maps, grid_config, upsample=0.25):
    """
    :param pred_maps: tuple[list[dict]]
    :return:
    """

    
    pred_seg_map = pred_maps['seg']
    pred_seg_map = torch.argmax(pred_seg_map, dim=1)[0]

    dx, bx, nx = gen_dx_bx(**grid_config)

    road_points_idxs = torch.nonzero(pred_seg_map > 0.001)
    v, u = road_points_idxs[:, 0], road_points_idxs[:, 1]

    road_points_lidar = torch.zeros((u.size(0), 3))
    road_points_lidar[:, 0] = u * dx[0] * upsample + bx[0]
    road_points_lidar[:, 1] = v * dx[1] * upsample + bx[1]
    return road_points_lidar.cpu()


def show_multicam_roadsegpoints_infer(
        imgs,
        intrins,
        rots,
        trans,
        roadseg_points_map,
        cams,
        grid_config,
        timestamp,
        show_dir,
        img_bev=None,
):
    os.makedirs(show_dir, exist_ok=True)

    if img_bev is None:
        

        dx, bx, nx = gen_dx_bx(**grid_config)
        ratio = merge_h / int(nx[1])
        bev_painter_h = merge_h
        bev_painter_w = int(nx[0] * ratio)
        # img_bev = np.full((bev_painter_h, bev_painter_w, 3), 128, dtype=np.uint8)
        img_bev = gen_bev_painter(bev_painter_h, bev_painter_w, grid_config, ratio)
    

    roadseg_points_t = get_road_points(roadseg_points_map, grid_config)
    roadseg_points = np.array(roadseg_points_t)
    img_bev = draw_bev_points(img_bev, grid_config, roadseg_points)
    num_cams = len(rots)
    rots = np.array(rots)

    cam2lidar = np.eye(4)
    cam2lidar = np.tile(cam2lidar[np.newaxis], (num_cams, 1, 1))
    # cam2lidar = np.tile(cam2lidar[np.newaxis], (12, 1, 1))
    cam2lidar[:, :3, :3] = rots
    cam2lidar[:, :3, 3] = trans

    roadseg_points = np.concatenate(
        [roadseg_points, np.ones((roadseg_points.shape[0], 1))], axis=1
    )
    roadseg_points = np.tile(roadseg_points[np.newaxis], (num_cams, 1, 1))
    # roadseg_points = np.tile(roadseg_points[np.newaxis], (12, 1, 1))

    cam_roadseg_points = np.linalg.inv(cam2lidar) @ roadseg_points.transpose((0, 2, 1))
    ims_roadseg_points = intrins @ cam_roadseg_points[:, :3, :]
    ims_roadseg_points = ims_roadseg_points.transpose((0, 2, 1))  # [6, N, 3]
    ims_roadseg_points[:, :, :2] = ims_roadseg_points[:, :, :2] / ims_roadseg_points[:, :, 2:3]

    draw_imgs = []
    for cam_i, cam in enumerate(cams):
        img = imgs[cam_i]
        img_h, img_w = img.shape[:2]
        im_roadseg_points = ims_roadseg_points[cam_i]

        valid_mask = (
                (im_roadseg_points[:, 0] > 0)
                & (im_roadseg_points[:, 0] < img_w)
                & (im_roadseg_points[:, 1] > 0)
                & (im_roadseg_points[:, 1] < img_h)
                & (im_roadseg_points[:, 2] > 0)
        )
        im_roadseg_points = im_roadseg_points[valid_mask, ...]

        us = im_roadseg_points[:, 1].astype(np.int)
        vs = im_roadseg_points[:, 0].astype(np.int)
        # img[us, vs] = img[us, vs] + [[255, 0, 0]]
        img[us, vs] = [_YELLOW]
        # import cv2
        # for u, v in zip(us, vs):
        #     cv2.circle(img, (int(v), int(u)), 3, _RED, -1)
        draw_imgs.append(img)

    img_to_show = multicam_show_imgs(cams, draw_imgs)

    # all_img_show = np.concatenate([img_to_show, img_bev], axis=1)
    #
    # save_path = os.path.join(show_dir, str(timestamp) + ".jpg")
    # cv2.imwrite(save_path, all_img_show)
    return img_to_show, img_bev


def show_multicam_hlinesegpoints(
        imgs,
        intrins,
        rots,
        trans,
        roadseg_points,
        cams,
        grid_config,
        timestamp,
        show_dir,
        img_bev=None,
):
    os.makedirs(show_dir, exist_ok=True)

    if img_bev is None:
        

        dx, bx, nx = gen_dx_bx(**grid_config)
        ratio = merge_h / int(nx[1])
        bev_painter_h = merge_h
        bev_painter_w = int(nx[0] * ratio)
        # img_bev = np.full((bev_painter_h, bev_painter_w, 3), 128, dtype=np.uint8)
        img_bev = gen_bev_painter(bev_painter_h, bev_painter_w, grid_config, ratio)

    img_bev = draw_bev_points(img_bev, grid_config, roadseg_points.copy())

    cam2lidar = np.eye(4)
    # cam2lidar = np.tile(cam2lidar[np.newaxis], (6, 1, 1))
    cam2lidar = np.tile(cam2lidar[np.newaxis], (12, 1, 1))
    cam2lidar[:, :3, :3] = rots
    cam2lidar[:, :3, 3] = trans

    roadseg_points = np.concatenate(
        [roadseg_points, np.ones((roadseg_points.shape[0], 1))], axis=1
    )
    # roadseg_points = np.tile(roadseg_points[np.newaxis], (6, 1, 1))
    roadseg_points = np.tile(roadseg_points[np.newaxis], (12, 1, 1))

    cam_roadseg_points = np.linalg.inv(cam2lidar) @ roadseg_points.transpose((0, 2, 1))
    ims_roadseg_points = intrins @ cam_roadseg_points[:, :3, :]
    ims_roadseg_points = ims_roadseg_points.transpose((0, 2, 1))  # [6, N, 3]
    ims_roadseg_points[:, :, :2] = ims_roadseg_points[:, :, :2] / ims_roadseg_points[:, :, 2:3]

    draw_imgs = []
    for cam_i, cam in enumerate(cams):
        img = imgs[cam_i]
        img_h, img_w = img.shape[:2]
        im_roadseg_points = ims_roadseg_points[cam_i]

        valid_mask = (
                (im_roadseg_points[:, 0] > 0)
                & (im_roadseg_points[:, 0] < img_w)
                & (im_roadseg_points[:, 1] > 0)
                & (im_roadseg_points[:, 1] < img_h)
                & (im_roadseg_points[:, 2] > 0)
        )
        im_roadseg_points = im_roadseg_points[valid_mask, ...]

        us = im_roadseg_points[:, 1].astype(np.int)
        vs = im_roadseg_points[:, 0].astype(np.int)
        # img[us, vs] = img[us, vs] + [[255, 0, 0]]
        # img[us, vs] = [_YELLOW]
        img[us, vs] = [_RED]

        points_ins = []
        for u, v in zip(us, vs):
            points_ins.append([int(v), int(u)])
            cv2.circle(img, (int(v), int(u)), 3, _RED, -1)
        points_ins = np.array(points_ins)
        cv2.polylines(img, np.int32([points_ins]), isClosed=False, color=_RED, thickness=1)

        draw_imgs.append(img)

    img_to_show = multicam_show_imgs(cams, draw_imgs)

    # all_img_show = np.concatenate([img_to_show, img_bev], axis=1)
    #
    # save_path = os.path.join(show_dir, str(timestamp) + ".jpg")
    # cv2.imwrite(save_path, all_img_show)
    return img_to_show, img_bev


def show_multicam_hdflags(imgs, intrins, rots, trans, hd_flag_out, cams, grid_config, img_bev=None, bev_h=None):
    
    scores, inds, clses, ys, xs = decode_roadmap_det(hd_flag_out["hm"].sigmoid_())
    bev = hd_flag_out["hm"]
    img_bev = np.zeros((bev.shape[2], bev.shape[3], 3))

    

    grid_config_od = {
        'xbound': [-48, 48, 2],
        'ybound': [-152, 152, 2],
        'zbound': [-10.0, 10.0, 20.0],
        'dbound': [1.0, 150.0, 2.0], }

    dx, bx, nx = gen_dx_bx(**grid_config_od)
    if len(cams) > 6:
        bev_h = merge_h // 2 * 3
    else:
        bev_h = merge_h

    ratio = bev_h / int(nx[1])
    bev_painter_h = bev_h
    bev_painter_w = int(nx[0] * ratio)
    img_bev = np.full((bev_painter_h, bev_painter_w, 3), 0, dtype=np.uint8)
    # img_bev = gen_bev_painter(bev_painter_h, bev_painter_w, grid_config_od, ratio, background=0)
    import cv2
    # draw selfcar
    selfcar_position = np.array([[-0.75, 1.5], [0.75, -1.5]])
    selfcar_bev_x = ((selfcar_position[:, 0] - bx[0]) / dx[0] * ratio).astype(np.int)
    selfcar_bev_y = bev_painter_h - ((selfcar_position[:, 1] - bx[1]) / dx[1] * ratio).astype(np.int)
    cv2.rectangle(img_bev, (selfcar_bev_x[0], selfcar_bev_y[0]),
                  (selfcar_bev_x[1], selfcar_bev_y[1]), (255, 255, 255), -1)

    grid_config_lane = {
        'xbound': [-56, 56, 0.2],
        'ybound': [-32, 120, 0.2],
        'zbound': [-10.0, 10.0, 20.0],
        'dbound': [1.0, 150.0, 2.0], }
    ldx, lbx, lnx = gen_dx_bx(**grid_config_lane)

    points = []
    thres = 0.2
    for sc, c, v, u in zip(scores, clses, ys, xs):
        lu1 = v * ldx[1] + lbx[1]
        lv1 = u * ldx[0] + lbx[0]
        lu = (lv1 - bx[0]) / dx[0]
        lv = (lu1 - bx[1]) / dx[1]

        if sc > thres:
            cv2.circle(img_bev, (int(lu * ratio), int(bev_painter_h - lv * ratio)), radius=2, color=_YELLOW,
                       thickness=2)
            cv2.putText(
                img_bev,
                str(int(c)),
                # (int(x + 2), int(y)),
                (int(lu * ratio - 4), int(bev_painter_h - lv * ratio) + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                _WHITE,
                1,
            )
            # points.append([int(x), int(y)])
            # for v, u in zip(v_edge, u_edge):
            #
            #     lu1 = v * ldx[1] + lbx[1]
            #     lv1 = u * ldx[0] + lbx[0]
            #     lu = (lv1 - bx[0]) / dx[0]
            #     lv = (lu1 - bx[1]) / dx[1]
            #     if int(bev_painter_h - lv * ratio - 1) >= 900 or int(lu * ratio) >= 284:
            #         continue
            #     img_bev[int(bev_painter_h - lv * ratio - 1), int(lu * ratio)] = (float(solid_line[v, u]) * 255.0, 0.0, 0.0)

    show_img_bev = img_bev
    # show_img_bev, _ = draw_dets(img_bev, scores, clses, ys, xs)
    dx, bx, nx = gen_dx_bx(**grid_config)
    draw_imgs = []
    for cam_i, cam in enumerate(cams):
        img = imgs[cam_i]
        draw_imgs.append(img)
    img_to_show = multicam_show_imgs(cams, draw_imgs)
    return img_to_show, show_img_bev


def show_hdmap_seg(imgs, intrins, rots, trans, hdmap_out, cams, grid_config,
                   img_bev=None):
    road_edge, dash_line, solid_line, stop_line, cross_walk = (
        hdmap_out["road_edge"],
        hdmap_out["dash_line"],
        hdmap_out["solid_line"],
        hdmap_out["stop_line"],
        hdmap_out["cross_walk"],
    )

    road_edge = road_edge.squeeze().sigmoid().cpu().numpy()
    dash_line = dash_line.squeeze().sigmoid().cpu().numpy()
    solid_line = solid_line.squeeze().sigmoid().cpu().numpy()
    stop_line = stop_line.squeeze().sigmoid().cpu().numpy()
    cross_walk = cross_walk.squeeze().argmax(0).cpu().numpy()
    h, w = road_edge.shape
    im_bev = np.zeros([h, w, 3])

    im_bev[..., 0] = solid_line ** 2 * 255
    im_bev[..., 1] = dash_line ** 2 * 255
    im_bev[..., 0] += stop_line ** 2 * 255
    im_bev[..., 2] += stop_line ** 2 * 255

    v_solid_line, u_solid_line = np.where(solid_line > 0.5)
    v_dash_line, u_dash_line = np.where(dash_line > 0.5)
    v_stop_line, u_stop_line = np.where(stop_line > 0.5)
    v_road_edge, u_road_edge = np.where(road_edge > 0.5)

    road_edge_points = [[u, v] for v, u in zip(v_road_edge, u_road_edge)]
    solid_lines_points = [[u, v] for v, u in zip(v_solid_line, u_solid_line)]
    dash_lines_points = [[u, v] for v, u in zip(v_dash_line, u_dash_line)]
    stop_lines_points = [[u, v] for v, u in zip(v_stop_line, u_stop_line)]
    road_lines_points = solid_lines_points + dash_lines_points + stop_lines_points + road_edge_points

    v_edge, u_edge = np.where(road_edge > 0.2)
    for v, u in zip(v_edge, u_edge):
        im_bev[int(v), int(u)] = (0.0, 0.0, float((np.exp(road_edge[v, u]) - 1) / 2.7) * 255.0)

    
    dx, bx, nx = gen_dx_bx(**grid_config)
    nums = len(rots)
    cam2lidar = np.eye(4)
    cam2lidar = np.tile(cam2lidar[np.newaxis], (nums, 1, 1))
    cam2lidar[:, :3, :3] = rots
    cam2lidar[:, :3, 3] = trans
    draw_imgs = []
    if np.array(road_lines_points).shape[0] > 0:
        ims_lineseg_points = get_lines_points_cams(road_lines_points, dx, bx, cam2lidar, intrins, upsample=0.25)
        for cam_i, cam in enumerate(cams):
            img = imgs[cam_i]
            img_h, img_w = img.shape[:2]
            im_roadseg_points = ims_lineseg_points[cam_i]
            valid_mask = (
                    (im_roadseg_points[:, 0] > 0)
                    & (im_roadseg_points[:, 0] < img_w)
                    & (im_roadseg_points[:, 1] > 0)
                    & (im_roadseg_points[:, 1] < img_h)
                    & (im_roadseg_points[:, 2] > 0)
            )
            im_roadseg_points = im_roadseg_points[valid_mask, ...]

            us = im_roadseg_points[:, 1].astype(np.int)
            vs = im_roadseg_points[:, 0].astype(np.int)
            # img[us, vs] = img[us, vs] + [[255, 0, 0]]

            img[us, vs] = [_YELLOW]
            import cv2
            for u, v in zip(us, vs):
                cv2.circle(img, (int(v), int(u)), 1, _RED, -1)
            draw_imgs.append(img)
    else:
        for cam_i, cam in enumerate(cams):
            img = imgs[cam_i]
            draw_imgs.append(img)

    img_to_show = multicam_show_imgs(cams, draw_imgs)

    im_bev = np.flip(im_bev, 0)

    return img_to_show, im_bev


def show_center_seg(imgs, intrins, rots, trans, hdmap_out, cams, grid_config,
                    img_bev=None):
    center_line = hdmap_out["center_line"]

    center_line = center_line.squeeze().sigmoid().cpu().numpy()
    h, w = center_line.shape
    im_bev = np.zeros([h, w, 3])

    im_bev[..., 0] += center_line * 255
    im_bev[..., 1] += center_line * 127
    im_bev[..., 2] += center_line * 255
    # im_bev = np.flip(im_bev, 0)

    # 
    #
    # grid_config_od = {
    #     'xbound': [-48, 48, 2],
    #     'ybound': [-152, 152, 2],
    #     'zbound': [-10.0, 10.0, 20.0],
    #     'dbound': [1.0, 150.0, 2.0], }
    #
    # dx, bx, nx = gen_dx_bx(**grid_config_od)
    # if len(cams) > 6:
    #     bev_h = merge_h // 2 * 3
    # else:
    #     bev_h = merge_h
    #
    # ratio = bev_h / int(nx[1])
    # bev_painter_h = bev_h
    # bev_painter_w = int(nx[0] * ratio)
    #
    # img_bev = np.full((bev_painter_h, bev_painter_w, 3), 0, dtype=np.uint8)
    # # img_bev = gen_bev_painter(bev_painter_h, bev_painter_w, grid_config_od, ratio, background=0)
    #
    # import cv2
    # # draw selfcar
    # selfcar_position = np.array([[-0.75, 1.5], [0.75, -1.5]])
    # selfcar_bev_x = ((selfcar_position[:, 0] - bx[0]) / dx[0] * ratio).astype(np.int)
    # selfcar_bev_y = bev_painter_h - ((selfcar_position[:, 1] - bx[1]) / dx[1] * ratio).astype(np.int)
    # cv2.rectangle(img_bev, (selfcar_bev_x[0], selfcar_bev_y[0]),
    #               (selfcar_bev_x[1], selfcar_bev_y[1]), (255, 255, 255), -1)
    #
    # grid_config_lane = {
    #     'xbound': [-56, 56, 0.2],
    #     'ybound': [-32, 120, 0.2],
    #     'zbound': [-10.0, 10.0, 20.0],
    #     'dbound': [1.0, 150.0, 2.0], }
    # ldx, lbx, lnx = gen_dx_bx(**grid_config_lane)
    # v_edge, u_edge = np.where(center_line > 0.45)
    # for v, u in zip(v_edge, u_edge):
    #     lu1 = v * ldx[1] + lbx[1]
    #     lv1 = u * ldx[0] + lbx[0]
    #     lu = (lv1 - bx[0]) / dx[0]
    #     lv = (lu1 - bx[1]) / dx[1]
    #     if int(bev_painter_h - lv * ratio - 1) >= 900 or int(lu * ratio) >= 284:
    #         continue
    #     img_bev[int(bev_painter_h - lv * ratio - 1), int(lu * ratio)] = (
    #         float(center_line[v, u]) * 255, float(center_line[v, u]) * 127, float(center_line[v, u]) * 255.0)

    v_center_line, u_center_line = np.where(center_line > 0.5)
    road_center_line_points = [[u, v] for v, u in zip(v_center_line, u_center_line)]
    
    dx, bx, nx = gen_dx_bx(**grid_config)
    nums = len(rots)
    cam2lidar = np.eye(4)
    cam2lidar = np.tile(cam2lidar[np.newaxis], (nums, 1, 1))
    cam2lidar[:, :3, :3] = rots
    cam2lidar[:, :3, 3] = trans
    draw_imgs = []
    if np.array(road_center_line_points).shape[0] > 0:
        ims_lineseg_points = get_lines_points_cams(road_center_line_points, dx, bx, cam2lidar, intrins, upsample=0.25)
        for cam_i, cam in enumerate(cams):
            img = imgs[cam_i]
            img_h, img_w = img.shape[:2]
            im_roadseg_points = ims_lineseg_points[cam_i]

            valid_mask = (
                    (im_roadseg_points[:, 0] > 0)
                    & (im_roadseg_points[:, 0] < img_w)
                    & (im_roadseg_points[:, 1] > 0)
                    & (im_roadseg_points[:, 1] < img_h)
                    & (im_roadseg_points[:, 2] > 0)
            )
            im_roadseg_points = im_roadseg_points[valid_mask, ...]

            us = im_roadseg_points[:, 1].astype(np.int)
            vs = im_roadseg_points[:, 0].astype(np.int)
            # img[us, vs] = img[us, vs] + [[255, 0, 0]]
            img[us, vs] = [_YELLOW]
            import cv2
            for u, v in zip(us, vs):
                cv2.circle(img, (int(v), int(u)), 1, _GREEN, -1)
            draw_imgs.append(img)
    else:
        draw_imgs = []
        for cam_i, cam in enumerate(cams):
            img = imgs[cam_i]
            draw_imgs.append(img)

    img_to_show = multicam_show_imgs(cams, draw_imgs)

    return img_to_show, im_bev


def show_multicam_hdmap(imgs, intrins, rots, trans, hdmap_out, cams, grid_config,
                        img_bev=None):
    COLOR = (_PURPLE, _YELLOW, _RED, _BLUE, _BLACK, _GREEN)

    if img_bev is None:
        

        if len(cams) > 6:
            bev_h = merge_h // 2 * 3
        else:
            bev_h = merge_h

        dx, bx, nx = gen_dx_bx(**grid_config)
        ratio = bev_h / int(nx[1])
        bev_painter_h = bev_h
        bev_painter_w = int(nx[0] * ratio)
        # img_bev = np.full((bev_painter_h, bev_painter_w, 3), 128, dtype=np.uint8)
        img_bev = gen_bev_painter(bev_painter_h, bev_painter_w, grid_config, ratio)

    # pred_vector: [n_lines, 40, 2]
    # pred_label: [n_lines]
    
    dx, bx, nx = gen_dx_bx(**grid_config)

    bev_painter_h = img_bev.shape[0]
    ratio = bev_painter_h / int(nx[1])

    pred_vector, score, pred_label = hdmap_out
    pred_vector = pred_vector.cpu().detach().numpy()
    pred_label = pred_label.cpu().detach().numpy()
    #
    pred_vector_bev = pred_vector.copy()
    pred_vector_bev[:, :, 0] = (pred_vector_bev[:, :, 0] - bx[0]) / dx[0]
    pred_vector_bev[:, :, 1] = (pred_vector_bev[:, :, 1] - bx[1]) / dx[1]

    pred_vector_bev = (pred_vector_bev * ratio).astype(np.int)
    pred_vector_bev[:, :, 1] = bev_painter_h - pred_vector_bev[:, :, 1]

    for vector, label in zip(pred_vector_bev, pred_label):
        color = COLOR[label.item()]
        cv2.polylines(
            img_bev,
            np.int32([vector]),
            isClosed=False,
            color=color,
            thickness=1,
        )

    cam2lidar = np.eye(4)
    cam2lidar = np.tile(cam2lidar[np.newaxis], (len(cams), 1, 1))
    cam2lidar[:, :3, :3] = rots
    cam2lidar[:, :3, 3] = trans

    # draw img
    line_num, sample_points_num = pred_vector.shape[:2]
    # [line_n, sample_n, 4]
    pred_vector_lidar = np.concatenate([pred_vector,
                                        np.zeros([line_num, sample_points_num, 1]),
                                        np.ones([line_num, sample_points_num, 1])], axis=-1)

    for line_points, label in zip(pred_vector_lidar, pred_label):
        color = COLOR[label.item()]
        line_points = np.tile(line_points[np.newaxis], (len(cams), 1, 1))
        line_points = np.linalg.inv(cam2lidar) @ line_points.transpose((0, 2, 1))

        ims_line_points = intrins @ line_points[:, :3, :]
        ims_line_points = ims_line_points.transpose((0, 2, 1))  # [6, N, 3]
        ims_line_points[:, :, :2] = ims_line_points[:, :, :2] / ims_line_points[:, :, 2:3]

        # draw_imgs = []
        for cam_i, cam in enumerate(cams):
            img = imgs[cam_i]
            img_h, img_w = img.shape[:2]
            im_line_points = ims_line_points[cam_i]  # [N, 3]

            valid_mask = (
                    (im_line_points[:, 0] > 0)
                    & (im_line_points[:, 0] < img_w)
                    & (im_line_points[:, 1] > 0)
                    & (im_line_points[:, 1] < img_h)
                    & (im_line_points[:, 2] > 0)
            )
            im_line_points = im_line_points[valid_mask, ...]
            im_line_points = im_line_points[:, :2]

            cv2.polylines(
                img,
                np.int32([im_line_points]),
                isClosed=False,
                color=color,
                thickness=2,
            )

    draw_imgs = [imgs[cam_i] for cam_i, cam in enumerate(cams)]
    img_to_show = multicam_show_imgs(cams, draw_imgs)

    return img_to_show, img_bev


def show_multicam_multielements(imgs, intrins, rots, trans, cams, grid_config,
                                timestamp, tasks, show_dir, pred_dict, filter_cids=[5],nuscenes=True,
                                 roadseg_points=None, hdline_out=None, center_seg=None, hdflags=None,
                                hdmap_out=None, jira_info=None, rot_uncerts=None, curr_pose=None,
                                hdline_grid_config=None):
    os.makedirs(show_dir, exist_ok=True)

    img_to_show, img_bev = None, None

    if pred_dict is not None:
        img_to_show, img_bev = show_multicam_bboxes(imgs, intrins, rots, trans, cams,
                                                    grid_config, timestamp, tasks, show_dir, pred_dict,
                                                    filter_cids=filter_cids,
                                                    nuscenes=nuscenes,
                                                    rot_uncerts=rot_uncerts
                                                    )
        img_bev = cv2.resize(img_bev, (300, 900))
    # if roadseg_points is not None:
    #     img_to_show, img_bev = show_multicam_roadsegpoints(imgs, intrins, rots, trans, roadseg_points,
    #                                                        cams, grid_config, timestamp,
    #                                                        show_dir=show_dir,
    #                                                        img_bev=img_bev)

    if hdline_grid_config is None:
        hdline_grid_config = grid_config

    img_bev_road = None
    if roadseg_points is not None:
        img_to_show, img_bev_road = show_multicam_roadsegpoints_infer(imgs, intrins, rots, trans, roadseg_points,
                                                                      cams, hdline_grid_config, timestamp,
                                                                      show_dir=show_dir,
                                                                      img_bev=img_bev_road)
        img_bev_road = cv2.resize(img_bev_road, (300, 900))

    hdline_img_bev = None
    if hdline_out is not None:
        _, hdline_img_bev = show_multicam_hdmap(
            imgs, intrins, rots, trans, hdline_out,
            cams, grid_config, img_bev=None)
    img_bev_hdmap_seg = None
    if hdmap_out is not None:
        img_to_show, img_bev_hdmap_seg = show_hdmap_seg(imgs, intrins, rots, trans,
                                                        hdmap_out=hdmap_out,
                                                        cams=cams,
                                                        grid_config=hdline_grid_config,
                                                        img_bev=img_bev)
        img_bev_hdmap_seg = cv2.resize(img_bev_hdmap_seg, (350, 900))
    img_bev_center = None
    if center_seg is not None:
        img_to_show, img_bev_center = show_center_seg(imgs, intrins, rots, trans,
                                                      hdmap_out=center_seg,
                                                      cams=cams,
                                                      grid_config=hdline_grid_config,
                                                      img_bev=img_bev)
        img_bev_center = cv2.resize(img_bev_center, (350, 900))
    img_bev_flag = None
    if hdflags is not None:
        img_to_show, img_bev_flag = show_multicam_hdflags(imgs, intrins, rots, trans, hdflags, cams,
                                                          hdline_grid_config,
                                                          img_bev=img_bev)
        img_bev_flag = cv2.resize(img_bev_flag, (350, 900))
    all_img_show = img_to_show
    # all_img_show = np.concatenate([all_img_show, img_bev_hdmap_seg, img_bev_center, img_bev_flag], axis=1)

    for bev_img in [img_bev, img_bev_hdmap_seg, img_bev_center, img_bev_flag, img_bev_road]:
        if bev_img is not None:
            all_img_show = np.concatenate([all_img_show, bev_img], axis=1)

    if hdline_img_bev is not None:
        hdline_img_bev = cv2.resize(hdline_img_bev, (merge_h, merge_h))
        hdline_img_bev = cv2.resize(hdline_img_bev, (360, 900))
        all_img_show = np.concatenate([all_img_show, hdline_img_bev], axis=1)

    # if img_bev_road is not None:
    #     all_img_show = np.concatenate([all_img_show, img_bev_road], axis=1)

    if jira_info is not None:
        text_painter = np.full((20, all_img_show.shape[1], 3), 128, dtype=np.uint8)
        all_img_show = np.concatenate([text_painter, all_img_show], axis=0)
        all_img_show = all_img_show.astype(np.uint8)
        all_img_show = Image.fromarray(cv2.cvtColor(all_img_show, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(all_img_show)
        fontStyle = ImageFont.truetype(
            "./tools/font/SimHei.ttf", 20, encoding="utf-8")
        draw.text((0, 0), jira_info, _RED, font=fontStyle)
        all_img_show = cv2.cvtColor(np.asarray(all_img_show), cv2.COLOR_RGB2BGR)

    save_path = os.path.join(show_dir, str(timestamp) + ".jpg")
    cv2.imwrite(save_path, all_img_show)
