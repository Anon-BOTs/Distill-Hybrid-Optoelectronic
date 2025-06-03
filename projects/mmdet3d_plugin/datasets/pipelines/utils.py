from numba import jit
import numpy as np
import random
from decimal import ROUND_HALF_UP, Decimal
import os
from pyquaternion import Quaternion
from shapely import affinity, ops
from shapely.geometry import LineString, MultiLineString, MultiPoint, MultiPolygon, box

import math
import os
from functools import reduce

import cv2
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits as nus_splits
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import transform_matrix
from PIL import Image
from pyquaternion import Quaternion
from shapely import affinity, ops
from shapely.geometry import LineString, MultiLineString, MultiPoint, MultiPolygon, box, Point, Polygon
from tqdm import tqdm
import copy
# from dreamer_models import DriveDreamer2Pipeline
import sys
import argparse
from datetime import datetime

def quaternion_yaw(q: Quaternion):
    """Calculate the yaw angle from a quaternion.

    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame. It does not
    work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """
    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))
    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])
    return yaw

@jit
def get_map_geom(patch_box, patch_angle, layer_names, nusc_map, map_explorer):
    map_geom = []
    for layer_name in layer_names:
        if layer_name in ['road_divider', 'lane_divider']:
            geoms = map_explorer._get_layer_line(patch_box, patch_angle, layer_name)
            map_geom.append((layer_name, geoms))
        elif layer_name in ['road_segment', 'lane']:
            geoms = map_explorer._get_layer_polygon(patch_box, patch_angle, layer_name)
            map_geom.append((layer_name, geoms))
        elif layer_name in ['ped_crossing']:
            geoms = get_ped_crossing_line(patch_box, patch_angle, nusc_map, map_explorer)
            map_geom.append((layer_name, geoms))
    return map_geom

def get_all_map_geom(layer_names, nusc_map, map_explorer):
    """
    获取整张地图中所有 layer 的几何信息
    :param nusc_map: NuScenesMap 对象
    :param layer_names: 要提取的 layer，比如 ['road_segment', 'lane']
    :return: dict: {layer_name: list of shapely Polygon/LineString}
    """
    from shapely.geometry import Polygon, LineString

    map_geom = []
    for layer in layer_names:
        if layer in ['ped_crossing']:
            geometries = get_all_ped_crossing_line(nusc_map, map_explorer)
        else:
            elements = getattr(nusc_map, layer)
            geometries = []
            for el in elements:
                if 'line_token' in el:
                    line = nusc_map.extract_line(el['line_token'])
                    if line.is_empty:  # Skip lines without nodes.
                        continue
                    geometries.append(line)
                if 'polygon_token' in el:
                    polygon = nusc_map.extract_polygon(el['polygon_token'])
                    if polygon.is_valid and not polygon.is_empty:
                        geometries.append(polygon)

        map_geom.append((layer, geometries))
    return map_geom


def get_ped_crossing_line(patch_box, patch_angle, nusc_map, map_explorer):
    def add_line(poly_xy, idx, patch, patch_angle, patch_x, patch_y, line_list):
        points = [(p0, p1) for p0, p1 in zip(poly_xy[0, idx : idx + 2], poly_xy[1, idx : idx + 2])]
        line = LineString(points)
        line = line.intersection(patch)
        if not line.is_empty:
            line = affinity.rotate(line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
            line = affinity.affine_transform(line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
            line_list.append(line)

    patch_x = patch_box[0]
    patch_y = patch_box[1]
    patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
    line_list = []
    records = nusc_map.ped_crossing
    for record in records:
        polygon = map_explorer.extract_polygon(record['polygon_token'])
        poly_xy = np.array(polygon.exterior.xy)
        dist = np.square(poly_xy[:, 1:] - poly_xy[:, :-1]).sum(0)
        x1, x2 = np.argsort(dist)[-2:]
        add_line(poly_xy, x1, patch, patch_angle, patch_x, patch_y, line_list)
        add_line(poly_xy, x2, patch, patch_angle, patch_x, patch_y, line_list)
    return line_list

def get_all_ped_crossing_line(nusc_map, map_explorer):
    def add_line(poly_xy, idx, line_list):
        points = [(p0, p1) for p0, p1 in zip(poly_xy[0, idx : idx + 2], poly_xy[1, idx : idx + 2])]
        line = LineString(points)
        if not line.is_empty:
            line_list.append(line)

    line_list = []
    records = nusc_map.ped_crossing
    for record in records:
        polygon = map_explorer.extract_polygon(record['polygon_token'])
        poly_xy = np.array(polygon.exterior.xy)
        dist = np.square(poly_xy[:, 1:] - poly_xy[:, :-1]).sum(0)
        x1, x2 = np.argsort(dist)[-2:]
        add_line(poly_xy, x1, line_list)
        add_line(poly_xy, x2, line_list)
    return line_list


def line_geoms_to_vectors(line_geom):
    line_vectors_dict = dict()
    for line_type, a_type_of_lines in line_geom:
        one_type_vectors = one_type_line_geom_to_vectors(a_type_of_lines)
        line_vectors_dict[line_type] = one_type_vectors
    return line_vectors_dict


def one_type_line_geom_to_vectors(line_geom):
    line_vectors = []
    for line in line_geom:
        if not line.is_empty:
            if line.geom_type == 'MultiLineString':
                for line_i in line.geoms:
                    line_vectors.append(sample_pts_from_line(line_i))
            elif line.geom_type == 'LineString':
                line_vectors.append(sample_pts_from_line(line))
            else:
                raise NotImplementedError
    return line_vectors


def sample_pts_from_line(line):
    distances = np.arange(0, line.length, 1)
    sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
    num_valid = len(sampled_points)
    return sampled_points, num_valid


def poly_geoms_to_vectors(polygon_geom):
    roads = polygon_geom[0][1]
    lanes = polygon_geom[1][1]
    union_roads = ops.unary_union(roads)
    union_lanes = ops.unary_union(lanes)
    union_segments = ops.unary_union([union_roads, union_lanes])
    max_x = 102.4 / 2  # FIXME 102.4 hardcode
    max_y = 102.4 / 2
    local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
    exteriors = []
    interiors = []
    if union_segments.geom_type != 'MultiPolygon':
        union_segments = MultiPolygon([union_segments])
    for poly in union_segments.geoms:
        exteriors.append(poly.exterior)
        for inter in poly.interiors:
            interiors.append(inter)
    results = []
    for ext in exteriors:
        if ext.is_ccw:
            ext.coords = list(ext.coords)[::-1]
        lines = ext.intersection(local_patch)
        if isinstance(lines, MultiLineString):
            lines = ops.linemerge(lines)
        results.append(lines)
    for inter in interiors:
        if not inter.is_ccw:
            inter.coords = list(inter.coords)[::-1]
        lines = inter.intersection(local_patch)
        if isinstance(lines, MultiLineString):
            lines = ops.linemerge(lines)
        results.append(lines)
    return one_type_line_geom_to_vectors(results)


def preprocess_map(vectors, canvas_size, max_channel, thickness):
    confidence_levels = [-1]
    vector_num_list = {}
    for i in range(max_channel):
        vector_num_list[i] = []
    for vector in vectors:
        if vector['pts_num'] >= 2:
            vector_num_list[vector['type']].append(LineString(vector['pts'][: vector['pts_num']]))
    idx = 1
    filter_masks = []
    instance_masks = []
    for i in range(max_channel):
        map_mask, idx = line_geom_to_mask(vector_num_list[i], confidence_levels, canvas_size, thickness, idx)
        instance_masks.append(map_mask)
        filter_mask, _ = line_geom_to_mask(vector_num_list[i], confidence_levels, canvas_size, thickness + 4, 1)
        filter_masks.append(filter_mask)
    filter_masks = np.stack(filter_masks)
    instance_masks = np.stack(instance_masks)
    instance_masks = overlap_filter(instance_masks, filter_masks)
    semantic_masks = instance_masks != 0
    return semantic_masks


def line_geom_to_mask(layer_geom, confidence_levels, canvas_size, thickness, idx, type='index', angle_class=36):
    patch = box(0, 0, canvas_size[1], canvas_size[0])
    map_mask = np.zeros(canvas_size, np.uint8)
    for line in layer_geom:
        if isinstance(line, tuple):
            line, confidence = line
        else:
            confidence = None
        new_line = line.intersection(patch)
        if not new_line.is_empty:
            confidence_levels.append(confidence)
            if new_line.geom_type == 'MultiLineString':
                for new_single_line in new_line.geoms:
                    map_mask, idx = mask_for_lines(new_single_line, map_mask, thickness, idx, type, angle_class)
            else:
                map_mask, idx = mask_for_lines(new_line, map_mask, thickness, idx, type, angle_class)
    return map_mask, idx


def overlap_filter(mask, filter_mask):
    C, _, _ = mask.shape
    for c in range(C - 1, -1, -1):
        filter = np.repeat((filter_mask[c] != 0)[None, :], c, axis=0)
        mask[:c][filter] = 0
    return mask


def round_dec(n, d):
    s = '0.' + '0' * d
    return Decimal(str(n)).quantize(Decimal(s), ROUND_HALF_UP)


def mask_for_lines(lines, mask, thickness, idx, type='index', angle_class=36):
    lines = lines.coords
    lines = np.array(lines)
    coords = np.zeros_like(lines)
    for i in range(lines.shape[0]):
        for j in range(lines.shape[1]):
            lin = lines[i][j]
            lr = round_dec(lin, 0)
            coords[i][j] = lr
    coords = coords.astype(np.int32)
    if len(coords) < 2:
        return mask, idx
    if type == 'backward':
        coords = np.flip(coords, 0)
    if type == 'index':
        cv2.polylines(mask, [coords], False, color=idx, thickness=thickness)
        idx += 1
    else:
        for i in range(len(coords) - 1):
            cv2.polylines(
                mask,
                [coords[i:]],
                False,
                color=get_discrete_degree(coords[i + 1] - coords[i], angle_class=angle_class),
                thickness=thickness,
            )
    return mask, idx