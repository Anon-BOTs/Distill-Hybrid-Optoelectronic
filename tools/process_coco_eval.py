import json
import os
import pickle

from tqdm import tqdm, trange

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

ori_coco_file = '/nuscenes_full/nuScenes/nuscenes_infos_val_mono3d.coco.json'
cur_coco_file = '/nuscenes_full/nuScenes/nuscenes_infos_val_mono3d_10cls.coco.json'

ff = {}
with open(ori_coco_file, 'r') as f:
    ori_coco = json.load(f)

for info in ori_coco['annotations']:
    ff[info['iscrowd']] = ff.get(info['iscrowd'], 0) + 1

print(ff)
with open('/lf7-jceph/open_source/nuscenes2d_temporal_infos_val.pkl', 'rb') as f:
    ann_info = pickle.load(f)

new_coco = {'annotations' : [], 'images' : []}
print(len(ori_coco['annotations']))

id = 0
for info in tqdm(ann_info['infos']):
    token = info['token']
    for cam_id, (cam, cam_info) in enumerate(info['cams'].items()):
        boxes = info['bboxes2d'][cam_id]
        labels = info['labels2d'][cam_id]
        filename = cam_info['data_path']
        sample_token = cam_info['sample_data_token']
        new_coco['images'].append(
            {
                'filename' : filename,
                'id' : sample_token,
                'token' : token,
                'width' : 1600,
                'height' : 900,
            }
        )
        for box, label in zip(boxes, labels):
            new_coco['annotations'].append({
                'filename' : filename,
                'image_id' : sample_token,
                # 'area'
                'category_name': class_names[label],
                'category_id' : label,
                'bbox' : box,
                'iscrowd' : 0,
                'id' : id
            })
            id += 1

