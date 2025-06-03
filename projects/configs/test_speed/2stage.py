_base_ = [
    '../base/nus-3d.py',
    '../base/default_runtime.py',
    '../base/schedule_1x.py'
]

plugin=True
plugin_dir='projects/mmdet3d_plugin/'

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
queue_length = 1
num_frame_losses = 1
collect_keys=['lidar2img', 'intrinsics', 'extrinsics','timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv']
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

# model settings
model = dict(
    type='TestTime',
    # backbone=dict(
    #     type="TestMMRegNet",
    #     choice="800MF",
    #     valid_layers=[0, 1],
    #     out_strides=[4, 8, 16, 32],
    #     pretrained_path=None,
    #     freeze=False,
    # ),
    backbone=dict(
        type='TestMMRegNet',
        arch='regnetx_800mf',
        out_indices=(0, 1, 2, 3),
        valid_layers=[0, 1],
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='regnetx_800mf-1f4be4c7.pth')),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Resize', img_scale=(667, 400), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        img_scale=(667, 400),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'], meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg', 'image_id')),
        ])
]

dataset_type = 'CustomNuScenesDataset2D'
data_root = '/high_perf_store/surround-view/datasets/'

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        use_front_cam=True,
        ann_file='/high_perf_store/surround-view/datasets/train_infos/nuscenes2d_temporal_infos_train.pkl',
        num_frame_losses=num_frame_losses,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        collect_keys=collect_keys + ['img', 'prev_exists', 'img_metas'],
        queue_length=queue_length,
        test_mode=False,
        filter_empty_gt=True,
        use_valid_flag=True,
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type, 
             data_root=data_root, 
             use_front_cam=False,
             pipeline=test_pipeline, 
             collect_keys=collect_keys + ['img', 'img_metas'], 
             queue_length=queue_length, 
             ann_file='/high_perf_store/surround-view/datasets/train_infos/nuscenes2d_temporal_infos_val.pkl', 
             classes=class_names, 
             modality=input_modality),
    test=dict(type=dataset_type, 
              data_root=data_root, 
              use_front_cam=True,
              pipeline=test_pipeline, 
              collect_keys=collect_keys + ['img', 'img_metas'], 
              queue_length=queue_length, 
              ann_file='/high_perf_store/surround-view/datasets/train_infos/nuscenes2d_temporal_infos_val.pkl', 
              classes=class_names, 
              modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
    )
# optimizer
optimizer = dict(
    lr=0.01, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
