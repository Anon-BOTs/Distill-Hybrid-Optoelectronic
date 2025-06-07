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


onn_config = {
    "lmbda": 1050e-9,  ## 1050e-9
    "dx": 1.75e-6,
    "nd": 128, # 50 ## ~2 * N
    "d1": 100e-6,
    "d2": 100e-6,
    "N": 64, # 50 ## ~2 * N
    "layersCount": 2  ## 4
}

# model settings
norm_cfg = dict(type='LN', requires_grad=True)
model=dict(
    type="CustomFCOS",
    pretrained="mae_pretrain_vit_base.pth",
    backbone=dict(
        type='ONNViT',
        img_size=1024,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        use_abs_pos_emb=True,

        merge_to_one=True,
        up_channels=True,
        onn_channel_ratio=6,
        onn_depth=[10, 11],
        use_bias=True,
        use_abs=False,
        use_square=True,
        onn_cfg=onn_config
        ),
    neck=dict(
        type='CustomFPNV2',
        in_channels=[768, 768, 768, 768],
        out_channels=256,
        norm_cfg=norm_cfg,
        use_residual=False,
        num_outs=5),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=10,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        # strides=[8, 16, 32, 64, 128],
        strides=[4, 8, 16, 32, 64],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
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
        max_per_img=100)
)


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
image_size = (1024, 1024)

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(
        type='Resize',
        img_scale=image_size,
        ratio_range=(0.1, 2.0),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=image_size),  # padding to image_size leads 0.5+ mAP
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1024),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'], meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg', 'image_id')),
        ])
]
dataset_type = 'CustomNuScenesDataset2D'
data_root = 'path_to_your_dataset'

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        use_front_cam=True,
        ann_file=data_root + '/train_infos/nuscenes2d_temporal_infos_train.pkl',
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
             use_front_cam=True,
             pipeline=test_pipeline, 
             collect_keys=collect_keys + ['img', 'img_metas'], 
             queue_length=queue_length, 
             ann_file=data_root + '/train_infos/nuscenes2d_temporal_infos_val.pkl', 
             classes=class_names, 
             modality=input_modality),
    test=dict(type=dataset_type, 
              data_root=data_root, 
              use_front_cam=True,
              pipeline=test_pipeline, 
              collect_keys=collect_keys + ['img', 'img_metas'], 
              queue_length=queue_length, 
              ann_file=data_root + '/train_infos/nuscenes2d_temporal_infos_val.pkl', 
              classes=class_names, 
              modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
    )
optimizer_config = dict(grad_clip=None)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.1,
    constructor='LayerDecayOptimizerConstructor', 
                 paramwise_cfg=dict(
                        num_layers=12, 
                        layer_decay_rate=0.7,
                        custom_keys={
                            'bias': dict(decay_multi=0.),
                            'pos_embed': dict(decay_mult=0.),
                            'relative_position_bias_table': dict(decay_mult=0.),
                            'norm': dict(decay_mult=0.),
                            "rel_pos_h": dict(decay_mult=0.),
                            "rel_pos_w": dict(decay_mult=0.),
                            }
                            )
                 )
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=250,
    warmup_ratio=0.067,
    step=[22, 24])
runner = dict(type='EpochBasedRunner', max_epochs=25)
checkpoint_config = dict(interval=4, max_keep_ckpts=7, save_last=True)
evaluation = dict(interval=25, metric='bbox')