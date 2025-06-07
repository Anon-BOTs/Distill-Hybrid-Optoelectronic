_base_ = './fcos_r50_caffe_fpn_gn-head_1x_coco.py'

plugin=True
plugin_dir = "StreamPETR/projects/mmdet3d_plugin/"

onn_config = {
    "lmbda": 1050e-9,  ## 1050e-9
    "dx": 1.75e-6,
    "ndMap": {0 : 400, 1 : 400, 2 : 200, 3 : 80}, # 50 ## ~2 * N
    "d1": 100e-6,
    "d2": 100e-6,
    "NMap": {0 : 300, 1 : 200, 2 : 100, 3 : 40}, # 50 ## ~2 * N
    "layersCountMap": {0 : 4, 1 : 4, 2 : 2, 3 : 2}  ## 4
}


model=dict(
    type="FCOS",
    backbone=dict(
        type='ONNMMRegNet',
        _delete_=True,
        arch='regnetx_800mf',
        out_indices=(0, 1, 2, 3),
        onn_channels= [64, 128, 288, 256],
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        before_downsample=True,
        delete_extra=True,
        use_square=True,
        use_v2=True,
        use_bias=True,
        onn_stage=[3],
        onn_cfg=onn_config,
        init_cfg=dict(
            type='Pretrained', checkpoint='regnetx_800mf-1f4be4c7.pth')),
    neck=dict(
        in_channels=[64, 128, 288, 256],
    )
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# img_norm_cfg = dict(
#     mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(677, 400), keep_ratio=True),
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
        img_scale=(677, 400),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data=dict(
    samples_per_gpu=1, # 2
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)