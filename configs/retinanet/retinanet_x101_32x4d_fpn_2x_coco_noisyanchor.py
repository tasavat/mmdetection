_base_ = './retinanet_x101_32x4d_fpn_2x_coco.py'
model = dict(
    bbox_head=dict(
        type='NoisyAnchorRetinaHead',
        max_n_bbox_per_gt=30,
        alpha=0.75,
        gamma=1.0,
    ),
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='checkpoints/resnext101_32x4d-a5af3160.pth'))
)