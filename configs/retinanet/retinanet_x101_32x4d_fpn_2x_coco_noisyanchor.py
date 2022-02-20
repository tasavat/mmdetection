_base_ = './retinanet_x101_32x4d_fpn_2x_coco.py'
model = dict(
    bbox_head=dict(
        type='NoisyAnchorRetinaHead',
        max_n_bbox_per_gt=30,
        alpha=0.75,
        gamma=1.0,
    )
)