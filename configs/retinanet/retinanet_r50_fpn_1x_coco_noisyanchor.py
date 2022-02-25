_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    bbox_head=dict(
        type='NoisyAnchorRetinaHead',
        max_n_bbox_per_gt=30,
        cleanliness_alpha=0.75,
        reweight_gamma=1.0,
        focal_loss_alpha=0.50,
        focal_loss_gamma=2.0,
    ),
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint='checkpoints/resnet50-19c8e357.pth'))
    )
# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
