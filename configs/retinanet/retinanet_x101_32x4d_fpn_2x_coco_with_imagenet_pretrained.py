_base_ = './retinanet_r50_fpn_2x_coco.py'
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint='checkpoints/resnext101_32x4d-a5af3160.pth')))
