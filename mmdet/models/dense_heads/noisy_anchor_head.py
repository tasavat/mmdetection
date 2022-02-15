"""
Implementation of noisy anchor head.
https://arxiv.org/pdf/1912.05086.pdf
"""

from .anchor_head import AnchorHead


@HEADS.register_module()
class NoisyAnchorHead(AnchorHead):
    """Anchor-based head used in `Learning from Noisy Anchor for One-stage Object Detection <https://arxiv.org/abs/1912.05086>`.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
    Args (Optional):
        feat_channels (int): Number of hidden channels. Used in child classes.
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Default False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """ 

    def __init__(self, num_classes, in_channels, **kwargs):
        super(NoisyAnchorHead, self).__init__(num_classes, in_channels, **kwargs)