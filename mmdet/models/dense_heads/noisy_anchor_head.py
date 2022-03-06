"""
Implementation of noisy anchor head.
https://arxiv.org/pdf/1912.05086.pdf
"""
import torch
import torch.nn as nn
from mmcv.runner import force_fp32

from mmdet.core import (
    anchor_inside_flags, 
    build_assigner, 
    build_bbox_coder,
    build_prior_generator, 
    build_sampler, 
    images_to_levels,
    multi_apply, 
    unmap
)
from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead


@HEADS.register_module()
class NoisyAnchorHead(AnchorHead):
    """Anchor-based head used in `Learning from Noisy Anchor for One-stage Object Detection <https://arxiv.org/abs/1912.05086>`.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.

    Args (Optional):
        max_n_bbox_per_gt (int): max number of bbox per gt
        cleanliness_alpha (int): cleanliness score weight
        reweight_gamma (float): reweight loss weight
        focal_loss_alpha (float): focal loss weight
        focal_loss_gamma: (float): focal loss weight

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

    def __init__(self, 
                 num_classes, 
                 in_channels, 
                 max_n_bbox_per_gt=30, 
                 cleanliness_alpha=0.90, 
                 reweight_gamma=1.0,
                 focal_loss_alpha=0.50,
                 focal_loss_gamma=2.0,
                 **kwargs):
        self.max_n_bbox_per_gt = max_n_bbox_per_gt
        self.cleanliness_alpha = cleanliness_alpha
        self.reweight_gamma = reweight_gamma
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma

        """
        In Detectron1, loss is normalized by number of foreground samples in the batch.
        When batch size is 1 per GPU, #foreground has a large variance and
        using it lead to lower performance. Here we maintain an EMA of #foreground to
        stabilize the normalizer.
        """
        self.loss_normalizer = 100  # initialize with any reasonable #fg that's not too small
        self.loss_normalizer_momentum = 0.9
        self.logits_clip = 5.0  # clip logits before applying sigmoid to prevent numerical issues

        # overide default assigner with TopNIoUAssigner
        kwargs['train_cfg']['assigner'] = dict(
                                        type='TopNIoUAssigner',
                                        max_n_bbox_per_gt=self.max_n_bbox_per_gt,
                                        min_pos_iou=0)
        # overide default bbox loss with SmoothL1Loss
        kwargs['loss_bbox'] = dict(
                            type='SmoothL1Loss', 
                            beta=0.1, 
                            loss_weight=1.0)

        super(NoisyAnchorHead, self).__init__(num_classes, in_channels, **kwargs)

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                bbox_targets_list (list[Tensor]): BBox targets of each level
                bbox_weights_list (list[Tensor]): BBox weights of each level
                num_total_pos (int): Number of positive samples in all images
                num_total_neg (int): Number of negative samples in all images
                anchor_ious (list[Tensor]): IOU between all anchors and to their assigned gt_bboxes
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]
        assign_result = self.assigner.assign(
            anchors, gt_bboxes, gt_bboxes_ignore,
            None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors, gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        anchor_ious = assign_result.max_overlaps
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
            anchor_ious = unmap(anchor_ious, num_total_anchors, inside_flags, fill=0)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result, anchor_ious)

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each
                  level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - num_total_pos (int): Number of positive samples in all
                  images.
                - num_total_neg (int): Number of negative samples in all
                  images.

            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]

        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        rest_results = list(results[7:])  # user-added return values

        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, num_total_pos, num_total_neg)
        if return_sampling_results:
            res = res + (sampling_results_list, )
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)

    def loss_single(self, cls_score, bbox_pred, anchors, anchor_ious, 
                    labels, label_weights, bbox_targets, bbox_weights):
        """Compute loss

        Args:
            cls_score (Tensor): Box scores for all scale levels
                Has shape (N * num_anchors * H * W, num_classes).
            bbox_pred (Tensor): Box energies / deltas for all scale levels 
                with shape (N * num_anchors * H * W, 4).
            anchors (Tensor): Box reference for all scale levels with shape
                (N * num_total_anchors, 4).
            anchor_ious (Tensor): IOU between anchors and their assigned gt bboxes 
                with shape (N * num_total_anchors)
            labels (Tensor): Labels of each anchors with shape
                (N * num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N * num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N * num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N * num_total_anchors, 4).

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        def calculate_soft_labels(
            cls_s_list: torch.Tensor, 
            loc_acc_list: torch.Tensor, 
            pos_inds: torch.Tensor, 
            alpha: float) -> torch.Tensor:
            """
            Generate soft labels
            Args:
                cls_s_list (Tensor): classification confidence of each anchor.
                loc_acc_list (Tensor): IoU of each anchor to its assigned gt (0 if negative).
                pos_inds (Tensor): indices of positive anchors.
                alpha (float): weight of localization loss [0, 1].
            Returns:
                Tensor: soft labels of each anchor
            """
            cls_s_pos = cls_s_list[pos_inds]
            loc_acc_pos = loc_acc_list[pos_inds]

            cls_s_pos_s = cls_s_pos / cls_s_pos.mean()
            loc_acc_pos_s = loc_acc_pos / loc_acc_pos.mean()
            soft_labels_pos = alpha * loc_acc_pos_s + (1 - alpha) * cls_s_pos_s

            # bouding soft_labels_pos back to range [C, 1] where C is loc_acc_pos.mean()
            scaling = (1.0 - loc_acc_pos.mean()) / (soft_labels_pos.max() - soft_labels_pos.min())
            soft_labels_pos = loc_acc_pos.mean() + scaling * (soft_labels_pos - soft_labels_pos.min())

            soft_labels = torch.zeros_like(cls_s_list)
            soft_labels[pos_inds] = soft_labels_pos

            return soft_labels

        def calculate_reweight_coeff(
            cls_s_list: torch.Tensor,
            loc_acc_list: torch.Tensor, 
            pos_inds: torch.Tensor, 
            alpha: float, 
            gamma: float) -> torch.Tensor:
            """
            Generate re-weighting coefficients
            Args:
                cls_s_list (Tensor): classification confidence of each anchor.
                loc_acc_list (Tensor): IoU of each anchor to its assigned gt (0 if negative).
                pos_inds (Tensor): indices of positive anchors.
                alpha (float): weight of localization loss [0, 1].
                gamma (float): reweight variance score
            Returns:
                Tensor: loss weights of each anchor
            """
            cls_s_pos = cls_s_list[pos_inds]
            loc_acc_pos = loc_acc_list[pos_inds]

            cls_s_pos_r = 1 / (1 - cls_s_pos) / (1 / (1 - cls_s_pos)).mean()
            loc_acc_pos_r = 1 / (1 - loc_acc_pos) / (1 / (1 - loc_acc_pos)).mean()
            reweight_coeff_pos = alpha * loc_acc_pos_r + (1 - alpha) * cls_s_pos_r
            reweight_coeff_pos = reweight_coeff_pos ** gamma

            reweight_coeff = torch.ones_like(cls_s_list)
            reweight_coeff[pos_inds] = reweight_coeff_pos

            return reweight_coeff

        def soft_sigmoid_focal_loss(
            inputs: torch.Tensor,
            gt_classes: torch.Tensor,
            soft_targets: torch.Tensor,
            foreground_idxs: torch.Tensor,
            alpha: float = 0.50,
            gamma: float = 2,
            num_classes: int = 80) -> torch.Tensor:
            """
            Focal Loss which accepts soft label as well, other than hard binary label. 
            """
            # prepare soft labels 
            sl = soft_targets.clone().unsqueeze(1).repeat(1, num_classes).fill_(0.0)
            sl[foreground_idxs, gt_classes[foreground_idxs].long()] = soft_targets[foreground_idxs]
            bg_st = 1.0 - sl

            # focal loss
            p = torch.sigmoid(inputs)
            term1 = (1 - p) ** gamma * torch.log(p)
            term2 = p ** gamma * torch.log(1 - p)

            loss = - sl * term1 * alpha - bg_st * term2 * (1 - alpha)
            return loss

        pos_inds = torch.nonzero(anchor_ious > 0).squeeze(1)
        cls_score = torch.clamp(cls_score, -self.logits_clip, self.logits_clip)  # clamp logits to avoid inf/nan
        cls_conf = torch.sigmoid(cls_score)
        label_inds = torch.clamp(labels, max=self.num_classes-1) # for bg class we use indices of the last class for now
        cls_label_conf = torch.gather(cls_conf, dim=1, index=label_inds.unsqueeze(1)).squeeze(1)

        # calculate soft labels and re-weighting coefficients
        soft_labels = calculate_soft_labels(
            cls_label_conf, 
            anchor_ious, 
            pos_inds, 
            self.cleanliness_alpha)
        reweight_coeff = calculate_reweight_coeff(
            cls_label_conf, 
            anchor_ious, 
            pos_inds, 
            self.cleanliness_alpha, 
            self.reweight_gamma)

        # classification loss
        loss_cls = soft_sigmoid_focal_loss(
            cls_score, 
            labels, 
            soft_labels, 
            pos_inds, 
            alpha=self.focal_loss_alpha, 
            gamma=self.focal_loss_gamma, 
            num_classes=self.cls_out_channels)

        # regression loss
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred[pos_inds],
            bbox_targets[pos_inds],
            reduction_override='none')

        # re-weight the losses
        loss_scale_cls = loss_cls.sum().data / (loss_cls * reweight_coeff[:, None]).sum().data
        loss_scale_bbox = loss_bbox.sum().data / (loss_bbox * reweight_coeff[pos_inds, None]).sum().data
        loss_cls *= reweight_coeff[:, None] * loss_scale_cls
        loss_bbox *= reweight_coeff[pos_inds, None] * loss_scale_bbox

        # normalize losses
        self.loss_normalizer = (
            self.loss_normalizer_momentum * self.loss_normalizer
            + (1 - self.loss_normalizer_momentum) * len(pos_inds)
        )
        loss_cls = loss_cls.sum()
        loss_bbox = loss_bbox.sum()
        loss_cls /= max(1, self.loss_normalizer)
        loss_bbox /= max(1, self.loss_normalizer)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, anchor_ious) = cls_reg_targets
        num_total_samples = (num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list, num_level_anchors)

        # concat all levels into a single tensor
        cls_scores = torch.cat(
            [cs.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels) for cs in cls_scores], dim=0)
        bbox_preds = torch.cat(
            [bbp.permute(0, 2, 3, 1).reshape(-1, 4) for bbp in bbox_preds], dim=0)
        all_anchor_list = torch.cat(
            [aa.reshape(-1, 4) for aa in all_anchor_list], dim=0)
        anchor_ious = torch.cat(
            [aa.reshape(-1) for aa in anchor_ious], dim=0)
        labels_list = torch.cat(
            [ll.reshape(-1) for ll in labels_list], dim=0)
        label_weights_list = torch.cat(
            [llw.reshape(-1) for llw in label_weights_list], dim=0)
        bbox_targets_list = torch.cat(
            [bbt.reshape(-1, 4) for bbt in bbox_targets_list], dim=0)
        bbox_weights_list = torch.cat(
            [bbw.reshape(-1, 4) for bbw in bbox_weights_list], dim=0)

        losses_cls, losses_bbox = self.loss_single(
            cls_scores,
            bbox_preds,
            all_anchor_list,
            anchor_ious,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
