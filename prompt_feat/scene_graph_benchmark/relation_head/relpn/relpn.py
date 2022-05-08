# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
# Reimplemtened by Jianwei Yang (jianwyan@microsoft.com)
# Adapted from https://github.com/jwyang/graph-rcnn.pytorch (Jianwei Yang)

import torch
import torch.nn as nn
import torch.nn.functional as F
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from ..pair_matcher import PairMatcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.bounding_box_pair import BoxPairList
from ..balanced_positive_negative_pair_sampler import BalancedPositiveNegativePairSampler
from maskrcnn_benchmark.modeling.utils import cat
from .relationshipness import Relationshipness

class RelPN(nn.Module):
    def __init__(
        self,
        cfg,
        proposal_matcher,
        fg_bg_pair_sampler,
        box_coder,
        cls_agnostic_bbox_reg=False,
        use_matched_pairs_only=False,
        minimal_matched_pairs=0,
    ):
        super(RelPN, self).__init__()
        self.cfg = cfg
        self.proposal_pair_matcher = proposal_matcher
        self.fg_bg_pair_sampler = fg_bg_pair_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.use_matched_pairs_only = use_matched_pairs_only
        self.minimal_matched_pairs = minimal_matched_pairs
        self.relationshipness = Relationshipness(self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES, pos_encoding=True)

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        temp = []
        target_box_pairs = []
        for i in range(match_quality_matrix.shape[0]):
            for j in range(match_quality_matrix.shape[0]):
                match_i = match_quality_matrix[i].view(-1, 1)
                match_j = match_quality_matrix[j].view(1, -1)
                match_ij = ((match_i + match_j) / 2)
                # rmeove duplicate index
                match_ij = match_ij.view(-1) # [::match_quality_matrix.shape[1]] = 0
                # non_duplicate_idx = (torch.eye(match_ij.shape[0]).view(-1) == 0).nonzero(as_tuple=False).view(-1).to(match_ij.device)
                # match_ij = match_ij[non_duplicate_idx]
                temp.append(match_ij)
                boxi = target.bbox[i]; boxj = target.bbox[j]
                box_pair = torch.cat((boxi, boxj), 0)
                target_box_pairs.append(box_pair)

        match_pair_quality_matrix = torch.stack(temp, 0).view(len(temp), -1)
        target_box_pairs = torch.stack(target_box_pairs, 0)
        target_pair = BoxPairList(target_box_pairs, target.size, target.mode)
        target_pair.add_field("labels", target.get_field("pred_labels").view(-1))

        box_subj = proposal.bbox
        box_obj = proposal.bbox
        box_subj = box_subj.unsqueeze(1).repeat(1, box_subj.shape[0], 1)
        box_obj = box_obj.unsqueeze(0).repeat(box_obj.shape[0], 1, 1)
        proposal_box_pairs = torch.cat((box_subj.view(-1, 4), box_obj.view(-1, 4)), 1)

        idx_subj = torch.arange(box_subj.shape[0]).view(-1, 1, 1).repeat(1, box_obj.shape[0], 1).to(proposal.bbox.device)
        idx_obj = torch.arange(box_obj.shape[0]).view(1, -1, 1).repeat(box_subj.shape[0], 1, 1).to(proposal.bbox.device)
        proposal_idx_pairs = torch.cat((idx_subj.view(-1, 1), idx_obj.view(-1, 1)), 1)

        # non_duplicate_idx = (proposal_idx_pairs[:, 0] != proposal_idx_pairs[:, 1]).nonzero(as_tuple=False)
        # proposal_box_pairs = proposal_box_pairs[non_duplicate_idx.view(-1)]
        # proposal_idx_pairs = proposal_idx_pairs[non_duplicate_idx.view(-1)]

        proposal_pairs = BoxPairList(proposal_box_pairs, proposal.size, proposal.mode)
        proposal_pairs.add_field("idx_pairs", proposal_idx_pairs)

        # matched_idxs = self.proposal_matcher(match_quality_matrix)
        matched_idxs = self.proposal_pair_matcher(match_pair_quality_matrix)

        # Fast RCNN only need "labels" field for selecting the targets
        # target = target.copy_with_fields("pred_labels")
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds

        if self.use_matched_pairs_only and \
            (matched_idxs >= 0).sum() > self.minimal_matched_pairs:
            # filter all matched_idxs < 0
            proposal_pairs = proposal_pairs[matched_idxs >= 0]
            matched_idxs = matched_idxs[matched_idxs >= 0]

        matched_targets = target_pair[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets, proposal_pairs

    def prepare_targets(self, proposals, targets):
        labels = []
        proposal_pairs = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets, proposal_pairs_per_image = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )

            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            # regression_targets_per_image = self.box_coder.encode(
            #     matched_targets.bbox, proposals_per_image.bbox
            # )

            labels.append(labels_per_image)
            proposal_pairs.append(proposal_pairs_per_image)

            # regression_targets.append(regression_targets_per_image)

        return labels, proposal_pairs


    def _relpnsample_train(self, proposals, targets):
        """
        perform relpn based sampling during training
        """

        labels, proposal_pairs = self.prepare_targets(proposals, targets)
        proposal_pairs = list(proposal_pairs)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, proposal_pairs_per_image in zip(
            labels, proposal_pairs
        ):
            proposal_pairs_per_image.add_field("labels", labels_per_image)

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_pair_sampler(labels)

        losses = 0
        for img_idx, (proposals_per_image, pos_inds_img, neg_inds_img) in \
            enumerate(zip(proposals, sampled_pos_inds, sampled_neg_inds)):
            obj_logits = proposals_per_image.get_field('scores_all')
            obj_bboxes = proposals_per_image.bbox
            relness = self.relationshipness(obj_logits, obj_bboxes, proposals_per_image.size)
            
            relness_sorted, order = torch.sort(relness.view(-1), descending=True)

            img_sampled_inds = order[:self.cfg.MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE].view(-1)
            proposal_pairs_per_image = proposal_pairs[img_idx][img_sampled_inds]
            proposal_pairs[img_idx] = proposal_pairs_per_image

            losses += F.binary_cross_entropy(relness, (labels[img_idx] > 0).view(-1, 1).float())

        self._proposal_pairs = proposal_pairs

        return proposal_pairs, losses

    def _fullsample_test(self, proposals):
        """
        This method get all subject-object pairs, and return the proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
        """
        proposal_pairs = []
        for i, proposals_per_image in enumerate(proposals):
            box_subj = proposals_per_image.bbox
            box_obj = proposals_per_image.bbox

            box_subj = box_subj.unsqueeze(1).repeat(1, box_subj.shape[0], 1)
            box_obj = box_obj.unsqueeze(0).repeat(box_obj.shape[0], 1, 1)
            proposal_box_pairs = torch.cat((box_subj.view(-1, 4), box_obj.view(-1, 4)), 1)

            idx_subj = torch.arange(box_subj.shape[0]).view(-1, 1, 1).repeat(1, box_obj.shape[0], 1).to(proposals_per_image.bbox.device)
            idx_obj = torch.arange(box_obj.shape[0]).view(1, -1, 1).repeat(box_subj.shape[0], 1, 1).to(proposals_per_image.bbox.device)
            proposal_idx_pairs = torch.cat((idx_subj.view(-1, 1), idx_obj.view(-1, 1)), 1)

            keep_idx = (proposal_idx_pairs[:, 0] != proposal_idx_pairs[:, 1]).nonzero(as_tuple=False).view(-1)

            # if we filter non overlap bounding boxes
            if self.cfg.MODEL.ROI_RELATION_HEAD.FILTER_NON_OVERLAP:
                ious = boxlist_iou(proposals_per_image, proposals_per_image).view(-1)
                ious = ious[keep_idx]
                keep_idx = keep_idx[(ious > 0).nonzero(as_tuple=False).view(-1)]
            proposal_idx_pairs = proposal_idx_pairs[keep_idx]
            proposal_box_pairs = proposal_box_pairs[keep_idx]
            proposal_pairs_per_image = BoxPairList(proposal_box_pairs, proposals_per_image.size, proposals_per_image.mode)
            proposal_pairs_per_image.add_field("idx_pairs", proposal_idx_pairs)

            proposal_pairs.append(proposal_pairs_per_image)
        return proposal_pairs

    def _relpnsample_test(self, proposals):
        """
        perform relpn based sampling during testing
        """
        proposals[0] = proposals[0]
        proposal_pairs = self._fullsample_test(proposals)
        proposal_pairs = list(proposal_pairs)

        relnesses = []
        for img_idx, proposals_per_image in enumerate(proposals):
            obj_logits = proposals_per_image.get_field('scores_all')
            obj_bboxes = proposals_per_image.bbox
            relness = self.relationshipness(obj_logits, obj_bboxes, proposals_per_image.size)
            keep_idx = (1 - torch.eye(obj_logits.shape[0]).to(relness.device)).view(-1).nonzero(as_tuple=False).view(-1)
            if self.cfg.MODEL.ROI_RELATION_HEAD.FILTER_NON_OVERLAP:
                ious = boxlist_iou(proposals_per_image, proposals_per_image).view(-1)
                ious = ious[keep_idx]
                keep_idx = keep_idx[(ious > 0).nonzero(as_tuple=False).view(-1)]
            relness = relness.view(-1)[keep_idx]
            relness_sorted, order = torch.sort(relness.view(-1), descending=True)

            img_sampled_inds = order[:self.cfg.MODEL.ROI_RELATION_HEAD.POST_RELPN_PREPOSALS].view(-1)
            relness = relness_sorted[:self.cfg.MODEL.ROI_RELATION_HEAD.POST_RELPN_PREPOSALS].view(-1)

            proposal_pairs_per_image = proposal_pairs[img_idx][img_sampled_inds]
            proposal_pairs[img_idx] = proposal_pairs_per_image
            relnesses.append(relness)

        self._proposal_pairs = proposal_pairs

        return proposal_pairs

    def forward(self, proposals, targets=None):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """
        if self.training:
            return self._relpnsample_train(proposals, targets)
        else:
            return self._relpnsample_test(proposals)

    def pred_classification_loss(self, class_logits, freq_prior=None):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])

        Returns:
            classification_loss (Tensor)
        """
        class_logits = cat(class_logits, dim=0)
        device = class_logits.device

        if not hasattr(self, "_proposal_pairs"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposal_pairs
        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)

        rel_fg_cnt = len(labels.nonzero(as_tuple=False))
        rel_bg_cnt = labels.shape[0] - rel_fg_cnt
        ce_weights = labels.new(class_logits.size(1)).fill_(1).float()
        ce_weights[0] = float(rel_fg_cnt) / (rel_bg_cnt + 1e-5)
        classification_loss = F.cross_entropy(class_logits, labels, weight=ce_weights)
        return classification_loss


def make_relation_proposal_network(cfg):
    matcher = PairMatcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativePairSampler(
        cfg.MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE,
        cfg.MODEL.ROI_RELATION_HEAD.POSITIVE_FRACTION
    )

    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    relpn = RelPN(
        cfg,
        matcher,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg
    )
    return relpn
