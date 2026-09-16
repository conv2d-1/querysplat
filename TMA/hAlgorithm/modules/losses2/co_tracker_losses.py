# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Loss

EPS = 1e-6


def reduce_masked_mean(input, mask, dim=None, keepdim=False):
    r"""Masked mean

    `reduce_masked_mean(x, mask)` computes the mean of a tensor :attr:`input`
    over a mask :attr:`mask`, returning

    .. math::
        \text{output} =
        \frac
        {\sum_{i=1}^N \text{input}_i \cdot \text{mask}_i}
        {\epsilon + \sum_{i=1}^N \text{mask}_i}

    where :math:`N` is the number of elements in :attr:`input` and
    :attr:`mask`, and :math:`\epsilon` is a small constant to avoid
    division by zero.

    `reduced_masked_mean(x, mask, dim)` computes the mean of a tensor
    :attr:`input` over a mask :attr:`mask` along a dimension :attr:`dim`.
    Optionally, the dimension can be kept in the output by setting
    :attr:`keepdim` to `True`. Tensor :attr:`mask` must be broadcastable to
    the same dimension as :attr:`input`.

    The interface is similar to `torch.mean()`.

    Args:
        inout (Tensor): input tensor.
        mask (Tensor): mask.
        dim (int, optional): Dimension to sum over. Defaults to None.
        keepdim (bool, optional): Keep the summed dimension. Defaults to False.

    Returns:
        Tensor: mean tensor.
    """

    mask = mask.expand_as(input)

    prod = input * mask

    if dim is None:
        numer = torch.sum(prod)
        denom = torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / (EPS + denom)
    return mean


def sequence_loss(
    flow_preds,
    flow_gt,
    valids,
    vis=None,
    gamma=0.8,
    add_huber_loss=False,
    loss_only_for_visible=False,
):
    """Loss function defined over sequence of flow predictions"""
    total_flow_loss = 0.0
    for j in range(len(flow_gt)):
        B, S, N, D = flow_gt[j].shape
        B, S2, N = valids[j].shape
        assert S == S2
        n_predictions = len(flow_preds[j])
        flow_loss = 0.0
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i - 1)
            flow_pred = flow_preds[j][i]
            if add_huber_loss:
                i_loss = huber_loss(flow_pred, flow_gt[j], delta=6.0)
            else:
                i_loss = (flow_pred - flow_gt[j]).abs()  # B, S, N, 2
            i_loss = torch.mean(i_loss, dim=3)  # B, S, N
            valid_ = valids[j].clone()
            if loss_only_for_visible:
                valid_ = valid_ * vis[j]
            flow_loss += i_weight * reduce_masked_mean(i_loss, valid_)
        flow_loss = flow_loss / n_predictions
        total_flow_loss += flow_loss
    return total_flow_loss / len(flow_gt)


def huber_loss(x, y, delta=1.0):
    """Calculate element-wise Huber loss between x and y"""
    diff = x - y
    abs_diff = diff.abs()
    flag = (abs_diff <= delta).float()
    return flag * 0.5 * diff**2 + (1 - flag) * delta * (abs_diff - 0.5 * delta)


def sequence_BCE_loss(vis_preds, vis_gts):
    total_bce_loss = 0.0
    for j in range(len(vis_preds)):
        n_predictions = len(vis_preds[j])
        bce_loss = 0.0
        for i in range(n_predictions):
            vis_loss = F.binary_cross_entropy(vis_preds[j][i], vis_gts[j])
            bce_loss += vis_loss
        bce_loss = bce_loss / n_predictions
        total_bce_loss += bce_loss
    return total_bce_loss / len(vis_preds)


def sequence_prob_loss(
    tracks: torch.Tensor,
    confidence: torch.Tensor,
    target_points: torch.Tensor,
    visibility: torch.Tensor,
    expected_dist_thresh: float = 12.0,
):
    """Loss for classifying if a point is within pixel threshold of its target."""
    # Points with an error larger than 12 pixels are likely to be useless; marking
    # them as occluded will actually improve Jaccard metrics and give
    # qualitatively better results.
    total_logprob_loss = 0.0
    for j in range(len(tracks)):
        n_predictions = len(tracks[j])
        logprob_loss = 0.0
        for i in range(n_predictions):
            err = torch.sum((tracks[j][i].detach() - target_points[j]) ** 2, dim=-1)
            valid = (err <= expected_dist_thresh**2).float()
            logprob = F.binary_cross_entropy(confidence[j][i], valid, reduction="none")
            logprob *= visibility[j]
            logprob = torch.mean(logprob, dim=[1, 2])
            logprob_loss += logprob
        logprob_loss = logprob_loss / n_predictions
        total_logprob_loss += logprob_loss
    return total_logprob_loss / len(tracks)


def masked_mean(data: torch.Tensor, mask: torch.Tensor | None, dim: List[int]):
    if mask is None:
        return data.mean(dim=dim, keepdim=True)
    mask = mask.float()
    mask_sum = torch.sum(mask, dim=dim, keepdim=True)
    mask_mean = torch.sum(data * mask, dim=dim, keepdim=True) / torch.clamp(mask_sum, min=1.0)
    return mask_mean


def masked_mean_var(data: torch.Tensor, mask: torch.Tensor, dim: List[int]):
    if mask is None:
        return data.mean(dim=dim, keepdim=True), data.var(dim=dim, keepdim=True)
    mask = mask.float()
    mask_sum = torch.sum(mask, dim=dim, keepdim=True)
    mask_mean = torch.sum(data * mask, dim=dim, keepdim=True) / torch.clamp(mask_sum, min=1.0)
    mask_var = torch.sum(mask * (data - mask_mean) ** 2, dim=dim, keepdim=True) / torch.clamp(mask_sum, min=1.0)
    return mask_mean.squeeze(dim), mask_var.squeeze(dim)


class TrackSequenceLoss(Loss):

    def __init__(
        self,
        gamma=0.8,
        add_huber_loss=False,
        loss_only_for_visible=False,
        loss_weight=0.05,
        pred_weight=1.0,
        vis_weight=1.0,
        conf_weight=1.0,
        vis_split_pose_and_neg=False,
    ):
        super().__init__(loss_weight=loss_weight)

        self.gamma = gamma
        self.add_huber_loss = add_huber_loss
        self.loss_only_for_visible = loss_only_for_visible
        self.loss_weight = loss_weight
        self.pred_weight = pred_weight
        self.vis_weight = vis_weight
        self.conf_weight = conf_weight
        self.vis_split_pose_and_neg = vis_split_pose_and_neg

    def check_nan_and_inf(self, pred):
        has_nan = torch.isnan(pred).any()
        has_inf = torch.isinf(pred).any()
        return has_nan or has_inf

    def forward(
        self,
        name,
        track_preds,
        vis_preds,
        conf_preds,
        track_gt,
        valids,
        vis,
        **kwargs,
    ):
        """Loss function defined over sequence of flow predictions"""

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(torch.nan_to_num(track_preds[0])), dict()

        if self.check_nan_and_inf(track_gt):
            return 0 * torch.sum(torch.nan_to_num(track_preds[0])), dict()

        if not isinstance(track_preds, (list, tuple)):
            track_preds = [track_preds]

        loss_dict = dict()

        n_predictions = len(track_preds)
        track_loss = 0.0
        for i in range(n_predictions):
            i_weight = self.gamma ** (n_predictions - i - 1)
            track_pred = track_preds[i]
            if self.add_huber_loss:
                i_loss = huber_loss(track_pred, track_gt, delta=6.0)
            else:
                i_loss = (track_pred - track_gt).abs()  # B, S, N, 2
            i_loss = torch.mean(i_loss, dim=3)  # B, S, N
            valid_ = valids.clone().unsqueeze(1)
            if self.loss_only_for_visible:
                valid_ = valid_ * vis
            track_loss += i_weight * torch.nan_to_num(reduce_masked_mean(i_loss, valid_))
        track_loss = track_loss / n_predictions

        err = torch.sum((track_preds[-1].detach() - track_gt) ** 2, dim=-1)
        valid = (err <= 12.0**2).float()

        conf_preds = torch.nan_to_num(conf_preds)  # NOTE
        conf_loss = F.binary_cross_entropy(conf_preds, valid, reduction="none")
        conf_loss = conf_loss * vis
        conf_loss = torch.mean(conf_loss, dim=[1, 2]).mean()

        if self.vis_split_pose_and_neg:
            vis_preds = torch.nan_to_num(vis_preds)  # NOTE
            vis_loss = F.binary_cross_entropy(vis_preds, vis, reduction="none")
            pos_vis_loss = (vis_loss[vis > 0]).mean()
            neg_vis_loss = (vis_loss[vis == 0]).mean()
            # vis_loss = (pos_vis_loss + neg_vis_loss) * 0.5

            loss_dict["tk_loss"] = track_loss * loss_weight * self.pred_weight
            loss_dict["tk_conf_loss"] = conf_loss * loss_weight * self.conf_weight
            loss_dict["tk_pvis_loss"] = pos_vis_loss * 0.5 * loss_weight * self.vis_weight
            loss_dict["tk_nvis_loss"] = neg_vis_loss * 0.5 * loss_weight * self.vis_weight
            loss = loss_dict["tk_loss"] + loss_dict["tk_conf_loss"] + loss_dict["tk_pvis_loss"] + loss_dict["tk_nvis_loss"]

        else:
            vis_preds = torch.nan_to_num(vis_preds)  # NOTE
            vis_loss = F.binary_cross_entropy(vis_preds, vis)

            loss_dict["tk_loss"] = track_loss * loss_weight * self.pred_weight
            loss_dict["tk_conf_loss"] = conf_loss * loss_weight * self.conf_weight
            loss_dict["tk_vis_loss"] = vis_loss * loss_weight * self.vis_weight

            loss = loss_dict["tk_loss"] + loss_dict["tk_conf_loss"] + loss_dict["tk_vis_loss"]

        return loss, loss_dict
