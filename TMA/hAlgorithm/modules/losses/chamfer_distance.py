import logging

# import numpy as np
import torch
import torch.nn as nn

# import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance


class ChamferDistance(nn.Module):
    def __init__(
        self,
        loss_weight=1.0,
        edge_mask=False,
        norm=2,
        debug=False,
        sample_nums=None,
        conf_thresh=None,
        **kwargs,
    ):
        super(ChamferDistance, self).__init__()

        self.loss_weight = loss_weight
        self.edge_mask = edge_mask
        self.norm = norm
        self.sample_nums = sample_nums
        self.conf_thresh = conf_thresh

        self.debug = debug

        if isinstance(self.loss_weight, dict):
            assert "default" in self.loss_weight

    def get_loss_weight(self, name=None):
        if name is not None and isinstance(self.loss_weight, dict) and name in self.loss_weight:
            loss_weight = self.loss_weight[name]
        elif isinstance(self.loss_weight, dict) and name not in self.loss_weight:
            loss_weight = self.loss_weight["default"]
        else:
            loss_weight = self.loss_weight
        return loss_weight

    def forward(self, pred, gt, mask, name=None, edge_mask=None, confidence=None, **kwargs):

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(torch.nan_to_num(pred))

        if self.edge_mask and edge_mask is not None:
            cur_mask = mask & edge_mask
        else:
            cur_mask = mask

        if self.conf_thresh is not None and confidence is not None:
            cur_mask = cur_mask & (confidence.squeeze(-1) < self.conf_thresh)

        if self.sample_nums is not None:
            valid_nums = cur_mask.reshape(cur_mask.shape[0], -1).sum(dim=-1).min()
            sample_nums = min(self.sample_nums, valid_nums)
            sample_index = torch.randperm(valid_nums)[:sample_nums]

            x = torch.stack([pred[bi, cur_mask[bi]][sample_index] for bi in range(pred.shape[0])])
            y = torch.stack([gt[bi, cur_mask[bi]][sample_index] for bi in range(gt.shape[0])])

            loss = chamfer_distance(
                x,
                y,
                norm=self.norm,
            )[0]
        else:
            loss = []
            for bi in range(pred.shape[0]):
                loss.append(
                    chamfer_distance(
                        pred[bi : bi + 1, cur_mask[bi]],
                        gt[bi : bi + 1, cur_mask[bi]],
                        norm=self.norm,
                    )[0]
                )
            loss = torch.stack(loss)

        loss = torch.nan_to_num(loss)
        loss = loss.mean()

        if torch.isinf(loss).item():
            loss = 0 * torch.sum(torch.nan_to_num(pred))
            logging.warning(f"Data {name}, ChamferDistance Inf, {loss}")

        return loss * loss_weight
