import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Loss, filter_by_quantile


class QueryMotionMaskLoss(Loss):
    def __init__(self, loss_weight=1.0, valid_range=None):
        super().__init__(loss_weight=loss_weight)
        self.bce_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        self.valid_range = valid_range

    def forward(self, pred_mask, gt_mask, valid_mask=None, name=None, **kwargs):
        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(pred_mask), {}

        if torch.isnan(pred_mask).any() or torch.isinf(pred_mask).any():
            zero = 0 * torch.sum(torch.nan_to_num(pred_mask))
            return zero, {}

        loss = self.bce_loss_fn(pred_mask.float(), gt_mask.float())
        loss = torch.nan_to_num(loss)

        if valid_mask is not None:
            valid_mask = valid_mask.bool()
            if valid_mask.any():
                loss = loss[valid_mask.expand_as(loss)]
            else:
                zero = 0 * torch.sum(pred_mask)
                return zero, {}

        if self.valid_range is not None:
            loss = filter_by_quantile(loss, self.valid_range)

        loss = loss.mean() * loss_weight
        loss_dict = dict(bce_loss=loss)

        return loss, loss_dict

