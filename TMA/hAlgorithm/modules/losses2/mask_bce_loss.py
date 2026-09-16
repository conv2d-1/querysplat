import torch
import torch.nn as nn

from .base import Loss, filter_by_quantile


class MaskLoss(Loss):
    def __init__(self, loss_weight=1, valid_range=None):
        super(MaskLoss, self).__init__(loss_weight=loss_weight)

        self.eps = 1e-6
        self.bce_loss_fn = nn.BCELoss(reduction="mean")

        self.valid_range = valid_range

    def forward(self, pred_invalid_mask, target_invalid_mask, name=None, **kwargs):
        """
        计算Laplace损失。

        参数:
        - target (torch.Tensor): 真实值张量。 B,1,H,W, bool
        - pred_invalid_mask (torch.Tensor): 预测值张量。B,1,H,W, float
        - filter_mask (torch.Tensor): 与target和prediction相同维度的filter

        返回:
        - loss (torch.Tensor): 计算得到的损失值。
        """
        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(pred_invalid_mask)

        if torch.isnan(pred_invalid_mask).sum() > 0 or torch.isinf(pred_invalid_mask).sum() > 0 :
            return 0 * torch.sum(torch.nan_to_num(pred_invalid_mask))

        predict_sig = nn.functional.sigmoid(pred_invalid_mask)
        loss = self.bce_loss_fn(predict_sig, target_invalid_mask.float())
        loss = torch.nan_to_num(loss)

        return loss * loss_weight

class MaskLossV2(Loss):
    def __init__(self, loss_weight=1, valid_range=None):
        super(MaskLossV2, self).__init__(loss_weight=loss_weight)

        self.eps = 1e-6
        self.bce_loss_fn = nn.BCELoss(reduction="none")

        self.valid_range = valid_range

    def forward(self, pred_invalid_mask, target_invalid_mask, name=None, **kwargs):
        """
        计算Laplace损失。

        参数:
        - target (torch.Tensor): 真实值张量。 B,1,H,W, bool
        - pred_invalid_mask (torch.Tensor): 预测值张量。B,1,H,W, float
        - filter_mask (torch.Tensor): 与target和prediction相同维度的filter

        返回:
        - loss (torch.Tensor): 计算得到的损失值。
        """
        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(pred_invalid_mask)

        if torch.isnan(pred_invalid_mask).sum() > 0 or torch.isinf(pred_invalid_mask).sum() > 0 :
            return 0 * torch.sum(torch.nan_to_num(pred_invalid_mask))

        predict_sig = nn.functional.sigmoid(pred_invalid_mask)
        loss = self.bce_loss_fn(predict_sig, target_invalid_mask.float())
        loss = torch.nan_to_num(loss)

        if self.valid_range is not None:
            loss = filter_by_quantile(loss, self.valid_range)
        loss = loss.mean()
        return loss * loss_weight
