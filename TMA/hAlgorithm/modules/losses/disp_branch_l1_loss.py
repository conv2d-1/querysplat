import torch
import torch.nn as nn


class DispL1Loss(nn.Module):
    def __init__(self, loss_weight=1, **kwargs):
        super(DispL1Loss, self).__init__()
        self.loss_weight = loss_weight
        self.eps = 1e-6
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

    def forward(self, disp_pred, disp_gt, valid_mask, name=None):
        """
        Args:
            disp_pred (torch.Tensor): 预测深度图, shape [B, H, W]
            disp_gt (torch.Tensor): 真实深度图, shape [B, H, W]
            valid_mask (torch.Tensor): 有效掩码, shape [B, H, W], 值为 0 或 1
        Returns:
            loss (torch.Tensor): 仿射不变损失
        """
        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(disp_pred)

        loss = torch.abs(disp_pred - disp_gt)[valid_mask]
        loss = loss.mean()

        return loss * loss_weight


class SeqDispL1Loss(DispL1Loss):
    def __init__(self, loss_gamma=0.9, **kwargs):
        super(SeqDispL1Loss, self).__init__(**kwargs)
        self.loss_gamma = loss_gamma

    def forward(self, seq_disp_pred, disp_gt, valid_mask, name=None):
        n_predictions = len(seq_disp_pred)

        seq_loss = 0
        for i in range(n_predictions):
            adjusted_loss_gamma = self.loss_gamma ** (15 / (n_predictions - 1))
            i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
            i_loss = super().forward(seq_disp_pred[i], disp_gt, valid_mask, name)
            seq_loss += i_loss * i_weight

        return seq_loss
