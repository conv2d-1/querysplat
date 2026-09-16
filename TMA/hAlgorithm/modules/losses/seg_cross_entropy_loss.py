import torch
import torch.nn as nn


class SegLoss(nn.Module):
    def __init__(self, loss_weight=1):
        super(SegLoss, self).__init__()
        self.loss_weight = loss_weight
        self.eps = 1e-6
        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")
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

    def forward(self, pred, target, name=None, **kwargs):
        """
        计算Laplace损失。

        参数:
        - target (torch.Tensor): 真实值张量。 B,C,H,W
        - pred (torch.Tensor): 预测值张量。B,C,H,W

        返回:
        - loss (torch.Tensor): 计算得到的损失值。
        """
        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(pred)

        loss = 0
        # predict_sig = nn.functional.sigmoid(pred) # prob: [0,1]
        loss = self.loss_fn(pred, target.squeeze(1))

        return loss * loss_weight