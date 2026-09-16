import torch
import torch.nn as nn
'''
Adapted from MOGE
'''

class MaskL2Loss(nn.Module):
    def __init__(self, loss_weight=1, threshold=0.5, reverse_gt=False):
        super(MaskL2Loss, self).__init__()
        self.loss_weight = loss_weight
        self.eps = 1e-6
        self.threshold = threshold
        self.reverse_gt = reverse_gt
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

    def forward(self, pred, target, filter_mask=None, name=None, **kwargs):
        """
        计算Laplace损失。

        参数:
        - target (torch.Tensor): 真实值张量。 B,1,H,W, bool
        - pred (torch.Tensor): 预测值张量。B,1,H,W, float
        - filter_mask (torch.Tensor): 与target和prediction相同维度的filter

        返回:
        - loss (torch.Tensor): 计算得到的损失值。
        """
        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(pred)

        loss = 0
        predict_sig = nn.functional.sigmoid(pred)
        predict_mask = predict_sig > self.threshold
        # target = torch.masked_fill(target, ~filter_mask, True) # fill in filter_mask
        if self.reverse_gt:
            loss = (1 - target.float()) * predict_mask.float().square() +  target.float() * (1 - predict_mask.float()).square()
        else:
            loss = target.float() * predict_mask.float().square() + (1 - target.float()) * (1 - predict_mask.float()).square()
        loss = loss.mean(dim=(-2, -1)).mean()

        return loss * loss_weight

