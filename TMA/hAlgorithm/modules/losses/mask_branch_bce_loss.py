import torch
import torch.nn as nn


class MaskLoss(nn.Module):
    def __init__(self, loss_weight=1, threshold=0.5, iou_weight=1.0, bce_weight=0, min_union=20):
        super(MaskLoss, self).__init__()
        self.loss_weight = loss_weight
        self.eps = 1e-6
        self.threshold = threshold
        self.min_union = min_union
        self.bce_loss_fn = nn.BCELoss(reduction="mean")
        self.iou_weight = iou_weight
        self.bce_weight = bce_weight
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

        if self.iou_weight > 0:
            intersect = torch.sum(predict_mask[target])
            union = torch.sum(predict_mask[~target]) + torch.sum(target)
            if union >= self.min_union:
                loss_iou = 1.0 - intersect / union
                loss += self.iou_weight * loss_iou

        if self.bce_weight > 0:
            loss_bce = self.bce_loss_fn(predict_sig, target.float())
            loss += self.bce_weight * loss_bce

        return loss * loss_weight


if __name__ == "__main__":
    pred = torch.randn(2, 20, 20)
    target = torch.randn(2, 20, 20) > 0
    val = torch.randn(2, 20, 20) > 0.1

    loss_fn = MaskLoss()
    print(f"loss: {loss_fn(pred, target, val)}")
