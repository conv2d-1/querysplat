import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfInputLoss(nn.Module):
    def __init__(self, loss_weight=1, threshold=0.05):
        super(ConfInputLoss, self).__init__()
        self.loss_weight = loss_weight
        self.eps = 1e-6
        self.threshold = threshold
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

    def forward(self, input_confidence_pred, prompt_depth, depth_target, valid_mask, name=None):
        """
        input_confidence_pred: B,H,W,1
        prompt_depth: B,H,W,3
        depth_target: B,H,W,3
        valid_mask: B,H,W
        """
        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(input_confidence_pred)

        if input_confidence_pred.shape[3] == 1:
            input_confidence_pred = input_confidence_pred.squeeze(3).unsqueeze(1)  # for interpolate
        _, H_gt, W_gt, _ = depth_target.shape
        input_confidence_pred = F.interpolate(input_confidence_pred, (H_gt, W_gt)).squeeze(1)
        predict_sig = nn.functional.sigmoid(input_confidence_pred)  # B,H,W

        l1_diff = torch.abs(prompt_depth - depth_target).sum(dim=-1, keepdim=True).squeeze(3)
        gt_noise_postive = (l1_diff > self.threshold).float()  # B,H,W

        bce = F.binary_cross_entropy(predict_sig, gt_noise_postive, reduction="none")
        loss = bce[valid_mask].mean()
        return loss * loss_weight


class ConfInputBceIouLossV2(nn.Module):
    def __init__(
        self,
        loss_weight=1,
        threshold=0.05,
        bce_weight=1.0,
        iou_weight=0.0,
        norm_type="L1",
        **kwargs,
    ):
        super(ConfInputBceIouLossV2, self).__init__()
        self.loss_weight = loss_weight
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        self.eps = 1e-6
        self.threshold = threshold
        self.norm_type = norm_type
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

    def forward(self, input_confidence_pred, prompt_diffmap, valid_mask, name=None, **kwargs):
        """
        input_confidence_pred: B,1,H,W
        prompt_diffmap: B,H,W,3
        valid_mask: B,H,W
        """
        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(torch.nan_to_num(input_confidence_pred))

        if input_confidence_pred.shape[3] == 1:
            input_confidence_pred = input_confidence_pred.squeeze(3).unsqueeze(1)  # for interpolate
        _, H_gt, W_gt, _ = prompt_diffmap.shape
        input_confidence_pred = F.interpolate(input_confidence_pred, (H_gt, W_gt)).squeeze(
            1
        )  # B,H,W
        predict_sig = nn.functional.sigmoid(input_confidence_pred)  # B,H,W

        if self.norm_type in ["l1", "L1"]:
            diff = prompt_diffmap.sum(dim=-1, keepdim=True).squeeze(3)
            gt_noise_postive = (diff > self.threshold).float()  # B,H,W
        elif self.norm_type in ["l2", "L2"]:
            diff = torch.torch.pow(prompt_diffmap, 2)
            diff = diff.sum(dim=-1, keepdim=True).squeeze(3)
            gt_noise_postive = (diff > self.threshold * self.threshold).float()  # B,H,W
        else:
            raise NotImplementedError("norm_type in ConfInputLoss should be L1 or L2 !")

        gt_noise_negative = 1.0 - gt_noise_postive

        loss = 0.0
        if self.iou_weight > 0:
            predict_mask = predict_sig > 0.5
            target_mask = gt_noise_negative.bool()
            intersect = torch.sum(predict_mask[target_mask & valid_mask])
            union = torch.sum(predict_mask[~(target_mask & valid_mask)]) + torch.sum(
                target_mask & valid_mask
            )
            loss_iou = 1.0 - intersect / union
            loss += self.iou_weight * loss_iou

        if self.bce_weight > 0:
            # noise-free areas should have a 1.0 confidence
            bce = F.binary_cross_entropy(predict_sig, gt_noise_negative, reduction="none")
            loss += self.bce_weight * bce[valid_mask].mean()

        return loss * loss_weight


class ConfOutputBceIouLoss(ConfInputBceIouLossV2):
    def __init__(
        self,
        loss_weight=1,
        threshold=0.05,
        bce_weight=1,
        iou_weight=0,
        norm_type="L1",
        soft_weight=False,
        soft_threshold=None,
        **kwargs,
    ):
        super().__init__(loss_weight, threshold, bce_weight, iou_weight, norm_type, **kwargs)

        self.soft_weight = soft_weight
        self.soft_threshold = soft_threshold if soft_threshold is not None else threshold

    def denormalize(self, depth, scale, shift):
        if shift is not None:
            return depth * scale + shift
        else:
            return depth * scale

    def forward(
        self,
        confidence_pred,
        depth_pred,
        depth_target,
        scale,
        shift,
        valid_mask,
        name=None,
        **kwargs,
    ):
        with torch.no_grad():
            if self.soft_weight:
                soft_weight = min(
                    self.soft_threshold / (depth_target - depth_pred).abs().mean(), 1.0
                )
            else:
                soft_weight = 1.0
            target = self.denormalize(depth_target, scale, shift)
            pred = self.denormalize(depth_pred, scale, shift)
            prompt_diffmap = torch.abs(target - pred)
        loss = super().forward(confidence_pred, prompt_diffmap, valid_mask, name)
        return loss * soft_weight


# 示例使用
if __name__ == "__main__":
    # 创建损失函数实例
    criterion = ConfInputLoss()

    # 示例数据
    D_gt = torch.randn(3, 640, 480, 3)
    D_hat = D_gt + torch.randn(3, 640, 480, 3) / 5
    mask = torch.randn(3, 640, 480) > -0.5
    b = torch.randn(3, 272, 360, 1)
    # b = torch.nn.Sigmoid().forward(b)

    # 计算损失
    import time

    start = time.perf_counter()
    for i in range(100):
        loss = criterion(b, D_gt, D_hat, mask)
    end = time.perf_counter()
    elapsed = (end - start) / 100
    print(f"Loss: {loss.item()}, time:{elapsed}s.")
