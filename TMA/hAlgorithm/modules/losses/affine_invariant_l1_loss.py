import torch
import torch.nn as nn

from hAlgorithm.modules.utils.alignment import align_least_square_batch


def depth_to_disparity(depth, valid_mask, normalize=False):
    """
    将深度图转换为归一化的视差图。

    Args:
        depth (torch.Tensor): 真实深度图, shape [B, H, W]

    Returns:
        disparity_normalized (torch.Tensor): 归一化的视差图, shape [B, H, W]
    """
    # Step 1: 将深度转换为视差
    # 避免除零错误，忽略深度为 0 的像素
    valid_mask = valid_mask.float()
    disparity = torch.where(valid_mask > 0, 1.0 / depth, torch.tensor(0.0, device=depth.device))

    if normalize:
        # Step 2: 计算每个样本的视差最大值和最小值
        B, H, W = depth.shape
        disparity_flat = disparity.view(B, -1)
        disparity_max = torch.max(disparity_flat, dim=1, keepdim=True)[0]
        disparity_min = torch.min(disparity_flat, dim=1, keepdim=True)[0]

        # Step 3: 归一化视差
        # 避免分母为零的情况
        range_d = torch.where(
            disparity_max > disparity_min,
            disparity_max - disparity_min,
            torch.tensor(1e-6, device=depth.device),
        )
        disparity_normalized = (disparity - disparity_min.view(B, 1, 1)) / range_d.view(B, 1, 1)
        return disparity_normalized

    return disparity


class AffineInvariantL1Loss(nn.Module):
    def __init__(self, loss_weight=1, disp_space=True):
        super(AffineInvariantL1Loss, self).__init__()
        self.loss_weight = loss_weight
        self.eps = 1e-6
        self.disp_space = disp_space
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

    def forward(self, pred, gt, valid_mask, name=None, **kwargs):
        """
        Args:
            pred (torch.Tensor): 预测深度图, shape [B, H, W]
            gt (torch.Tensor): 真实深度图, shape [B, H, W]
            valid_mask (torch.Tensor): 有效掩码, shape [B, H, W], 值为 0 或 1
        Returns:
            loss (torch.Tensor): 仿射不变损失
        """

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(pred)

        B, H, W = pred.shape

        if self.disp_space:
            gt_disp = depth_to_disparity(gt, valid_mask)
            max_values, _ = gt_disp.view(gt_disp.size(0), -1).max(dim=1)
            gt_disp_norm = gt_disp / max_values.view(-1, 1, 1)
            # align pred to gt
            pred_align_disp = align_least_square_batch(pred, gt_disp_norm, valid_mask)
            valid_mask = valid_mask & (pred_align_disp > 0.001)

            pred_align = pred_align_disp
            gt = gt_disp_norm
        else:
            pred_align = align_least_square_batch(pred, gt, valid_mask)

        # Flatten the tensors to [B, HW]
        pred_flat = pred_align.view(B, -1)
        gt_flat = gt.view(B, -1)
        valid_mask_flat = valid_mask.view(B, -1)

        # # Scale and shift both prediction and ground truth
        # pred_scaled = self.scale_shift_norm(pred_flat)
        # gt_scaled = self.scale_shift_norm(gt_flat)

        # Compute absolute error
        abs_error = torch.abs(pred_flat - gt_flat)

        # Apply valid mask
        masked_abs_error = abs_error * valid_mask_flat

        # Compute mean over valid pixels
        num_valid_pixels = torch.sum(valid_mask_flat, dim=1)
        loss = torch.sum(masked_abs_error, dim=1) / (num_valid_pixels + self.eps)

        # Average over the batch
        loss = torch.mean(loss)

        return loss * loss_weight

    def scale_shift_norm(self, depth):
        # Compute median (t(d)) for each sample in the batch
        t_d = torch.median(depth, dim=1, keepdim=True)[0]

        # Compute scale factor (s(d)) for each sample in the batch
        s_d = torch.mean(torch.abs(depth - t_d), dim=1, keepdim=True)

        # Avoid division by zero
        s_d = torch.where(s_d > 0, s_d, torch.tensor(self.eps, device=s_d.device))

        # Scale and shift both prediction and ground truth
        depth_scaled = (depth - t_d) / s_d

        return depth_scaled
