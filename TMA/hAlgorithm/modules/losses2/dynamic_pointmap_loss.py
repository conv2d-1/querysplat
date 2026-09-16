import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Loss, filter_by_quantile


class DynamicPointmapLoss(nn.Module):
    """
    Dynamic Pointmap Loss - supervises absolute 3D positions in camera coordinates.
    
    This loss is designed to be consistent with GlobalPointZWeightedLoss (static pointmap),
    using the same L1 loss formulation with optional Z-weighting.
    
    Key Design:
    - Same loss formula as static pointmap: |pred_3d - gt_3d| with optional Z-weighting
    - GT positions are in SOURCE CAMERA frame (frame 0) after pipeline transformation
    - GT is ALREADY normalized by scale in get_inputs to match normalized extrinsics
    
    Coordinate Flow:
    1. Pipeline normalizes trajs_3d by scale in get_inputs (to match normalized extrinsics)
    2. Pipeline transforms trajs_3d from WORLD coords to CAMERA coords (frame 0) using normalized extrinsics
    3. This loss directly computes L1 loss between normalized GT and predictions (both in normalized scale)
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        offset: float = 0.0,              # Offset for Z-weighting (same as GlobalPointZWeightedLoss)
        threshold: float = 3.0,           # Threshold for outlier filtering
        zweighted: bool = True,           # Whether to use Z-weighting (divide by depth)
        eps: float = 1e-6,
        dynamic_threshold: float = 0.02,  # Motion threshold (meters) for dynamic/static classification
        balance_weight: float = 0.5,      # Weight for dynamic loss (for class balance)
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.offset = offset
        self.threshold = threshold
        self.zweighted = zweighted
        self.eps = eps
        self.dynamic_threshold = dynamic_threshold
        self.balance_weight = balance_weight

    def _sample_prediction(
        self,
        pred_points: torch.Tensor,  # [B*Pairs, 3, H_feat, W_feat]
        trajs_2d: torch.Tensor,     # [B*Pairs, N, 2] (x, y) - at original resolution
        original_h: int,
        original_w: int,
    ) -> torch.Tensor:
        """
        Sample predicted 3D positions at 2D trajectory locations using bilinear interpolation.
        
        trajs_2d is at original image resolution (original_h x original_w).
        pred_points is at model output resolution (H_feat x W_feat).
        
        We scale trajs_2d to pred_points resolution, then normalize to [-1, 1] for grid_sample.
        This ensures correct pixel-to-pixel correspondence regardless of resolution differences.
        """
        H_feat, W_feat = pred_points.shape[-2:]
        
        # Scale trajs_2d from original resolution to pred_points resolution
        scale_x = W_feat / original_w
        scale_y = H_feat / original_h
        
        trajs_2d_scaled = trajs_2d.clone()
        trajs_2d_scaled[..., 0] = trajs_2d[..., 0] * scale_x
        trajs_2d_scaled[..., 1] = trajs_2d[..., 1] * scale_y
        
        # Normalize to [-1, 1] using pred_points resolution
        trajs_2d_grid = trajs_2d_scaled.unsqueeze(2)
        grid = torch.zeros_like(trajs_2d_grid)
        grid[..., 0] = 2.0 * (trajs_2d_grid[..., 0] / max(W_feat - 1, 1.0)) - 1.0
        grid[..., 1] = 2.0 * (trajs_2d_grid[..., 1] / max(H_feat - 1, 1.0)) - 1.0

        sampled = F.grid_sample(
            pred_points, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )

        return sampled.squeeze(-1).permute(0, 2, 1)  # [B*Pairs, N, 3]

    def forward(
        self,
        pred_points: torch.Tensor,
        trajs_3d_tgt: torch.Tensor,
        trajs_2d_src: torch.Tensor,   # 确保这里是参考帧坐标
        valid_mask: torch.Tensor,
        original_h: int,
        original_w: int,
        trajs_3d_src: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        device = pred_points.device

        # 1. 获取维度
        B, Pairs, C, H_feat, W_feat = pred_points.shape
        N = trajs_3d_tgt.shape[2]
        batch_pairs = B * Pairs

        # 2. 统一展平 (全部使用 src 坐标进行采样和边界检查)
        flat_pred = pred_points.reshape(batch_pairs, C, H_feat, W_feat)
        flat_trajs_2d_src = trajs_2d_src.reshape(batch_pairs, N, 2) # 修改变量名
        flat_trajs_3d_tgt = trajs_3d_tgt.reshape(batch_pairs, N, 3)
        flat_valid = valid_mask.reshape(batch_pairs, N)

        # 2.1 修正：基于采样坐标 (src) 进行边界检查
        in_bounds = (
            (flat_trajs_2d_src[..., 0] >= 0)
            & (flat_trajs_2d_src[..., 0] < original_w)
            & (flat_trajs_2d_src[..., 1] >= 0)
            & (flat_trajs_2d_src[..., 1] < original_h)
        )
        flat_valid = flat_valid * in_bounds.float()

        # 3. 采样：使用 src 坐标作为“档案索引”
        pred_sampled = self._sample_prediction(
            flat_pred, 
            flat_trajs_2d_src, # 确保传入的是 src
            original_h, 
            original_w
        )

        # 4. 计算差异 (GT 已在 Pipeline 中归一化)
        gt_norm = flat_trajs_3d_tgt
        valid_bool = flat_valid > 0.5
        
        
        # Keep scale_expanded for logging/debugging purposes
        if scale is not None:
            scale_flat = scale.flatten()
            scale_len = scale_flat.shape[0]
            if scale_len == 1:
                scale_expanded = scale_flat.view(1, 1, 1).expand(batch_pairs, N, 1)
            elif scale_len == B:
                num_pairs = batch_pairs // B
                scale_expanded = (
                    scale_flat.view(B, 1, 1)
                    .expand(B, num_pairs * N, 1)
                    .reshape(batch_pairs, N, 1)
                )
            elif scale_len == batch_pairs:
                scale_expanded = scale_flat.view(batch_pairs, 1, 1).expand(batch_pairs, N, 1)
            else:
                scale_expanded = scale_flat.mean().view(1, 1, 1).expand(batch_pairs, N, 1)
        else:
            scale_expanded = torch.ones(batch_pairs, N, 1, device=device)

        # 5. Compute L1 difference (same as GlobalPointZWeightedLoss)
        valid_bool = flat_valid > 0.5
        
        diff_raw = torch.abs(pred_sampled - gt_norm)  # [B*Pairs, N, 3]
        diff_raw = diff_raw.sum(dim=-1)  # [B*Pairs, N] - L1 norm over (X, Y, Z)

        # Z-weighting (optional, same as GlobalPointZWeightedLoss)
        if self.zweighted:
            gt_z = gt_norm[..., 2].abs()  # [B*Pairs, N]
            diff = (diff_raw + self.offset) / (gt_z + self.offset + self.eps)
        else:
            diff = diff_raw

        # 6. Dynamic/Static classification (for class balancing)
        # NOTE: dynamic_threshold is in ABSOLUTE scale (meters), so we need to convert
        # normalized motion to absolute scale for threshold comparison.
        if trajs_3d_src is not None:
            flat_trajs_3d_src = trajs_3d_src.reshape(batch_pairs, N, 3)
            motion_norm = flat_trajs_3d_tgt - flat_trajs_3d_src  # normalized scale
            
            # Convert to absolute scale for threshold comparison
            if scale is not None:
                motion_abs = motion_norm * scale_expanded
            else:
                motion_abs = motion_norm
            
            motion_mag_abs = torch.norm(motion_abs, dim=-1)  # absolute scale magnitude
            is_dynamic = motion_mag_abs > self.dynamic_threshold
            mask_static = valid_bool & (~is_dynamic)
            mask_dynamic = valid_bool & is_dynamic
        else:
            mask_static = valid_bool
            mask_dynamic = torch.zeros_like(valid_bool)

        # 7. Compute loss with threshold filtering (same as GlobalPointZWeightedLoss)
        def compute_masked_loss(diff_tensor, mask):
            if mask.sum() == 0:
                return torch.tensor(0.0, device=device)
            masked_diff = diff_tensor[mask]
            if self.threshold is not None:
                masked_diff = masked_diff[masked_diff < self.threshold]
            if masked_diff.numel() == 0:
                return torch.tensor(0.0, device=device)
            return masked_diff.mean()

        loss_static = compute_masked_loss(diff, mask_static)
        loss_dynamic = compute_masked_loss(diff, mask_dynamic)

        # Combine with class balancing
        if mask_static.sum() > 0 and mask_dynamic.sum() > 0:
            loss = (1.0 - self.balance_weight) * loss_static + self.balance_weight * loss_dynamic
        else:
            loss = loss_static + loss_dynamic

        # Handle NaN/Inf
        if torch.isnan(loss).item() or torch.isinf(loss).item():
            loss = 0 * torch.sum(pred_points)
            logging.warning("DynamicPointmapLoss: NaN/Inf detected")

        final_loss = loss * self.loss_weight

        # 8. Statistics (report in ABSOLUTE scale for interpretability, consistent with any4d_loss)
        with torch.no_grad():
            stats = {
                "pointmap_loss_total": final_loss.item(),
                "pointmap_dyn_ratio": (mask_dynamic.sum().float() / (valid_bool.sum() + 1e-6)).item(),
            }
            if valid_bool.sum() > 0:
                # Convert to absolute scale for logging
                gt_abs = gt_norm * scale_expanded if scale is not None else gt_norm
                pred_abs = pred_sampled * scale_expanded if scale is not None else pred_sampled
                stats["pointmap_gt_z_mean"] = gt_abs[valid_bool][..., 2].mean().item()
                stats["pointmap_pred_z_mean"] = pred_abs[valid_bool][..., 2].mean().item()
                
                # Motion statistics (if trajs_3d_src provided)
                if trajs_3d_src is not None:
                    stats["pointmap_motion_mag_mean"] = motion_mag_abs[valid_bool].mean().item()

        return final_loss, stats



class GlobalPointZWeightedLoss(Loss):
    def __init__(
        self,
        loss_weight=1,
        offset=0,
        threshold=3.0,
        zweighted=True,
        with_conf=False,
        reg_loss_scale=1.0,
        conf_loss_scale=1.0,
        conf_expp1=True,
        conf_weight=0.1,
        valid_range=None,
        with_weight=False,
    ):
        super().__init__(loss_weight=loss_weight)

        self.offset = offset
        self.threshold = threshold
        self.zweighted = zweighted
        self.with_conf = with_conf
        self.with_weight = with_weight

        self.reg_loss_scale = reg_loss_scale
        self.conf_loss_scale = conf_loss_scale
        self.eps = 1e-6
        
        self.conf_expp1 = conf_expp1
        self.conf_weight = conf_weight
        
        self.valid_range = valid_range

    def forward(self, pred_depth, target_depth, valid_mask, pred_conf=None, name=None, weight=None, **kwargs):

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return {"l1_loss": 0 * torch.sum(pred_depth)}

        diff_raw = torch.abs(pred_depth[valid_mask] - target_depth[valid_mask])
        diff_raw = diff_raw.sum(dim=-1, keepdim=True)

        if self.zweighted:
            diff = (diff_raw + self.offset) / (target_depth[valid_mask][:, 2:3].abs() + self.offset + self.eps)
        else:
            diff = diff_raw

        point_weight = None
        if self.with_weight and weight is not None:
            point_weight = weight[valid_mask]
            if point_weight.ndim == 1:
                point_weight = point_weight.unsqueeze(-1)
            elif point_weight.ndim > 1 and point_weight.shape[-1] != 1:
                # Convert potential vector weights to per-point scalar weights.
                point_weight = point_weight.reshape(point_weight.shape[0], -1).mean(dim=-1, keepdim=True)
            point_weight = point_weight.clamp_min(0.0)

        valid_range = None
        if self.valid_range is not None:
            if isinstance(self.valid_range, dict):
                valid_range = self.valid_range.get(name, None)
            else:
                valid_range = self.valid_range

        threshold_mask = None
        if self.threshold is not None:
            if valid_range is not None:
                diff = filter_by_quantile(diff, valid_range)
            if isinstance(self.threshold, dict):
                threshold = self.threshold.get(name, self.threshold.get("default", 3.0))
                threshold_mask = diff < threshold
                diff = diff[threshold_mask]
            else:
                threshold_mask = diff < self.threshold
                diff = diff[threshold_mask]
        else:
            if valid_range is not None:
                diff = filter_by_quantile(diff, valid_range)

        if diff.numel() == 0:
            loss = 0 * torch.sum(pred_depth)
        elif point_weight is not None:
            selected_weight = point_weight
            if threshold_mask is not None:
                selected_weight = selected_weight[threshold_mask]
            if selected_weight.numel() == 0 or selected_weight.sum() <= self.eps:
                loss = diff.mean()
            else:
                loss = (diff * selected_weight).mean()
        else:
            loss = diff.mean()

        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(pred_depth)
            logging.warning(f"Data {name}, GlobalPointZLoss NAN error, {loss}")

            loss_dict = dict(l1_loss=loss, conf_loss=loss)
            return loss_dict

        if self.with_conf and pred_conf is not None:
            pred_conf = pred_conf[valid_mask]

            # Some datasets can produce a sample with no valid query points.
            # Calling mean() on that empty confidence tensor returns NaN.
            if pred_conf.numel() == 0 or diff_raw.numel() == 0:
                conf_loss = 0 * torch.sum(pred_depth)
                return dict(
                    l1_loss=loss * loss_weight * self.reg_loss_scale,
                    conf_loss=conf_loss * loss_weight * self.conf_loss_scale,
                )

            if self.conf_expp1:
                # Prevent exp overflow under fp16/bf16 mixed precision.
                pred_conf = 1 + torch.exp(
                    pred_conf.float().clamp(min=-20.0, max=20.0)
                )

            conf, log_conf = pred_conf, torch.log(pred_conf)
            conf_loss = diff_raw * conf - self.conf_weight * log_conf
            if valid_range is not None and conf_loss.numel() > 0:
                conf_loss = filter_by_quantile(conf_loss, valid_range)
            if conf_loss.numel() == 0:
                conf_loss = 0 * torch.sum(pred_depth)
            elif point_weight is not None:
                selected_weight = point_weight
                if selected_weight.numel() == 0 or selected_weight.sum() <= self.eps:
                    conf_loss = conf_loss.mean()
                else:
                    conf_loss = (conf_loss * selected_weight).mean()
            else:
                conf_loss = conf_loss.mean()

            if torch.isnan(conf_loss).item() | torch.isinf(conf_loss).item():
                conf_loss = 0 * torch.sum(pred_depth)
                logging.warning(f"Data {name}, GlobalPointZLoss NAN error, {conf_loss}")

            loss_dict = dict(
                l1_loss=loss * loss_weight * self.reg_loss_scale,
                conf_loss=conf_loss * loss_weight * self.conf_loss_scale,
            )

        else:
            loss_dict = dict(l1_loss=loss * loss_weight)

        return loss_dict
