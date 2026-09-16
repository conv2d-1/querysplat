import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

class Any4DSceneFlowLoss(nn.Module):
    """
    Robust Any4D Scene Flow Loss with Class Balancing.
    
    IMPORTANT: This loss computes in RELATIVE (NORMALIZED) SCALE for balanced supervision.
    - pred_flow: Expected in normalized scale from model output (no denormalization needed)
    - trajs_3d_*: Expected in NORMALIZED scale from get_inputs (already normalized by scale)
    - Loss computation: Both in normalized scale - NO further normalization needed
    
    Rationale: Computing loss in relative scale ensures:
    1. Small and large motions contribute equally to the loss
    2. Network learns motion patterns, not absolute magnitudes
    3. Training is more stable across scenes with different scales
    """
    def __init__(
        self, 
        dynamic_threshold: float = 0.02,  # 动态阈值 (Absolute Scale, in meters) - used to classify motion magnitude
        eps: float = 1e-6,
        loss_weight: float = 1.0,
        balance_weight: float = 0.5,      # 动态/静态 Loss 平衡因子
    ):
        super().__init__()
        self.dynamic_threshold = dynamic_threshold
        self.eps = eps
        self.loss_weight = loss_weight
        self.balance_weight = balance_weight

    def any4d_log_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Log transformation: f(x) = (x / ||x||) * log(1 + ||x||)
        
        Args:
            x: [..., 3] Motion vectors
            
        Returns:
            [..., 3] Log-transformed motion
        """
        norm = torch.norm(x, dim=-1, keepdim=True)
        # f(x) = x * (log(1+n) / n). Limit is 1 when n->0.
        scale_factor = torch.log(1 + norm) / (norm + self.eps)
        return x * scale_factor

    def _sample_prediction(
        self, 
        pred_flow: torch.Tensor, 
        trajs_2d: torch.Tensor, 
        H: int, 
        W: int
    ) -> torch.Tensor:
        """
        Sample prediction at 2D track locations using bilinear interpolation.
        
        Args:
            pred_flow: [B*Pairs, 3, H_feat, W_feat] Dense prediction
            trajs_2d:  [B*Pairs, N, 2] Sparse 2D coordinates (x, y)
            H, W:      Original image dimensions (for normalization)
            
        Returns:
            sampled_flow: [B*Pairs, N, 3]
        """
        # [Fix] 1. 调整形状以支持广播和 Grid Sample 要求
        # trajs_2d: [B*Pairs, N, 2] -> [B*Pairs, N, 1, 2]
        trajs_2d_grid = trajs_2d.unsqueeze(2)
        
        # 2. 创建 Grid 并归一化到 [-1, 1]
        grid = torch.zeros_like(trajs_2d_grid)
        
        # grid[..., 0] shape: [B*Pairs, N, 1]
        # trajs_2d_grid[..., 0] shape: [B*Pairs, N, 1]
        grid[..., 0] = 2.0 * (trajs_2d_grid[..., 0] / max(W - 1, 1.0)) - 1.0
        grid[..., 1] = 2.0 * (trajs_2d_grid[..., 1] / max(H - 1, 1.0)) - 1.0
        
        # 3. Grid Sample
        # Output: [B*Pairs, 3, N, 1]
        sampled = F.grid_sample(
            pred_flow, 
            grid, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=True
        )
        
        # 4. 调整回 [B*Pairs, N, 3]
        return sampled.squeeze(-1).permute(0, 2, 1)

    def forward(
        self, 
        pred_flow: torch.Tensor,          # [B, Pairs, 3, H_feat, W_feat] (normalized scale)
        trajs_3d_src: torch.Tensor,       # [B, Pairs, N, 3] (normalized scale, camera frame)
        trajs_3d_tgt: torch.Tensor,       # [B, Pairs, N, 3] (normalized scale, camera frame)
        trajs_2d_src: torch.Tensor,       # [B, Pairs, N, 2] (pixel coordinates)
        valid_mask: torch.Tensor,         # [B, Pairs, N] (boolean mask)
        original_h: int,                  # Original image height
        original_w: int,                  # Original image width
        scale: Optional[torch.Tensor] = None,  # [B] or [B*Pairs] - scene scale (for logging/threshold only)
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Any4D scene flow loss in RELATIVE (NORMALIZED) SCALE.
        
        Loss Computation Strategy:
        1. pred_flow: Already in normalized scale (model output)
        2. trajs_3d_*: Already in normalized scale (from get_inputs)
        3. Compute loss directly in normalized space - NO further normalization needed
        
        Mathematical Formulation:
            pred_motion_norm = pred_flow (already normalized)
            gt_motion_norm = trajs_3d_tgt - trajs_3d_src (already normalized)
            
            loss = |log_transform(pred_motion_norm) - log_transform(gt_motion_norm)|
        
        Args:
            pred_flow: Predicted scene flow in NORMALIZED scale
            trajs_3d_src: Source GT 3D positions in NORMALIZED scale (camera frame)
            trajs_3d_tgt: Target GT 3D positions in NORMALIZED scale (camera frame)
            trajs_2d_src: Source 2D pixel coordinates for sampling pred_flow
            valid_mask: Validity mask for trajectories
            original_h: Original image height
            original_w: Original image width
            scale: Scene scale (for converting to absolute scale for threshold/logging only)
            **kwargs: Additional unused parameters
        
        Returns:
            loss: Scalar loss value
            stats: Dictionary with statistics (motion magnitudes in ABSOLUTE scale for logging)
        """
        
        device = pred_flow.device
        
        # 1. 获取基础维度
        B, Pairs, C, H_feat, W_feat = pred_flow.shape
        N = trajs_3d_src.shape[2]
        batch_pairs = B * Pairs
        
        # 2. 展平数据 [B, Pairs, ...] -> [B*Pairs, ...]
        flat_pred_flow = pred_flow.reshape(batch_pairs, C, H_feat, W_feat)
        flat_trajs_2d = trajs_2d_src.reshape(batch_pairs, N, 2)
        flat_trajs_3d_src = trajs_3d_src.reshape(batch_pairs, N, 3)
        flat_trajs_3d_tgt = trajs_3d_tgt.reshape(batch_pairs, N, 3)
        flat_valid = valid_mask.reshape(batch_pairs, N)
        
        # 2.1 Compute in-bounds mask for trajs_2d_src
        # Points outside image bounds should NOT participate in supervision
        # because F.grid_sample with padding_mode='zeros' returns 0 for OOB points,
        # leading to incorrect supervision if GT motion is non-zero.
        in_bounds = (
            (flat_trajs_2d[..., 0] >= 0) & (flat_trajs_2d[..., 0] < original_w) &
            (flat_trajs_2d[..., 1] >= 0) & (flat_trajs_2d[..., 1] < original_h)
        )  # [B*Pairs, N]
        flat_valid = flat_valid * in_bounds.float()  # Exclude OOB points from supervision

        # 3. 采样预测值 (pred_flow is in normalized scale)
        pred_motion = self._sample_prediction(
            flat_pred_flow, flat_trajs_2d, original_h, original_w
        )  # [B*Pairs, N, 3] - normalized scale

        # 4. 计算 GT motion (already in normalized scale from get_inputs)
        gt_motion_norm = flat_trajs_3d_tgt - flat_trajs_3d_src  # [B*Pairs, N, 3] - normalized scale
        
        # 5. Convert to absolute scale for dynamic/static classification threshold
        # The threshold should be in absolute scale (e.g., 0.02m) for semantic consistency
        if scale is not None:
            # Handle different scale shapes: [B], [B, 1, 1, 1], etc.
            scale_flat = scale.flatten()  # [B] or [B*Pairs]
            scale_len = scale_flat.shape[0]
            
            if scale_len == 1:
                # Single scale value, expand to all batch_pairs
                scale_expanded = scale_flat.view(1, 1).expand(batch_pairs, 1)
            elif scale_len == B:
                # Scale per batch, expand to pairs
                # Pairs is computed as num_pairs from src_flat/tgt_flat
                num_pairs = batch_pairs // B
                scale_expanded = scale_flat.view(B, 1).expand(B, num_pairs).reshape(batch_pairs, 1)
            elif scale_len == batch_pairs:
                # Already expanded
                scale_expanded = scale_flat.view(batch_pairs, 1)
            else:
                # Fallback: use mean
                scale_expanded = scale_flat.mean().view(1, 1).expand(batch_pairs, 1)
            
            # Convert to absolute scale only for threshold comparison: [B*Pairs, N, 3]
            gt_motion_abs = gt_motion_norm * scale_expanded.unsqueeze(1)
        else:
            # No scale provided, use normalized motion for threshold (less ideal)
            gt_motion_abs = gt_motion_norm
        
        # 6. 动态/静态 分割 (use ABSOLUTE scale for threshold to maintain semantic meaning)
        # Dynamic threshold should be in absolute scale (e.g., 0.02m) for semantic consistency
        motion_mag_abs = torch.norm(gt_motion_abs, dim=-1)  # [B*P, N] - absolute magnitude
        is_dynamic = motion_mag_abs > self.dynamic_threshold
        
        valid_bool = flat_valid > 0.5
        mask_static = valid_bool & (~is_dynamic)
        mask_dynamic = valid_bool & is_dynamic
        
        # 7. Log 变换 & Loss 计算 (both in NORMALIZED scale)
        pred_log = self.any4d_log_transform(pred_motion)      # normalized scale
        gt_log = self.any4d_log_transform(gt_motion_norm)     # normalized scale
        
        diff = torch.abs(pred_log - gt_log).sum(dim=-1)
        
        # Static Loss
        if mask_static.sum() > 0:
            loss_static = diff[mask_static].mean()
        else:
            loss_static = torch.tensor(0.0, device=device)
            
        # Dynamic Loss
        if mask_dynamic.sum() > 0:
            loss_dynamic = diff[mask_dynamic].mean()
        else:
            loss_dynamic = torch.tensor(0.0, device=device)
            
        # 混合 Loss
        if mask_static.sum() > 0 and mask_dynamic.sum() > 0:
            loss = (1.0 - self.balance_weight) * loss_static + self.balance_weight * loss_dynamic
        else:
            loss = loss_static + loss_dynamic

        final_loss = loss * self.loss_weight

        # 8. 统计信息 (report in ABSOLUTE scale for interpretability)
        with torch.no_grad():
            # Convert pred_motion to absolute scale for logging
            if scale is not None:
                pred_motion_abs = pred_motion * scale_expanded.unsqueeze(1)
            else:
                pred_motion_abs = pred_motion
            
            n_valid = valid_bool.sum()
            gt_mag_valid = motion_mag_abs[valid_bool]
            pred_mag_valid = torch.norm(pred_motion_abs, dim=-1)[valid_bool]
            stats = {
                "loss_motion_total": final_loss.item(),
                "motion_dyn_ratio": (mask_dynamic.sum().float() / (n_valid + 1e-6)).item(),
                # Guard against empty or NaN-contaminated valid sets (e.g. when a
                # batch has no visible tracks or the scene scale resolved to NaN).
                "motion_gt_mag_mean": gt_mag_valid[torch.isfinite(gt_mag_valid)].mean().item()
                    if (n_valid > 0 and torch.isfinite(gt_mag_valid).any()) else 0.0,
                "motion_pred_mag_mean": pred_mag_valid[torch.isfinite(pred_mag_valid)].mean().item()
                    if (n_valid > 0 and torch.isfinite(pred_mag_valid).any()) else 0.0,
            }

        return final_loss, stats