"""
D4RT Auxiliary Losses
=====================
Auxiliary losses inspired by D4RT (arXiv:2512.08924) to regularize the
query-based sparse motion decoder.

1. DepthSelfConsistencyLoss:
   The Z-component of decoder(u, v, t, t, t) should match the dense depth
   prediction at pixel (u, v) in frame t.

2. ReprojectionLoss:
   The predicted 3D point, projected to 2D via camera intrinsics, should
   land at the correct pixel location.

3. CycleConsistencyLoss:
   Tracking forward A->B then backward B->A should return to the starting
   3D position.
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Depth Self-Consistency Loss
# =============================================================================

class DepthSelfConsistencyLoss(nn.Module):
    """Enforce decoder depth == dense depth head depth for identity queries.

    A "depth query" is ``(u, v, t, t, t)`` — the point should be at its
    current position in its own camera frame.  The Z-component of the
    decoder output must match the dense depth prediction at that pixel.

    Args:
        loss_weight: scalar multiplier applied to the final loss.
        eps: small constant for numerical stability.
    """

    def __init__(self, loss_weight: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.loss_weight = loss_weight
        self.eps = eps

    def forward(
        self,
        pred_z_from_query: torch.Tensor,   # [B, N] Z-component from decoder
        pred_z_from_dense: torch.Tensor,   # [B, N] sampled from dense depth map
        valid_mask: Optional[torch.Tensor] = None,  # [B, N] bool
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with key ``depth_consist_loss``.
        """
        diff = (pred_z_from_query - pred_z_from_dense).abs()

        if valid_mask is not None:
            valid = valid_mask.bool()
            if valid.sum() == 0:
                zero = torch.tensor(0.0, device=diff.device, requires_grad=True)
                return dict(depth_consist_loss=zero)
            loss = diff[valid].mean()
        else:
            loss = diff.mean()

        return dict(depth_consist_loss=loss * self.loss_weight)


# =============================================================================
# 2. Reprojection Loss
# =============================================================================

class ReprojectionLoss(nn.Module):
    """Enforce 3D prediction re-projects to the correct 2D location.

    For **static** queries ``(u, v, t, t, t)`` the re-projection target is
    ``(u, v)`` itself.  For **tracking** queries the target is the GT 2D
    trajectory coordinate at ``t_tgt``.

    Args:
        loss_weight: scalar multiplier.
        normalize_by_image: if True, normalize pixel error by image diagonal.
        eps: clamping floor for predicted depth to avoid division by zero.
    """

    def __init__(
        self,
        loss_weight: float = 0.5,
        normalize_by_image: bool = True,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.normalize_by_image = normalize_by_image
        self.eps = eps

    def forward(
        self,
        pred_3d: torch.Tensor,              # [B, N, 3] in camera coordinates of t_cam
        intrinsics: torch.Tensor,            # [B, 3, 3] intrinsics of t_cam
        gt_2d: torch.Tensor,                 # [B, N, 2] target pixel coords
        valid_mask: Optional[torch.Tensor] = None,  # [B, N] bool
        image_size: Optional[tuple] = None,  # (H, W) for normalisation
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with key ``reproj_loss``.
        """
        # Safety: Check for NaN/Inf in inputs
        if torch.isnan(pred_3d).any() or torch.isinf(pred_3d).any():
            logger.warning(f"[ReprojectionLoss] pred_3d contains NaN/Inf: nan={torch.isnan(pred_3d).sum()}, inf={torch.isinf(pred_3d).sum()}")
            zero = torch.tensor(0.0, device=pred_3d.device, requires_grad=True)
            return dict(reproj_loss=zero)
        
        if torch.isnan(gt_2d).any() or torch.isinf(gt_2d).any():
            logger.warning(f"[ReprojectionLoss] gt_2d contains NaN/Inf")
            zero = torch.tensor(0.0, device=pred_3d.device, requires_grad=True)
            return dict(reproj_loss=zero)
        
        # Perspective projection: uv = K[:2,:2] @ (xy / z) + K[:2, 2]
        # Clamp depth to reasonable range to prevent division issues
        z = pred_3d[..., 2:3].clamp(min=self.eps, max=100.0)  # [B, N, 1] - added max clamp
        xy_over_z = pred_3d[..., :2] / z                       # [B, N, 2]
        
        # Check for NaN/Inf after division
        if torch.isnan(xy_over_z).any() or torch.isinf(xy_over_z).any():
            logger.warning(f"[ReprojectionLoss] xy_over_z contains NaN/Inf after division")
            zero = torch.tensor(0.0, device=pred_3d.device, requires_grad=True)
            return dict(reproj_loss=zero)

        fx = intrinsics[:, 0, 0].unsqueeze(1)  # [B, 1]
        fy = intrinsics[:, 1, 1].unsqueeze(1)
        cx = intrinsics[:, 0, 2].unsqueeze(1)
        cy = intrinsics[:, 1, 2].unsqueeze(1)

        uv_pred_x = xy_over_z[..., 0] * fx + cx  # [B, N]
        uv_pred_y = xy_over_z[..., 1] * fy + cy
        uv_pred = torch.stack([uv_pred_x, uv_pred_y], dim=-1)  # [B, N, 2]
        
        # Check projected coordinates
        if torch.isnan(uv_pred).any() or torch.isinf(uv_pred).any():
            logger.warning(f"[ReprojectionLoss] uv_pred contains NaN/Inf after projection")
            zero = torch.tensor(0.0, device=pred_3d.device, requires_grad=True)
            return dict(reproj_loss=zero)

        diff = (uv_pred - gt_2d).abs().sum(dim=-1)  # [B, N]

        # Optional normalisation by image diagonal
        if self.normalize_by_image and image_size is not None:
            H, W = image_size
            diag = (H ** 2 + W ** 2) ** 0.5
            diff = diff / max(diag, 1.0)

        if valid_mask is not None:
            valid = valid_mask.bool()
            if valid.sum() == 0:
                zero = torch.tensor(0.0, device=diff.device, requires_grad=True)
                return dict(reproj_loss=zero)
            loss = diff[valid].mean()
        else:
            loss = diff.mean()
        
        # Final NaN check
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"[ReprojectionLoss] Final loss is NaN/Inf: {loss}")
            zero = torch.tensor(0.0, device=pred_3d.device, requires_grad=True)
            return dict(reproj_loss=zero)

        return dict(reproj_loss=loss * self.loss_weight)


# =============================================================================
# 3. Cycle Consistency Loss
# =============================================================================

class CycleConsistencyLoss(nn.Module):
    """Enforce forward-backward cycle returns to the starting 3D position.

    Given a point P_A at frame A, we track it to frame B to get P_B, then
    track P_B back to frame A to get P_A'.  The loss is ``|P_A - P_A'|``.

    The pipeline is responsible for running the two decoder calls and
    providing ``original_3d`` and ``cycled_3d``.

    Args:
        loss_weight: scalar multiplier.
        threshold: L1 threshold for outlier filtering.
    """

    def __init__(
        self,
        loss_weight: float = 0.1,
        threshold: Optional[float] = 5.0,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.threshold = threshold

    def forward(
        self,
        original_3d: torch.Tensor,          # [B, N, 3]
        cycled_3d: torch.Tensor,             # [B, N, 3]
        valid_mask: Optional[torch.Tensor] = None,  # [B, N] bool
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with key ``cycle_loss``.
        """
        diff = (original_3d - cycled_3d).abs().sum(dim=-1)  # [B, N]

        if valid_mask is not None:
            valid = valid_mask.bool()
            if valid.sum() == 0:
                zero = torch.tensor(0.0, device=diff.device, requires_grad=True)
                return dict(cycle_loss=zero)
            diff = diff[valid]
        else:
            diff = diff.reshape(-1)

        # Outlier filtering
        if self.threshold is not None and diff.numel() > 0:
            diff = diff[diff < self.threshold]

        if diff.numel() == 0:
            zero = torch.tensor(0.0, device=original_3d.device, requires_grad=True)
            return dict(cycle_loss=zero)

        return dict(cycle_loss=diff.mean() * self.loss_weight)
