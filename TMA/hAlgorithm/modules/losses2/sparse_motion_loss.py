"""Sparse Motion Losses
======================

Two loss functions for query-based sparse motion prediction:

- ``SparseMotionLoss``  – Z-weighted L1 for **world_position** prediction.
- ``SparseDisplacementLoss`` – Log-transformed L1 for **displacement** prediction
  (ported from ``Any4DSceneFlowLoss`` to sparse [B, N, 3] interface).
- ``SparseDisplacementQueryMotion3dDeltaLoss`` – thin adapter from GlobalPoint-style
  ``query_motion3d_delta_loss`` call sites to ``SparseDisplacementLoss``.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hAlgorithm.modules.losses2.base import Loss


class SparseMotionLoss(nn.Module):
    """Loss for query-based sparse motion prediction.
    
    Key fix: Z-weighting offset only in denominator, not numerator.
    diff = L1 / (z + offset + eps)
    """
    
    def __init__(
        self,
        loss_weight: float = 1.0,
        offset: float = 0.5,
        threshold: float = 10.0,
        zweighted: bool = True,
        eps: float = 1e-6,
        dynamic_threshold: float = 0.02,
        balance_weight: float = 0.5,
        use_uncertainty: bool = False,
        logvar_min: float = -6.0,
        logvar_max: float = 6.0,
        uncertainty_weight: float = 1.0,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.offset = offset
        self.threshold = threshold
        self.zweighted = zweighted
        self.eps = eps
        self.dynamic_threshold = dynamic_threshold
        self.balance_weight = balance_weight
        self.use_uncertainty = use_uncertainty
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.uncertainty_weight = uncertainty_weight
    
    def forward(
        self,
        pred_positions: torch.Tensor,
        gt_positions: torch.Tensor,
        valid_mask: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        gt_positions_src: Optional[torch.Tensor] = None,
        pred_log_variance: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = pred_positions.device
        B, N, _ = pred_positions.shape
        per_q_w: Optional[torch.Tensor] = kwargs.get("per_query_weight")
        if per_q_w is not None:
            if per_q_w.shape[:2] != (B, N):
                raise ValueError(
                    f"per_query_weight must align with [B, N]=[{B}, {N}], "
                    f"got {tuple(per_q_w.shape)}."
                )
            per_q_w = per_q_w.to(device=device, dtype=pred_positions.dtype)

        valid_bool = valid_mask > 0.5
        if valid_bool.sum() == 0:
            zero_loss = pred_positions.sum() * 0.0
            return zero_loss, {
                "sparse_motion_loss": 0.0,
                "sparse_dyn_ratio": 0.0,
                "sparse_num_queries": 0,
            }

        # Normalize GT by scale if provided
        if scale is not None:
            if scale.ndim == 0 or scale.numel() == 1:
                scale_expanded = scale.view(1, 1, 1).expand(B, N, 1)
            elif scale.numel() == B:
                scale_expanded = scale.view(B, 1, 1).expand(B, N, 1)
            else:
                scale_expanded = scale.mean().view(1, 1, 1).expand(B, N, 1)
            gt_norm = gt_positions / (scale_expanded + self.eps)
        else:
            gt_norm = gt_positions

        # L1 error
        diff_raw = torch.abs(pred_positions - gt_norm).sum(dim=-1)  # [B, N]
        
        # FIX: Z-weighting — offset only in denominator
        # Old (buggy): diff = (diff_raw + self.offset) / (gt_z + self.offset + self.eps)
        # This added a constant ~1.0 to the loss even for perfect predictions!
        if self.zweighted:
            gt_z = gt_norm[..., 2].abs()
            diff = diff_raw / (gt_z + self.offset + self.eps)
        else:
            diff = diff_raw

        logvar = None
        if self.use_uncertainty and pred_log_variance is not None:
            logvar = pred_log_variance
            if logvar.ndim == 3 and logvar.shape[-1] == 1:
                logvar = logvar.squeeze(-1)
            logvar = logvar.clamp(min=self.logvar_min, max=self.logvar_max)
            diff = torch.exp(-logvar) * diff + self.uncertainty_weight * logvar
        
        # Dynamic/static classification
        if gt_positions_src is not None:
            gt_src_norm = gt_positions_src / (scale_expanded + self.eps) if scale is not None else gt_positions_src
            motion_mag = torch.norm(gt_norm - gt_src_norm, dim=-1)
            is_dynamic = motion_mag > self.dynamic_threshold
            mask_static = valid_bool & (~is_dynamic)
            mask_dynamic = valid_bool & is_dynamic
        else:
            mask_static = valid_bool
            mask_dynamic = torch.zeros_like(valid_bool)
        
        def compute_masked_loss(diff_tensor, mask):
            if mask.sum() == 0:
                return torch.tensor(0.0, device=device)
            masked_diff = diff_tensor[mask]
            if per_q_w is not None:
                w = per_q_w[mask]
                if self.threshold is not None:
                    keep = masked_diff < self.threshold
                    masked_diff = masked_diff[keep]
                    w = w[keep]
                if masked_diff.numel() == 0:
                    return torch.tensor(0.0, device=device)
                w = w / w.sum().clamp(min=self.eps)
                return (masked_diff * w).sum()
            if self.threshold is not None:
                masked_diff = masked_diff[masked_diff < self.threshold]
            if masked_diff.numel() == 0:
                return torch.tensor(0.0, device=device)
            return masked_diff.mean()
        
        loss_static = compute_masked_loss(diff, mask_static)
        loss_dynamic = compute_masked_loss(diff, mask_dynamic)
        
        if mask_static.sum() > 0 and mask_dynamic.sum() > 0:
            loss = (1.0 - self.balance_weight) * loss_static + self.balance_weight * loss_dynamic
        else:
            loss = loss_static + loss_dynamic
        
        if torch.isnan(loss).item() or torch.isinf(loss).item():
            loss = 0 * torch.sum(pred_positions)
            logging.warning("SparseMotionLoss: NaN/Inf detected")
        
        final_loss = loss * self.loss_weight
        
        with torch.no_grad():
            stats = {
                "sparse_motion_loss": final_loss.item(),
                "sparse_dyn_ratio": (mask_dynamic.sum().float() / (valid_bool.sum() + 1e-6)).item(),
                "sparse_num_queries": valid_bool.sum().item(),
            }
            if valid_bool.sum() > 0:
                stats["sparse_gt_z_mean"] = gt_norm[valid_bool][..., 2].mean().item()
                stats["sparse_gt_z_min"] = gt_norm[valid_bool][..., 2].min().item()
                stats["sparse_gt_z_max"] = gt_norm[valid_bool][..., 2].max().item()
                stats["sparse_gt_neg_z_ratio"] = (gt_norm[valid_bool][..., 2] < 0).float().mean().item()
                stats["sparse_pred_z_mean"] = pred_positions[valid_bool][..., 2].mean().item()
                stats["sparse_pred_z_min"] = pred_positions[valid_bool][..., 2].min().item()
                stats["sparse_pred_z_max"] = pred_positions[valid_bool][..., 2].max().item()
                stats["sparse_l1_raw"] = diff_raw[valid_bool].mean().item()
                stats["sparse_l1_weighted"] = diff[valid_bool].mean().item()
                if logvar is not None:
                    stats["sparse_logvar_mean"] = logvar[valid_bool].mean().item()
                    stats["sparse_logvar_min"] = logvar[valid_bool].min().item()
                    stats["sparse_logvar_max"] = logvar[valid_bool].max().item()
        
        return final_loss, stats


class SparseDisplacementLoss(nn.Module):
    """Log-transformed loss for sparse displacement prediction.

    Ported from ``Any4DSceneFlowLoss`` (dense, grid-sample-based) to the
    sparse ``[B, N, 3]`` interface used by ``SparseMotionPipeline``.

    Core formulation::

        log_transform(x) = x * log(1 + ||x||) / ||x||
        loss = | log_transform(pred) - log_transform(gt) |

    Why log-transform for displacement:
        - Compresses large motion magnitudes → large/small motions contribute equally
        - No Z-weighting needed (displacement Z can be negative / near-zero)
        - Better gradient behaviour across motion scales

    ``batch_reduction="per_example_mean"`` (V-DPM-style):
        When ``pool_static_dynamic`` is False, first form the static/dynamic
        ``balance_weight`` mixture **within** each batch example, then average
        those per-example scalars across examples that have at least one valid
        query. When ``pool_static_dynamic`` is True, each per-example scalar is
        instead the masked mean of ``diff`` over **all** valid queries in that
        example, then averaged across the batch. Dense static scenes no longer
        dominate purely by query count when ``batch_size > 1``; with
        ``batch_size == 1`` this matches the corresponding global reduction up to
        numerical order of operations.

    ``pool_static_dynamic=True``:
        Do **not** split the main term into separate static vs dynamic means or
        apply ``balance_weight``. The primary loss is a single masked mean of
        ``diff`` over all valid queries (same threshold / optional per-query
        weights as the global path). ``dynamic_threshold`` is still used only
        for ``sparse_dyn_ratio`` diagnostics and auxiliary mag/dir terms.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        dynamic_threshold: float = 0.02,
        balance_weight: float = 0.5,
        threshold: float = 10.0,
        eps: float = 1e-6,
        detach_log_scale: bool = False,
        use_uncertainty: bool = False,
        logvar_min: float = -6.0,
        logvar_max: float = 6.0,
        uncertainty_weight: float = 1.0,
        dynamic_weight_by_mag: bool = False,
        dynamic_weight_cap: float = 5.0,
        mag_loss_weight: float = 0.0,
        dir_loss_weight: float = 0.0,
        dir_loss_mag_threshold: float = 0.05,
        batch_reduction: str = "global",
        pool_static_dynamic: bool = False,
    ):
        super().__init__()
        if batch_reduction not in ("global", "per_example_mean"):
            raise ValueError(
                f"batch_reduction must be 'global' or 'per_example_mean', got {batch_reduction!r}"
            )
        self.loss_weight = loss_weight
        self.dynamic_threshold = dynamic_threshold
        self.balance_weight = balance_weight
        self.threshold = threshold
        self.eps = eps
        self.detach_log_scale = detach_log_scale
        self.use_uncertainty = use_uncertainty
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.uncertainty_weight = uncertainty_weight
        self.dynamic_weight_by_mag = dynamic_weight_by_mag
        self.dynamic_weight_cap = dynamic_weight_cap
        self.mag_loss_weight = mag_loss_weight
        self.dir_loss_weight = dir_loss_weight
        self.dir_loss_mag_threshold = dir_loss_mag_threshold
        self.batch_reduction = batch_reduction
        self.pool_static_dynamic = pool_static_dynamic

    # ── helpers ──────────────────────────────────────────────────────────

    def _log_transform(self, x: torch.Tensor) -> torch.Tensor:
        """f(x) = x * log(1 + ||x||) / ||x||.  Limit → x when ||x|| → 0.

        The log transform compresses large magnitudes so that the loss treats
        small and large displacements more equally.

        Numerical safety:
        ``torch.norm`` has a 0/0 backward at zero input; chaining it with
        ``.clamp(min=eps)`` still yields a NaN gradient because PyTorch's
        clamp saturation gradient is ``0`` and ``0 * NaN = NaN``. We instead
        compute ``sqrt(sum(x²) + eps²)`` so the denominator is mathematically
        bounded away from zero, giving a finite gradient even when the
        predicted displacement is exactly zero (a regular occurrence at
        initialization for spherical/MLP motion heads).

        Gradient behaviour vs ``detach_log_scale``:
        - ``True``: gradient ∝ log(||x||)/||x|| (decreases with ||x||) —
          may cause mode collapse toward zero predictions.
        - ``False``: full gradient through the log transform — more
          aggressive but avoids mode collapse.
        """
        # Stable norm: use squared-sum + eps² to avoid 0/0 backward at zero input.
        sq_sum = (x * x).sum(dim=-1, keepdim=True)
        norm = (sq_sum + self.eps * self.eps).sqrt().clamp(max=10.0)

        if self.detach_log_scale:
            scale = torch.log1p(norm.detach()) / norm.detach()
        else:
            scale = torch.log1p(norm) / norm

        return x * scale

    def _masked_mean_queries(
        self,
        diff_row: torch.Tensor,
        mask_row: torch.Tensor,
        weights_row: Optional[torch.Tensor] = None,
        per_query_row: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Mean of ``diff_row`` on positions where ``mask_row`` is True (one sample)."""
        device = diff_row.device
        if mask_row.sum() == 0:
            return torch.tensor(0.0, device=device, dtype=diff_row.dtype)
        vals = diff_row[mask_row]
        w_mag = weights_row[mask_row] if weights_row is not None else None
        w_q = per_query_row[mask_row] if per_query_row is not None else None
        if w_mag is not None and w_q is not None:
            w = w_mag * w_q
        elif w_mag is not None:
            w = w_mag
        elif w_q is not None:
            w = w_q
        else:
            w = None
        if self.threshold is not None:
            keep = vals < self.threshold
            vals = vals[keep]
            if w is not None:
                w = w[keep]
        if vals.numel() == 0:
            return torch.tensor(0.0, device=device, dtype=diff_row.dtype)
        if w is not None:
            w = w / w.sum().clamp(min=self.eps)
            return (vals * w).sum()
        return vals.mean()

    # ── forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        pred_displacement: torch.Tensor,        # [B, N, C]
        gt_displacement: torch.Tensor,           # [B, N, C]
        valid_mask: torch.Tensor,                # [B, N]
        scale: Optional[torch.Tensor] = None,    # for abs-scale threshold only
        pred_log_variance: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = pred_displacement.device
        B, N, _ = pred_displacement.shape
        valid_bool = valid_mask > 0.5
        per_q_w: Optional[torch.Tensor] = kwargs.get("per_query_weight")
        if per_q_w is not None:
            if per_q_w.shape[:2] != (B, N):
                raise ValueError(
                    f"per_query_weight must align with [B, N]=[{B}, {N}], "
                    f"got {tuple(per_q_w.shape)}."
                )
            per_q_w = per_q_w.to(device=device, dtype=pred_displacement.dtype)

        # ── Input sanitization: filter out NaN/Inf values per-query ────────
        pred_finite = torch.isfinite(pred_displacement).all(dim=-1)  # [B, N]
        gt_finite = torch.isfinite(gt_displacement).all(dim=-1)      # [B, N]
        valid_bool = valid_bool & pred_finite & gt_finite

        # CRITICAL: replace NaN/Inf entries with 0 BEFORE any tensor op that
        # participates in autograd. Even when masked out from the forward
        # reduction, non-finite entries propagate NaN through the chain rule
        # of ``_log_transform`` / ``torch.norm`` / etc. and poison
        # ``grad_norm``, which the trainer then sets to 0 — i.e. the entire
        # step skips weight updates. ``nan_to_num`` is differentiable: the
        # gradient is 1 at originally-finite positions and 0 at the patched
        # positions, so legitimate gradients are preserved.
        pred_displacement = torch.nan_to_num(
            pred_displacement, nan=0.0, posinf=0.0, neginf=0.0
        )
        gt_displacement = torch.nan_to_num(
            gt_displacement, nan=0.0, posinf=0.0, neginf=0.0
        )
        
        if valid_bool.sum() == 0:
            # No valid samples - return zero loss connected to pred for gradient graph
            zero_loss = 0.0 * pred_displacement.sum()
            return zero_loss, {
                "sparse_disp_loss": 0.0,
                "sparse_dyn_ratio": 0.0,
                "sparse_num_queries": 0,
            }

        # ── dynamic / static split (threshold in absolute scale) ────────
        if scale is not None:
            # Align scale to [B] using the reference view when multi-view
            # scale tensors are provided (e.g. [B, V, 1, 1, 1]).
            if scale.ndim == 0 or scale.numel() == 1:
                s = scale.view(1).expand(B)
            elif scale.shape[0] == B:
                s_ref = scale[:, 0] if scale.ndim >= 2 else scale
                s = s_ref.reshape(B, -1)[:, 0]
            else:
                s_flat = scale.flatten()
                s = s_flat if s_flat.numel() == B else s_flat.mean().expand(B)
            gt_abs = gt_displacement * s.view(B, 1, 1)
        else:
            gt_abs = gt_displacement

        motion_mag = torch.norm(gt_abs, dim=-1)          # [B, N]
        is_dynamic = motion_mag > self.dynamic_threshold
        mask_static  = valid_bool & (~is_dynamic)
        mask_dynamic = valid_bool & is_dynamic

        # ── log-transformed L1 ──────────────────────────────────────────
        pred_log = self._log_transform(pred_displacement)
        gt_log   = self._log_transform(gt_displacement)
        diff = torch.abs(pred_log - gt_log).sum(dim=-1)   # [B, N]

        logvar = None
        if self.use_uncertainty and pred_log_variance is not None:
            logvar = pred_log_variance
            if logvar.ndim == 3 and logvar.shape[-1] == 1:
                logvar = logvar.squeeze(-1)
            logvar = logvar.clamp(min=self.logvar_min, max=self.logvar_max)
            diff = torch.exp(-logvar) * diff + self.uncertainty_weight * logvar

        # ── Per-query NaN/Inf filtering (more robust than whole-batch rejection) ──
        diff_valid = torch.isfinite(diff)
        if not diff_valid.all():
            num_invalid = (~diff_valid).sum().item()
            logging.warning(
                f"SparseDisplacementLoss: {num_invalid} queries have NaN/Inf diff, filtering them out. "
                f"pred range=[{pred_displacement.min():.4f},{pred_displacement.max():.4f}], "
                f"gt range=[{gt_displacement.min():.4f},{gt_displacement.max():.4f}]"
            )
            # Update masks to exclude NaN/Inf queries
            mask_static = mask_static & diff_valid
            mask_dynamic = mask_dynamic & diff_valid
            valid_bool = valid_bool & diff_valid
            
            if valid_bool.sum() == 0:
                zero_loss = 0.0 * pred_displacement.sum()
                return zero_loss, {
                    "sparse_disp_loss": 0.0,
                    "sparse_dyn_ratio": 0.0,
                    "sparse_num_queries": 0,
                }

        # Per-query magnitude weighting (only when using static/dynamic split).
        dyn_weights = None
        if (
            not self.pool_static_dynamic
            and self.dynamic_weight_by_mag
            and mask_dynamic.sum() > 0
        ):
            mag_dyn = motion_mag[mask_dynamic].detach()
            mean_mag = mag_dyn.mean().clamp(min=self.eps)
            raw_w = (motion_mag / mean_mag).clamp(max=self.dynamic_weight_cap)
            dyn_weights = raw_w * mask_dynamic.float()
        elif self.pool_static_dynamic and self.dynamic_weight_by_mag:
            logging.warning(
                "SparseDisplacementLoss: pool_static_dynamic=True ignores "
                "dynamic_weight_by_mag for the main displacement term."
            )

        def _masked_mean_global(
            diff_t: torch.Tensor,
            mask: torch.Tensor,
            weights: Optional[torch.Tensor] = None,
            pqw: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            if mask.sum() == 0:
                return torch.tensor(0.0, device=device)
            vals = diff_t[mask]
            w_mag = weights[mask] if weights is not None else None
            w_q = pqw[mask] if pqw is not None else None
            if w_mag is not None and w_q is not None:
                w = w_mag * w_q
            elif w_mag is not None:
                w = w_mag
            elif w_q is not None:
                w = w_q
            else:
                w = None
            if self.threshold is not None:
                keep = vals < self.threshold
                vals = vals[keep]
                if w is not None:
                    w = w[keep]
            if vals.numel() == 0:
                return torch.tensor(0.0, device=device)
            if w is not None:
                w = w / w.sum().clamp(min=self.eps)
                return (vals * w).sum()
            return vals.mean()

        if self.pool_static_dynamic:
            if self.batch_reduction == "global":
                loss = _masked_mean_global(diff, valid_bool, None, per_q_w)
            else:
                per_example_losses = []
                for b in range(B):
                    if not valid_bool[b].any():
                        continue
                    pq_row = per_q_w[b] if per_q_w is not None else None
                    per_example_losses.append(
                        self._masked_mean_queries(
                            diff[b], valid_bool[b], None, pq_row,
                        )
                    )
                if not per_example_losses:
                    loss = 0.0 * pred_displacement.sum()
                else:
                    loss = torch.stack(per_example_losses).mean()
        elif self.batch_reduction == "global":
            loss_static = _masked_mean_global(diff, mask_static, None, per_q_w)
            loss_dynamic = _masked_mean_global(diff, mask_dynamic, dyn_weights, per_q_w)
            if mask_static.sum() > 0 and mask_dynamic.sum() > 0:
                loss = ((1.0 - self.balance_weight) * loss_static
                        + self.balance_weight * loss_dynamic)
            else:
                loss = loss_static + loss_dynamic
        else:
            # Mean within each batch example, then mean across examples (V-DPM-style).
            per_example_losses = []
            bw = self.balance_weight
            for b in range(B):
                if not valid_bool[b].any():
                    continue
                pq_row = per_q_w[b] if per_q_w is not None else None
                dyn_row = dyn_weights[b] if dyn_weights is not None else None
                ls_b = self._masked_mean_queries(
                    diff[b], mask_static[b], None, pq_row,
                )
                ld_b = self._masked_mean_queries(
                    diff[b], mask_dynamic[b], dyn_row, pq_row,
                )
                if mask_static[b].any() and mask_dynamic[b].any():
                    l_b = (1.0 - bw) * ls_b + bw * ld_b
                else:
                    l_b = ls_b + ld_b
                per_example_losses.append(l_b)
            if not per_example_losses:
                loss = 0.0 * pred_displacement.sum()
            else:
                loss = torch.stack(per_example_losses).mean()

        if torch.isnan(loss) or torch.isinf(loss):
            loss = 0 * pred_displacement.sum()
            logging.warning("SparseDisplacementLoss: NaN/Inf detected")

        final_loss = loss * self.loss_weight

        # ── Auxiliary magnitude loss: penalise magnitude mismatch in log1p-space ──
        # Uses log1p(||x||) instead of log(||x||) to avoid gradient explosion
        # near zero (gradient bounded at 1.0 instead of diverging to infinity).
        mag_loss_val = torch.tensor(0.0, device=device)
        if self.mag_loss_weight > 0 and mask_dynamic.sum() > 0:
            # Stable magnitudes: ``torch.norm`` is 0/0 in backward at zero input;
            # use ``sqrt(sum(x²) + eps²)`` to keep gradients finite for predictions
            # that collapse toward zero early in training.
            pred_mag = (
                (pred_displacement * pred_displacement).sum(dim=-1) + self.eps * self.eps
            ).sqrt()
            gt_mag_norm = (
                (gt_displacement * gt_displacement).sum(dim=-1) + self.eps * self.eps
            ).sqrt()
            mag_diff = torch.abs(torch.log1p(pred_mag) - torch.log1p(gt_mag_norm))
            md = mag_diff[mask_dynamic]
            if per_q_w is not None:
                w_m = per_q_w[mask_dynamic]
                w_m = w_m / w_m.sum().clamp(min=self.eps)
                mag_loss_val = (md * w_m).sum()
            else:
                mag_loss_val = md.mean()
            if torch.isfinite(mag_loss_val):
                final_loss = final_loss + self.mag_loss_weight * mag_loss_val
            else:
                mag_loss_val = torch.tensor(0.0, device=device)

        # ── Auxiliary direction loss: cosine similarity for dynamic points ──
        dir_loss_val = torch.tensor(0.0, device=device)
        if self.dir_loss_weight > 0 and mask_dynamic.sum() > 0:
            gt_mag_for_dir = gt_displacement.norm(dim=-1)
            dir_mask = mask_dynamic & (gt_mag_for_dir > self.dir_loss_mag_threshold)
            if dir_mask.sum() > 0:
                pred_dir = pred_displacement[dir_mask]
                gt_dir = gt_displacement[dir_mask]
                cos_sim = F.cosine_similarity(pred_dir, gt_dir, dim=-1)
                d_err = 1.0 - cos_sim
                if per_q_w is not None:
                    w_d = per_q_w[dir_mask]
                    w_d = w_d / w_d.sum().clamp(min=self.eps)
                    dir_loss_val = (d_err * w_d).sum()
                else:
                    dir_loss_val = d_err.mean()
                if torch.isfinite(dir_loss_val):
                    final_loss = final_loss + self.dir_loss_weight * dir_loss_val
                else:
                    dir_loss_val = torch.tensor(0.0, device=device)

        # ── stats (absolute scale for interpretability) ─────────────────
        with torch.no_grad():
            pred_abs = (pred_displacement * s.view(B, 1, 1)
                        if scale is not None else pred_displacement)
            stats = {
                "sparse_disp_loss": final_loss.item(),
                "sparse_dyn_ratio": (
                    mask_dynamic.sum().float() / (valid_bool.sum() + 1e-6)
                ).item(),
                "sparse_num_queries": valid_bool.sum().item(),
            }
            if self.mag_loss_weight > 0:
                stats["sparse_mag_loss"] = mag_loss_val.item()
            if self.dir_loss_weight > 0:
                stats["sparse_dir_loss"] = dir_loss_val.item()
            if valid_bool.sum() > 0:
                pred_abs_norm = torch.norm(pred_abs, dim=-1)  # [B, N]
                stats["sparse_gt_disp_mag"] = motion_mag[valid_bool].mean().item()
                stats["sparse_pred_disp_mag"] = pred_abs_norm[valid_bool].mean().item()
                stats["sparse_l1_log"] = diff[valid_bool].mean().item()
                if logvar is not None:
                    stats["sparse_logvar_mean"] = logvar[valid_bool].mean().item()
                    stats["sparse_logvar_min"] = logvar[valid_bool].min().item()
                    stats["sparse_logvar_max"] = logvar[valid_bool].max().item()

                # ── prediction diversity diagnostics ──────────────────
                # Magnitude diversity: std of predicted magnitudes (low = magnitude collapse)
                stats["sparse_pred_mag_std"] = pred_abs_norm[valid_bool].std().item()

                if mask_dynamic.sum() >= 2:
                    dyn_pred = pred_abs[mask_dynamic]  # [N_dyn, C]
                    dyn_gt   = gt_abs[mask_dynamic]    # [N_dyn, C]
                    dyn_pred_norm = dyn_pred.norm(dim=-1)
                    dyn_gt_norm   = dyn_gt.norm(dim=-1)

                    # Dynamic-only pred/gt magnitude ratio
                    stats["sparse_mag_ratio_dyn"] = (
                        dyn_pred_norm.mean() / dyn_gt_norm.mean().clamp(min=1e-6)
                    ).item()

                    # Direction collapse: mean resultant length of unit-predicted vectors
                    # (1.0 = all same direction = full collapse, ~0 = diverse)
                    nz = dyn_pred_norm > 1e-6
                    if nz.sum() >= 2:
                        unit_pred = dyn_pred[nz] / dyn_pred_norm[nz].unsqueeze(-1)
                        mean_dir = unit_pred.mean(dim=0)
                        stats["sparse_dir_collapse"] = mean_dir.norm().item()

        return final_loss, stats


class SparseDisplacementQueryMotion3dDeltaLoss(Loss):
    """Adapter: ``query_motion3d_delta_loss`` contract (GlobalPoint-like) backed by SparseDisplacementLoss.

    The motion pipeline passes ``pred_depth`` / ``target_depth`` shaped like ``(Q, 1, 3)``
    with ``valid_mask`` and optional ``weight`` per query. This wrapper reshapes them to
    ``[B, N, 3]`` with ``B=1`` for ``SparseDisplacementLoss`` and returns a scalar dict
    ``{"l1_loss": ...}`` so callers can aggregate with the ``delta_`` prefix unchanged.

    Outer ``loss_weight`` (including dataset-keyed dicts via ``get_loss_weight``) scales the
    returned scalar exactly once; the inner ``SparseDisplacementLoss`` uses ``loss_weight=1.0``.
    """

    def __init__(self, loss_weight=1.0, **sparse_kwargs):
        sk = {k: v for k, v in sparse_kwargs.items() if k != "type"}
        sk["loss_weight"] = 1.0
        super().__init__(loss_weight=loss_weight)
        self._sparse = SparseDisplacementLoss(**sk)

    @staticmethod
    def _squeeze_mid_singleton(t: torch.Tensor) -> torch.Tensor:
        """Drop a singleton dimension before the last axis, e.g. (Q,1,3)->(Q,3)."""

        x = t
        while x.ndim >= 3 and x.shape[-2] == 1:
            x = x.squeeze(-2)
        return x

    def _to_bn3(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        valid_mask: torch.Tensor,
        weight: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        pd = self._squeeze_mid_singleton(pred)
        gd = self._squeeze_mid_singleton(gt)
        if pd.shape != gd.shape:
            raise ValueError(
                f"pred_depth and target_depth must align after reshape; "
                f"got pred {tuple(pd.shape)}, target {tuple(gd.shape)}."
            )
        if pd.ndim != 2 or pd.shape[-1] != 3:
            raise ValueError(
                "Expected displacement with shape (Q, 3) after removing singleton "
                f"axes; got {tuple(pd.shape)}."
            )
        q = pd.shape[0]
        pd = pd.unsqueeze(0)
        gd = gd.unsqueeze(0)

        vm = valid_mask.to(device=pd.device, dtype=torch.float32)
        while vm.ndim >= 2 and vm.shape[-1] == 1:
            vm = vm.squeeze(-1)
        if vm.ndim == 1:
            if vm.numel() != q:
                raise ValueError(
                    f"valid_mask length {vm.numel()} does not match Q={q} queries."
                )
            vm = vm.unsqueeze(0)
        elif vm.ndim == 2:
            if vm.shape != (1, q):
                raise ValueError(
                    f"valid_mask shape {tuple(vm.shape)} incompatible with Q={q}."
                )
        else:
            raise ValueError(
                f"valid_mask must be (Q,) or (1, Q) after squeezing; got {tuple(vm.shape)}."
            )

        pqw: Optional[torch.Tensor] = None
        if weight is not None:
            pqw = weight.to(device=pd.device, dtype=pd.dtype)
            while pqw.ndim >= 2 and pqw.shape[-1] == 1:
                pqw = pqw.squeeze(-1)
            if pqw.ndim == 1:
                if pqw.numel() != q:
                    raise ValueError(
                        f"weight length {pqw.numel()} does not match Q={q} queries."
                    )
                pqw = pqw.unsqueeze(0)
            elif pqw.ndim == 2:
                if pqw.shape != (1, q):
                    raise ValueError(
                        f"weight shape {tuple(pqw.shape)} incompatible with Q={q}."
                    )
            else:
                raise ValueError(
                    f"weight must be (Q,) or (1, Q) after squeezing; got {tuple(pqw.shape)}."
                )

        return pd, gd, vm, pqw

    def forward(
        self,
        pred_depth,
        target_depth,
        valid_mask,
        pred_conf=None,
        name=None,
        weight=None,
        scale=None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        lw = self.get_loss_weight(name)
        if lw == 0:
            z = 0.0 * pred_depth.sum()
            return {"l1_loss": z}

        pd, gd, vm, pqw = self._to_bn3(pred_depth, target_depth, valid_mask, weight)

        pred_log_var = None
        if self._sparse.use_uncertainty and pred_conf is not None:
            pred_log_var = pred_conf

        core, _stats = self._sparse(
            pred_displacement=pd,
            gt_displacement=gd,
            valid_mask=vm,
            scale=scale,
            pred_log_variance=pred_log_var,
            per_query_weight=pqw,
        )
        out = core * lw
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return {"l1_loss": out}


class SparseDisplacementL1Loss(nn.Module):
    """Plain L1 loss for sparse displacement prediction (no log transform).

    This serves as a stable baseline to compare against ``SparseDisplacementLoss``
    when diagnosing optimization behavior under log-space supervision.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        dynamic_threshold: float = 0.02,
        balance_weight: float = 0.5,
        threshold: float = 10.0,
        eps: float = 1e-6,
        use_uncertainty: bool = False,
        logvar_min: float = -6.0,
        logvar_max: float = 6.0,
        uncertainty_weight: float = 1.0,
        dynamic_weight_by_mag: bool = False,
        dynamic_weight_cap: float = 5.0,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.dynamic_threshold = dynamic_threshold
        self.balance_weight = balance_weight
        self.threshold = threshold
        self.eps = eps
        self.use_uncertainty = use_uncertainty
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.uncertainty_weight = uncertainty_weight
        self.dynamic_weight_by_mag = dynamic_weight_by_mag
        self.dynamic_weight_cap = dynamic_weight_cap

    def forward(
        self,
        pred_displacement: torch.Tensor,        # [B, N, 3]
        gt_displacement: torch.Tensor,          # [B, N, 3]
        valid_mask: torch.Tensor,               # [B, N]
        scale: Optional[torch.Tensor] = None,   # for abs-scale threshold/stat only
        pred_log_variance: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = pred_displacement.device
        B, N, _ = pred_displacement.shape
        valid_bool = valid_mask > 0.5
        per_q_w: Optional[torch.Tensor] = kwargs.get("per_query_weight")
        if per_q_w is not None:
            if per_q_w.shape[:2] != (B, N):
                raise ValueError(
                    f"per_query_weight must align with [B, N]=[{B}, {N}], "
                    f"got {tuple(per_q_w.shape)}."
                )
            per_q_w = per_q_w.to(device=device, dtype=pred_displacement.dtype)

        # Input sanitization: filter non-finite samples per query.
        pred_finite = torch.isfinite(pred_displacement).all(dim=-1)
        gt_finite = torch.isfinite(gt_displacement).all(dim=-1)
        valid_bool = valid_bool & pred_finite & gt_finite

        if valid_bool.sum() == 0:
            zero_loss = 0.0 * pred_displacement.sum()
            return zero_loss, {
                "sparse_disp_loss": 0.0,
                "sparse_dyn_ratio": 0.0,
                "sparse_num_queries": 0,
            }

        if scale is not None:
            # Align scale to [B] using ref view when multi-view scale is provided.
            if scale.ndim == 0 or scale.numel() == 1:
                s = scale.view(1).expand(B)
            elif scale.shape[0] == B:
                s_ref = scale[:, 0] if scale.ndim >= 2 else scale
                s = s_ref.reshape(B, -1)[:, 0]
            else:
                s_flat = scale.flatten()
                s = s_flat if s_flat.numel() == B else s_flat.mean().expand(B)
            pred_abs = pred_displacement * s.view(B, 1, 1)
            gt_abs = gt_displacement * s.view(B, 1, 1)
        else:
            pred_abs = pred_displacement
            gt_abs = gt_displacement

        # Dynamic/static split uses absolute displacement magnitude.
        motion_mag = torch.norm(gt_abs, dim=-1)  # [B, N]
        is_dynamic = motion_mag > self.dynamic_threshold
        mask_static = valid_bool & (~is_dynamic)
        mask_dynamic = valid_bool & is_dynamic

        # Plain displacement L1 in normalized training space.
        diff = torch.abs(pred_displacement - gt_displacement).sum(dim=-1)  # [B, N]

        logvar = None
        if self.use_uncertainty and pred_log_variance is not None:
            logvar = pred_log_variance
            if logvar.ndim == 3 and logvar.shape[-1] == 1:
                logvar = logvar.squeeze(-1)
            logvar = logvar.clamp(min=self.logvar_min, max=self.logvar_max)
            diff = torch.exp(-logvar) * diff + self.uncertainty_weight * logvar

        # Per-query NaN/Inf filtering for robustness.
        diff_valid = torch.isfinite(diff)
        if not diff_valid.all():
            num_invalid = (~diff_valid).sum().item()
            logging.warning(
                "SparseDisplacementL1Loss: %d queries have NaN/Inf diff, filtering them out.",
                num_invalid,
            )
            mask_static = mask_static & diff_valid
            mask_dynamic = mask_dynamic & diff_valid
            valid_bool = valid_bool & diff_valid
            if valid_bool.sum() == 0:
                zero_loss = 0.0 * pred_displacement.sum()
                return zero_loss, {
                    "sparse_disp_loss": 0.0,
                    "sparse_dyn_ratio": 0.0,
                    "sparse_num_queries": 0,
                }

        def _masked_mean(
            diff_t: torch.Tensor,
            mask: torch.Tensor,
            weights: Optional[torch.Tensor] = None,
            pqw: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            if mask.sum() == 0:
                return torch.tensor(0.0, device=device)
            vals = diff_t[mask]
            w_mag = weights[mask] if weights is not None else None
            w_q = pqw[mask] if pqw is not None else None
            if w_mag is not None and w_q is not None:
                w = w_mag * w_q
            elif w_mag is not None:
                w = w_mag
            elif w_q is not None:
                w = w_q
            else:
                w = None
            if self.threshold is not None:
                keep = vals < self.threshold
                vals = vals[keep]
                if w is not None:
                    w = w[keep]
            if vals.numel() == 0:
                return torch.tensor(0.0, device=device)
            if w is not None:
                w = w / w.sum().clamp(min=self.eps)
                return (vals * w).sum()
            return vals.mean()

        dyn_weights = None
        if self.dynamic_weight_by_mag and mask_dynamic.sum() > 0:
            mag_dyn = motion_mag[mask_dynamic].detach()
            mean_mag = mag_dyn.mean().clamp(min=self.eps)
            raw_w = (motion_mag / mean_mag).clamp(max=self.dynamic_weight_cap)
            dyn_weights = raw_w * mask_dynamic.float()

        loss_static  = _masked_mean(diff, mask_static, None, per_q_w)
        loss_dynamic = _masked_mean(diff, mask_dynamic, dyn_weights, per_q_w)

        if mask_static.sum() > 0 and mask_dynamic.sum() > 0:
            loss = (
                (1.0 - self.balance_weight) * loss_static
                + self.balance_weight * loss_dynamic
            )
        else:
            loss = loss_static + loss_dynamic

        if torch.isnan(loss) or torch.isinf(loss):
            loss = 0 * pred_displacement.sum()
            logging.warning("SparseDisplacementL1Loss: NaN/Inf detected")

        final_loss = loss * self.loss_weight

        with torch.no_grad():
            stats = {
                "sparse_disp_loss": final_loss.item(),
                "sparse_dyn_ratio": (
                    mask_dynamic.sum().float() / (valid_bool.sum() + 1e-6)
                ).item(),
                "sparse_num_queries": valid_bool.sum().item(),
            }
            if valid_bool.sum() > 0:
                stats["sparse_gt_disp_mag"] = motion_mag[valid_bool].mean().item()
                stats["sparse_pred_disp_mag"] = (
                    torch.norm(pred_abs, dim=-1)[valid_bool].mean().item()
                )
                stats["sparse_l1_disp"] = diff[valid_bool].mean().item()
                if logvar is not None:
                    stats["sparse_logvar_mean"] = logvar[valid_bool].mean().item()
                    stats["sparse_logvar_min"] = logvar[valid_bool].min().item()
                    stats["sparse_logvar_max"] = logvar[valid_bool].max().item()
                
                # DEBUG: Detailed displacement statistics
                gt_valid = gt_displacement[valid_bool]  # [N_valid, 3]
                pred_valid = pred_abs[valid_bool]  # [N_valid, 3]
                
                # Per-component statistics (log only, don't add to stats dict)
                gt_xyz_mean = gt_valid.mean(dim=0).tolist()
                pred_xyz_mean = pred_valid.mean(dim=0).tolist()
                
                # Direction similarity (cosine similarity between pred and gt)
                gt_norm = gt_valid / (gt_valid.norm(dim=-1, keepdim=True) + 1e-8)
                pred_norm = pred_valid / (pred_valid.norm(dim=-1, keepdim=True) + 1e-8)
                cos_sim = (gt_norm * pred_norm).sum(dim=-1)  # [N_valid]
                cos_sim_mean = cos_sim.mean().item()
                cos_sim_std = cos_sim.std().item()
                
                # Prediction range
                pred_mag = pred_valid.norm(dim=-1)
                pred_mag_min = pred_mag.min().item()
                pred_mag_max = pred_mag.max().item()
                
                # GT range
                gt_mag = gt_valid.norm(dim=-1)
                gt_mag_min = gt_mag.min().item()
                gt_mag_max = gt_mag.max().item()
                
                # Log detailed info
                logging.info(
                    "[DEBUG Loss] GT xyz_mean=[%.4f,%.4f,%.4f] | Pred xyz_mean=[%.4f,%.4f,%.4f] | "
                    "cos_sim=%.4f±%.4f | pred_mag=[%.4f,%.4f] gt_mag=[%.4f,%.4f]",
                    *gt_xyz_mean, *pred_xyz_mean,
                    cos_sim_mean, cos_sim_std,
                    pred_mag_min, pred_mag_max,
                    gt_mag_min, gt_mag_max,
                )
                
                # Print a few sample predictions vs GT
                n_show = min(5, gt_valid.shape[0])
                for i in range(n_show):
                    gt_i = gt_valid[i].tolist()
                    pred_i = pred_valid[i].tolist()
                    logging.info(
                        "[DEBUG Sample %d] GT=[%.4f,%.4f,%.4f] mag=%.4f | "
                        "Pred=[%.4f,%.4f,%.4f] mag=%.4f",
                        i, gt_i[0], gt_i[1], gt_i[2], gt_valid[i].norm().item(),
                        pred_i[0], pred_i[1], pred_i[2], pred_valid[i].norm().item(),
                    )

        return final_loss, stats


class SparseDisplacementL1MagLoss(nn.Module):
    """L1 + symmetric log-magnitude matching.

    Addresses the zero-collapse problem that plagues both MSE and plain L1:
    the model learns to predict near-zero displacement because most query points
    are static, and predicting zero minimises average error on static points.

    Two-part design:
    1. L1 base loss: stable constant-gradient supervision for direction and value.
    2. Log-magnitude ratio (dynamic points only):
           L_mag = | log(||pred|| + eps) - log(||gt|| + eps) |
       Properties:
       - SYMMETRIC in log-space: 2x over-prediction and 0.5x under-prediction
         get the same penalty (log 2 ≈ 0.69).  Unlike the relative L1 term in
         SparseDisplacementHuberLoss which was asymmetric and caused 21x
         over-prediction in v5.
       - Strong anti-collapse gradient: when pred→0, ∂L_mag/∂pred ∝ 1/||pred||,
         producing increasingly strong gradients that push predictions away from
         zero.  This directly prevents the magnitude collapse observed in v2/v8.
       - Scale-invariant: works uniformly across all motion magnitudes.

    Loss formula:
        L_dyn  = (1 - mag_alpha) * L1(pred, gt) + mag_alpha * L_mag(pred, gt)
        L_stat = L1(pred, gt)
        L      = balance_weight * mean(L_dyn) + (1-balance_weight) * mean(L_stat)
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        dynamic_threshold: float = 0.02,
        balance_weight: float = 0.7,
        mag_alpha: float = 0.1,
        mag_clamp: float = 5.0,
        mag_eps: float = 1e-4,
        threshold: float = 10.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.dynamic_threshold = dynamic_threshold
        self.balance_weight = balance_weight
        self.mag_alpha = mag_alpha
        self.mag_clamp = mag_clamp
        self.mag_eps = mag_eps
        self.threshold = threshold
        self.eps = eps

        logging.info(
            "[SparseDisplacementL1MagLoss] weight=%.2f, dyn_thresh=%.4f, "
            "balance=%.2f, mag_alpha=%.2f, mag_clamp=%.1f",
            loss_weight, dynamic_threshold, balance_weight, mag_alpha, mag_clamp,
        )

    def forward(
        self,
        pred_displacement: torch.Tensor,
        gt_displacement: torch.Tensor,
        valid_mask: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        pred_log_variance: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        B, Q, _ = pred_displacement.shape
        device = pred_displacement.device

        valid_bool = valid_mask > 0.5
        pred_finite = torch.isfinite(pred_displacement).all(dim=-1)
        gt_finite = torch.isfinite(gt_displacement).all(dim=-1)
        valid_bool = valid_bool & pred_finite & gt_finite

        if valid_bool.sum() == 0:
            return pred_displacement.sum() * 0.0, {
                "sparse_disp_loss": 0.0, "sparse_num_queries": 0,
            }

        # ── Scale extraction for stats ───────────────────────────────────────
        s = None
        if scale is not None:
            if scale.ndim == 0 or scale.numel() == 1:
                s = scale.view(1).expand(B)
            elif scale.shape[0] == B:
                s_ref = scale[:, 0] if scale.ndim >= 2 else scale
                s = s_ref.reshape(B, -1)[:, 0]
            else:
                s_flat = scale.flatten()
                s = s_flat if s_flat.numel() == B else s_flat.mean().expand(B)

        # ── Per-query L1 ─────────────────────────────────────────────────────
        l1_q = torch.abs(pred_displacement - gt_displacement).sum(dim=-1)  # [B, Q]

        # ── Per-query log-magnitude ratio ────────────────────────────────────
        pred_mag = pred_displacement.norm(dim=-1).clamp(min=self.mag_eps)  # [B, Q]
        gt_mag = gt_displacement.norm(dim=-1).clamp(min=self.mag_eps)      # [B, Q]
        log_ratio = (torch.log(pred_mag) - torch.log(gt_mag)).clamp(
            -self.mag_clamp, self.mag_clamp
        )
        mag_q = log_ratio.abs()  # [B, Q]

        # ── Dynamic / static split ───────────────────────────────────────────
        gt_norm = gt_displacement.norm(dim=-1)
        is_dynamic = gt_norm > self.dynamic_threshold
        mask_dynamic = valid_bool & is_dynamic
        mask_static = valid_bool & ~is_dynamic

        # NaN/Inf guard
        diff_valid = torch.isfinite(l1_q) & torch.isfinite(mag_q)
        if not diff_valid.all():
            mask_static = mask_static & diff_valid
            mask_dynamic = mask_dynamic & diff_valid
            valid_bool = valid_bool & diff_valid
            if valid_bool.sum() == 0:
                return pred_displacement.sum() * 0.0, {
                    "sparse_disp_loss": 0.0, "sparse_num_queries": 0,
                }

        def _mean(tensor, mask):
            if mask.sum() == 0:
                return torch.tensor(0.0, device=device)
            vals = tensor[mask]
            if self.threshold is not None:
                vals = vals[vals < self.threshold]
            if vals.numel() == 0:
                return torch.tensor(0.0, device=device)
            return vals.mean()

        # Dynamic: L1 + magnitude matching
        dyn_loss_q = (1.0 - self.mag_alpha) * l1_q + self.mag_alpha * mag_q
        loss_dynamic = _mean(dyn_loss_q, mask_dynamic)
        # Static: L1 only (no magnitude term — static points have gt≈0)
        loss_static = _mean(l1_q, mask_static)

        if mask_dynamic.sum() > 0 and mask_static.sum() > 0:
            loss = (self.balance_weight * loss_dynamic
                    + (1.0 - self.balance_weight) * loss_static)
        elif mask_dynamic.sum() > 0:
            loss = loss_dynamic
        else:
            loss = loss_static

        if torch.isnan(loss) or torch.isinf(loss):
            loss = 0 * pred_displacement.sum()
            logging.warning("SparseDisplacementL1MagLoss: NaN/Inf detected")

        final_loss = loss * self.loss_weight

        # ── Stats ────────────────────────────────────────────────────────────
        with torch.no_grad():
            gt_abs = gt_displacement * s.view(B, 1, 1) if s is not None else gt_displacement
            pred_abs = pred_displacement * s.view(B, 1, 1) if s is not None else pred_displacement
            stats = {
                "sparse_disp_loss": final_loss.item(),
                "sparse_dyn_ratio": (
                    mask_dynamic.sum().float() / (valid_bool.sum() + 1e-6)
                ).item(),
                "sparse_num_queries": valid_bool.sum().item(),
            }
            if valid_bool.sum() > 0:
                gt_abs_mag = gt_abs.norm(dim=-1)[valid_bool]
                pred_abs_mag = pred_abs.norm(dim=-1)[valid_bool]
                stats["sparse_gt_disp_mag"] = gt_abs_mag.mean().item()
                stats["sparse_pred_disp_mag"] = pred_abs_mag.mean().item()
                stats["sparse_l1_disp"] = _mean(l1_q, valid_bool).item()
                stats["sparse_mag_loss"] = _mean(mag_q, mask_dynamic).item() if mask_dynamic.sum() > 0 else 0.0
                # Key diagnostic: magnitude ratio (pred/gt) for dynamic points
                if mask_dynamic.sum() > 0:
                    dyn_pred_mag = pred_abs.norm(dim=-1)[mask_dynamic]
                    dyn_gt_mag = gt_abs.norm(dim=-1)[mask_dynamic]
                    ratio = (dyn_pred_mag / (dyn_gt_mag + 1e-6)).clamp(max=100.0)
                    stats["sparse_mag_ratio"] = ratio.mean().item()

        return final_loss, stats


class SparseDisplacementMSELoss(nn.Module):
    """MSE loss for sparse displacement prediction.
    
    MSE loss has larger gradients when error is large, leading to faster
    convergence in the early stages of training compared to L1 loss.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        dynamic_threshold: float = 0.02,
        balance_weight: float = 0.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.dynamic_threshold = dynamic_threshold
        self.balance_weight = balance_weight
        self.eps = eps
        
        logging.info(
            "[SparseDisplacementMSELoss] weight=%.2f, dyn_thresh=%.4f, balance=%.2f",
            loss_weight, dynamic_threshold, balance_weight,
        )

    def forward(
        self,
        pred_displacement: torch.Tensor,
        gt_displacement: torch.Tensor,
        valid_mask: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        pred_log_variance: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred_displacement: [B, Q, 3] predicted displacement
            gt_displacement: [B, Q, 3] ground truth displacement
            valid_mask: [B, Q] validity mask
            scale: [B, 1, 1] or [B] scene scale for absolute metrics
            pred_log_variance: unused, for API compatibility
        """
        B, Q, _ = pred_displacement.shape
        device = pred_displacement.device
        
        valid_bool = valid_mask > 0.5
        if valid_bool.sum() == 0:
            return pred_displacement.sum() * 0.0, {
                "sparse_disp_loss": 0.0, "sparse_num_queries": 0,
            }
        
        # Compute per-query MSE
        diff_sq = (pred_displacement - gt_displacement).pow(2).sum(dim=-1)  # [B, Q]
        
        # Motion magnitude for dynamic/static classification
        motion_mag = gt_displacement.norm(dim=-1)  # [B, Q]
        
        # Dynamic/static split
        is_dynamic = motion_mag > self.dynamic_threshold
        mask_dynamic = valid_bool & is_dynamic
        mask_static = valid_bool & ~is_dynamic
        
        # Compute losses
        def compute_masked_loss(mse_tensor, mask):
            if mask.sum() == 0:
                return torch.tensor(0.0, device=device)
            return mse_tensor[mask].mean()
        
        loss_dynamic = compute_masked_loss(diff_sq, mask_dynamic)
        loss_static = compute_masked_loss(diff_sq, mask_static)
        
        # Combine with balance weight
        if self.balance_weight > 0 and mask_dynamic.sum() > 0 and mask_static.sum() > 0:
            loss = self.balance_weight * loss_dynamic + (1 - self.balance_weight) * loss_static
        else:
            loss = diff_sq[valid_bool].mean()
        
        final_loss = loss * self.loss_weight
        
        # Stats
        with torch.no_grad():
            if scale is not None:
                # Align scale to [B] — handles multi-view tensors like [B, V, 1, 1, 1]
                if scale.ndim == 0 or scale.numel() == 1:
                    s = scale.view(1).expand(B)
                elif scale.shape[0] == B:
                    s_ref = scale[:, 0] if scale.ndim >= 2 else scale
                    s = s_ref.reshape(B, -1)[:, 0]
                else:
                    s_flat = scale.flatten()
                    s = s_flat if s_flat.numel() == B else s_flat.mean().expand(B)
                pred_abs = pred_displacement * s.view(B, 1, 1)
            else:
                pred_abs = pred_displacement

            stats = {
                "sparse_disp_loss": final_loss.item(),
                "sparse_dyn_ratio": (mask_dynamic.sum().float() / (valid_bool.sum() + 1e-6)).item(),
                "sparse_num_queries": valid_bool.sum().item(),
            }
            if valid_bool.sum() > 0:
                gt_abs = gt_displacement * s.view(B, 1, 1) if scale is not None else gt_displacement
                stats["sparse_gt_disp_mag"] = gt_abs.norm(dim=-1)[valid_bool].mean().item()
                stats["sparse_pred_disp_mag"] = pred_abs.norm(dim=-1)[valid_bool].mean().item()
                stats["sparse_mse"] = diff_sq[valid_bool].mean().item()
        
        return final_loss, stats


class SparseDisplacementHuberLoss(nn.Module):
    """Improved displacement loss that prevents zero-collapse.

    Two-part design:
    1. Huber (smooth-L1) base: linear gradients for large errors so that
       large-displacement samples stay impactful throughout training instead
       of being rapidly driven toward zero by MSE's quadratic growth.
    2. Relative L1 term (dynamic points only): penalises predictions that are
       a wrong *fraction* of the GT magnitude.  Specifically,
           rel = |pred - gt|_2 / (|gt|_2 + eps)
       is high (~1) when the model predicts near-zero for a large-motion
       query and drops to zero when prediction equals GT.  This directly
       counters the zero-collapse identified in v2 training.

    Loss formula:
        L_dyn  = (1 - rel_alpha) * huber(pred, gt) + rel_alpha * rel(pred, gt)
        L_stat = huber(pred, gt)
        L      = balance_weight * mean(L_dyn) + (1-balance_weight) * mean(L_stat)
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        dynamic_threshold: float = 0.02,
        balance_weight: float = 0.85,
        huber_delta: float = 0.1,
        rel_loss_alpha: float = 0.4,
        rel_loss_max: float = 10.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.dynamic_threshold = dynamic_threshold
        self.balance_weight = balance_weight
        self.huber_delta = huber_delta
        self.rel_loss_alpha = rel_loss_alpha
        self.rel_loss_max = rel_loss_max
        self.eps = eps

        logging.info(
            "[SparseDisplacementHuberLoss] weight=%.2f, dyn_thresh=%.4f, "
            "balance=%.2f, huber_delta=%.3f, rel_alpha=%.2f",
            loss_weight, dynamic_threshold, balance_weight, huber_delta, rel_loss_alpha,
        )

    def forward(
        self,
        pred_displacement: torch.Tensor,
        gt_displacement: torch.Tensor,
        valid_mask: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        pred_log_variance: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred_displacement: [B, Q, 3] predicted displacement (normalised)
            gt_displacement:   [B, Q, 3] GT displacement (normalised, same space)
            valid_mask:        [B, Q]    1 = valid, 0 = ignore
            scale:             optional, used only for logging world-space magnitudes
        """
        import torch.nn.functional as F_torch

        B, Q, _ = pred_displacement.shape
        device = pred_displacement.device

        valid_bool = valid_mask > 0.5
        if valid_bool.sum() == 0:
            return pred_displacement.sum() * 0.0, {
                "sparse_disp_loss": 0.0, "sparse_num_queries": 0,
            }

        # ── per-query Huber (sum over 3 spatial dims) ─────────────────────────
        huber_per_comp = F_torch.huber_loss(
            pred_displacement, gt_displacement,
            reduction='none', delta=self.huber_delta,
        )  # [B, Q, 3]
        huber_q = huber_per_comp.sum(dim=-1)  # [B, Q]

        # ── relative L1 per query ─────────────────────────────────────────────
        diff_norm = (pred_displacement - gt_displacement).norm(dim=-1)  # [B, Q]
        gt_norm   = gt_displacement.norm(dim=-1)                         # [B, Q]
        rel_q     = (diff_norm / (gt_norm + self.eps)).clamp(max=self.rel_loss_max)

        # ── dynamic/static split ──────────────────────────────────────────────
        is_dynamic   = gt_norm > self.dynamic_threshold
        mask_dynamic = valid_bool & is_dynamic
        mask_static  = valid_bool & ~is_dynamic

        def _mean(tensor, mask):
            if mask.sum() == 0:
                return torch.tensor(0.0, device=device)
            return tensor[mask].mean()

        # Dynamic: Huber + relative term combined
        dyn_loss_q   = (1.0 - self.rel_loss_alpha) * huber_q + self.rel_loss_alpha * rel_q
        loss_dynamic = _mean(dyn_loss_q, mask_dynamic)
        loss_static  = _mean(huber_q,    mask_static)

        if mask_dynamic.sum() > 0 and mask_static.sum() > 0:
            loss = self.balance_weight * loss_dynamic + (1.0 - self.balance_weight) * loss_static
        elif mask_dynamic.sum() > 0:
            loss = loss_dynamic
        else:
            loss = loss_static

        final_loss = loss * self.loss_weight

        # ── stats (world-space magnitudes for log readability) ─────────────────
        with torch.no_grad():
            s = None
            if scale is not None:
                if scale.ndim == 0 or scale.numel() == 1:
                    s = scale.view(1).expand(B)
                elif scale.shape[0] == B:
                    s_ref = scale[:, 0] if scale.ndim >= 2 else scale
                    s = s_ref.reshape(B, -1)[:, 0]
                else:
                    s_flat = scale.flatten()
                    s = s_flat if s_flat.numel() == B else s_flat.mean().expand(B)

            stats = {
                "sparse_disp_loss":   final_loss.item(),
                "sparse_dyn_ratio":   (mask_dynamic.sum().float() / (valid_bool.sum() + 1e-6)).item(),
                "sparse_num_queries": valid_bool.sum().item(),
            }
            if valid_bool.sum() > 0:
                gt_abs   = gt_displacement   * s.view(B, 1, 1) if s is not None else gt_displacement
                pred_abs = pred_displacement * s.view(B, 1, 1) if s is not None else pred_displacement
                stats["sparse_gt_disp_mag"]   = gt_abs.norm(dim=-1)[valid_bool].mean().item()
                stats["sparse_pred_disp_mag"] = pred_abs.norm(dim=-1)[valid_bool].mean().item()
                stats["sparse_huber"]         = _mean(huber_q, valid_bool).item()
                stats["sparse_rel_loss"]      = _mean(rel_q, mask_dynamic).item() if mask_dynamic.sum() > 0 else 0.0

        return final_loss, stats


class SparseDisplacementMoVieSLoss(nn.Module):
    """MoVieS-style sparse 3D displacement loss (query-based).

    Matches ``MotionLoss`` in MoVieS: per-query L1 on ``(Δx, Δy, Δz)`` with optional
    heteroscedastic-style confidence weighting::

        L = mean( conf * |pred - gt| - α * log(conf) )

    over all valid queries. Optional pairwise Gram-matrix term (same as MoVieS
    ``motion_dist_weight``) on a random subset of valid queries.

    ``pred_conf`` is expected positive (e.g. ``softplus`` head); internally we
    clamp before ``log`` for numerical stability.

    **Shapes:** ``valid_mask`` must be broadcast-compatible with ``[B, Q]`` (the
    query layout of ``pred_displacement``). A degenerate axis such as ``[B, 1, Q]``
    still broadcasts against ``[B, Q]`` masks to ``[B, Q, Q]``, which then breaks
    boolean indexing. Such layouts are squeezed here. A square ``pred_conf``
    tensor ``[B, Q, Q]`` (unexpected for scalar confidence) is reduced with
    ``diagonal`` so it cannot broadcast ``valid_mask`` the same way.

    Args:
        loss_weight: Scalar multiplier on the returned loss (same role as other
            sparse losses in this file).
        motion_conf_alpha: Coefficient on ``-log(conf)`` (MoVieS ``motion_conf_alpha``).
            ``0`` disables the log term (pure confidence-weighted L1 when conf is
            predicted; use ``pred_conf=None`` or all-ones for plain L1).
        motion_dist_weight: Weight on the optional pairwise distance (Gram) term.
        motion_dist_sample_number: Max queries subsampled for the Gram term.
        eps: Clamp for ``log(conf)``.
    """

    def __init__(
        self,
        loss_weight: float = 100.0,
        motion_conf_alpha: float = 0.0,
        motion_dist_weight: float = 0.0,
        motion_dist_sample_number: int = 1000,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.motion_conf_alpha = motion_conf_alpha
        self.motion_dist_weight = motion_dist_weight
        self.motion_dist_sample_number = int(motion_dist_sample_number)
        self.eps = eps

    def forward(
        self,
        pred_displacement: torch.Tensor,
        gt_displacement: torch.Tensor,
        valid_mask: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        pred_conf: Optional[torch.Tensor] = None,
        pred_log_variance: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        del scale, pred_log_variance, kwargs  # API compatibility with other sparse losses

        device = pred_displacement.device
        valid_bool = valid_mask > 0.5
        # Drop singleton axes like (B, 1, Q) so `&` with (B, Q) tensors never
        # widens to (B, Q, Q) (PyTorch broadcast rules).
        if valid_bool.ndim == 3:
            if valid_bool.shape[1] == 1:
                valid_bool = valid_bool.squeeze(1)
            elif valid_bool.shape[-1] == 1:
                valid_bool = valid_bool.squeeze(-1)
        pred_finite = torch.isfinite(pred_displacement).all(dim=-1)
        gt_finite = torch.isfinite(gt_displacement).all(dim=-1)
        valid_bool = valid_bool & pred_finite & gt_finite

        if pred_conf is not None:
            if pred_conf.ndim == 4 and pred_conf.shape[-1] == 1:
                pred_conf = pred_conf.squeeze(-1)
            while pred_conf.ndim > 2 and pred_conf.shape[-1] == 1:
                pred_conf = pred_conf.squeeze(-1)
            # Do **not** use ``pred_conf.shape[:2] == valid_bool.shape``: for e.g.
            # ``pred_conf`` (B, Q, Q) that is wrong but common when Q == N, the first
            # two dimensions still match (B, Q), and ``valid_bool & cf_ok`` would
            # broadcast ``valid_bool`` (B, Q) to (B, Q, Q), breaking
            # ``pred_displacement[valid_bool]``.
            if pred_conf.ndim > 2:
                if (
                    pred_conf.shape[0] == valid_bool.shape[0]
                    and pred_conf.shape[1] == valid_bool.shape[1]
                ):
                    if pred_conf.shape[-1] == pred_conf.shape[-2]:
                        pred_conf = torch.diagonal(pred_conf, dim1=-2, dim2=-1)
                    else:
                        logging.warning(
                            "SparseDisplacementMoVieSLoss: pred_conf rank=%d shape=%s; "
                            "using [... , 0] to align with valid_mask %s",
                            pred_conf.ndim,
                            tuple(pred_conf.shape),
                            tuple(valid_bool.shape),
                        )
                        pred_conf = pred_conf[..., 0]
                else:
                    logging.warning(
                        "SparseDisplacementMoVieSLoss: pred_conf shape=%s incompatible "
                        "with valid_mask %s; dropping conf",
                        tuple(pred_conf.shape),
                        tuple(valid_bool.shape),
                    )
                    pred_conf = None
            if pred_conf is not None:
                cf_ok = torch.isfinite(pred_conf) & (pred_conf > 0)
                if pred_conf.shape == valid_bool.shape:
                    valid_bool = valid_bool & cf_ok

        if valid_bool.sum() == 0:
            z = pred_displacement.sum() * 0.0
            return z, {
                "sparse_movies_motion_loss": 0.0,
                "sparse_num_queries": 0,
            }

        pred_v = pred_displacement[valid_bool]
        gt_v = gt_displacement[valid_bool]
        l1 = torch.abs(pred_v - gt_v)

        if pred_conf is not None:
            c = pred_conf[valid_bool]
            if c.dim() > 1:
                c = c.reshape(-1)
            c = c.unsqueeze(-1).expand_as(l1)
        else:
            c = torch.ones_like(l1)

        c_safe = c.clamp(min=self.eps)
        weighted = c * l1 - self.motion_conf_alpha * torch.log(c_safe)
        loss = weighted.mean()

        if self.motion_dist_weight > 0.0 and pred_v.shape[0] > 1:
            m = pred_v.shape[0]
            if m > self.motion_dist_sample_number:
                idx = torch.randperm(m, device=device)[: self.motion_dist_sample_number]
            else:
                idx = torch.arange(m, device=device)
            p = pred_v[idx]
            g = gt_v[idx]
            pred_mm = p @ p.T
            gt_mm = g @ g.T
            dist_loss = torch.abs(pred_mm - gt_mm).mean()
            loss = loss + self.motion_dist_weight * dist_loss
        else:
            dist_loss = torch.tensor(0.0, device=device)

        if torch.isnan(loss) or torch.isinf(loss):
            loss = pred_displacement.sum() * 0.0
            logging.warning("SparseDisplacementMoVieSLoss: NaN/Inf detected")

        final_loss = loss * self.loss_weight
        with torch.no_grad():
            stats = {
                "sparse_movies_motion_loss": final_loss.item(),
                "sparse_num_queries": int(valid_bool.sum().item()),
            }
            if self.motion_dist_weight > 0.0:
                stats["sparse_movies_dist_loss"] = float(
                    (dist_loss * self.motion_dist_weight).item()
                )
        return final_loss, stats


class SparseDisplacementLossUncertainty(SparseDisplacementLoss):
    """Displacement loss with heteroscedastic uncertainty weighting.
    
    Uses predicted log-variance to weight the loss per-query:
        L = exp(-log_var) * L1 + reg_weight * log_var
    
    This allows the model to predict higher uncertainty for difficult
    cases (occlusions, textureless regions) and focus learning on
    confident predictions.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        dynamic_threshold: float = 0.02,
        balance_weight: float = 0.5,
        detach_log_scale: bool = False,
        use_heteroscedastic: bool = True,
        uncertainty_regularization: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            loss_weight=loss_weight,
            dynamic_threshold=dynamic_threshold,
            balance_weight=balance_weight,
            detach_log_scale=detach_log_scale,
            use_uncertainty=use_heteroscedastic,
            uncertainty_weight=uncertainty_regularization,
            **kwargs,
        )
        self.use_heteroscedastic = use_heteroscedastic


class CycleConsistencyLoss(nn.Module):
    """Cycle consistency loss for bidirectional motion.
    
    Penalizes inconsistency: |forward_motion + backward_motion|
    Points with large inconsistency are likely occluded.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        threshold: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.threshold = threshold
        self.eps = eps

    def forward(
        self,
        forward_displacement: torch.Tensor,
        backward_displacement: torch.Tensor,
        valid_mask: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute cycle consistency loss.
        
        Args:
            forward_displacement: [B, N, 3] forward motion (src→tgt).
            backward_displacement: [B, N, 3] backward motion (tgt→src).
            valid_mask: [B, N] validity mask.
            
        Returns:
            Loss tensor and statistics dict.
        """
        device = forward_displacement.device
        valid_bool = valid_mask > 0.5
        
        if valid_bool.sum() == 0:
            return forward_displacement.sum() * 0.0, {"cycle_loss": 0.0}
        
        cycle_error = (forward_displacement + backward_displacement).norm(dim=-1)
        
        consistent_mask = valid_bool & (cycle_error < self.threshold)
        
        if consistent_mask.sum() == 0:
            loss = torch.tensor(0.0, device=device)
        else:
            loss = cycle_error[consistent_mask].mean()
        
        final_loss = loss * self.loss_weight
        
        with torch.no_grad():
            stats = {
                "cycle_loss": final_loss.item(),
                "cycle_error_mean": cycle_error[valid_bool].mean().item() if valid_bool.sum() > 0 else 0.0,
                "cycle_consistent_ratio": (consistent_mask.sum().float() / (valid_bool.sum() + self.eps)).item(),
            }
        
        return final_loss, stats


class SparseDirectionLoss(Loss):
    """Direction + static-magnitude loss for SphericalHead.

    Uses :meth:`Loss.get_loss_weight` on the dataset ``name`` (same convention as other
    query motion losses): scale tuning lives in ``loss_weight`` in the pipeline config.

    Two complementary terms:

    1. **Dynamic points** (GT mag > ``mag_threshold``):
       ``1 - cosine_similarity(pred_dir, gt_dir)`` — supervises direction
       branch only, gradient does not flow through magnitude.

    2. **Static points** (GT mag ≤ ``mag_threshold``):
       ``mean(pred_magnitude)`` — pushes magnitude branch toward zero.
       Uses ``displacement_magnitude`` directly so gradient flows
       exclusively through the magnitude branch, not direction.

    Together they ensure both branches of SphericalHead receive clean
    supervision for all point types.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        mag_threshold: float = 0.02,
        static_mag_weight: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__(loss_weight=loss_weight)
        self.mag_threshold = mag_threshold
        self.static_mag_weight = static_mag_weight
        self.eps = eps

    def _resolve_scale(self, scale: Optional[torch.Tensor], B: int) -> Optional[torch.Tensor]:
        if scale is None:
            return None
        if scale.ndim == 0 or scale.numel() == 1:
            return scale.view(1).expand(B)
        if scale.shape[0] == B:
            s_ref = scale[:, 0] if scale.ndim >= 2 else scale
            return s_ref.reshape(B, -1)[:, 0]
        s_flat = scale.flatten()
        return s_flat if s_flat.numel() == B else s_flat.mean().expand(B)

    def forward(
        self,
        pred_direction: torch.Tensor,       # [B, Q, 3] unit vectors
        gt_displacement: torch.Tensor,       # [B, Q, 3] GT displacement
        valid_mask: torch.Tensor,            # [B, Q]
        scale: Optional[torch.Tensor] = None,
        pred_magnitude: Optional[torch.Tensor] = None,  # [B, Q, 1]
        name=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        lw = self.get_loss_weight(name)
        if lw == 0:
            z = pred_direction.sum() * 0.0
            return z, {}

        B, Q, _ = pred_direction.shape
        per_q_w: Optional[torch.Tensor] = kwargs.get("per_query_weight")
        if per_q_w is not None:
            if per_q_w.shape[:2] != (B, Q):
                raise ValueError(
                    f"per_query_weight must align with [B, Q]=[{B}, {Q}], "
                    f"got {tuple(per_q_w.shape)}."
                )
            per_q_w = per_q_w.to(device=pred_direction.device, dtype=pred_direction.dtype)

        valid_bool = valid_mask > 0.5
        valid_bool = valid_bool & torch.isfinite(pred_direction).all(dim=-1)
        valid_bool = valid_bool & torch.isfinite(gt_displacement).all(dim=-1)

        # See SparseDisplacementLoss for rationale: prevent NaN/Inf in pred or
        # gt from contaminating the autograd graph even at masked-out positions.
        pred_direction = torch.nan_to_num(
            pred_direction, nan=0.0, posinf=0.0, neginf=0.0
        )
        gt_displacement = torch.nan_to_num(
            gt_displacement, nan=0.0, posinf=0.0, neginf=0.0
        )
        if pred_magnitude is not None:
            pred_magnitude = torch.nan_to_num(
                pred_magnitude, nan=0.0, posinf=0.0, neginf=0.0
            )

        gt_mag = gt_displacement.norm(dim=-1)

        s = self._resolve_scale(scale, B)
        gt_abs_mag = gt_mag * s.view(B, 1) if s is not None else gt_mag

        dynamic_mask = valid_bool & (gt_abs_mag > self.mag_threshold)
        static_mask = valid_bool & (gt_abs_mag <= self.mag_threshold)

        # ── Direction loss on dynamic points ──────────────────────────────
        # Perf: F.normalize + dot only on dynamic queries (skip static B*Q).
        # pred_direction is already unit length from SphericalHead; GT unit
        # vector from F.normalize(gt_disp) makes cosine = elementwise dot.
        dir_loss = pred_direction.sum() * 0.0
        cos_sim_mean_t = None
        n_dynamic = int(dynamic_mask.sum().item())

        if n_dynamic > 0:
            pred_dyn = pred_direction[dynamic_mask]
            gt_dir_dyn = F.normalize(
                gt_displacement[dynamic_mask], dim=-1, eps=self.eps,
            )
            # Numerically match ``F.cosine_similarity(pred, gt_dir, dim=-1, eps=eps)``:
            # dot / (||pred|| * ||gt||).clamp_min(eps).  (``gt_dir`` is ~unit but we
            # still use its norm like PyTorch does.)
            dot = (pred_dyn * gt_dir_dyn).sum(dim=-1)
            # Default ``eps`` for ``F.cosine_similarity`` is 1e-8 (config ``self.eps``
            # is only for ``F.normalize`` on GT). Use ``sqrt(sum(x²) + eps²)`` rather
            # than ``torch.norm + clamp_min`` to keep the backward finite when
            # ``pred_dyn`` is exactly zero (``torch.norm`` is 0/0 at zero input and
            # clamp saturation does not protect the chain rule).
            _cos_eps = 1e-8
            pred_norm = (
                (pred_dyn * pred_dyn).sum(dim=-1) + _cos_eps * _cos_eps
            ).sqrt()
            gt_norm = (
                (gt_dir_dyn * gt_dir_dyn).sum(dim=-1) + _cos_eps * _cos_eps
            ).sqrt()
            denom = pred_norm * gt_norm
            cos_sim = dot / denom
            d_err = 1.0 - cos_sim
            if per_q_w is not None:
                w = per_q_w[dynamic_mask]
                w = w / w.sum().clamp(min=self.eps)
                dl = (d_err * w).sum()
            else:
                dl = d_err.mean()
            if torch.isfinite(dl):
                dir_loss = dl
                cos_sim_mean_t = cos_sim.mean()
            else:
                logging.warning("SparseDirectionLoss: NaN/Inf in direction term")

        # ── Magnitude regularisation on static points ─────────────────────
        static_mag_loss = pred_direction.sum() * 0.0
        n_static = int(static_mask.sum().item())

        if (
            n_static > 0
            and self.static_mag_weight > 0.0
            and pred_magnitude is not None
        ):
            # Mean over static queries; softplus(mag) is finite in normal runs.
            pm = pred_magnitude.squeeze(-1)[static_mask]
            if per_q_w is not None:
                w = per_q_w[static_mask]
                w = w / w.sum().clamp(min=self.eps)
                static_mag_loss = (pm * w).sum()
            else:
                static_mag_loss = pm.mean()

        final_loss = (
            dir_loss * lw + static_mag_loss * self.static_mag_weight
        )

        with torch.no_grad():
            stats = {
                "dir_loss": (dir_loss * lw).item(),
                "dir_cos_sim": (
                    float(cos_sim_mean_t.item())
                    if cos_sim_mean_t is not None
                    else 0.0
                ),
                "dir_num_queries": n_dynamic,
                "static_mag_loss": (static_mag_loss * self.static_mag_weight).item(),
                "static_num_queries": n_static,
            }

        return final_loss, stats


class SpatialSmoothnessLoss(nn.Module):
    """Spatial smoothness loss: nearby points should have similar motion."""

    def __init__(
        self,
        loss_weight: float = 1.0,
        k_neighbors: int = 8,
        sigma: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.k_neighbors = k_neighbors
        self.sigma = sigma
        self.eps = eps

    def forward(
        self,
        pred_displacement: torch.Tensor,
        uv: torch.Tensor,
        valid_mask: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute spatial smoothness loss.
        
        Args:
            pred_displacement: [B, N, 3] predicted displacements.
            uv: [B, N, 2] query UV coordinates.
            valid_mask: [B, N] validity mask.
            
        Returns:
            Loss tensor and statistics dict.
        """
        B, N, _ = pred_displacement.shape
        device = pred_displacement.device
        valid_bool = valid_mask > 0.5
        
        if valid_bool.sum() < 2:
            return pred_displacement.sum() * 0.0, {"spatial_smooth_loss": 0.0}
        
        dist = torch.cdist(uv, uv)
        _, indices = dist.topk(self.k_neighbors + 1, dim=-1, largest=False)
        indices = indices[:, :, 1:]
        
        weights = torch.exp(-dist.gather(-1, indices) / (2 * self.sigma ** 2))
        weights = weights / (weights.sum(dim=-1, keepdim=True) + self.eps)
        
        idx_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, 3)
        neighbor_disp = pred_displacement.unsqueeze(1).expand(-1, N, -1, -1).gather(2, idx_expanded)
        
        center_disp = pred_displacement.unsqueeze(2)
        diff = (center_disp - neighbor_disp).norm(dim=-1)
        
        weighted_diff = (diff * weights).sum(dim=-1)
        
        valid_expanded = valid_bool.float()
        loss = (weighted_diff * valid_expanded).sum() / (valid_expanded.sum() + self.eps)
        
        final_loss = loss * self.loss_weight
        
        with torch.no_grad():
            stats = {
                "spatial_smooth_loss": final_loss.item(),
            }
        
        return final_loss, stats


class TemporalSmoothnessLoss(nn.Module):
    """Temporal smoothness loss: penalize acceleration (2nd order) or jerk (3rd order)."""

    def __init__(
        self,
        loss_weight: float = 1.0,
        order: int = 2,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.order = order
        self.eps = eps

    def forward(
        self,
        pred_trajectory: torch.Tensor,
        valid_mask: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute temporal smoothness loss.
        
        Args:
            pred_trajectory: [B, T, N, 3] predicted trajectory over T frames.
            valid_mask: [B, T, N] validity mask.
            
        Returns:
            Loss tensor and statistics dict.
        """
        device = pred_trajectory.device
        B, T, N, _ = pred_trajectory.shape
        
        if T < self.order + 1:
            return pred_trajectory.sum() * 0.0, {"temporal_smooth_loss": 0.0}
        
        diff = pred_trajectory
        for _ in range(self.order):
            diff = diff[:, 1:] - diff[:, :-1]
        
        diff_norm = diff.norm(dim=-1)
        
        valid_diff = valid_mask[:, self.order:]
        valid_count = valid_diff.sum()
        
        if valid_count == 0:
            return pred_trajectory.sum() * 0.0, {"temporal_smooth_loss": 0.0}
        
        loss = (diff_norm * valid_diff.float()).sum() / (valid_count + self.eps)
        
        final_loss = loss * self.loss_weight
        
        with torch.no_grad():
            stats = {
                "temporal_smooth_loss": final_loss.item(),
            }
        
        return final_loss, stats