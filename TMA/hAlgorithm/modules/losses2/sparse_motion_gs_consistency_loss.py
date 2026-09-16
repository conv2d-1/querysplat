"""Consistency between sparse motion-branch and dense Gaussian-branch displacements."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from hAlgorithm.modules.losses2.base import Loss
from hAlgorithm.modules.pipelines2.utils.sparse_pair_gaussian_utils import (
    _motion_uv_at_ref,
    canonicalize_scene_flow,
)


def snap_motion_uv_to_gaussian_indices(
    motion_uv: Tensor,
    gaussian_uv: Tensor,
    ref_frame: int,
    width: int,
    height: int,
    full_uv: bool,
    batch_size: int = 1,
    num_views: Optional[int] = None,
) -> Tensor:
    """Map motion query UVs to nearest dense Gaussian query indices.

    Args:
        motion_uv: ``[B, T, Qm, 2]``, ``[B, Qm, 2]``, ``[T, Qm, 2]``, or ``[Qm, 2]``.
        gaussian_uv: ``[Qg, 2]`` dense query UVs.
        ref_frame: Reference frame index when ``motion_uv`` is 4-D.
        width, height: Input resolution for full-UV grid snapping.
        full_uv: Whether ``gaussian_uv`` is a full H×W grid (raster order).
        batch_size: Batch size for ``motion_flow`` / gather.
        num_views: Clip length when ``motion_uv`` is ``[T, Qm, 2]``.

    Returns:
        ``[B, Qm]`` long indices into the dense Gaussian query axis.
    """
    muv = _motion_uv_at_ref(
        motion_uv,
        ref_frame=ref_frame,
        batch_size=batch_size,
        num_views=num_views,
    )
    b, qm, _ = muv.shape
    if full_uv and width > 0 and height > 0:
        ix = (muv[..., 0] * max(width - 1, 1)).round().long().clamp(0, width - 1)
        iy = (muv[..., 1] * max(height - 1, 1)).round().long().clamp(0, height - 1)
        return (iy * width + ix).long()

    guv = gaussian_uv.float()
    if guv.dim() == 2:
        guv = guv.unsqueeze(0).expand(b, -1, -1)
    gw = width if width > 0 else int(guv.shape[1] ** 0.5)
    gh = height if height > 0 else gw
    if gw > 1 and gh > 1 and guv.shape[1] == gw * gh:
        ix = (muv[..., 0] * (gw - 1)).round().long().clamp(0, gw - 1)
        iy = (muv[..., 1] * (gh - 1)).round().long().clamp(0, gh - 1)
        return (iy * gw + ix).long()

    diff = muv.unsqueeze(2) - guv.unsqueeze(1)
    dist = (diff ** 2).sum(dim=-1)
    return dist.argmin(dim=-1).long()


class SparseMotionGsDisplacementConsistencyLoss(Loss):
    """Align dense GS scene flow with sparse motion-branch flow at motion UVs.

    Only computed when dual-query outputs expose both
    ``motion_sparse_scene_flow`` and ``sparse_scene_flow``.
    """

    def __init__(
        self,
        loss_weight: float = 0.25,
        dynamic_threshold: float = 0.02,
        huber_delta: float = 0.05,
    ):
        super().__init__(loss_weight=loss_weight)
        self.dynamic_threshold = dynamic_threshold
        self.huber_delta = huber_delta

    def forward(
        self,
        results: dict,
        motion_query,
        gaussian_query,
        meta_data: dict,
        valid_mask: Optional[Tensor] = None,
        name: str | None = None,
        **kwargs,
    ) -> Tuple[Tensor, Dict[str, float]]:
        lw = self.get_loss_weight(name)
        zero = results["sparse_scene_flow"].sum() * 0.0
        if lw == 0:
            return zero, {}

        motion_flow = results.get("motion_sparse_scene_flow")
        gs_flow = results.get("sparse_scene_flow")
        if motion_flow is None or gs_flow is None:
            return zero, {}
        if motion_query is None or gaussian_query is None:
            return zero, {}

        ref_frame = int(results.get("src_frame_idx", 0))
        motion_flow = canonicalize_scene_flow(motion_flow.float())
        gs_flow = canonicalize_scene_flow(gs_flow.float())
        b, n_views, qm_flow, _ = motion_flow.shape

        width = int(meta_data["input_width"][0])
        height = int(meta_data["input_height"][0])
        full_uv = bool(getattr(gaussian_query, "full_uv", True))
        g_uv = gaussian_query.uv
        if g_uv.dim() == 4:
            g_uv = g_uv[:, ref_frame]
        if g_uv.dim() == 3:
            g_uv = g_uv[0]

        idx = snap_motion_uv_to_gaussian_indices(
            motion_query.uv,
            g_uv,
            ref_frame=ref_frame,
            width=width,
            height=height,
            full_uv=full_uv,
            batch_size=b,
            num_views=n_views,
        )

        qm = min(qm_flow, idx.shape[1], motion_flow.shape[2])
        if idx.shape[1] != qm:
            idx = idx[:, :qm]
        motion_flow = motion_flow[:, :, :qm]
        idx = idx.clamp(0, gs_flow.shape[2] - 1)
        gs_at_motion = torch.gather(
            gs_flow,
            2,
            idx.unsqueeze(1).unsqueeze(-1).expand(b, n_views, qm, 3),
        )

        diff = motion_flow - gs_at_motion
        per_q_mag = motion_flow.norm(dim=-1)
        mask = per_q_mag > self.dynamic_threshold
        if valid_mask is not None:
            if valid_mask.shape[-1] > qm:
                valid_mask = valid_mask[..., :qm]
            if valid_mask.dim() == 3:
                mask = mask & (valid_mask[:, ref_frame] > 0.5)
            elif valid_mask.dim() == 2:
                mask = mask & (valid_mask > 0.5)

        if mask.sum() < 1:
            return zero, {
                "motion_gs_consistency": 0.0,
                "_motion_gs_consistency_mask": 0,
                "_motion_gs_consistency_diff": 0.0,
            }

        err = F.smooth_l1_loss(
            diff[mask],
            torch.zeros_like(diff[mask]),
            beta=self.huber_delta,
            reduction="mean",
        )
        err = err * lw
        with torch.no_grad():
            diff_mean = diff[mask].abs().mean().item()
            mask_count = int(mask.sum().item())
        return err, {
            "motion_gs_consistency": round(err.item(), 4),
            "_motion_gs_consistency_mask": mask_count,
            "_motion_gs_consistency_diff": round(diff_mean, 4),
        }
