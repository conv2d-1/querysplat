"""
Shared Motion Utilities
========================
Common functions for scene flow / dynamic pointmap processing,
used by both MoviesPipeline and MVFRMotionPipeline (and any future
motion pipeline) to ensure consistent behavior.

Functions are pure (no self/pipeline dependency) so they can be
imported and called from any pipeline class.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Decoupled pair head (mlp_head direction + magnitude -> warp3d_delta)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_scene_flow_keys_inplace(out: Optional[Dict[str, Any]]) -> None:
    """Mirror canonical ``warp3d_delta*`` tensors under legacy ``displacement*`` names."""
    if not out:
        return
    if "warp3d_delta" in out:
        out.setdefault("displacement", out["warp3d_delta"])
    elif "displacement" in out:
        out.setdefault("warp3d_delta", out["displacement"])
    if "warp3d_delta_direction" in out:
        out.setdefault("displacement_direction", out["warp3d_delta_direction"])
    elif "displacement_direction" in out:
        out.setdefault("warp3d_delta_direction", out["displacement_direction"])
    if "warp3d_delta_magnitude" in out:
        out.setdefault("displacement_magnitude", out["warp3d_delta_magnitude"])
    elif "displacement_magnitude" in out:
        out.setdefault("warp3d_delta_magnitude", out["displacement_magnitude"])


def finalize_decoupled_warp3d_mlp_logits_inplace(results: Optional[Dict[str, Any]]) -> None:
    """Scene-flow branch only: no in-place transforms here.

    ``warp3d`` (world point) and ``warp3d_delta`` (scene flow) are independent.  Direction /
    magnitude activations for the delta path are owned by the head (e.g. ``mlp_head`` ``norm`` /
    ``softplus`` on ``warp3d_delta_*``).  This helper intentionally does **not** rewrite tensors.

    Exits early when ``warp3d_delta`` already exists or when canonical
    ``warp3d_delta_direction`` + ``warp3d_delta_magnitude`` are present (nothing to do).
    """
    if not results:
        return
    if results.get("warp3d_delta") is not None:
        return
    if results.get("warp3d_delta_direction") is not None and results.get(
        "warp3d_delta_magnitude"
    ) is not None:
        return


def inject_warp3d_delta_from_decoupled_mlp_head(results: Optional[Dict[str, Any]]) -> None:
    """Compose ``warp3d_delta`` from ``warp3d_delta_direction`` * ``warp3d_delta_magnitude`` only.

    Does **not** apply activations or normalization (head must already produce valid tensors).
    Ignores ``warp3d`` / world-point outputs: only the scene-flow delta keys participate.

    Writes ``warp3d_delta``, mirrors ``warp3d_delta_*`` if missing, and runs
    :func:`normalize_scene_flow_keys_inplace`.
    """
    if not results:
        return
    if results.get("warp3d_delta") is not None:
        return
    if results.get("warp3d_delta_direction") is not None and results.get(
        "warp3d_delta_magnitude"
    ) is not None:
        direction = results["warp3d_delta_direction"]
        mag = results["warp3d_delta_magnitude"]
    else:
        return
    results["warp3d_delta"] = direction * mag
    results.setdefault("warp3d_delta_direction", direction)
    results.setdefault("warp3d_delta_magnitude", mag)
    normalize_scene_flow_keys_inplace(results)


def prepare_decoupled_warp3d_delta_inplace(results: Optional[Dict[str, Any]]) -> None:
    """Compose scene-flow ``warp3d_delta`` from canonical ``warp3d_delta_direction`` / magnitude."""
    finalize_decoupled_warp3d_mlp_logits_inplace(results)
    inject_warp3d_delta_from_decoupled_mlp_head(results)


# ─────────────────────────────────────────────────────────────────────────────
# Scale helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_ref_scale(scale: torch.Tensor) -> torch.Tensor:
    """Extract reference view (view 0) scale for trajectory normalization.

    Args:
        scale: [B, V, 1, 1, 1]
    Returns:
        [B, 1, 1, 1] reference scale tensor (broadcastable to [B, V, N, 3])
    """
    ref = scale[:, 0:1, :, :, :]        # [B, 1, 1, 1, 1]
    return ref.view(ref.shape[0], 1, 1, 1)  # [B, 1, 1, 1]


def get_ref_scale_flat(scale: torch.Tensor) -> torch.Tensor:
    """Extract reference view scale as [B, 1] for broadcasting to [B, V].

    Args:
        scale: [B, V, 1, 1, 1]
    Returns:
        [B, 1] reference scale tensor
    """
    return scale[:, 0, 0, 0, 0].unsqueeze(1)  # [B, 1]


def align_scale_for_loss(scale: Optional[torch.Tensor],
                         target_batch_size: int) -> Optional[torch.Tensor]:
    """Align scale tensor to 1-D [target_batch_size] for loss computation.

    Handles multi-view scale by taking reference view (view 0).
    """
    if scale is None:
        return None

    if scale.dim() == 5 and scale.shape[1] > 1:
        scale = scale[:, 0]  # [B, 1, 1, 1]

    scale_flat = scale.flatten()
    n = scale_flat.shape[0]

    if n == 1:
        return scale_flat.expand(target_batch_size)
    elif n == target_batch_size:
        return scale_flat
    else:
        logger.warning(f"[Scale Align] Unexpected shape {scale.shape}, using mean")
        return scale_flat.mean().expand(target_batch_size)


# ─────────────────────────────────────────────────────────────────────────────
# GT preparation  (train time)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_scene_flow_gt(
    trajs_3d: torch.Tensor,       # [B, V, N, 3]  world coords (normalized)
    trajs_2d: torch.Tensor,       # [B, V, N, 2]  pixel coords
    visibs: torch.Tensor,         # [B, V, N]
    valids: Optional[torch.Tensor],
    extrinsics: Optional[torch.Tensor],  # [B, V, 4, 4]  w2c
    src_idx: torch.Tensor,        # [V_tgt]  source frame indices
    tgt_idx: torch.Tensor,        # [V_tgt]  target frame indices
    coord_frame_idx: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare ground truth for **scene flow** loss.

    Transforms world-space trajectories into camera coordinates.

    Args:
        coord_frame_idx: If specified, the world→camera transform always uses
            this frame's extrinsics (e.g. 0 to express all displacements in
            frame-0's camera coordinate system, regardless of which frame is
            the source).  When None (default), the source frame's own
            extrinsics are used (original behavior).

    Returns:
        src_cam:    [B, V_tgt, N, 3]  source positions in camera coords
        tgt_cam:    [B, V_tgt, N, 3]  target positions in camera coords
        src_2d:     [B, V_tgt, N, 2]  source 2D positions (pixel coords)
        valid_mask: [B, V_tgt, N]     combined visibility & validity mask
    """
    B = trajs_3d.shape[0]
    src_3d = trajs_3d[:, src_idx]   # [B, V_tgt, N, 3]
    tgt_3d = trajs_3d[:, tgt_idx]
    src_2d = trajs_2d[:, src_idx]

    valid_mask = torch.logical_and(visibs, valids if valids is not None else visibs).float()
    valid_mask = valid_mask[:, src_idx] * valid_mask[:, tgt_idx]

    if extrinsics is not None:
        if extrinsics.shape[0] != B:
            extrinsics = extrinsics.expand(B, -1, -1, -1)

        if coord_frame_idx is not None:
            # Fixed coordinate frame: always use frame coord_frame_idx's
            # extrinsics so that displacement vectors are expressed in a
            # consistent camera coordinate system across all source frames.
            V_tgt = src_idx.shape[0]
            coord_idx = torch.full(
                (V_tgt,), coord_frame_idx,
                dtype=torch.long, device=extrinsics.device,
            )
            R = extrinsics[:, coord_idx, :3, :3]   # [B, V_tgt, 3, 3]
            T = extrinsics[:, coord_idx, :3, 3]     # [B, V_tgt, 3]
        else:
            # Original behavior: use the source frame's own coordinate system
            R = extrinsics[:, src_idx, :3, :3]
            T = extrinsics[:, src_idx, :3, 3]

        cam_disp = torch.einsum('bvij,bvnj->bvni', R, tgt_3d - src_3d)
        src_cam = torch.einsum('bvij,bvnj->bvni', R, src_3d) + T.unsqueeze(2)
        tgt_cam = src_cam + cam_disp
    else:
        src_cam, tgt_cam = src_3d, tgt_3d

    return src_cam, tgt_cam, src_2d, valid_mask


def prepare_dynamic_pointmap_gt(
    trajs_3d: torch.Tensor,
    trajs_2d: torch.Tensor,
    visibs: torch.Tensor,
    valids: Optional[torch.Tensor],
    extrinsics: Optional[torch.Tensor],
    src_idx: torch.Tensor,
    tgt_idx: torch.Tensor,
    original_h: Optional[int] = None,
    original_w: Optional[int] = None,
    target_depth: Optional[torch.Tensor] = None,
    coord_frame_idx: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare ground truth for **dynamic pointmap** loss.

    Same coordinate transform as scene flow GT, but both src and tgt
    positions are independently transformed (not via displacement).

    Args:
        coord_frame_idx: See prepare_scene_flow_gt.

    Returns:
        src_cam, tgt_cam, src_2d, valid_mask  (same shapes as scene flow GT)
    """
    B = trajs_3d.shape[0]
    src_3d = trajs_3d[:, src_idx]
    tgt_3d = trajs_3d[:, tgt_idx]
    src_2d = trajs_2d[:, src_idx]

    valid_mask = torch.logical_and(visibs, valids if valids is not None else visibs).float()
    valid_mask = valid_mask[:, src_idx] * valid_mask[:, tgt_idx]

    if extrinsics is not None:
        if extrinsics.shape[0] != B:
            extrinsics = extrinsics.expand(B, -1, -1, -1)

        if coord_frame_idx is not None:
            V_tgt = src_idx.shape[0]
            coord_idx = torch.full(
                (V_tgt,), coord_frame_idx,
                dtype=torch.long, device=extrinsics.device,
            )
            R = extrinsics[:, coord_idx, :3, :3]
            T = extrinsics[:, coord_idx, :3, 3]
        else:
            R = extrinsics[:, src_idx, :3, :3]
            T = extrinsics[:, src_idx, :3, 3]

        src_cam = torch.einsum('bvij,bvnj->bvni', R, src_3d) + T.unsqueeze(2)
        tgt_cam = torch.einsum('bvij,bvnj->bvni', R, tgt_3d) + T.unsqueeze(2)
    else:
        src_cam, tgt_cam = src_3d, tgt_3d
        logger.warning("[Dynamic Pointmap GT] No extrinsics provided, using trajs_3d directly")

    # Optional debug verification
    if target_depth is not None and original_h is not None and original_w is not None:
        verify_trajs3d_pointmap_consistency(
            tgt_cam, src_2d, target_depth, valid_mask,
            original_h, original_w, B, trajs_3d.device,
        )

    return src_cam, tgt_cam, src_2d, valid_mask


# ─────────────────────────────────────────────────────────────────────────────
# Validation & verification
# ─────────────────────────────────────────────────────────────────────────────

def validate_motion_inputs(
    pred_raw: Optional[torch.Tensor],
    trajs_3d: Optional[torch.Tensor],
    trajs_2d: Optional[torch.Tensor],
    visibs: Optional[torch.Tensor],
    valids: Optional[torch.Tensor],
) -> Tuple[Optional[Tuple], Optional[Tuple], Optional[int]]:
    """Validate motion loss inputs and return dimensions.

    Returns:
        pred_dims:  (B_pred, V_src, V_tgt)  or None
        traj_dims:  (B_traj, V_traj, N_tracks)  or None
        B:          effective batch size  or None
    """
    if pred_raw is None or trajs_3d is None or trajs_2d is None:
        return None, None, None

    B_pred, V_src, V_tgt = pred_raw.shape[:3]
    B_traj, V_traj, N_tracks = trajs_3d.shape[:3]

    errors = []
    if V_src != 1:
        errors.append(f"Expected V_src=1 for query-based training, got {V_src}")
    if V_tgt != V_traj:
        errors.append(f"Frame mismatch: pred={V_tgt}, traj={V_traj}")
    if trajs_3d.dim() != 4:
        errors.append(
            f"trajs_3d expected 4D [B, V, N, 3], got {trajs_3d.dim()}D {trajs_3d.shape}. "
            f"Check scale broadcasting in get_inputs."
        )
    if trajs_2d.dim() != 4:
        errors.append(f"trajs_2d expected 4D [B, V, N, 2], got {trajs_2d.dim()}D {trajs_2d.shape}.")
    if B_pred != B_traj and B_pred != 1 and B_traj != 1:
        errors.append(f"Batch mismatch: pred={B_pred}, traj={B_traj}")

    if errors:
        for err in errors:
            logger.error(f"[Motion Loss] {err}")
        return None, None, None

    return (B_pred, V_src, V_tgt), (B_traj, V_traj, N_tracks), max(B_pred, B_traj)


def verify_trajs3d_pointmap_consistency(
    tgt_cam: torch.Tensor,          # [B, V_tgt, N, 3]
    trajs_2d: torch.Tensor,         # [B, V_tgt, N, 2]
    target_depth: torch.Tensor,     # [B, V, C, H, W]
    valid_mask: torch.Tensor,       # [B, V_tgt, N]
    original_h: int,
    original_w: int,
    B: int,
    device: torch.device,
):
    """Debug: verify transformed trajs_3d matches pointmap at frame 0."""
    td_frame0 = target_depth[:, 0]       # [B, C, H, W]
    pt_2d_frame0 = trajs_2d[:, 0]        # [B, N, 2]
    H_pm, W_pm = td_frame0.shape[-2:]

    scale_x = W_pm / original_w
    scale_y = H_pm / original_h
    pt_2d_scaled = pt_2d_frame0.clone()
    pt_2d_scaled[:, :, 0] *= scale_x
    pt_2d_scaled[:, :, 1] *= scale_y

    N_pts = pt_2d_frame0.shape[1]
    grid = torch.zeros(B, N_pts, 1, 2, device=device, dtype=pt_2d_frame0.dtype)
    grid[:, :, 0, 0] = 2.0 * (pt_2d_scaled[:, :, 0] / max(W_pm - 1, 1.0)) - 1.0
    grid[:, :, 0, 1] = 2.0 * (pt_2d_scaled[:, :, 1] / max(H_pm - 1, 1.0)) - 1.0

    sampled = F.grid_sample(
        td_frame0, grid, mode='bilinear', align_corners=True,
    ).squeeze(-1).permute(0, 2, 1)     # [B, N, C]

    tgt_cam_f0 = tgt_cam[:, 0]          # [B, N, 3]
    valid_f0 = valid_mask[:, 0] > 0

    in_bounds = (
        (pt_2d_scaled[:, :, 0] >= 0) & (pt_2d_scaled[:, :, 0] < W_pm) &
        (pt_2d_scaled[:, :, 1] >= 0) & (pt_2d_scaled[:, :, 1] < H_pm)
    )
    valid_f0 = valid_f0 & in_bounds

    if valid_f0.any():
        ratio = tgt_cam_f0[valid_f0] / (sampled[valid_f0] + 1e-8)
        med = ratio.median(dim=0).values
        logger.info(
            f"[Dynamic Pointmap GT] Verification: "
            f"X={med[0]:.4f}, Y={med[1]:.4f}, Z={med[2]:.4f} (expect ~1.0)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Type conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert_pointmap_to_scene_flow(
    dynamic_pointmap: torch.Tensor,
    ref_frame_idx: int = 0,
) -> torch.Tensor:
    """Convert dynamic pointmap predictions to scene flow (displacement) format.

    Args:
        dynamic_pointmap: [B, V_src, V_tgt, 3, H, W]  absolute positions
        ref_frame_idx: reference frame index (default 0)
    Returns:
        scene_flow: [B, V_src, V_tgt, 3, H, W]  displacements from ref
    """
    ref = dynamic_pointmap[:, :, ref_frame_idx:ref_frame_idx + 1, :, :, :]
    return dynamic_pointmap - ref


# ─────────────────────────────────────────────────────────────────────────────
# Scene flow loss  (train time)
# ─────────────────────────────────────────────────────────────────────────────

def compute_scene_flow_loss(
    *,
    pred_scene_flow: torch.Tensor,       # [B, V_src, V_tgt, 3, H, W]
    trajs_3d: torch.Tensor,              # [B, V, N, 3]  world coords (normalized)
    trajs_2d: torch.Tensor,              # [B, V, N, 2]  pixel coords
    visibs: torch.Tensor,                # [B, V, N]
    valids: Optional[torch.Tensor],
    original_h: int,
    original_w: int,
    scale: Optional[torch.Tensor],
    extrinsics: Optional[torch.Tensor],  # [B, V, 4, 4]  w2c
    loss_fn,                             # callable (Any4DSceneFlowLoss.forward interface)
    device: Optional[torch.device] = None,
    exclude_self_pairs: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute scene flow loss using sparse trajectory supervision.

    Generates frame pairs from pred_scene_flow shape, prepares GT using
    prepare_scene_flow_gt (einsum-based coordinate transform), handles
    batch dimension mismatch, aligns scale, and calls the loss function.

    Supports both pairing strategies:
      - All-pairs:     pred [B, V, V, 3, H, W]  exclude_self_pairs=True
      - Single-source: pred [B, 1, V, 3, H, W]  exclude_self_pairs=False

    Returns:
        loss, loss_dict
    """
    if loss_fn is None:
        return torch.tensor(0.0, device=device or pred_scene_flow.device), {}

    device = device or pred_scene_flow.device
    B_pred, V_src, V_tgt = pred_scene_flow.shape[:3]
    B_traj = trajs_3d.shape[0]

    # Move inputs to device
    trajs_3d = trajs_3d.to(device)
    trajs_2d = trajs_2d.to(device)
    visibs = visibs.to(device)
    if valids is not None:
        valids = valids.to(device)

    # Generate frame pairs from prediction tensor shape
    src_grid, tgt_grid = torch.meshgrid(
        torch.arange(V_src, device=device),
        torch.arange(V_tgt, device=device),
        indexing='ij',
    )
    if exclude_self_pairs:
        pair_mask = src_grid != tgt_grid
    else:
        pair_mask = torch.ones_like(src_grid, dtype=torch.bool)

    src_flat = src_grid[pair_mask]   # indices into pred dim 1
    tgt_flat = tgt_grid[pair_mask]   # indices into pred dim 2

    if len(src_flat) == 0:
        return torch.tensor(0.0, device=device), {}

    # Clamp to trajectory frame count
    num_frames_traj = trajs_3d.shape[1]
    src_traj_idx = src_flat.clamp(0, num_frames_traj - 1)
    tgt_traj_idx = tgt_flat.clamp(0, num_frames_traj - 1)

    # Prepare GT (coordinate transform to source camera frame)
    src_cam, tgt_cam, src_2d, valid_mask = prepare_scene_flow_gt(
        trajs_3d, trajs_2d, visibs, valids, extrinsics,
        src_traj_idx, tgt_traj_idx,
    )

    # Extract pred for selected pairs
    pred = pred_scene_flow[:, src_flat, tgt_flat]   # [B_pred, num_pairs, 3, H, W]

    # Handle batch dimension mismatch
    effective_B = max(B_pred, B_traj)
    if B_pred == 1 and B_traj > 1:
        pred = pred.expand(B_traj, -1, -1, -1, -1)
    elif B_pred > 1 and B_traj == 1:
        src_cam = src_cam.expand(B_pred, -1, -1, -1)
        tgt_cam = tgt_cam.expand(B_pred, -1, -1, -1)
        src_2d = src_2d.expand(B_pred, -1, -1, -1)
        valid_mask = valid_mask.expand(B_pred, -1, -1)

    # Align scale
    scale_aligned = align_scale_for_loss(scale, effective_B)

    try:
        loss, loss_dict = loss_fn(
            pred_flow=pred,
            trajs_3d_src=src_cam,
            trajs_3d_tgt=tgt_cam,
            trajs_2d_src=src_2d,
            valid_mask=valid_mask,
            original_h=original_h,
            original_w=original_w,
            scale=scale_aligned,
        )
        return loss, loss_dict
    except Exception as e:
        logger.error(f"[Scene Flow Loss] Failed: {e}")
        return torch.tensor(0.0, device=device), {}


# ─────────────────────────────────────────────────────────────────────────────
# Motion loss dispatcher  (train time, movies.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_motion_head_loss(
    *,
    name: str,
    pred_scene_flow: Optional[torch.Tensor],       # [B, 1, V, 3, H, W]
    pred_dynamic_pointmap: Optional[torch.Tensor],  # [B, 1, V, 3, H, W]
    motion_prediction_type: str,                    # "scene_flow" or "dynamic_pointmap"
    trajs_3d: Optional[torch.Tensor],
    trajs_2d: Optional[torch.Tensor],
    visibs: Optional[torch.Tensor],
    valids: Optional[torch.Tensor],
    original_h: int,
    original_w: int,
    scale: Optional[torch.Tensor],
    extrinsics: Optional[torch.Tensor],
    motion_any4d_loss=None,
    motion_pointmap_loss=None,
    device: torch.device = torch.device("cpu"),
    target_depth: Optional[torch.Tensor] = None,
    ref_frame_idx: int = 0,
    **kwargs,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute motion head loss for single reference view training.

    Strategy: frame ref_frame_idx → all frames [0, 1, ..., V-1].
    Dispatches to either scene flow loss or dynamic pointmap loss.

    COORDINATE SYSTEM: Regardless of which frame is the source/query,
    all displacement vectors are always expressed in **frame 0's camera
    coordinate system**.  This is ensured by passing coord_frame_idx=0
    to the GT preparation functions.

    Args:
        ref_frame_idx: Which frame is the source/reference (default 0).
            Determines which frame's 2D locations are used for sampling and
            which 3D positions serve as the starting point.
            Set by the motion head when random_src_frame is enabled.
    """
    pred_raw = (pred_dynamic_pointmap
                if motion_prediction_type == "dynamic_pointmap"
                else pred_scene_flow)

    pred_dims, traj_dims, B = validate_motion_inputs(
        pred_raw, trajs_3d, trajs_2d, visibs, valids,
    )
    if pred_dims is None:
        return torch.tensor(0.0, device=device), {}

    _, V_src, V_tgt = pred_dims
    device = pred_raw.device

    src_idx = torch.full((V_tgt,), ref_frame_idx, dtype=torch.long, device=device)
    tgt_idx = torch.arange(V_tgt, dtype=torch.long, device=device)

    pred = pred_raw[:, 0, tgt_idx]  # [B, V_tgt, 3, H, W]

    if motion_prediction_type == "dynamic_pointmap":
        src_cam, tgt_cam, src_2d, valid_mask = prepare_dynamic_pointmap_gt(
            trajs_3d, trajs_2d, visibs, valids, extrinsics,
            src_idx, tgt_idx,
            original_h=original_h, original_w=original_w,
            target_depth=target_depth,
            coord_frame_idx=0,
        )
    else:
        src_cam, tgt_cam, src_2d, valid_mask = prepare_scene_flow_gt(
            trajs_3d, trajs_2d, visibs, valids, extrinsics,
            src_idx, tgt_idx,
            coord_frame_idx=0,
        )

    scale_aligned = align_scale_for_loss(scale, B)

    try:
        if motion_prediction_type == "dynamic_pointmap":
            if motion_pointmap_loss is None:
                return torch.tensor(0.0, device=device), {}
            loss, stats = motion_pointmap_loss(
                pred_points=pred,
                trajs_3d_tgt=tgt_cam,
                trajs_2d_src=src_2d,
                valid_mask=valid_mask,
                original_h=original_h,
                original_w=original_w,
                trajs_3d_src=src_cam,
                scale=scale_aligned,
            )
        else:
            if motion_any4d_loss is None:
                logger.warning("[Motion Loss] motion_any4d_loss is None, skipping")
                return torch.tensor(0.0, device=device), {}
            loss, stats = motion_any4d_loss(
                pred, src_cam, tgt_cam, src_2d, valid_mask,
                original_h, original_w, scale=scale_aligned,
            )

        return loss, {f"motion_{k}": v for k, v in stats.items()
                      if "mean" in k or "ratio" in k}

    except Exception as e:
        logger.error(f"[Motion Loss] Failed: {e}")
        return torch.tensor(0.0, device=device), {}


# ─────────────────────────────────────────────────────────────────────────────
# Inference output helpers
# ─────────────────────────────────────────────────────────────────────────────

def denormalize_scene_flow_for_eval(
    pred_scene_flow: Optional[torch.Tensor],
    scale: Optional[torch.Tensor],
    denormalize_fn,
) -> Optional[torch.Tensor]:
    """Denormalize predicted scene flow to absolute scale for evaluation.

    Args:
        pred_scene_flow: [B, V_src, V_tgt, 3, H, W]
        scale: [B, V, 1, 1, 1]
        denormalize_fn: callable(x, scale=s) → x * s
    """
    if pred_scene_flow is None or scale is None:
        return pred_scene_flow
    scale_for_flow = scale[:, 0:1, :, :, :]
    return denormalize_fn(pred_scene_flow, scale=scale_for_flow)


def attach_motion_data_to_output(
    single,
    index: int,
    pred_scene_flow: Optional[torch.Tensor],
    trajs_3d: Optional[torch.Tensor],
    trajs_2d: Optional[torch.Tensor],
    visibs: Optional[torch.Tensor],
    valids: Optional[torch.Tensor],
    motion_extrinsics: Optional[torch.Tensor],
    scale: Optional[torch.Tensor],
    meta_data: dict,
    is_training: bool,
    denormalize_fn=None,
):
    """Attach motion-related data to a single-view output object.

    Handles denormalization of trajs_3d and motion_extrinsics during
    inference so that GT and pred are in the same absolute scale.

    Args:
        single: output object (e.g. SimpleNamespace) to attach attributes to.
        index: view index within the batch.
        denormalize_fn: callable(x, scale=s) → x * s.
        is_training: if True, skip denormalization (not needed at train time).
    """
    # Scene flow prediction (already denormalized by caller)
    if pred_scene_flow is not None:
        flow = pred_scene_flow[0, 0].cpu()  # [V_tgt, 3, H_flow, W_flow]
        target_h = getattr(single, 'pointmap_h', None)
        target_w = getattr(single, 'pointmap_w', None)
        if (target_h and target_w and
                (flow.shape[-2] != target_h or flow.shape[-1] != target_w)):
            flow = F.interpolate(
                flow, size=(target_h, target_w),
                mode='bilinear', align_corners=False,
            )
        single.scene_flow_pred = flow
        single.origin_height = (
            meta_data["origin_height"][0].item()
            if "origin_height" in meta_data else None
        )
        single.origin_width = (
            meta_data["origin_width"][0].item()
            if "origin_width" in meta_data else None
        )

    # GT trajectories — denormalize to absolute scale for evaluation
    if trajs_3d is not None and trajs_2d is not None:
        trajs_3d_out = trajs_3d
        if scale is not None and not is_training and denormalize_fn is not None:
            ref_scale = get_ref_scale(scale)
            trajs_3d_out = denormalize_fn(trajs_3d, scale=ref_scale)

        single.trajs_3d = trajs_3d_out[0].cpu()
        single.trajs_2d = trajs_2d[0].cpu()
        if visibs is not None:
            single.trajs_visibs = visibs[0].cpu()
        if valids is not None:
            single.trajs_valids = valids[0].cpu()

    # Motion extrinsics — denormalize translation for evaluation
    if motion_extrinsics is not None:
        me_out = motion_extrinsics.clone()
        if scale is not None and not is_training and denormalize_fn is not None:
            ref_scale_flat = get_ref_scale_flat(scale)
            me_out[..., :3, 3] = denormalize_fn(
                motion_extrinsics[..., :3, 3], scale=ref_scale_flat,
            )
        single.motion_extrinsics = me_out[0].cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Sparse Motion Pipeline Utilities
# ─────────────────────────────────────────────────────────────────────────────

def sanitize(pred: torch.Tensor, clamp: float = 100.0) -> torch.Tensor:
    """Clamp and NaN-guard for numerical stability.

    Uses a gradient-safe replacement strategy to prevent NaN/Inf values in
    the model output from contaminating the backward pass.

    The key issue with ``torch.nan_to_num(pred.clamp(...))`` is that
    ``clamp`` treats NaN as "in-bounds" (since ``NaN >= min`` and
    ``NaN <= max`` are both False in IEEE 754), so its backward passes
    the incoming gradient straight through NaN positions.  Downstream,
    ``0 * NaN = NaN`` in weight-gradient accumulations even when the
    output gradient is zeroed, because PyTorch's matrix-multiply (cuBLAS)
    does not guarantee ``0 * NaN = 0``.

    ``torch.where(finite_mask, x, detached_zeros)`` avoids this:
    - Forward:  NaN/Inf positions get value 0 (from the detached branch).
    - Backward: ``torch.where`` sends gradient only to the branch that was
      selected.  At NaN/Inf positions (condition=False) the gradient goes
      to ``detached_zeros``, which has no grad_fn → gradient at those
      positions in ``pred`` is cleanly 0, never NaN.
    """
    finite_mask = torch.isfinite(pred)
    if finite_mask.all():
        return pred
    n_bad = (~finite_mask).sum().item()
    ratio = n_bad / max(pred.numel(), 1)
    logger.warning("[SparseMotion] %d NaN/Inf values (%.1f%%) — clamping", n_bad, 100 * ratio)
    # Detached zero tensor: torch.where backward sends no gradient here,
    # so positions replaced by zero contribute 0 (not NaN) to param gradients.
    safe_zeros = torch.zeros_like(pred.detach())
    return torch.where(finite_mask, pred.clamp(-clamp, clamp), safe_zeros)


def world_to_camera(pts: torch.Tensor, ext: torch.Tensor, frame: int) -> torch.Tensor:
    """[B, Q, 3] world → camera coords at `frame`."""
    R = ext[:, frame, :3, :3]
    t = ext[:, frame, :3, 3]
    return torch.bmm(R, pts.transpose(1, 2)).transpose(1, 2) + t.unsqueeze(1)


def cat_query_dicts(batches: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Concatenate query dicts along the query dimension (dim=1)."""
    if len(batches) == 1:
        return batches[0]
    combined = {}
    for key in batches[0]:
        vals = [q[key] for q in batches if key in q]
        if vals:
            combined[key] = torch.cat(vals, dim=1 if vals[0].ndim >= 2 else 0)
    return combined


def scalar(value, default=None):
    """Extract a scalar from tensor / list / int / None."""
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        return value.flatten()[0].item()
    if isinstance(value, (int, float)):
        return value
    try:
        return int(value[0])
    except (TypeError, IndexError):
        return default


def maybe_instantiate(cfg):
    """Instantiate from config dict if not None."""
    from hAlgorithm.utils import instantiate_from_config
    return instantiate_from_config(cfg) if cfg else None


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field

@dataclass
class FieldNames:
    """Mapping from semantic roles to batch key names."""
    scale: Optional[str] = None
    intrinsics: Optional[str] = None
    extrinsics: Optional[str] = None
    motion_extrinsics: str = "extrinsics"
    prompt_depth: Optional[str] = None
    target_local_depth: Optional[str] = None
    target_global_points: Optional[str] = None
    target_depth_mask: Optional[str] = None

    def get(self, key: str) -> Optional[str]:
        return getattr(self, key, None)


# ─────────────────────────────────────────────────────────────────────────────
# Loss Accumulator
# ─────────────────────────────────────────────────────────────────────────────

class LossAccumulator:
    """Accumulates weighted losses into a total + breakdown dict."""

    def __init__(self, device: torch.device, task_weights: Dict[str, float]):
        self.total = torch.tensor(0.0, device=device)
        self.breakdown: Dict[str, object] = {}
        self._weights = task_weights

    def add(self, name: str, loss) -> None:
        w = self._weights.get(name, 1.0)
        if isinstance(loss, dict):
            for k, v in loss.items():
                weighted = v * w
                self.total = self.total + weighted
                key = f"{name}_{k}"
                self.breakdown[key] = self.breakdown.get(key, 0) + weighted
            task_sum = sum(v * w for v in loss.values())
            self.breakdown[name] = self.breakdown.get(name, 0) + task_sum
        else:
            self.total = self.total + loss * w
            self.breakdown[name] = self.breakdown.get(name, 0) + loss * w

    def add_meta(self, key: str, value) -> None:
        self.breakdown[key] = value

    def result(self) -> Tuple[torch.Tensor, Dict]:
        return self.total, self.breakdown
