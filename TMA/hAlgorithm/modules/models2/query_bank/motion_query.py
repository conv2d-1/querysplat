"""
D4RT Motion Query Construction
================================
Query building utilities for sparse motion prediction.

This module contains:
- QueryBatch: Typed container for D4RT queries
- QueryConfig: Configuration for query construction
- QueryBuilder: Constructs all query batches for training/inference
- MotionQueryBank: nn.Module wrapper around QueryBuilder (sv_query-style interface)
- create_queries_from_trajectories: Creates queries from GT trajectory annotations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Query Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QueryBatch:
    """Typed container for a batch of D4RT queries."""
    uv: torch.Tensor                 # [B, Q, 2] normalized coords in [0, 1]
    src_frame_idx: torch.Tensor      # [B, Q]    int frame indices
    tgt_frame_idx: torch.Tensor      # [B, Q]    int frame indices
    tgt_camera_idx: torch.Tensor     # [B, Q]    int frame indices
    src_time: Optional[torch.Tensor] = None
    tgt_time: Optional[torch.Tensor] = None
    cam_time: Optional[torch.Tensor] = None
    gt_2d_at_tgt: Optional[torch.Tensor] = None
    gt_3d_at_src: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, torch.Tensor]:
        d = {
            'uv': self.uv,
            'src_frame_idx': self.src_frame_idx,
            'tgt_frame_idx': self.tgt_frame_idx,
            'tgt_camera_idx': self.tgt_camera_idx,
        }
        for key in ('src_time', 'tgt_time', 'cam_time',
                     'gt_2d_at_tgt', 'gt_3d_at_src'):
            val = getattr(self, key)
            if val is not None:
                d[key] = val
        return d

    @staticmethod
    def from_dict(d: Dict[str, torch.Tensor]) -> 'QueryBatch':
        return QueryBatch(
            uv=d['uv'],
            src_frame_idx=d['src_frame_idx'],
            tgt_frame_idx=d['tgt_frame_idx'],
            tgt_camera_idx=d['tgt_camera_idx'],
            src_time=d.get('src_time'),
            tgt_time=d.get('tgt_time'),
            cam_time=d.get('cam_time'),
            gt_2d_at_tgt=d.get('gt_2d_at_tgt'),
            gt_3d_at_src=d.get('gt_3d_at_src'),
        )

    @property
    def num_queries(self) -> int:
        return self.uv.shape[1]


@dataclass
class QueryConfig:
    """Parameters controlling query construction."""
    num_queries_per_frame: int = 256
    src_frame_idx: int = 0
    default_max_frames: int = 1000
    deterministic: bool = False
    num_depth_consistency_samples: int = 64
    cycle_consistency_prob: float = 0.3
    # Stratified dynamic/static sampling:
    # 0.0 = existing soft-bias; >0 = guarantee at least this fraction are dynamic.
    min_dynamic_ratio: float = 0.0
    # Normalized-space motion threshold used for dynamic classification in sampler.
    # trajs_3d is already divided by scene scale, so 0.002 ≈ 0.02–0.06 m absolute
    # at typical scales of 10–30.
    sampler_dynamic_threshold: float = 0.002
    # If True, training uses ALL tracks × ALL target frames (same coverage as inference).
    # Useful for overfitting experiments to ensure train and eval see identical queries.
    use_all_points: bool = False
    # If True, uniformly sample a random source frame each training step instead of
    # always using src_frame_idx.  Mirrors the random_src_frame flag in MotionHead4RC.
    random_src_frame: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Query construction from trajectory annotations
# ─────────────────────────────────────────────────────────────────────────────

def _make_empty_queries(
    B: int, Q: int, src_safe: int, T: int,
    time_idx: Optional[torch.Tensor], device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Create Q dummy queries with valid_mask=0.

    Used when no usable tracks exist so that the model forward still runs
    (touching all parameters) and DDP gradient reduction completes normally.
    The zero valid_mask ensures these queries contribute no gradient signal.
    """
    t_norm = max(T - 1, 1)
    src_idx = torch.full((B, Q), src_safe, dtype=torch.long, device=device)
    tgt_idx = torch.zeros(B, Q, dtype=torch.long, device=device)
    cam_idx = torch.full((B, Q), src_safe, dtype=torch.long, device=device)
    if time_idx is None:
        src_time = torch.full((B, Q), src_safe / t_norm, dtype=torch.float32, device=device)
        tgt_time = torch.zeros(B, Q, dtype=torch.float32, device=device)
    else:
        src_time = time_idx[:, src_safe].unsqueeze(1).expand(B, Q)
        tgt_time = torch.zeros(B, Q, dtype=torch.float32, device=device)

    queries = {
        'uv':             torch.zeros(B, Q, 2, device=device),
        'src_frame_idx':  src_idx,
        'tgt_frame_idx':  tgt_idx,
        'tgt_camera_idx': cam_idx,
        'src_time':       src_time,
        'tgt_time':       tgt_time,
        'cam_time':       src_time.clone(),
        'gt_2d_at_tgt':   torch.zeros(B, Q, 2, device=device),
        'gt_3d_at_src':   torch.zeros(B, Q, 3, device=device),
    }
    gt_3d      = torch.zeros(B, Q, 3, device=device)
    valid_mask = torch.zeros(B, Q, device=device)
    return queries, gt_3d, valid_mask


def create_queries_from_trajectories(
    trajs_2d: torch.Tensor,
    trajs_3d: torch.Tensor,
    time_idx: torch.Tensor,
    visibs: torch.Tensor,
    valids: torch.Tensor,
    src_frame: int = 0,
    num_queries_per_frame: int = 256,
    img_size: Optional[Tuple[int, int]] = None,
    deterministic: bool = False,
    use_all_points: bool = False,
    min_dynamic_ratio: float = 0.0,
    sampler_dynamic_threshold: float = 0.002,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Create D4RT query data from trajectory annotations.

    IMPORTANT: tgt_camera_idx is set to src_frame (NOT tgt_frame), so the
    model predicts 3D positions in the SOURCE camera coordinate system.
    This must match the GT coordinate system used in the loss.

    Never returns None.  When no usable tracks exist, returns dummy queries
    with ``valid_mask = 0`` so that the model forward still executes and
    DDP gradient reduction completes normally.

    Args:
        deterministic: If True, use deterministic sampling for reproducibility.
        use_all_points: If True, use ALL valid tracks × ALL target frames.
            This creates Q = N_valid * (T-1) queries so that sparse evaluation
            is directly comparable to dense evaluation.  Ignores
            ``num_queries_per_frame`` and ``deterministic`` when enabled.
    """
    B, T, N, _ = trajs_2d.shape
    device = trajs_2d.device
    Q = num_queries_per_frame
    src_safe = min(max(src_frame, 0), T - 1)

    src_usable = visibs[:, src_safe] * valids[:, src_safe]
    if src_usable.sum() == 0:
        logger.warning("[create_queries] No usable tracks at src frame %d — "
                       "returning %d dummy queries with valid_mask=0.", src_safe, Q)
        return _make_empty_queries(B, Q, src_safe, T, time_idx, device)

    tgt_frames = [t for t in range(T) if t != src_safe]
    if not tgt_frames:
        logger.warning("[create_queries] Only 1 frame (T=%d) — "
                       "returning %d dummy queries with valid_mask=0.", T, Q)
        return _make_empty_queries(B, Q, src_safe, T, time_idx, device)
    tgt_frames_t = torch.tensor(tgt_frames, device=device)
    t_norm = max(T - 1, 1)

    def _src_time_values(num_queries: int) -> torch.Tensor:
        if time_idx is None:
            return torch.full(
                (B, num_queries), src_safe / t_norm, dtype=torch.float32, device=device,
            )
        return time_idx[:, src_safe].unsqueeze(1).expand(B, num_queries)

    # ══════════════════════════════════════════════════════════════════
    # Full-coverage mode: ALL valid tracks × ALL target frames
    # ══════════════════════════════════════════════════════════════════
    if use_all_points:
        n_tgt = len(tgt_frames)
        Q_total = N * n_tgt

        # Build (track, target_frame) pairs.
        # Layout: track0×tgt0, track0×tgt1, …, track1×tgt0, …
        all_track = torch.arange(N, device=device).repeat_interleave(n_tgt)  # [N*n_tgt]
        all_tgt   = tgt_frames_t.repeat(N)                                   # [N*n_tgt]
        all_track = all_track.unsqueeze(0).expand(B, -1)   # [B, Q_total]
        all_tgt   = all_tgt.unsqueeze(0).expand(B, -1)     # [B, Q_total]
        bi        = torch.arange(B, device=device).unsqueeze(1).expand(B, Q_total)

        # Gather GT 3D / 2D
        gt_3d     = trajs_3d[bi, all_tgt, all_track].nan_to_num(0.0)       # [B, Q_total, 3]
        uv_coords = trajs_2d[bi, src_safe, all_track].nan_to_num(0.0)      # [B, Q_total, 2]
        gt_2d_tgt = trajs_2d[bi, all_tgt, all_track].nan_to_num(0.0)       # [B, Q_total, 2]
        gt_3d_src = trajs_3d[bi, src_safe, all_track].nan_to_num(0.0)      # [B, Q_total, 3]

        # Normalize UV
        if img_size is not None:
            H_orig, W_orig = img_size
            uv_s = torch.tensor([max(W_orig - 1, 1), max(H_orig - 1, 1)],
                                dtype=uv_coords.dtype, device=device)
            uv_coords = uv_coords / uv_s
            gt_2d_tgt = gt_2d_tgt / uv_s

        # Validity: src_usable AND tgt_visible AND tgt_valid
        src_ok = (src_usable.unsqueeze(2)
                  .expand(B, N, n_tgt)
                  .reshape(B, Q_total) > 0.5)                              # [B, Q_total]
        tgt_vis = visibs[bi, all_tgt, all_track]
        tgt_val = valids[bi, all_tgt, all_track]
        final_valid = src_ok.float() * tgt_vis * tgt_val

        mask = final_valid.unsqueeze(-1)
        uv_coords = uv_coords * mask
        gt_3d     = gt_3d * mask

        src_idx    = torch.full((B, Q_total), src_safe, dtype=torch.long, device=device)
        cam_idx    = torch.full((B, Q_total), src_safe, dtype=torch.long, device=device)
        tgt_masked = all_tgt * (final_valid > 0.5).long()
        src_time = _src_time_values(Q_total)
        if time_idx is None:
            tgt_time = tgt_masked.float() / t_norm
        else:
            tgt_time = time_idx[bi, all_tgt] * (final_valid > 0.5).float()
        cam_time = src_time

        queries = {
            'uv':              uv_coords,
            'src_frame_idx':   src_idx,
            'tgt_frame_idx':   tgt_masked,
            'tgt_camera_idx':  cam_idx,
            'src_time':        src_time,
            'tgt_time':        tgt_time,
            'cam_time':        cam_time,
            'gt_2d_at_tgt':    gt_2d_tgt,
            'gt_3d_at_src':    gt_3d_src,
        }
        n_valid = int(final_valid.sum())
        logger.info("[create_queries] use_all_points: %d tracks × %d targets "
                    "= %d queries (%d valid)", N, n_tgt, Q_total, n_valid)
        return queries, gt_3d, final_valid

    # ══════════════════════════════════════════════════════════════════
    # Subsampled mode (training): Q queries with motion-aware priority
    # ══════════════════════════════════════════════════════════════════

    # Per-track motion magnitude for priority scoring
    src_3d = trajs_3d[:, src_safe : src_safe + 1, :, :]   # [B, 1, N, 3]
    motion_mag = (trajs_3d - src_3d).norm(dim=-1).nan_to_num(0.0)  # [B, T, N]
    max_motion = motion_mag.max(dim=1).values                       # [B, N]

    is_usable = src_usable > 0.5  # [B, N]

    if deterministic:
        # Motion-aware deterministic: sort by motion magnitude (descending),
        # break ties with track index for reproducibility.
        tie_break = torch.arange(N, 0, -1, device=device).float().unsqueeze(0) * 1e-7
        priority = max_motion + tie_break
        priority[~is_usable] = -float('inf')

        k = min(Q, N)
        _, topk_idx = priority.topk(k, dim=1)
        if k < Q:
            topk_idx = F.pad(topk_idx, (0, Q - k), value=0)
        topk_idx = topk_idx[:, :Q]

        num_valid = is_usable.sum(dim=1)
        actual_Q  = torch.clamp(num_valid, max=Q)
        query_valid = torch.arange(Q, device=device).unsqueeze(0) < actual_Q.unsqueeze(1)

    elif min_dynamic_ratio > 0.0:
        # ── Stratified sampling: hard guarantee on minimum dynamic fraction ───
        # trajs_3d is already normalized by scene scale, so sampler_dynamic_threshold
        # (default 0.002) corresponds to roughly 0.02–0.06 m at typical scales 10–30.
        is_dynamic = (max_motion > sampler_dynamic_threshold) & is_usable  # [B, N]

        n_dyn_slots   = int(round(Q * min_dynamic_ratio))
        n_rest_slots  = Q - n_dyn_slots

        # Step 1 – dynamic pool: topk by motion magnitude + tiny noise
        dyn_prio = torch.where(is_dynamic,
                               max_motion + torch.rand(B, N, device=device) * 1e-4,
                               torch.full_like(max_motion, -float('inf')))
        k_dyn = min(n_dyn_slots, N)
        _, dyn_idx = dyn_prio.topk(k_dyn, dim=1)          # [B, k_dyn]
        if k_dyn < n_dyn_slots:
            dyn_idx = F.pad(dyn_idx, (0, n_dyn_slots - k_dyn), value=0)

        # Step 2 – rest pool: random from ALL usable, excluding already-selected
        rest_prio = torch.rand(B, N, device=device) * is_usable.float()
        rest_prio.scatter_(1, dyn_idx.clamp(0, N - 1), -1.0)  # de-prioritize selected
        k_rest = min(n_rest_slots, N)
        _, rest_idx = rest_prio.topk(k_rest, dim=1)            # [B, k_rest]
        if k_rest < n_rest_slots:
            rest_idx = F.pad(rest_idx, (0, n_rest_slots - k_rest), value=0)

        topk_idx = torch.cat([dyn_idx, rest_idx], dim=1)       # [B, Q]

        # Validity: each slot is valid only if its track is truly usable
        dyn_valid  = is_usable.gather(1, dyn_idx.clamp(0, N - 1))
        if k_dyn < n_dyn_slots:
            dyn_valid[:, k_dyn:] = False
        rest_valid = is_usable.gather(1, rest_idx.clamp(0, N - 1))
        if k_rest < n_rest_slots:
            rest_valid[:, k_rest:] = False
        query_valid = torch.cat([dyn_valid, rest_valid], dim=1)  # [B, Q] bool

        n_dyn_actual = is_dynamic.sum(dim=1)
        logger.debug(
            "[create_queries] stratified: want %d dyn / %d rest, "
            "available dyn per item (min/max): %d / %d",
            n_dyn_slots, n_rest_slots,
            n_dyn_actual.min().item(), n_dyn_actual.max().item(),
        )

    else:
        # ── Soft motion-aware random bias (original behaviour) ───────────────
        motion_weight = 1.0 + torch.sqrt(
            max_motion / (max_motion.max(dim=1, keepdim=True).values + 1e-8)
        )
        priority = torch.rand(B, N, device=device) * motion_weight
        priority[~is_usable] = -float('inf')

        k = min(Q, N)
        _, topk_idx = priority.topk(k, dim=1)
        if k < Q:
            topk_idx = F.pad(topk_idx, (0, Q - k), value=0)
        topk_idx = topk_idx[:, :Q]

        num_valid  = is_usable.sum(dim=1)
        actual_Q   = torch.clamp(num_valid, max=Q)
        query_valid = torch.arange(Q, device=device).unsqueeze(0) < actual_Q.unsqueeze(1)

    if deterministic:
        rand_sel = torch.arange(Q, device=device).unsqueeze(0).expand(B, Q) % len(tgt_frames)
    else:
        rand_sel = torch.randint(len(tgt_frames), (B, Q), device=device)

    tgt_idx = tgt_frames_t[rand_sel]

    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, Q)

    # FIX #1: Use consistently clamped indices for ALL gather operations.
    # Previously gt_2d_tgt and gt_3d_at_src used unclamped topk_idx/tgt_idx,
    # which could index out of bounds when k < Q (padded indices = 0 may be
    # wrong) or if values exceed valid range.
    topk_safe = topk_idx.clamp(0, N - 1)
    tgt_safe = tgt_idx.clamp(0, T - 1)

    gt_3d = trajs_3d[batch_idx, tgt_safe, topk_safe].nan_to_num(0.0)
    uv_coords = trajs_2d[batch_idx, src_safe, topk_safe].nan_to_num(0.0)
    gt_2d_tgt = trajs_2d[batch_idx, tgt_safe, topk_safe].nan_to_num(0.0)   # FIX: was [batch_idx, tgt_idx, topk_idx]

    if img_size is not None:
        H_orig, W_orig = img_size
        uv_scale = torch.tensor(
            [max(W_orig - 1, 1), max(H_orig - 1, 1)],
            dtype=uv_coords.dtype, device=device,
        )
        uv_coords = uv_coords / uv_scale
        gt_2d_tgt = gt_2d_tgt / uv_scale
    else:
        logger.warning("create_queries_from_trajectories: img_size not provided")

    # FIX #2: Gather visibility/validity with clamped indices (already done),
    # BUT also force padded queries (beyond actual_Q) to invalid BEFORE
    # combining with tgt_vis/tgt_val. This prevents clamped garbage indices
    # from accidentally landing on visible tracks and getting nonzero weight.
    tgt_vis = visibs[batch_idx, tgt_safe, topk_safe]
    tgt_val = valids[batch_idx, tgt_safe, topk_safe]
    # Apply query_valid first so padded slots cannot inherit validity from
    # whatever track index 0 happens to have.
    final_valid = query_valid.float() * tgt_vis * tgt_val

    mask = final_valid.unsqueeze(-1)
    uv_coords = uv_coords * mask
    gt_3d = gt_3d * mask

    src_idx = torch.full((B, Q), src_safe, dtype=torch.long, device=device)
    tgt_idx_masked = tgt_idx * query_valid.long()
    cam_idx = torch.full((B, Q), src_safe, dtype=torch.long, device=device)
    src_time = _src_time_values(Q)
    if time_idx is None:
        tgt_time = tgt_idx_masked.float() / t_norm
    else:
        tgt_time = time_idx[batch_idx, tgt_safe] * query_valid.float()
    cam_time = src_time

    queries = {
        'uv': uv_coords,
        'src_frame_idx': src_idx,
        'tgt_frame_idx': tgt_idx_masked,
        'tgt_camera_idx': cam_idx,
        'src_time': src_time,
        'tgt_time': tgt_time,
        'cam_time': cam_time,
        'gt_2d_at_tgt': gt_2d_tgt,
        'gt_3d_at_src': trajs_3d[batch_idx, src_safe, topk_safe].nan_to_num(0.0),  # FIX: was topk_idx (unclamped)
    }
    return queries, gt_3d, final_valid


# ─────────────────────────────────────────────────────────────────────────────
# Query Builder (extracted from pipeline for testability)
# ─────────────────────────────────────────────────────────────────────────────

class QueryBuilder:
    """Constructs all query batches for training and inference."""

    def __init__(self, cfg: QueryConfig):
        self.cfg = cfg

    @property
    def src(self) -> int:
        return self.cfg.src_frame_idx

    # ── motion queries (from GT trajectories) ────────────────────────────

    def motion(self, trajs_2d, trajs_3d, time_idx, visibs, valids,
               img_size, deterministic=None, use_all_points=None):
        """Build motion queries from GT trajectories.

        Always returns (queries, gt_positions, valid_mask).  When no usable
        tracks exist, returns dummy queries with valid_mask=0 (never None).

        When ``QueryConfig.random_src_frame`` is True the source frame is
        drawn uniformly from [0, T) each call, mirroring the behaviour of
        ``MotionHead4RC`` with ``random_src_frame=True``.  At inference time
        this flag is False and ``src_frame_idx`` is used deterministically.
        """
        det = deterministic if deterministic is not None else self.cfg.deterministic
        all_pts = use_all_points if use_all_points is not None else self.cfg.use_all_points

        if self.cfg.random_src_frame and trajs_2d is not None:
            T = trajs_2d.shape[1]
            src_frame = int(torch.randint(0, max(T, 1), (1,)).item())
        else:
            src_frame = self.src

        return create_queries_from_trajectories(
            trajs_2d=trajs_2d, trajs_3d=trajs_3d,
            time_idx=time_idx, visibs=visibs, valids=valids,
            src_frame=src_frame,
            num_queries_per_frame=self.cfg.num_queries_per_frame,
            img_size=img_size, deterministic=det,
            use_all_points=all_pts,
            min_dynamic_ratio=self.cfg.min_dynamic_ratio,
            sampler_dynamic_threshold=self.cfg.sampler_dynamic_threshold,
        )

    # ── single-target motion queries (for per-frame inference) ───────────

    def motion_single_target(
        self,
        trajs_2d: torch.Tensor,
        trajs_3d: torch.Tensor,
        time_idx: Optional[torch.Tensor],
        visibs: torch.Tensor,
        valids: torch.Tensor,
        img_size: Tuple[int, int],
        tgt_frame: int,
    ) -> Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]]:
        """Build queries for ALL tracks but a SINGLE target frame.

        This produces N queries (one per track) instead of N×(T-1),
        enabling memory-efficient per-target-frame inference.

        Returns (queries_dict, gt_3d_at_tgt [B,N,3], valid_mask [B,N]) or None.
        """
        B, T, N, _ = trajs_2d.shape
        device = trajs_2d.device
        src = self.src
        src_safe = min(max(src, 0), T - 1)
        t_norm = max(T - 1, 1)

        # Source-frame usability
        src_usable = visibs[:, src_safe] * valids[:, src_safe]  # [B, N]
        if src_usable.sum() == 0:
            return None

        bi = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
        track_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, N)

        # Gather GT
        tgt_safe = min(max(tgt_frame, 0), T - 1)
        gt_3d = trajs_3d[bi, tgt_safe, track_idx].nan_to_num(0.0)          # [B, N, 3]
        uv_coords = trajs_2d[bi, src_safe, track_idx].nan_to_num(0.0)      # [B, N, 2]
        gt_2d_tgt = trajs_2d[bi, tgt_safe, track_idx].nan_to_num(0.0)      # [B, N, 2]
        gt_3d_src = trajs_3d[bi, src_safe, track_idx].nan_to_num(0.0)      # [B, N, 3]

        # Normalize UV to [0, 1]
        if img_size is not None:
            H_orig, W_orig = img_size
            uv_s = torch.tensor([max(W_orig - 1, 1), max(H_orig - 1, 1)],
                                dtype=uv_coords.dtype, device=device)
            uv_coords = uv_coords / uv_s
            gt_2d_tgt = gt_2d_tgt / uv_s

        # Validity: src usable AND target visible AND target valid
        tgt_vis = visibs[:, tgt_safe]   # [B, N]
        tgt_val = valids[:, tgt_safe]   # [B, N]
        final_valid = (src_usable > 0.5).float() * tgt_vis * tgt_val        # [B, N]

        # Zero out invalid entries
        mask = final_valid.unsqueeze(-1)  # [B, N, 1]
        uv_coords = uv_coords * mask
        gt_3d = gt_3d * mask

        # Build index tensors
        src_idx = torch.full((B, N), src_safe, dtype=torch.long, device=device)
        tgt_idx = torch.full((B, N), tgt_safe, dtype=torch.long, device=device)
        tgt_masked = tgt_idx * (final_valid > 0.5).long()
        cam_idx = torch.full((B, N), src_safe, dtype=torch.long, device=device)
        if time_idx is None:
            src_time = src_idx.float() / t_norm
            tgt_time = tgt_masked.float() / t_norm
        else:
            src_time = time_idx[:, src_safe].unsqueeze(1).expand(B, N)
            tgt_time = time_idx[:, tgt_safe].unsqueeze(1).expand(B, N) * (final_valid > 0.5).float()
        cam_time = src_time

        queries = {
            'uv':             uv_coords,
            'src_frame_idx':  src_idx,
            'tgt_frame_idx':  tgt_masked,
            'tgt_camera_idx': cam_idx,
            'src_time':       src_time,
            'tgt_time':       tgt_time,
            'cam_time':       cam_time,
            'gt_2d_at_tgt':   gt_2d_tgt,
            'gt_3d_at_src':   gt_3d_src,
        }
        return queries, gt_3d, final_valid

    # ── depth-consistency queries (identity: src→src) ────────────────────

    def depth_consistency(self, B, V, time_idx, device) -> Dict[str, torch.Tensor]:
        """UV in [0,1]; queries predict at (src, src) for self-consistency."""
        N = self.cfg.num_depth_consistency_samples
        src = self.src

        uv = torch.rand(B, N, 2, device=device)
        t = self._src_time(time_idx, V, B, N, device)

        return {
            'uv': uv,
            'src_frame_idx': torch.full((B, N), src, dtype=torch.long, device=device),
            'tgt_frame_idx': torch.full((B, N), src, dtype=torch.long, device=device),
            'tgt_camera_idx': torch.full((B, N), src, dtype=torch.long, device=device),
            'src_time': t, 'tgt_time': t.clone(), 'cam_time': t.clone(),
        }

    # ── reverse queries (for cycle consistency) ──────────────────────────

    @staticmethod
    def reverse(motion_queries: Dict) -> Optional[Dict]:
        """Swap src↔tgt for bidirectional cycle loss."""
        gt_2d = motion_queries.get('gt_2d_at_tgt')
        if gt_2d is None:
            return None
        return {
            'uv': gt_2d,
            'src_frame_idx': motion_queries['tgt_frame_idx'],
            'tgt_frame_idx': motion_queries['src_frame_idx'],
            'tgt_camera_idx': motion_queries['src_frame_idx'],
            'src_time': motion_queries['tgt_time'],
            'tgt_time': motion_queries['src_time'],
            'cam_time': motion_queries['src_time'],
        }

    # ── grid queries (inference without GT) ──────────────────────────────

    def grid(self, B, V, time_idx, device) -> Optional[Dict[str, torch.Tensor]]:
        """Uniform grid queries when GT trajectories are unavailable."""
        src = self.src
        src_safe = min(max(src, 0), V - 1)
        tgt_frames = [t for t in range(V) if t != src_safe]
        if not tgt_frames:
            return None

        pts = max(1, self.cfg.num_queries_per_frame // max(1, len(tgt_frames)))
        side = max(4, int(np.sqrt(pts)))

        u = torch.linspace(0, 1, side, device=device)
        v = torch.linspace(0, 1, side, device=device)
        gv, gu = torch.meshgrid(v, u, indexing='ij')
        uv_flat = torch.stack([gu.flatten(), gv.flatten()], dim=-1)  # [P, 2]

        n_pts, n_tgt = uv_flat.shape[0], len(tgt_frames)
        Q = n_pts * n_tgt

        # Tile UV across targets, then across batch
        uv = (uv_flat.unsqueeze(1).expand(n_pts, n_tgt, 2)
               .reshape(1, Q, 2).expand(B, Q, 2))

        tgt_t = torch.tensor(tgt_frames, device=device)
        tgt_idx = tgt_t.repeat(n_pts).unsqueeze(0).expand(B, Q)

        st = self._src_time(time_idx, V, B, Q, device)
        if time_idx is None:
            tgt_time = ((torch.arange(V, device=device).float() / max(V - 1, 1))[tgt_t]
                        .repeat(n_pts).unsqueeze(0).expand(B, Q))
        else:
            # Gather target times per batch sample, then tile across points.
            tgt_time = time_idx[:, tgt_t].unsqueeze(1).expand(B, n_pts, n_tgt).reshape(B, Q)

        return {
            'uv': uv,
            'src_frame_idx': torch.full((B, Q), src_safe, dtype=torch.long, device=device),
            'tgt_frame_idx': tgt_idx,
            # Keep camera frame aligned with source-frame prediction convention.
            'tgt_camera_idx': torch.full((B, Q), src_safe, dtype=torch.long, device=device),
            'src_time': st, 'tgt_time': tgt_time, 'cam_time': st.clone(),
        }

    # ── helpers ──────────────────────────────────────────────────────────

    def _src_time(self, time_idx, V, B, N, device):
        src_safe = min(max(self.src, 0), V - 1)
        if time_idx is None:
            return torch.full((B, N), src_safe / max(V - 1, 1), device=device)
        return time_idx[:, src_safe].unsqueeze(1).expand(B, N)


# ─────────────────────────────────────────────────────────────────────────────
# sv_query-style nn.Module interface
# ─────────────────────────────────────────────────────────────────────────────

class MotionQueryBank(nn.Module):
    """sv_query-style nn.Module wrapper around QueryBuilder.

    Holds all query-sampling hyper-parameters in a single config block that can
    be registered in the model config alongside ``query_feats_aggregator`` and
    ``query_decoder``, mirroring the three-module pattern of sv_query.

    .. attribute:: PIPELINE_MANAGED
       Signals to :class:`MVQueryUnified` that this query bank requires GT
       batch data and should be called by the pipeline, not inside forward().

    Unlike sv_query's QueryBank5 (which generates UV from pixels), motion
    queries depend on GT trajectory annotations and therefore the actual
    sampling still happens inside the pipeline's ``train_step`` / ``val_step``.
    This class owns the ``QueryBuilder`` instance so all sampling parameters
    live in one place, and exposes the same ``QueryBuilder`` API for the
    pipeline to call.

    Parameters mirror ``QueryConfig`` fields exactly so the config block is
    self-documenting.
    """

    PIPELINE_MANAGED = True

    def __init__(
        self,
        num_queries_per_frame: int = 256,
        src_frame_idx: int = 0,
        default_max_frames: int = 1000,
        deterministic: bool = False,
        num_depth_consistency_samples: int = 64,
        cycle_consistency_prob: float = 0.3,
        min_dynamic_ratio: float = 0.0,
        sampler_dynamic_threshold: float = 0.002,
        use_all_points: bool = False,
        random_src_frame: bool = False,
    ):
        super().__init__()
        cfg = QueryConfig(
            num_queries_per_frame=num_queries_per_frame,
            src_frame_idx=src_frame_idx,
            default_max_frames=default_max_frames,
            deterministic=deterministic,
            num_depth_consistency_samples=num_depth_consistency_samples,
            cycle_consistency_prob=cycle_consistency_prob,
            min_dynamic_ratio=min_dynamic_ratio,
            sampler_dynamic_threshold=sampler_dynamic_threshold,
            use_all_points=use_all_points,
            random_src_frame=random_src_frame,
        )
        # Store cfg for pipeline introspection (e.g. self.src shortcut).
        self.cfg = cfg
        self._builder = QueryBuilder(cfg)

    # ── delegate the full QueryBuilder API ───────────────────────────────

    @property
    def src(self) -> int:
        return self.cfg.src_frame_idx

    def motion(self, trajs_2d, trajs_3d, time_idx, visibs, valids,
               img_size, deterministic=None, use_all_points=None):
        return self._builder.motion(
            trajs_2d, trajs_3d, time_idx, visibs, valids,
            img_size, deterministic=deterministic, use_all_points=use_all_points,
        )

    def motion_single_target(self, trajs_2d, trajs_3d, time_idx,
                              visibs, valids, img_size, tgt_frame):
        return self._builder.motion_single_target(
            trajs_2d, trajs_3d, time_idx, visibs, valids, img_size, tgt_frame,
        )

    def depth_consistency(self, B, V, time_idx, device):
        return self._builder.depth_consistency(B, V, time_idx, device)

    def reverse(self, motion_queries):
        return QueryBuilder.reverse(motion_queries)

    def grid(self, B, V, time_idx, device):
        return self._builder.grid(B, V, time_idx, device)

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "MotionQueryBank has no forward(); call .motion() / .grid() etc. "
            "from the pipeline, exactly as with QueryBuilder."
        )
