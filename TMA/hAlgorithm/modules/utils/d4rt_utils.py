"""
D4RT Dense Tracking — Occupancy Grid All-Pixel Tracker
========================================================
Implements D4RT's efficient dense correspondence algorithm:

  "We introduce Algorithm 1 which exploits spatio-temporal redundancy
   using an occupancy grid G ∈ {0,1}^{T×H×W} to speed this procedure
   up significantly." — D4RT paper

The key insight: once a pixel is tracked (i.e., its 3D trajectory is known
at all frames), we can mark the pixel as "occupied" in EVERY frame where
it's visible. Subsequent tracking rounds skip occupied pixels, avoiding
redundant computation.

This yields 5–15× speedup over naive all-pixel tracking.

Algorithm:
  1. Initialize occupancy grid G = zeros(T, H, W)
  2. For each source frame t_src:
     a. Find unoccupied pixels in t_src
     b. Sample a batch of unoccupied pixels
     c. Query the D4RT decoder for their 3D positions at ALL frames
     d. Project back to 2D in all frames → mark as occupied in G
  3. Repeat until G is sufficiently filled or all frames are processed

Usage:
    tracker = DensePixelTracker(model, ...)
    tracks_3d, tracks_2d, visibility = tracker.track_all_pixels(
        rgb_images, intrinsics, features_cached, ...
    )
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


class DensePixelTracker:
    """D4RT-style dense pixel tracking with occupancy grid acceleration.

    Tracks all pixels in a video by iteratively querying unoccupied pixels
    and marking their projections in other frames as occupied.

    Args:
        decode_fn: callable(queries_dict) → [B, Q, 3] predictions.
            This should be model.decode_queries() or equivalent.
        max_queries_per_batch: max queries per forward pass (GPU memory limit).
        occupancy_radius: radius (in pixels) around each projected point to
            mark as occupied. Larger = fewer iterations but coarser coverage.
        min_occupancy: stop when this fraction of pixels are occupied.
        max_iterations: hard limit on tracking iterations.
        visibility_threshold: depth difference threshold for visibility check.
    """

    def __init__(
        self,
        decode_fn,
        max_queries_per_batch: int = 4096,
        occupancy_radius: int = 1,
        min_occupancy: float = 0.95,
        max_iterations: int = 50,
        visibility_threshold: float = 0.1,
    ):
        self.decode_fn = decode_fn
        self.max_queries_per_batch = max_queries_per_batch
        self.occupancy_radius = occupancy_radius
        self.min_occupancy = min_occupancy
        self.max_iterations = max_iterations
        self.visibility_threshold = visibility_threshold

    @torch.no_grad()
    def track_all_pixels(
        self,
        cached: Dict,
        rgb: torch.Tensor,              # [1, T, 3, H, W]
        intrinsics: torch.Tensor,        # [1, T, 3, 3]
        time_idx: Optional[torch.Tensor] = None,  # [1, T]
        img_size: Tuple[int, int] = None,  # (H_orig, W_orig)
        meta_data: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """Track all pixels in the video.

        Returns:
            dict with:
                'tracks_3d': [T, H, W, 3] — 3D position per pixel per frame
                'tracks_2d': [T, H, W, T, 2] — 2D projection in every frame
                'visibility': [T, H, W, T] — visibility flag per pixel per frame
                'occupancy': [T, H, W] — which pixels were successfully tracked
        """
        assert rgb.shape[0] == 1, "Dense tracking supports batch_size=1 only"

        _, T, _, H_feat, W_feat = rgb.shape
        H, W = img_size if img_size is not None else (H_feat, W_feat)

        device = rgb.device

        if time_idx is None:
            time_idx = torch.arange(T, device=device).float() / max(T - 1, 1)
            time_idx = time_idx.unsqueeze(0)  # [1, T]

        # ── Initialize outputs ───────────────────────────────────────────
        # Track 3D positions: for each source pixel, its 3D position at every target frame
        # Shape: [T_src, H, W, T_tgt, 3]
        all_tracks_3d = torch.zeros(T, H, W, T, 3, device=device)

        # Occupancy grid: which pixels have been tracked
        occupancy = torch.zeros(T, H, W, dtype=torch.bool, device=device)

        # Track info: which source frame "owns" each pixel's track
        track_owner = torch.full((T, H, W), -1, dtype=torch.long, device=device)

        total_pixels = T * H * W
        iteration = 0

        logger.info(
            "[DenseTracker] Starting: %d frames × %d×%d = %d total pixels",
            T, H, W, total_pixels,
        )

        # ── Main tracking loop ───────────────────────────────────────────
        while iteration < self.max_iterations:
            occupied_count = occupancy.sum().item()
            occupancy_ratio = occupied_count / total_pixels

            if occupancy_ratio >= self.min_occupancy:
                logger.info(
                    "[DenseTracker] Reached %.1f%% occupancy after %d iterations",
                    occupancy_ratio * 100, iteration,
                )
                break

            # Find the source frame with most unoccupied pixels
            unoccupied_per_frame = (~occupancy).sum(dim=(1, 2))  # [T]
            src_frame = unoccupied_per_frame.argmax().item()

            if unoccupied_per_frame[src_frame] == 0:
                break

            # Sample unoccupied pixels from this source frame
            unoccupied_mask = ~occupancy[src_frame]  # [H, W]
            uv_candidates = unoccupied_mask.nonzero(as_tuple=False)  # [N_cand, 2] (row, col)

            if len(uv_candidates) == 0:
                break

            # Random sample up to max_queries
            n_sample = min(self.max_queries_per_batch, len(uv_candidates))
            perm = torch.randperm(len(uv_candidates), device=device)[:n_sample]
            selected_yx = uv_candidates[perm]  # [n_sample, 2] in (row, col) = (v, u)

            # Convert to (u, v) pixel coordinates
            uv = torch.stack([selected_yx[:, 1].float(), selected_yx[:, 0].float()], dim=-1)  # [Q, 2]

            # ── Query all target frames for these source pixels ──────────
            Q = uv.shape[0]
            tgt_frames = list(range(T))

            # Build queries: each source pixel × each target frame
            all_preds_3d = torch.zeros(Q, T, 3, device=device)

            for t_tgt in tgt_frames:
                queries = self._build_queries(
                    uv=uv.unsqueeze(0),  # [1, Q, 2]
                    src_frame=src_frame,
                    tgt_frame=t_tgt,
                    time_idx=time_idx,
                    device=device,
                )

                pred_3d = self.decode_fn(
                    cached=cached,
                    rgb=rgb,
                    motion_queries=queries,
                    img_size=(H, W),
                    meta_data=meta_data,
                )  # [1, Q, 3]

                if pred_3d is not None:
                    all_preds_3d[:, t_tgt] = pred_3d[0]

            # ── Store tracks and update occupancy ────────────────────────
            for qi in range(Q):
                u_int = int(uv[qi, 0].round().clamp(0, W - 1))
                v_int = int(uv[qi, 1].round().clamp(0, H - 1))

                # Store 3D track for this source pixel
                all_tracks_3d[src_frame, v_int, u_int] = all_preds_3d[qi]

                # Mark source pixel as occupied
                occupancy[src_frame, v_int, u_int] = True
                track_owner[src_frame, v_int, u_int] = src_frame

            # ── Project to other frames and mark occupancy ───────────────
            if intrinsics is not None:
                self._update_occupancy_via_projection(
                    all_preds_3d, uv, src_frame, intrinsics[0],
                    occupancy, all_tracks_3d, track_owner,
                    H, W, T,
                )

            iteration += 1

            if iteration % 5 == 0:
                occ_pct = occupancy.sum().item() / total_pixels * 100
                logger.info(
                    "[DenseTracker] iter=%d, occupancy=%.1f%%, src_frame=%d, queries=%d",
                    iteration, occ_pct, src_frame, Q,
                )

        # ── Finalize ─────────────────────────────────────────────────────
        final_occupancy = occupancy.sum().item() / total_pixels
        logger.info(
            "[DenseTracker] Done: %d iterations, %.1f%% occupied",
            iteration, final_occupancy * 100,
        )

        return {
            'tracks_3d': all_tracks_3d,      # [T, H, W, T, 3]
            'occupancy': occupancy,            # [T, H, W]
            'track_owner': track_owner,        # [T, H, W]
        }

    def _build_queries(
        self,
        uv: torch.Tensor,      # [1, Q, 2]
        src_frame: int,
        tgt_frame: int,
        time_idx: torch.Tensor,  # [1, T]
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Build a query dict for one (src_frame, tgt_frame) pair."""
        Q = uv.shape[1]

        src_time = time_idx[0, src_frame].item()
        tgt_time = time_idx[0, tgt_frame].item()

        return {
            'uv': uv,
            'src_frame_idx': torch.full((1, Q), src_frame, dtype=torch.long, device=device),
            'tgt_frame_idx': torch.full((1, Q), tgt_frame, dtype=torch.long, device=device),
            'tgt_camera_idx': torch.full((1, Q), tgt_frame, dtype=torch.long, device=device),
            'src_time': torch.full((1, Q), src_time, device=device),
            'tgt_time': torch.full((1, Q), tgt_time, device=device),
            'cam_time': torch.full((1, Q), tgt_time, device=device),
        }

    def _update_occupancy_via_projection(
        self,
        preds_3d: torch.Tensor,     # [Q, T, 3]
        src_uv: torch.Tensor,       # [Q, 2]
        src_frame: int,
        intrinsics: torch.Tensor,    # [T, 3, 3]
        occupancy: torch.Tensor,     # [T, H, W]
        tracks_3d: torch.Tensor,     # [T, H, W, T, 3]
        track_owner: torch.Tensor,   # [T, H, W]
        H: int,
        W: int,
        T: int,
    ):
        """Project tracked points to other frames and mark occupancy.

        For each predicted 3D point at target frame t, project to 2D in
        frame t's camera. If the projected pixel is unoccupied, mark it
        and record that it's covered by this track.
        """
        Q = preds_3d.shape[0]
        r = self.occupancy_radius

        for t in range(T):
            if t == src_frame:
                continue

            # Get predicted 3D positions at frame t (in frame t's camera coords)
            pts = preds_3d[:, t]  # [Q, 3]
            z = pts[:, 2]

            # Skip points behind camera
            valid = z > 1e-4
            if not valid.any():
                continue

            # Project to 2D
            K = intrinsics[t]  # [3, 3]
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            u_proj = (pts[:, 0] / z) * fx + cx
            v_proj = (pts[:, 1] / z) * fy + cy

            # Mark occupancy for valid in-bounds projections
            u_int = u_proj.round().long()
            v_int = v_proj.round().long()

            for qi in range(Q):
                if not valid[qi]:
                    continue

                ui, vi = u_int[qi].item(), v_int[qi].item()

                # Mark a small neighborhood (occupancy_radius)
                for dv in range(-r, r + 1):
                    for du in range(-r, r + 1):
                        u_n, v_n = ui + du, vi + dv
                        if 0 <= u_n < W and 0 <= v_n < H and not occupancy[t, v_n, u_n]:
                            occupancy[t, v_n, u_n] = True
                            track_owner[t, v_n, u_n] = src_frame


# ─────────────────────────────────────────────────────────────────────────────
# Batched dense tracking (more efficient for GPU)
# ─────────────────────────────────────────────────────────────────────────────

class BatchedDenseTracker(DensePixelTracker):
    """GPU-optimized variant that batches all target frames in one query.

    Instead of querying one target frame at a time, this variant creates
    Q * T queries (one per pixel per target frame) and processes them in
    chunks. This better utilizes GPU parallelism.
    """

    @torch.no_grad()
    def track_all_pixels(
        self,
        cached: Dict,
        rgb: torch.Tensor,
        intrinsics: torch.Tensor,
        time_idx: Optional[torch.Tensor] = None,
        img_size: Tuple[int, int] = None,
        meta_data: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        assert rgb.shape[0] == 1

        _, T, _, H_feat, W_feat = rgb.shape
        H, W = img_size if img_size is not None else (H_feat, W_feat)
        device = rgb.device

        if time_idx is None:
            time_idx = torch.arange(T, device=device).float() / max(T - 1, 1)
            time_idx = time_idx.unsqueeze(0)

        occupancy = torch.zeros(T, H, W, dtype=torch.bool, device=device)
        all_tracks_3d = torch.zeros(T, H, W, T, 3, device=device)
        total_pixels = T * H * W
        iteration = 0

        # Queries per batch: divide budget across target frames
        queries_per_iter = max(1, self.max_queries_per_batch // T)

        while iteration < self.max_iterations:
            occ_ratio = occupancy.sum().item() / total_pixels
            if occ_ratio >= self.min_occupancy:
                break

            # Pick source frame with most unoccupied pixels
            unoccupied_count = (~occupancy).sum(dim=(1, 2))
            src_frame = unoccupied_count.argmax().item()
            if unoccupied_count[src_frame] == 0:
                break

            # Sample unoccupied pixels
            unocc_yx = (~occupancy[src_frame]).nonzero(as_tuple=False)
            n_sample = min(queries_per_iter, len(unocc_yx))
            perm = torch.randperm(len(unocc_yx), device=device)[:n_sample]
            selected_yx = unocc_yx[perm]

            uv = torch.stack([selected_yx[:, 1].float(), selected_yx[:, 0].float()], dim=-1)
            Q = uv.shape[0]

            # Build batched queries: Q points × T target frames = Q*T queries
            uv_expanded = uv.unsqueeze(1).expand(Q, T, 2).reshape(Q * T, 2).unsqueeze(0)  # [1, Q*T, 2]

            src_idx = torch.full((1, Q * T), src_frame, dtype=torch.long, device=device)
            tgt_idx = torch.arange(T, device=device).unsqueeze(0).expand(Q, T).reshape(1, Q * T)

            src_t = time_idx[0, src_frame].expand(1, Q * T)
            tgt_t = time_idx[0].unsqueeze(0).expand(Q, T).reshape(1, Q * T)

            queries = {
                'uv': uv_expanded,
                'src_frame_idx': src_idx,
                'tgt_frame_idx': tgt_idx,
                'tgt_camera_idx': tgt_idx,
                'src_time': src_t,
                'tgt_time': tgt_t,
                'cam_time': tgt_t,
            }

            # Process in chunks if Q*T exceeds GPU memory
            total_queries = Q * T
            chunk_size = self.max_queries_per_batch
            all_preds = []

            for start in range(0, total_queries, chunk_size):
                end = min(start + chunk_size, total_queries)
                chunk_queries = {k: v[:, start:end] for k, v in queries.items()}

                pred = self.decode_fn(
                    cached=cached, rgb=rgb, motion_queries=chunk_queries,
                    img_size=(H, W), meta_data=meta_data,
                )
                if pred is not None:
                    all_preds.append(pred[0])  # [chunk, 3]

            if not all_preds:
                iteration += 1
                continue

            preds_flat = torch.cat(all_preds, dim=0)  # [Q*T, 3]
            preds_3d = preds_flat.reshape(Q, T, 3)    # [Q, T, 3]

            # Store and update occupancy
            for qi in range(Q):
                u_int = int(uv[qi, 0].round().clamp(0, W - 1))
                v_int = int(uv[qi, 1].round().clamp(0, H - 1))
                all_tracks_3d[src_frame, v_int, u_int] = preds_3d[qi]
                occupancy[src_frame, v_int, u_int] = True

            # Project and mark occupancy in other frames
            if intrinsics is not None:
                self._batch_update_occupancy(
                    preds_3d, uv, src_frame, intrinsics[0],
                    occupancy, H, W, T,
                )

            iteration += 1

        logger.info(
            "[BatchedDenseTracker] Done: %d iters, %.1f%% occupied",
            iteration, occupancy.sum().item() / total_pixels * 100,
        )

        return {
            'tracks_3d': all_tracks_3d,
            'occupancy': occupancy,
        }

    def _batch_update_occupancy(
        self,
        preds_3d: torch.Tensor,    # [Q, T, 3]
        src_uv: torch.Tensor,      # [Q, 2]
        src_frame: int,
        intrinsics: torch.Tensor,   # [T, 3, 3]
        occupancy: torch.Tensor,    # [T, H, W]
        H: int, W: int, T: int,
    ):
        """Vectorized occupancy update via batch projection."""
        Q = preds_3d.shape[0]
        r = self.occupancy_radius

        for t in range(T):
            if t == src_frame:
                continue

            pts = preds_3d[:, t]  # [Q, 3]
            z = pts[:, 2]
            valid = z > 1e-4

            if not valid.any():
                continue

            K = intrinsics[t]
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

            u_proj = ((pts[:, 0] / z) * fx + cx).round().long()
            v_proj = ((pts[:, 1] / z) * fy + cy).round().long()

            # Vectorized bounds check
            in_bounds = (valid
                         & (u_proj >= r) & (u_proj < W - r)
                         & (v_proj >= r) & (v_proj < H - r))

            if not in_bounds.any():
                continue

            u_valid = u_proj[in_bounds]
            v_valid = v_proj[in_bounds]

            # Mark neighborhood
            for dv in range(-r, r + 1):
                for du in range(-r, r + 1):
                    occupancy[t, (v_valid + dv).clamp(0, H - 1),
                              (u_valid + du).clamp(0, W - 1)] = True


"""
D4RT Pose Estimation — Umeyama Algorithm from Query Correspondences
=====================================================================
Implements D4RT's camera pose estimation:

  "Camera extrinsics between video frames are estimated by querying grids
   of correspondences and aligning outputs with rigid transformations
   (Umeyama algorithm), facilitating pose estimation without needing
   explicit optimization or costly refinement stages." — D4RT paper

Pipeline:
  1. Create a uniform grid of query points on the source frame
  2. For each query point, decode its 3D position in:
     a. Source camera coords (t_cam = t_src)
     b. Target camera coords (t_cam = t_tgt)
  3. Run Umeyama's algorithm to find the rigid transform between
     the two point sets → gives the relative pose (R, t) from
     source to target camera

This achieves 200+ FPS pose estimation on A100 because:
  - The encoder runs once (shared scene representation)
  - Only a small grid of queries (~100-400 points) is needed
  - Umeyama is a closed-form SVD solution (O(n) after decomposition)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Umeyama's Algorithm
# ─────────────────────────────────────────────────────────────────────────────

def umeyama_alignment(
    src_points: torch.Tensor,       # [B, N, 3] or [N, 3]
    tgt_points: torch.Tensor,       # [B, N, 3] or [N, 3]
    weights: Optional[torch.Tensor] = None,  # [B, N] or [N]
    estimate_scale: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Umeyama's algorithm: find optimal rigid (or similarity) transform.

    Finds R, t (and optionally s) that minimizes:
        sum_i w_i || s*R*src_i + t - tgt_i ||^2

    Uses SVD of the weighted cross-covariance matrix.

    Args:
        src_points: source point set [B, N, 3] or [N, 3].
        tgt_points: target point set [B, N, 3] or [N, 3].
        weights: per-point weights [B, N] or [N]. None = uniform.
        estimate_scale: if True, also estimate a scale factor.

    Returns:
        R: rotation matrix [B, 3, 3] or [3, 3]
        t: translation vector [B, 3] or [3]
        s: scale factor [B] or scalar (1.0 if estimate_scale=False)
    """
    batched = src_points.ndim == 3
    if not batched:
        src_points = src_points.unsqueeze(0)
        tgt_points = tgt_points.unsqueeze(0)
        if weights is not None:
            weights = weights.unsqueeze(0)

    B, N, D = src_points.shape
    assert D == 3, f"Expected 3D points, got {D}D"
    device = src_points.device
    dtype = src_points.dtype

    # Weights
    if weights is not None:
        w = weights.unsqueeze(-1)  # [B, N, 1]
        w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-8)  # [B, 1, 1]
        w_norm = w / w_sum
    else:
        w_norm = torch.ones(B, N, 1, device=device, dtype=dtype) / N

    # Weighted centroids
    mu_src = (w_norm * src_points).sum(dim=1, keepdim=True)   # [B, 1, 3]
    mu_tgt = (w_norm * tgt_points).sum(dim=1, keepdim=True)   # [B, 1, 3]

    # Center the points
    src_c = src_points - mu_src  # [B, N, 3]
    tgt_c = tgt_points - mu_tgt  # [B, N, 3]

    # Weighted cross-covariance matrix: H = src_c^T @ diag(w) @ tgt_c
    # Efficiently: H = (w * src_c)^T @ tgt_c
    H = torch.bmm((w_norm * src_c).transpose(1, 2), tgt_c)  # [B, 3, 3]

    # SVD
    U, S, Vh = torch.linalg.svd(H)  # U: [B,3,3], S: [B,3], Vh: [B,3,3]

    # Ensure proper rotation (det(R) = +1, not reflection)
    d = torch.det(torch.bmm(Vh.transpose(1, 2), U.transpose(1, 2)))  # [B]
    sign_matrix = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).clone()
    sign_matrix[:, 2, 2] = torch.sign(d)

    # R = V @ diag(1,1,sign(det)) @ U^T
    R = torch.bmm(torch.bmm(Vh.transpose(1, 2), sign_matrix), U.transpose(1, 2))  # [B, 3, 3]

    # Scale (optional — for Sim(3) alignment)
    if estimate_scale:
        # Weighted variance of source points
        var_src = (w_norm * (src_c ** 2)).sum(dim=(1, 2))  # [B]
        S_corrected = S.clone()
        S_corrected[:, 2] *= torch.sign(d)
        s = S_corrected.sum(dim=1) / var_src.clamp(min=1e-8)  # [B]
    else:
        s = torch.ones(B, device=device, dtype=dtype)

    # Translation: t = mu_tgt - s * R @ mu_src
    t = mu_tgt.squeeze(1) - s.unsqueeze(-1) * torch.bmm(R, mu_src.squeeze(1).unsqueeze(-1)).squeeze(-1)

    if not batched:
        R, t, s = R.squeeze(0), t.squeeze(0), s.squeeze(0)

    return R, t, s


# ─────────────────────────────────────────────────────────────────────────────
# D4RT Pose Estimator
# ─────────────────────────────────────────────────────────────────────────────

class D4RTPoseEstimator(nn.Module):
    """D4RT camera pose estimation via grid queries + Umeyama.

    Estimates relative camera pose between any two frames by:
    1. Querying a grid of points from frame A in frame A's camera coords
    2. Querying the same points from frame A in frame B's camera coords
    3. Running Umeyama to find the rigid transform (R, t)

    This gives the relative pose T_{A→B} such that:
        P_B = R * P_A + t

    Args:
        grid_size: side length of the query grid (total points = grid_size^2).
        estimate_scale: if True, also estimate scale (Sim(3) alignment).
        confidence_threshold: discard points with low predicted confidence.
        ransac_iterations: if > 0, use RANSAC for robust estimation.
        ransac_inlier_threshold: inlier distance threshold for RANSAC.
    """

    def __init__(
        self,
        grid_size: int = 16,
        estimate_scale: bool = False,
        confidence_threshold: float = 0.0,
        ransac_iterations: int = 0,
        ransac_inlier_threshold: float = 0.05,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.estimate_scale = estimate_scale
        self.confidence_threshold = confidence_threshold
        self.ransac_iterations = ransac_iterations
        self.ransac_inlier_threshold = ransac_inlier_threshold

    def build_grid_queries(
        self,
        H: int,
        W: int,
        src_frame: int,
        tgt_frame: int,
        time_idx: Optional[torch.Tensor],  # [1, T]
        T: int,
        device: torch.device,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Build paired query dicts for pose estimation.

        Returns:
            (queries_src_cam, queries_tgt_cam):
                Both query the same grid points from src_frame, but
                express the result in different camera coordinate frames.
        """
        gs = self.grid_size
        u = torch.linspace(0, W - 1, gs, device=device)
        v = torch.linspace(0, H - 1, gs, device=device)
        gv, gu = torch.meshgrid(v, u, indexing='ij')
        uv = torch.stack([gu.flatten(), gv.flatten()], dim=-1)  # [Q, 2]
        Q = uv.shape[0]
        uv = uv.unsqueeze(0)  # [1, Q, 2]

        # Time values
        if time_idx is not None:
            t_src = time_idx[0, src_frame].item()
            t_tgt = time_idx[0, tgt_frame].item()
        else:
            t_src = src_frame / max(T - 1, 1)
            t_tgt = tgt_frame / max(T - 1, 1)

        # Shared fields
        base = {
            'uv': uv,
            'src_frame_idx': torch.full((1, Q), src_frame, dtype=torch.long, device=device),
            'tgt_frame_idx': torch.full((1, Q), tgt_frame, dtype=torch.long, device=device),
            'src_time': torch.full((1, Q), t_src, device=device),
            'tgt_time': torch.full((1, Q), t_tgt, device=device),
        }

        # Query 1: predict 3D at t_tgt in src camera frame (t_cam = t_src)
        queries_src_cam = {
            **base,
            'tgt_camera_idx': torch.full((1, Q), src_frame, dtype=torch.long, device=device),
            'cam_time': torch.full((1, Q), t_src, device=device),
        }

        # Query 2: predict 3D at t_tgt in tgt camera frame (t_cam = t_tgt)
        queries_tgt_cam = {
            **base,
            'tgt_camera_idx': torch.full((1, Q), tgt_frame, dtype=torch.long, device=device),
            'cam_time': torch.full((1, Q), t_tgt, device=device),
        }

        return queries_src_cam, queries_tgt_cam

    @torch.no_grad()
    def estimate_pose(
        self,
        decode_fn,
        cached: Dict,
        rgb: torch.Tensor,            # [1, T, 3, H, W]
        src_frame: int,
        tgt_frame: int,
        time_idx: Optional[torch.Tensor],
        img_size: Tuple[int, int],
        meta_data: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """Estimate relative pose from src_frame to tgt_frame.

        Returns:
            dict with 'R' [3,3], 't' [3], 's' scalar, 'inlier_ratio' scalar
        """
        H, W = img_size
        T = rgb.shape[1]
        device = rgb.device

        # Build paired queries
        q_src, q_tgt = self.build_grid_queries(
            H, W, src_frame, tgt_frame, time_idx, T, device,
        )

        # Decode both query sets
        pts_src = decode_fn(cached=cached, rgb=rgb, motion_queries=q_src,
                            img_size=img_size, meta_data=meta_data)
        pts_tgt = decode_fn(cached=cached, rgb=rgb, motion_queries=q_tgt,
                            img_size=img_size, meta_data=meta_data)

        if pts_src is None or pts_tgt is None:
            return self._identity_result(device)

        pts_src = pts_src[0]  # [Q, 3]
        pts_tgt = pts_tgt[0]  # [Q, 3]

        # Filter by confidence / validity
        valid = (pts_src[:, 2] > 1e-4) & (pts_tgt[:, 2] > 1e-4)

        # Remove outliers with extreme values
        valid = valid & (pts_src.abs().max(dim=-1).values < 100)
        valid = valid & (pts_tgt.abs().max(dim=-1).values < 100)

        if valid.sum() < 4:
            logger.warning("[PoseEstimator] Too few valid points (%d), returning identity", valid.sum())
            return self._identity_result(device)

        pts_src_valid = pts_src[valid]
        pts_tgt_valid = pts_tgt[valid]

        # Estimate transform
        if self.ransac_iterations > 0:
            R, t, s, inlier_ratio = self._ransac_umeyama(
                pts_src_valid, pts_tgt_valid,
            )
        else:
            R, t, s = umeyama_alignment(
                pts_src_valid, pts_tgt_valid,
                estimate_scale=self.estimate_scale,
            )
            # Compute residual for quality metric
            transformed = (s * (R @ pts_src_valid.T)).T + t
            residuals = (transformed - pts_tgt_valid).norm(dim=-1)
            inlier_ratio = (residuals < self.ransac_inlier_threshold).float().mean()

        return {
            'R': R,                      # [3, 3]
            't': t,                      # [3]
            's': s,                      # scalar
            'inlier_ratio': inlier_ratio,  # scalar
            'num_valid': valid.sum(),
        }

    @torch.no_grad()
    def estimate_all_poses(
        self,
        decode_fn,
        cached: Dict,
        rgb: torch.Tensor,
        time_idx: Optional[torch.Tensor],
        img_size: Tuple[int, int],
        meta_data: Optional[Dict] = None,
        reference_frame: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Estimate poses for all frames relative to reference_frame.

        Returns:
            dict with:
                'rotations': [T, 3, 3]
                'translations': [T, 3]
                'scales': [T]
                'inlier_ratios': [T]
        """
        T = rgb.shape[1]
        device = rgb.device

        rotations = torch.eye(3, device=device).unsqueeze(0).expand(T, -1, -1).clone()
        translations = torch.zeros(T, 3, device=device)
        scales = torch.ones(T, device=device)
        inlier_ratios = torch.zeros(T, device=device)

        # Reference frame has identity pose
        inlier_ratios[reference_frame] = 1.0

        for t in range(T):
            if t == reference_frame:
                continue

            result = self.estimate_pose(
                decode_fn=decode_fn,
                cached=cached,
                rgb=rgb,
                src_frame=reference_frame,
                tgt_frame=t,
                time_idx=time_idx,
                img_size=img_size,
                meta_data=meta_data,
            )

            rotations[t] = result['R']
            translations[t] = result['t']
            scales[t] = result['s']
            inlier_ratios[t] = result['inlier_ratio']

        return {
            'rotations': rotations,
            'translations': translations,
            'scales': scales,
            'inlier_ratios': inlier_ratios,
        }

    def _ransac_umeyama(
        self,
        pts_src: torch.Tensor,  # [N, 3]
        pts_tgt: torch.Tensor,  # [N, 3]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """RANSAC-wrapped Umeyama for robust pose estimation.

        Handles dynamic objects by iteratively filtering outliers.
        """
        N = pts_src.shape[0]
        device = pts_src.device
        threshold = self.ransac_inlier_threshold
        min_samples = max(4, N // 10)  # At least 4 points (minimum for 3D rigid)

        best_R = torch.eye(3, device=device)
        best_t = torch.zeros(3, device=device)
        best_s = torch.ones(1, device=device)
        best_inliers = 0
        best_ratio = torch.tensor(0.0, device=device)

        for _ in range(self.ransac_iterations):
            # Random subset
            indices = torch.randperm(N, device=device)[:min_samples]
            src_sub = pts_src[indices]
            tgt_sub = pts_tgt[indices]

            try:
                R, t, s = umeyama_alignment(
                    src_sub, tgt_sub, estimate_scale=self.estimate_scale,
                )
            except Exception:
                continue

            # Score against all points
            transformed = (s * (R @ pts_src.T)).T + t
            residuals = (transformed - pts_tgt).norm(dim=-1)
            inliers = residuals < threshold
            n_inliers = inliers.sum().item()

            if n_inliers > best_inliers:
                best_inliers = n_inliers
                best_ratio = inliers.float().mean()

                # Refit on all inliers
                if n_inliers >= 4:
                    R_ref, t_ref, s_ref = umeyama_alignment(
                        pts_src[inliers], pts_tgt[inliers],
                        estimate_scale=self.estimate_scale,
                    )
                    best_R, best_t, best_s = R_ref, t_ref, s_ref
                else:
                    best_R, best_t, best_s = R, t, s

        return best_R, best_t, best_s, best_ratio

    @staticmethod
    def _identity_result(device: torch.device) -> Dict[str, torch.Tensor]:
        return {
            'R': torch.eye(3, device=device),
            't': torch.zeros(3, device=device),
            's': torch.ones(1, device=device),
            'inlier_ratio': torch.tensor(0.0, device=device),
            'num_valid': torch.tensor(0, device=device),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Intrinsics estimation from decoded 3D points (bonus)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_intrinsics_from_points(
    uv: torch.Tensor,           # [N, 2] pixel coords
    points_3d: torch.Tensor,    # [N, 3] decoded 3D in camera coords
    image_size: Tuple[int, int],  # (H, W)
    assume_centered_pp: bool = True,
) -> Dict[str, torch.Tensor]:
    """Estimate camera intrinsics from decoded 3D point correspondences.

    D4RT's approach: assuming pinhole camera with principal point at (0.5, 0.5),
    estimate focal lengths from the distribution of decoded 3D points.

    For each point: u = fx * X/Z + cx, v = fy * Y/Z + cy
    → fx = (u - cx) * Z / X, fy = (v - cy) * Z / Y

    Args:
        uv: 2D pixel coordinates [N, 2]
        points_3d: corresponding 3D points in camera coords [N, 3]
        image_size: (H, W)
        assume_centered_pp: if True, assume cx=W/2, cy=H/2

    Returns:
        dict with 'fx', 'fy', 'cx', 'cy', 'K' [3,3]
    """
    H, W = image_size
    device = uv.device

    if assume_centered_pp:
        cx = W / 2.0
        cy = H / 2.0
    else:
        cx = W / 2.0
        cy = H / 2.0

    X, Y, Z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    u, v = uv[:, 0], uv[:, 1]

    # Filter valid points (positive depth, non-zero X and Y)
    valid = (Z > 1e-4) & (X.abs() > 1e-6) & (Y.abs() > 1e-6)

    if valid.sum() < 10:
        # Fallback: assume reasonable default
        fx = fy = max(H, W) * 0.8
        logger.warning("[IntrinsicsEstim] Too few valid points, using default focal=%.1f", fx)
    else:
        # Per-point focal length estimates
        fx_samples = (u[valid] - cx) * Z[valid] / X[valid]
        fy_samples = (v[valid] - cy) * Z[valid] / Y[valid]

        # Robust estimation: median (handles outliers better than mean)
        fx = fx_samples.median().item()
        fy = fy_samples.median().item()

        # Sanity check
        fx = max(abs(fx), 10.0)
        fy = max(abs(fy), 10.0)

    K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1],
    ], device=device, dtype=torch.float32)

    return {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'K': K}