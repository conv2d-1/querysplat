import logging
import os

import cv2
import numpy as np


def _generate_colors(n, seed=0):
    """Generate *n* distinct BGR colours via evenly-spaced hue in HSV."""
    if n <= 0:
        return []
    rng = np.random.RandomState(seed)
    hues = np.linspace(0, 179, n, endpoint=False).astype(np.uint8)
    rng.shuffle(hues)
    hsv = np.stack([hues, np.full(n, 220, dtype=np.uint8), np.full(n, 230, dtype=np.uint8)], axis=-1).reshape(-1, 1, 3)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3)
    return [tuple(int(c) for c in row) for row in bgr]


def _uv_to_pixel(uv, h, w):
    """Normalized [0, 1] UV → pixel coords (x, y)."""
    px = uv[..., 0] * (w - 1)
    py = uv[..., 1] * (h - 1)
    return np.stack([px, py], axis=-1)


def _to_np(t, min_ndim=1):
    """Tensor / ndarray → numpy, squeeze batch dim if present.

    Args:
        min_ndim: minimum number of dimensions to keep after squeezing.
            Use 2 for coordinate arrays (Q, 2/3) to avoid collapsing to 1-D.
    """
    if t is None:
        return None
    if hasattr(t, 'numpy'):
        t = t.numpy()
    while t.ndim > min_ndim and t.shape[0] == 1:
        t = t[0]
    return t.copy()


def _sample_rgb_at_uv(img_rgb, uv_norm):
    """Sample RGB colours from *img_rgb* at normalised UV positions.

    Args:
        img_rgb: (H, W, 3) float32 image in [0, 1].
        uv_norm: (N, 2) normalised coordinates in [0, 1] (u=x, v=y).

    Returns:
        (N, 3) float64 RGB colours in [0, 1].
    """
    H, W = img_rgb.shape[:2]
    px = np.clip((uv_norm[:, 0] * (W - 1)).round().astype(int), 0, W - 1)
    py = np.clip((uv_norm[:, 1] * (H - 1)).round().astype(int), 0, H - 1)
    return img_rgb[py, px].astype(np.float64)


def _grid_sample_at_uv(grid_data, uv_norm, grid_h, grid_w):
    """Sample from a flat grid array at normalised UV positions (nearest).

    Args:
        grid_data: (H*W, D) flat array of grid predictions.
        uv_norm:   (N, 2) normalised [0,1] coordinates (u=x, v=y).
        grid_h, grid_w: grid height / width so that H*W == grid_data.shape[0].

    Returns:
        (N, D) sampled values.
    """
    col = np.clip(np.round(uv_norm[:, 0] * (grid_w - 1)).astype(int), 0, grid_w - 1)
    row = np.clip(np.round(uv_norm[:, 1] * (grid_h - 1)).astype(int), 0, grid_h - 1)
    flat_idx = row * grid_w + col
    return grid_data[flat_idx]


def _subsample_ids(ids, max_n, seed=42):
    if max_n is None or len(ids) <= max_n:
        return ids
    rng = np.random.RandomState(seed)
    ids = rng.choice(ids, max_n, replace=False)
    ids.sort()
    return ids


def _grid_subsample_ids(grid_h, grid_w, max_n):
    """Select evenly spaced ids from a flattened HxW grid."""
    total_n = int(grid_h) * int(grid_w)
    if max_n is None or total_n <= max_n:
        return np.arange(total_n, dtype=int)

    grid_h = int(grid_h)
    grid_w = int(grid_w)
    n_rows = max(1, min(grid_h, int(round(np.sqrt(max_n * grid_h / max(grid_w, 1))))))
    n_cols = max(1, min(grid_w, int(np.ceil(max_n / n_rows))))

    while n_rows * n_cols > max_n:
        if n_cols >= n_rows and n_cols > 1:
            n_cols -= 1
        elif n_rows > 1:
            n_rows -= 1
        else:
            break

    rows = np.unique(np.round(np.linspace(0, grid_h - 1, n_rows)).astype(int))
    cols = np.unique(np.round(np.linspace(0, grid_w - 1, n_cols)).astype(int))
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return np.sort((rr * grid_w + cc).reshape(-1))


def _sigmoid(x):
    x = np.clip(x.astype(np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _overlay_motion_mask_on_bgr(
    src_bgr,
    motion_mask,
    query_uv_norm,
    query_h,
    query_w,
    overlay_color_bgr=(0, 60, 255),
    overlay_alpha=0.45,
):
    """Blend motion mask over source RGB (view A): dense grid → resize, else sparse at query_uv."""
    H_s, W_s = src_bgr.shape[:2]
    mm = np.asarray(motion_mask).reshape(-1)
    if mm.size == 0:
        return src_bgr

    if mm.dtype == bool or mm.dtype == np.bool_:
        mm_f = mm.astype(np.float64)
    elif mm.size and (mm.max() > 1.0 or mm.min() < 0.0):
        mm_f = _sigmoid(mm)
    else:
        mm_f = np.clip(mm.astype(np.float64), 0.0, 1.0)

    base = src_bgr.astype(np.float64)

    if (
        query_h is not None
        and query_w is not None
        and mm_f.size == int(query_h) * int(query_w)
    ):
        dense = mm_f.reshape(int(query_h), int(query_w)).astype(np.float32)
        mask_hw = cv2.resize(dense, (W_s, H_s), interpolation=cv2.INTER_LINEAR)
        mask_hw = np.clip(mask_hw, 0.0, 1.0)
        color = np.array(overlay_color_bgr, dtype=np.float64).reshape(1, 1, 3)
        alpha = (mask_hw[..., None] * overlay_alpha).astype(np.float64)
        out = base * (1.0 - alpha) + color * alpha
        return np.clip(out, 0, 255).astype(np.uint8)

    if query_uv_norm is None or len(query_uv_norm) == 0:
        return src_bgr
    n = min(len(mm_f), len(query_uv_norm))
    radius = max(2, min(H_s, W_s) // 160)
    overlay = np.zeros_like(src_bgr)
    for i in range(n):
        if mm_f[i] < 0.5:
            continue
        sx, sy = _uv_to_pixel(query_uv_norm[i : i + 1], H_s, W_s)[0]
        cv2.circle(overlay, (int(sx), int(sy)), radius, tuple(int(c) for c in overlay_color_bgr), -1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, overlay_alpha, src_bgr, 1.0 - overlay_alpha, 0)


def _draw_points_on_pair(
    src_img_bgr, tgt_img_bgr,
    src_uv, tgt_uv,
    H_s, W_s, H_t, W_t,
    point_ids, colors,
    src_vis=None, tgt_vis=None,
    src_is_normalized=True, tgt_is_normalized=True,
    marker="circle",
    line_alpha=0.3,
):
    """Draw coloured points on source / target canvases with connecting lines.

    Args:
        point_ids: indices into src_uv / tgt_uv to draw.
        colors: one colour per point_id.
        src_vis: boolean array indexed by point id; None = all visible.
        tgt_vis: boolean array indexed by point id; None = all visible.
        marker: "circle" → filled/hollow circles, "cross" → X marks on target.
        line_alpha: opacity of the connecting lines (0 = invisible, 1 = fully opaque).
    """
    canvas_src = src_img_bgr.copy()
    canvas_tgt = tgt_img_bgr.copy()
    radius = max(1, min(H_s, W_s) // 200)
    thickness = max(1, radius // 2)

    if H_s != H_t:
        canvas_tgt = cv2.resize(canvas_tgt, (int(W_t * H_s / H_t), H_s))

    combined = np.concatenate([canvas_src, canvas_tgt], axis=1)

    def _px(uv_row, h, w, is_norm):
        if is_norm:
            return _uv_to_pixel(uv_row, h, w)
        return uv_row

    # ── pass 1: semi-transparent connecting lines ──
    if line_alpha > 0 and src_uv is not None and tgt_uv is not None:
        overlay = combined.copy()
        for ci, qi in enumerate(point_ids):
            if qi >= len(src_uv) or qi >= len(tgt_uv):
                continue
            sx, sy = _px(src_uv[qi:qi+1], H_s, W_s, src_is_normalized)[0]
            tx, ty = _px(tgt_uv[qi:qi+1], H_t, W_t, tgt_is_normalized)[0]
            try:
                cv2.line(overlay, (int(sx), int(sy)), (int(tx) + W_s, int(ty)), colors[ci], 1, cv2.LINE_AA)
            except:
                logging.error(f"error: sx{sx} sy{sy} tx{tx} ty{ty}")
                continue
        combined = cv2.addWeighted(overlay, line_alpha, combined, 1 - line_alpha, 0)

    # ── pass 2: opaque points on top ──
    for ci, qi in enumerate(point_ids):
        color = colors[ci]
        s_visible = True if src_vis is None else bool(src_vis[qi])
        t_visible = True if tgt_vis is None else bool(tgt_vis[qi])

        if src_uv is not None and qi < len(src_uv):
            sx, sy = _px(src_uv[qi:qi+1], H_s, W_s, src_is_normalized)[0]
            fill_s = -1 if s_visible else thickness
            try:
                cv2.circle(combined, (int(sx), int(sy)), radius, color, fill_s, cv2.LINE_AA)
            except:
                logging.error(f"error: sx{sx} sy{sy}")
                continue

        if tgt_uv is not None and qi < len(tgt_uv):
            tx, ty = _px(tgt_uv[qi:qi+1], H_t, W_t, tgt_is_normalized)[0]
            try:
                tgt_x, tgt_y = int(tx) + W_s, int(ty)
                fill_t = -1 if t_visible else thickness
                if marker == "circle":
                    cv2.circle(combined, (tgt_x, tgt_y), radius, color, fill_t, cv2.LINE_AA)
                else:
                    d = radius
                    cv2.line(combined, (tgt_x - d, tgt_y - d), (tgt_x + d, tgt_y + d), color, thickness, cv2.LINE_AA)
                    cv2.line(combined, (tgt_x - d, tgt_y + d), (tgt_x + d, tgt_y - d), color, thickness, cv2.LINE_AA)
            except:
                logging.error(f"error: tx{tx} ty{ty}")
                continue

    return combined


def vis_motion_results(cfg, mv_outputs, motion_out_dir, data_idx, frame_num, view_num, gt_out_dir=None):
    """Visualize track_3d motion results: 2D reprojection + 3D point cloud.

    Predicted and GT are drawn separately (they may have different Q counts).
    Output per (src, tgt) pair:
      - track3d_2d_pred_tgt{i}_{idx}.jpg : [src query_uv | tgt warp3d_uv]
      - track3d_2d_gt_tgt{i}_{idx}.jpg   : [src src_2d_gt | tgt tgt_2d_gt]
      - track3d_3d_pred_tgt{i}_{idx}.ply  : predicted 3D
      - track3d_3d_gt_tgt{i}_{idx}.ply    : GT 3D
      - motion_mask_rgb_tgt{i}_{idx}.jpg : source (A) RGB with semi-transparent motion mask
        (cfg: motion_mask_overlay_alpha, optional motion_mask_overlay_bgr as [B,G,R] 0–255)
    """

    max_vis_points = cfg.get("motion_max_vis_points", 512)
    ref_frame = int(cfg.get("motion_vis_ref_frame", 0))
    skip_identity_tgt = bool(cfg.get("motion_vis_skip_identity_tgt", True))

    for frame_index in range(frame_num):
        if frame_index != ref_frame:
            continue
        for view_index in range(view_num):
            index = frame_index * view_num + view_index
            output = mv_outputs[index]

            if output.track_3d is None or len(output.track_3d) == 0:
                continue

            gt_src_prefix = f"{data_idx:03}_src{frame_index:03d}"
            pred_src_prefix = f"src{frame_index:03d}"

            src_rgb = output.rgb  # (H, W, 3), [0, 1] float RGB
            src_img_bgr = cv2.cvtColor((np.clip(src_rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            H_s, W_s = src_img_bgr.shape[:2]

            for tgt_idx, track in output.track_3d.items():
                tgt_flat = int(tgt_idx)
                if skip_identity_tgt and tgt_flat == ref_frame:
                    continue
                track_src = int(getattr(track, "src_index", ref_frame))
                if track_src != ref_frame:
                    continue

                tgt_output = mv_outputs[tgt_flat]
                tgt_img_bgr = cv2.cvtColor((np.clip(tgt_output.rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                H_t, W_t = tgt_img_bgr.shape[:2]

                cur_dir = os.path.join(motion_out_dir, pred_src_prefix)
                os.makedirs(cur_dir, exist_ok=True)

                query_uv = _to_np(track.query_uv, min_ndim=2)        # (Q_pred, 2) normalised
                warp3d_uv = _to_np(track.warp3d_uv, min_ndim=2)      # (Q_pred, 2) pixel
                warp3d = _to_np(track.warp3d, min_ndim=2)             # (Q_pred, 3)
                warp2d = _to_np(track.warp2d, min_ndim=2)
                warp3d_delta = _to_np(track.warp3d_delta, min_ndim=2)             # (Q_pred, 3)

                src_points = _to_np(track.src_points, min_ndim=2)      # (Q_gt, 3)
                src_2d_gt = _to_np(track.src_2d_gt, min_ndim=2)      # (Q_gt, 2) normalised
                tgt_2d_gt = _to_np(track.tgt_2d_gt, min_ndim=2)      # (Q_gt, 2) normalised
                src_3d_gt = _to_np(track.src_3d_gt, min_ndim=2)      # (Q_gt, 3)
                tgt_3d_gt = _to_np(track.tgt_3d_gt, min_ndim=2)      # (Q_gt, 3)
                has_gt = not (src_3d_gt is None and tgt_3d_gt is None)
                qh = int(track.query_height) if track.query_height is not None else None
                qw = int(track.query_width) if track.query_width is not None else None

                src_valids = _to_np(track.src_valids_gt)
                tgt_valids = _to_np(track.tgt_valids_gt)
                src_visibs = _to_np(track.src_visibs_gt)
                tgt_visibs = _to_np(track.tgt_visibs_gt)

                if warp3d_delta is not None and src_points is not None:
                    warp3d_delta += src_points
                elif warp3d_delta is not None and src_3d_gt is not None:
                    warp3d_delta += src_3d_gt
                else:
                    warp3d_delta = None

                # ────────── GT 2D (saved to gt_out_dir, skip if exists) ──────────
                gt_valid_ids = np.array([], dtype=int)
                Q_gt = src_2d_gt.shape[0] if src_2d_gt is not None else 0
                if Q_gt > 0 and gt_out_dir is not None:
                    gt_cur_dir = os.path.join(gt_out_dir, gt_src_prefix)
                    gt_2d_path = os.path.join(gt_cur_dir, f"track3d_2d_gt_tgt{tgt_idx:03d}.jpg")

                    gt_valid = np.ones(Q_gt, dtype=bool)
                    for m in (src_valids, tgt_valids):
                        if m is not None:
                            gt_valid &= (m > 0.5)
                    gt_src_vis = (src_visibs > 0.5) if src_visibs is not None else None
                    gt_tgt_vis = (tgt_visibs > 0.5) if tgt_visibs is not None else None

                    gt_valid_ids = np.where(gt_valid)[0]
                    if len(gt_valid_ids) > 0:
                        gt_valid_ids = _subsample_ids(gt_valid_ids, max_vis_points)

                        if not os.path.exists(gt_2d_path):
                            os.makedirs(gt_cur_dir, exist_ok=True)
                            gt_colors = _generate_colors(len(gt_valid_ids), seed=1)
                            combined_gt = _draw_points_on_pair(
                                src_img_bgr, tgt_img_bgr,
                                src_uv=src_2d_gt, tgt_uv=tgt_2d_gt,
                                H_s=H_s, W_s=W_s, H_t=H_t, W_t=W_t,
                                point_ids=gt_valid_ids, colors=gt_colors,
                                src_vis=gt_src_vis, tgt_vis=gt_tgt_vis,
                                src_is_normalized=True, tgt_is_normalized=True,
                                marker="circle",
                            )
                            cv2.imwrite(gt_2d_path, combined_gt)
                            logging.info(f"vis_motion save: {gt_2d_path}")
                
                # ────────── Predicted 2D ──────────
                Q_pred = query_uv.shape[0] if query_uv is not None else 0
                if has_gt and len(gt_valid_ids) > 0:
                    pred_ids = gt_valid_ids
                elif qh is not None and qw is not None and qh * qw == Q_pred:
                    pred_ids = _grid_subsample_ids(qh, qw, max_vis_points)
                else:
                    pred_ids = np.arange(min(Q_pred, max_vis_points), dtype=int) if max_vis_points is not None else np.arange(Q_pred, dtype=int)
                pred_colors = _generate_colors(len(pred_ids))

                if Q_pred > 0 and len(pred_ids) > 0:
                    # Prefer 3D→2D reprojection; fall back to direct warp2d when cameras are unavailable.
                    pred_tgt_uv = warp3d_uv
                    pred_tgt_is_normalized = False
                    if pred_tgt_uv is None and warp2d is not None:
                        pred_tgt_uv = warp2d
                        pred_tgt_is_normalized = True

                    combined_pred = _draw_points_on_pair(
                        src_img_bgr, tgt_img_bgr,
                        src_uv=query_uv, tgt_uv=pred_tgt_uv,
                        H_s=H_s, W_s=W_s, H_t=H_t, W_t=W_t,
                        point_ids=pred_ids, colors=pred_colors,
                        src_is_normalized=True, tgt_is_normalized=pred_tgt_is_normalized,
                        marker="cross",
                    )
                    path = os.path.join(cur_dir, f"track3d_2d_pred_tgt{tgt_idx:03d}.jpg")
                    cv2.imwrite(path, combined_pred)
                    logging.info(f"vis_motion save: {path}")

                    if warp2d is not None:
                        combined_pred = _draw_points_on_pair(
                            src_img_bgr, tgt_img_bgr,
                            src_uv=query_uv, tgt_uv=warp2d,
                            H_s=H_s, W_s=W_s, H_t=H_t, W_t=W_t,
                            point_ids=pred_ids, colors=pred_colors,
                            src_is_normalized=True, tgt_is_normalized=True,
                            marker="cross",
                        )
                        path = os.path.join(cur_dir, f"track2d_pred_tgt{tgt_idx:03d}.jpg")
                        cv2.imwrite(path, combined_pred)
                        # logging.info(f"vis_motion save: {path}")

                # ────────── 3D point clouds (separate files) ──────────
                try:
                    import open3d as o3d
                except ImportError:
                    o3d = None

                if o3d is not None and warp3d is not None and Q_pred > 0:
                    pred_ids_3d = _subsample_ids(np.arange(warp3d.shape[0]), max_n=None)
                    pred_ply = os.path.join(cur_dir, f"track3d_3d_pred_tgt{tgt_idx:03d}.ply")
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(warp3d[pred_ids_3d])
                    if query_uv is not None:
                        pcd.colors = o3d.utility.Vector3dVector(_sample_rgb_at_uv(src_rgb, query_uv[pred_ids_3d]))
                    else:
                        pcd.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.0, 0.0], (len(pred_ids_3d), 1)))
                    o3d.io.write_point_cloud(pred_ply, pcd)

                    motion_mask = _to_np(track.motion_mask)
                    if motion_mask is not None and tgt_3d_gt is not None and warp3d.shape[0] == tgt_3d_gt.shape[0]:
                        is_dynamic = motion_mask > 0  # logits > 0 ↔ sigmoid > 0.5
                        mixed_pts = np.where(is_dynamic[..., None], warp3d, tgt_3d_gt)
                        mixed_ply = os.path.join(cur_dir, f"track3d_3d_pred_masked_tgt{tgt_idx:03d}.ply")
                        pcd_mixed = o3d.geometry.PointCloud()
                        pcd_mixed.points = o3d.utility.Vector3dVector(mixed_pts[pred_ids_3d])
                        rgb = _sample_rgb_at_uv(src_rgb, query_uv[pred_ids_3d]) if query_uv is not None \
                            else np.tile([1.0, 0.0, 0.0], (len(pred_ids_3d), 1))
                        rgb[is_dynamic[pred_ids_3d]] = [1.0, 0.0, 0.0]
                        pcd_mixed.colors = o3d.utility.Vector3dVector(rgb)
                        o3d.io.write_point_cloud(mixed_ply, pcd_mixed)
                
                if o3d is not None and warp3d_delta is not None and Q_pred > 0:
                    pred_ids_3d = _subsample_ids(np.arange(warp3d_delta.shape[0]), max_n=None)
                    pred_ply = os.path.join(cur_dir, f"track3d_3d_delta_pred_tgt{tgt_idx:03d}.ply")
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(warp3d_delta[pred_ids_3d])
                    if query_uv is not None:
                        pcd.colors = o3d.utility.Vector3dVector(_sample_rgb_at_uv(src_rgb, query_uv[pred_ids_3d]))
                    else:
                        pcd.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.0, 0.0], (len(pred_ids_3d), 1)))
                    o3d.io.write_point_cloud(pred_ply, pcd)


                if o3d is not None and len(gt_valid_ids) > 0 and gt_out_dir is not None:
                    gt_cur_dir = os.path.join(gt_out_dir, gt_src_prefix)

                    if src_3d_gt is not None:
                        src_gt_ply = os.path.join(gt_cur_dir, "track3d_3d_gt.ply")
                        if not os.path.exists(src_gt_ply):
                            os.makedirs(gt_cur_dir, exist_ok=True)
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(src_3d_gt)
                            if src_2d_gt is not None:
                                pcd.colors = o3d.utility.Vector3dVector(
                                    _sample_rgb_at_uv(np.clip(src_rgb, 0, 1), src_2d_gt)
                                )
                            else:
                                pcd.colors = o3d.utility.Vector3dVector(np.tile([0.0, 0.0, 1.0], (len(tgt_2d_gt), 1)))
                            o3d.io.write_point_cloud(src_gt_ply, pcd)

                    if tgt_3d_gt is not None:
                        tgt_gt_ply = os.path.join(gt_cur_dir, f"track3d_3d_gt_tgt{tgt_idx:03d}.ply")
                        if not os.path.exists(tgt_gt_ply):
                            os.makedirs(gt_cur_dir, exist_ok=True)
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(tgt_3d_gt)
                            if tgt_2d_gt is not None and tgt_output.rgb is not None:
                                pcd.colors = o3d.utility.Vector3dVector(
                                    _sample_rgb_at_uv(np.clip(tgt_output.rgb, 0, 1), tgt_2d_gt)
                                )
                            else:
                                pcd.colors = o3d.utility.Vector3dVector(np.tile([0.0, 0.0, 1.0], (len(tgt_2d_gt), 1)))
                            o3d.io.write_point_cloud(tgt_gt_ply, pcd)

                # ────────── Motion mask on source RGB / view A (semi-transparent) ──────────
                motion_mask_np = _to_np(track.motion_mask)
                if motion_mask_np is not None:
                    overlay_alpha = float(cfg.get("motion_mask_overlay_alpha", 0.45))
                    color_cfg = cfg.get("motion_mask_overlay_bgr")
                    color_bgr = tuple(int(c) for c in color_cfg) if color_cfg is not None else (0, 60, 255)
                    mask_rgb_bgr = _overlay_motion_mask_on_bgr(
                        src_img_bgr,
                        motion_mask_np,
                        query_uv,
                        qh,
                        qw,
                        overlay_color_bgr=color_bgr,
                        overlay_alpha=overlay_alpha,
                    )
                    mask_path = os.path.join(cur_dir, f"motion_mask_rgb_tgt{tgt_idx:03d}.jpg")
                    cv2.imwrite(mask_path, mask_rgb_bgr)
                    logging.info(f"vis_motion save: {mask_path}")

            if cfg.get("save_motion_warp2d_video", False) or cfg.get(
                "save_motion_track3d_2d_video", False
            ):
                from hAlgorithm.modules.pipelines2.utils import visualize_dynamic_video

                if cfg.get("save_motion_warp2d_video", False):
                    visualize_dynamic_video.save_warp2d_trajectory_video_for_src(
                        cfg=cfg,
                        mv_outputs=mv_outputs,
                        motion_out_dir=motion_out_dir,
                        data_idx=data_idx,
                        frame_index=frame_index,
                        src_idx=index,
                        track_3d=output.track_3d,
                        max_vis_points=max_vis_points,
                    )

                if cfg.get("save_motion_track3d_2d_video", False):
                    visualize_dynamic_video.save_track3d_2d_trajectory_video_for_src(
                        cfg=cfg,
                        mv_outputs=mv_outputs,
                        motion_out_dir=motion_out_dir,
                        data_idx=data_idx,
                        frame_index=frame_index,
                        src_idx=index,
                        track_3d=output.track_3d,
                        max_vis_points=max_vis_points,
                    )
