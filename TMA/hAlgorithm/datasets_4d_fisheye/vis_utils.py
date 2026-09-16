"""Visualization utilities for fisheye tracking datasets.

These functions override the standard pinhole-based visualizations to use
correct fisheye polynomial unprojection for point cloud generation.
"""

import logging
import numpy as np
from typing import Optional

from hAlgorithm.datasets_4d.vis_utils import (
    compute_motion_colors,
    transform_points,
    sample_colors_from_rgb,
    prepare_rgb_for_vis,
    to_numpy,
    ensure_4x4,
)
from hAlgorithm.datasets_4d_fisheye.utils import fisheye_unproject


def unproject_depth_fisheye(
    depth: np.ndarray,
    cx: float,
    cy: float,
    k: np.ndarray,
    sensor_width: float,
    sensor_height: float,
    downsample: int = 1,
    depth_range: tuple = (0.01, 100.0),
) -> tuple:
    """Unproject depth map to 3D points using fisheye model.

    Args:
        depth: (H, W) depth map (radial depth).
        cx, cy: Principal point in pixels.
        k: Polynomial coefficients [k0, k1, k2, k3, k4].
        sensor_width, sensor_height: Sensor size in mm.
        downsample: Spatial downsample factor.
        depth_range: Valid depth range (min, max).

    Returns:
        pts_cam: (N, 3) 3D points in camera frame.
        valid_mask: (H_ds, W_ds) boolean mask of valid points.
        uu, vv: Pixel coordinate grids.
    """
    H, W = depth.shape
    us = np.arange(0, W, downsample)
    vs = np.arange(0, H, downsample)
    uu, vv = np.meshgrid(us, vs)
    dd = depth[vv, uu]

    valid_mask = (dd > depth_range[0]) & (dd < depth_range[1])

    if valid_mask.sum() == 0:
        return np.empty((0, 3)), valid_mask, uu, vv

    uf = uu[valid_mask].astype(np.float64)
    vf = vv[valid_mask].astype(np.float64)
    df = dd[valid_mask].astype(np.float64)

    # Blender fisheye outputs radial depth (distance along ray), not Z-depth
    X, Y, Z = fisheye_unproject(
        uf, vf, df, cx, cy, k, sensor_width, sensor_height, W, H,
        depth_is_along_ray=True
    )

    # Filter out NaN values (points with θ >= 90°)
    finite_mask = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    X, Y, Z = X[finite_mask], Y[finite_mask], Z[finite_mask]

    # Update valid_mask to reflect filtered points
    valid_indices = np.where(valid_mask.flatten())[0][finite_mask]
    new_valid_mask = np.zeros(valid_mask.size, dtype=bool)
    new_valid_mask[valid_indices] = True
    valid_mask = new_valid_mask.reshape(valid_mask.shape)

    pts_cam = np.stack([X, Y, Z], axis=1)
    return pts_cam, valid_mask, uu, vv


def compute_effective_sensor_size(
    original_sensor_size: list,
    original_image_size: tuple,
    current_image_size: tuple,
) -> list:
    """Compute effective sensor size after cropping and scaling.

    The pixel_size (mm/pixel) relationship must be preserved through
    cropping and scaling transformations.

    Args:
        original_sensor_size: [width, height] in mm for original image.
        original_image_size: (width, height) in pixels for original image.
        current_image_size: (width, height) in pixels for current image.

    Returns:
        Effective sensor size [width, height] in mm for current image.
    """
    orig_w, orig_h = original_image_size
    curr_w, curr_h = current_image_size

    orig_pixel_size_x = original_sensor_size[0] / orig_w
    orig_pixel_size_y = original_sensor_size[1] / orig_h

    eff_sensor_w = orig_pixel_size_x * curr_w
    eff_sensor_h = orig_pixel_size_y * curr_h

    return [eff_sensor_w, eff_sensor_h]


def save_tracking_gif_fisheye(
    dataset,
    sample_idx=0,
    output_path="tracking_debug.gif",
    downsample_pc=8,
    max_pc_points=5000,
    max_traj_vis=500,
    depth_range=(0.01, 20.0),
    fps=4,
    trail_len=8,
    rotate_3d=True,
    fisheye_k: Optional[np.ndarray] = None,
    sensor_size: Optional[list] = None,
    original_image_size: Optional[tuple] = None,
):
    """Render 2D + 3D tracking visualization as an animated GIF with fisheye support.

    This function uses correct fisheye polynomial unprojection for point cloud
    generation, unlike the standard pinhole-based version.

    Args:
        dataset: Dataset supporting ``__getitem__`` / ``__len__``.
        sample_idx: Index of sample to visualize.
        output_path: ``.gif`` output path.
        downsample_pc: Spatial downsample factor for depth → point cloud.
        max_pc_points: Cap on point-cloud points per frame.
        max_traj_vis: Cap on trajectories to draw.
        depth_range: (min, max) valid depth range.
        fps: GIF frame rate.
        trail_len: How many past frames to show as 2D trail.
        rotate_3d: Slowly rotate 3D viewpoint across frames.
        fisheye_k: Polynomial coefficients [k0, k1, k2, k3, k4].
        sensor_size: Sensor size [width, height] in mm for ORIGINAL image.
        original_image_size: Original image size (width, height) before crop/resize.
                             If provided, sensor_size will be adjusted accordingly.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from PIL import Image as PILImage
    import io

    logging.info(f"Generating fisheye tracking GIF for sample {sample_idx} ...")
    data = dataset[sample_idx]

    required = ["trajs_2d", "trajs_3d", "depth", "extrinsics", "intrinsics"]
    missing = [k for k in required if k not in data]
    if missing:
        logging.error(f"Missing required data: {missing}")
        return None

    trajs_2d = to_numpy(data["trajs_2d"])
    trajs_3d = to_numpy(data["trajs_3d"])
    visibs = to_numpy(data["visibs"]) if "visibs" in data else None
    valids = to_numpy(data["valids"]) if "valids" in data else None
    depth_all = to_numpy(data["depth"])
    extrinsics = to_numpy(data["extrinsics"])
    intrinsics = to_numpy(data["intrinsics"])
    rgb_np, _ = prepare_rgb_for_vis(data)

    if len(depth_all.shape) == 4 and depth_all.shape[1] == 1:
        depth_all = depth_all[:, 0]

    T, N = trajs_3d.shape[:2]
    H_img = rgb_np.shape[1] if rgb_np is not None else depth_all.shape[1]
    W_img = rgb_np.shape[2] if rgb_np is not None else depth_all.shape[2]

    meta = data.get("meta_data", {})

    def _s(v, d):
        return (v.item() if hasattr(v, "item") else v) if v is not None else d

    origin_h = _s(meta.get("origin_height"), H_img)
    origin_w = _s(meta.get("origin_width"), W_img)
    scale_uv = np.array([W_img / origin_w, H_img / origin_h])

    w2c_ref = ensure_4x4(extrinsics[0])

    if N > max_traj_vis:
        rng = np.random.RandomState(0)
        if visibs is not None:
            score = visibs.astype(np.float64).sum(0) + rng.uniform(0, 0.01, N)
            traj_sel = np.sort(np.argsort(-score)[:max_traj_vis])
        else:
            traj_sel = np.sort(rng.choice(N, max_traj_vis, replace=False))
    else:
        traj_sel = np.arange(N)

    per_track = []
    for j, pi in enumerate(traj_sel):
        segs, seg = [], []
        for t in range(T):
            ok = bool(valids[t, pi]) if valids is not None else True
            if not ok:
                if len(seg) >= 2:
                    segs.append(np.array(seg, dtype=np.float32))
                seg = []
                continue
            seg.append(transform_points(trajs_3d[t, pi : pi + 1], w2c_ref)[0])
        if len(seg) >= 2:
            segs.append(np.array(seg, dtype=np.float32))
        if segs:
            per_track.append((j, segs, segs[-1][-1] - segs[0][0]))

    if per_track:
        net_vecs = np.array([m for _, _, m in per_track])
        dir_uint8 = compute_motion_colors(net_vecs)
        dir_float = dir_uint8 / 255.0
    else:
        dir_float = np.empty((0, 3))

    strips_pts, strips_col = [], []
    for idx, (_, segs, _) in enumerate(per_track):
        for s in segs:
            for k_idx in range(len(s) - 1):
                strips_pts.append(s[k_idx : k_idx + 2])
                strips_col.append(dir_float[idx])

    traj_dir = np.full((len(traj_sel), 3), 0.5)
    for idx, (j, _, _) in enumerate(per_track):
        traj_dir[j] = dir_float[idx]

    cam_pos = np.zeros((T, 3), dtype=np.float32)
    for t in range(T):
        c2w = np.linalg.inv(ensure_4x4(extrinsics[t]))
        cam_pos[t] = (w2c_ref @ c2w)[:3, 3]

    # Compute effective sensor size for current image dimensions
    # This accounts for cropping and resizing transformations
    effective_sensor_size = sensor_size
    if sensor_size is not None and original_image_size is not None:
        current_image_size = (W_img, H_img)
        effective_sensor_size = compute_effective_sensor_size(
            sensor_size, original_image_size, current_image_size
        )
        logging.info(
            f"Adjusted sensor_size from {sensor_size} to {effective_sensor_size} "
            f"(original: {original_image_size}, current: {current_image_size})"
        )

    # Pre-compute per-frame point clouds using FISHEYE unprojection
    pc_frames = []
    for t in range(T):
        K = intrinsics[t] if len(intrinsics.shape) == 3 else intrinsics
        w2c_t = ensure_4x4(extrinsics[t])
        c2w_t = np.linalg.inv(w2c_t)
        rfc = w2c_ref @ c2w_t
        cx, cy = float(K[0, 2]), float(K[1, 2])

        d = depth_all[t]
        H_d, W_d = d.shape

        if fisheye_k is not None and effective_sensor_size is not None:
            pts_cam, valid_mask, uu, vv = unproject_depth_fisheye(
                d,
                cx,
                cy,
                fisheye_k,
                effective_sensor_size[0],
                effective_sensor_size[1],
                downsample=downsample_pc,
                depth_range=depth_range,
            )

            if len(pts_cam) == 0:
                pc_frames.append((np.empty((0, 3)), np.empty((0, 3))))
                continue

            pts = transform_points(pts_cam, rfc)

            if rgb_np is not None and t < rgb_np.shape[0]:
                Hr, Wr = rgb_np[t].shape[:2]
                if Hr == H_d and Wr == W_d:
                    col = rgb_np[t][vv[valid_mask], uu[valid_mask]]
                else:
                    ur = (
                        (uu[valid_mask].astype(np.float64) * Wr / W_d)
                        .astype(int)
                        .clip(0, Wr - 1)
                    )
                    vr = (
                        (vv[valid_mask].astype(np.float64) * Hr / H_d)
                        .astype(int)
                        .clip(0, Hr - 1)
                    )
                    col = rgb_np[t][vr, ur]
            else:
                col = np.full((len(pts), 3), 180, dtype=np.uint8)
        else:
            # Fallback to pinhole if no fisheye params
            fx, fy = float(K[0, 0]), float(K[1, 1])
            us = np.arange(0, W_d, downsample_pc)
            vs = np.arange(0, H_d, downsample_pc)
            uu, vv = np.meshgrid(us, vs)
            dd = d[vv, uu]
            dm = (dd > depth_range[0]) & (dd < depth_range[1])

            if dm.sum() == 0:
                pc_frames.append((np.empty((0, 3)), np.empty((0, 3))))
                continue

            uf, vf, df = (
                uu[dm].astype(np.float64),
                vv[dm].astype(np.float64),
                dd[dm].astype(np.float64),
            )
            pts_cam = np.stack(
                [(uf - cx) / fx * df, (vf - cy) / fy * df, df], axis=1
            )
            pts = transform_points(pts_cam, rfc)

            if rgb_np is not None and t < rgb_np.shape[0]:
                Hr, Wr = rgb_np[t].shape[:2]
                if Hr == H_d and Wr == W_d:
                    col = rgb_np[t][vv[dm], uu[dm]]
                else:
                    ur = (
                        (uu[dm].astype(np.float64) * Wr / W_d)
                        .astype(int)
                        .clip(0, Wr - 1)
                    )
                    vr = (
                        (vv[dm].astype(np.float64) * Hr / H_d)
                        .astype(int)
                        .clip(0, Hr - 1)
                    )
                    col = rgb_np[t][vr, ur]
            else:
                col = np.full((len(pts), 3), 180, dtype=np.uint8)

        if len(pts) > max_pc_points:
            sel = np.random.RandomState(t).choice(len(pts), max_pc_points, replace=False)
            pts, col = pts[sel], col[sel]

        pc_frames.append((pts.astype(np.float32), col))

    # 3D bounding box
    bbox_pts = [cam_pos]
    for pts, _ in pc_frames:
        if len(pts) > 0:
            bbox_pts.append(pts)
    for t in range(T):
        bbox_pts.append(
            transform_points(trajs_3d[t, traj_sel], w2c_ref).astype(np.float32)
        )
    bbox_all = np.concatenate(bbox_pts, axis=0)
    pmin, pmax = bbox_all.min(0), bbox_all.max(0)
    center = (pmin + pmax) / 2
    extent = (pmax - pmin).max() * 0.6 + 1e-3
    lims = [center - extent, center + extent]

    frames_pil = []
    base_elev, base_azim = 20, 45

    for t in range(T):
        fig = plt.figure(figsize=(14, 6), dpi=100)

        # Left: 2D tracking
        ax2d = fig.add_subplot(1, 2, 1)
        if rgb_np is not None and t < rgb_np.shape[0]:
            ax2d.imshow(rgb_np[t])
        ax2d.set_xlim(0, W_img)
        ax2d.set_ylim(H_img, 0)
        ax2d.set_aspect("equal")
        ax2d.axis("off")
        ax2d.set_title(f"2D Tracking (frame {t})")

        t_start = max(0, t - trail_len + 1)
        traj_sub_2d = trajs_2d[t_start : t + 1, traj_sel] * scale_uv

        for j in range(len(traj_sel)):
            pi = traj_sel[j]
            pts = traj_sub_2d[:, j]
            col = traj_dir[j]
            if len(pts) >= 2:
                segments = np.array([pts[:-1], pts[1:]]).transpose(1, 0, 2)
                lc = LineCollection(segments, colors=[col] * len(segments), linewidths=1)
                ax2d.add_collection(lc)
            x, y = pts[-1]
            vis_flag = visibs[t, pi] if visibs is not None else True
            val_flag = valids[t, pi] if valids is not None else True
            if val_flag and vis_flag:
                c = "lime"
            elif val_flag:
                c = "yellow"
            else:
                c = "red"
            ax2d.scatter(x, y, c=c, s=8, zorder=3)

        # Right: 3D world
        ax3d = fig.add_subplot(1, 2, 2, projection="3d")
        pts_pc, col_pc = pc_frames[t]
        if len(pts_pc) > 0:
            ax3d.scatter(
                pts_pc[:, 0],
                pts_pc[:, 1],
                pts_pc[:, 2],
                c=col_pc / 255.0,
                s=0.5,
                alpha=0.6,
            )

        if strips_pts:
            lc3d = Line3DCollection(strips_pts, colors=strips_col, linewidths=0.5, alpha=0.7)
            ax3d.add_collection3d(lc3d)

        cur_pts = transform_points(trajs_3d[t, traj_sel], w2c_ref)
        ax3d.scatter(
            cur_pts[:, 0],
            cur_pts[:, 1],
            cur_pts[:, 2],
            c=traj_dir,
            s=10,
            edgecolors="k",
            linewidths=0.3,
        )

        ax3d.plot(cam_pos[:, 0], cam_pos[:, 1], cam_pos[:, 2], "y-", linewidth=1.5)
        ax3d.scatter(*cam_pos[0], c="g", s=40, marker="^", label="start")
        ax3d.scatter(*cam_pos[t], c="b", s=60, marker="o")

        ax3d.set_xlim(lims[0][0], lims[1][0])
        ax3d.set_ylim(lims[0][1], lims[1][1])
        ax3d.set_zlim(lims[0][2], lims[1][2])
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")
        ax3d.set_title("3D World (fisheye)")

        if rotate_3d and T > 1:
            azim = base_azim + (t / (T - 1)) * 30
        else:
            azim = base_azim
        ax3d.view_init(elev=base_elev, azim=azim)

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        frames_pil.append(PILImage.open(buf).copy())
        plt.close(fig)

    if frames_pil:
        frames_pil[0].save(
            output_path,
            save_all=True,
            append_images=frames_pil[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        logging.info(f"Saved tracking GIF (fisheye) to {output_path}")
    return output_path


def visualize_tracking_debug_fisheye(
    dataset,
    sample_idx=0,
    output_path="tracking_debug.rrd",
    downsample_pc=4,
    max_pc_points=50000,
    max_traj_vis=1000,
    depth_range=(0.01, 20.0),
    fisheye_k: Optional[np.ndarray] = None,
    sensor_size: Optional[list] = None,
    original_image_size: Optional[tuple] = None,
):
    """Comprehensive multi-frame tracking visualization with fisheye support.

    Uses Rerun for visualization with correct fisheye polynomial unprojection
    for point cloud generation.

    Args:
        dataset: Dataset supporting ``__getitem__`` / ``__len__``.
        sample_idx: Index of sample to visualize.
        output_path: ``.rrd`` output path.
        downsample_pc: Spatial downsample factor for depth → point cloud.
        max_pc_points: Cap on points per frame.
        max_traj_vis: Cap on trajectory points to draw.
        depth_range: (min, max) valid depth range.
        fisheye_k: Polynomial coefficients [k0, k1, k2, k3, k4].
        sensor_size: Sensor size [width, height] in mm for ORIGINAL image.
        original_image_size: Original image size (width, height) before crop/resize.
    """
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError:
        logging.error("rerun-sdk not installed. pip install rerun-sdk")
        return None

    logging.info(f"Loading sample {sample_idx} for fisheye visualization ...")
    data = dataset[sample_idx]
    logging.info(f"Available keys: {list(data.keys())}")

    required = ["trajs_2d", "trajs_3d", "depth", "extrinsics", "intrinsics"]
    missing = [k for k in required if k not in data]
    if missing:
        logging.error(f"Missing required data: {missing}")
        return None

    trajs_2d = to_numpy(data["trajs_2d"])
    trajs_3d = to_numpy(data["trajs_3d"])
    visibs = to_numpy(data["visibs"]) if "visibs" in data else None
    valids = to_numpy(data["valids"]) if "valids" in data else None
    depth_all = to_numpy(data["depth"])
    extrinsics = to_numpy(data["extrinsics"])
    intrinsics = to_numpy(data["intrinsics"])

    rgb_np, _ = prepare_rgb_for_vis(data)

    if len(depth_all.shape) == 4 and depth_all.shape[1] == 1:
        depth_all = depth_all[:, 0]

    T, N = trajs_3d.shape[:2]
    H_img = rgb_np.shape[1] if rgb_np is not None else depth_all.shape[1]
    W_img = rgb_np.shape[2] if rgb_np is not None else depth_all.shape[2]

    logging.info(f"T={T}  N={N}  image={H_img}x{W_img}  depth={depth_all.shape}")

    meta = data.get("meta_data", {})

    def _s(v, default):
        if v is None:
            return default
        return v.item() if hasattr(v, "item") else v

    origin_h = _s(meta.get("origin_height"), H_img)
    origin_w = _s(meta.get("origin_width"), W_img)
    scale_uv = np.array([W_img / origin_w, H_img / origin_h], dtype=np.float64)

    w2c_ref = ensure_4x4(extrinsics[0])

    if N > max_traj_vis:
        rng = np.random.RandomState(0)
        if visibs is not None:
            score = visibs.astype(np.float64).sum(axis=0)
            score += rng.uniform(0, 0.01, N)
            traj_sel = np.sort(np.argsort(-score)[:max_traj_vis])
        else:
            traj_sel = np.sort(rng.choice(N, max_traj_vis, replace=False))
    else:
        traj_sel = np.arange(N)

    if rgb_np is not None:
        uv0_scaled = trajs_2d[0] * scale_uv
        traj_colors_all = sample_colors_from_rgb(rgb_np[0], uv0_scaled)
    else:
        traj_colors_all = np.random.RandomState(42).randint(50, 255, (N, 3)).astype(
            np.uint8
        )
    traj_colors = traj_colors_all[traj_sel]

    rr.init("tracking_debug", spawn=False)
    blueprint = rrb.Horizontal(
        rrb.Spatial3DView(name="3D World", origin="world"),
        rrb.Spatial2DView(name="2D Tracking", origin="images"),
    )
    rr.send_blueprint(blueprint)
    rr.save(output_path)

    track_pos_ref = np.zeros((T, len(traj_sel), 3), dtype=np.float32)
    track_valid = np.ones((T, len(traj_sel)), dtype=bool)
    for t in range(T):
        track_pos_ref[t] = transform_points(trajs_3d[t, traj_sel], w2c_ref)
        if valids is not None:
            track_valid[t] = valids[t, traj_sel].astype(bool)

    net_motions = np.zeros((len(traj_sel), 3), dtype=np.float32)
    for j in range(len(traj_sel)):
        valid_t = np.where(track_valid[:, j])[0]
        if len(valid_t) >= 2:
            net_motions[j] = track_pos_ref[valid_t[-1], j] - track_pos_ref[valid_t[0], j]
    traj_dir_colors = compute_motion_colors(net_motions)

    src_pts = track_pos_ref[0]
    rr.log(
        "world/tracks/source",
        rr.Points3D(positions=src_pts, colors=traj_colors, radii=0.010),
        static=True,
    )
    logging.info(f"Pre-computed {len(traj_sel)} track trajectories over {T} frames")

    cam_positions = np.zeros((T, 3), dtype=np.float32)
    for t in range(T):
        c2w_t = np.linalg.inv(ensure_4x4(extrinsics[t]))
        cam_positions[t] = (w2c_ref @ c2w_t)[:3, 3]

    if T > 1:
        rr.log(
            "world/camera_path",
            rr.LineStrips3D(strips=[cam_positions], colors=[[255, 255, 0]], radii=0.008),
            static=True,
        )
    rr.log(
        "world/camera_start",
        rr.Points3D(positions=[cam_positions[0]], colors=[[0, 255, 0]], radii=0.025),
        static=True,
    )
    if T > 1:
        rr.log(
            "world/camera_end",
            rr.Points3D(positions=[cam_positions[-1]], colors=[[255, 0, 0]], radii=0.025),
            static=True,
        )

    # Compute effective sensor size for current image dimensions
    effective_sensor_size = sensor_size
    if sensor_size is not None and original_image_size is not None:
        current_image_size = (W_img, H_img)
        effective_sensor_size = compute_effective_sensor_size(
            sensor_size, original_image_size, current_image_size
        )
        logging.info(
            f"Adjusted sensor_size from {sensor_size} to {effective_sensor_size} "
            f"(original: {original_image_size}, current: {current_image_size})"
        )

    for t in range(T):
        rr.set_time_sequence("frame", t)

        K = intrinsics[t] if len(intrinsics.shape) == 3 else intrinsics
        w2c_t = ensure_4x4(extrinsics[t])
        c2w_t = np.linalg.inv(w2c_t)
        ref_from_cam = w2c_ref @ c2w_t
        cx, cy = float(K[0, 2]), float(K[1, 2])

        d_frame = depth_all[t]
        H_d, W_d = d_frame.shape

        if fisheye_k is not None and effective_sensor_size is not None:
            pts_cam, valid_mask, uu, vv = unproject_depth_fisheye(
                d_frame,
                cx,
                cy,
                fisheye_k,
                effective_sensor_size[0],
                effective_sensor_size[1],
                downsample=downsample_pc,
                depth_range=depth_range,
            )

            if len(pts_cam) > 0:
                pts_ref = transform_points(pts_cam, ref_from_cam)

                if rgb_np is not None and t < rgb_np.shape[0]:
                    H_rgb, W_rgb = rgb_np[t].shape[:2]
                    if H_rgb == H_d and W_rgb == W_d:
                        pc_col = rgb_np[t][vv[valid_mask], uu[valid_mask]]
                    else:
                        uu_r = (
                            (uu[valid_mask].astype(np.float64) * W_rgb / W_d)
                            .astype(int)
                            .clip(0, W_rgb - 1)
                        )
                        vv_r = (
                            (vv[valid_mask].astype(np.float64) * H_rgb / H_d)
                            .astype(int)
                            .clip(0, H_rgb - 1)
                        )
                        pc_col = rgb_np[t][vv_r, uu_r]
                else:
                    pc_col = np.full((len(pts_ref), 3), 180, dtype=np.uint8)

                if len(pts_ref) > max_pc_points:
                    sel = np.random.RandomState(t).choice(
                        len(pts_ref), max_pc_points, replace=False
                    )
                    pts_ref, pc_col = pts_ref[sel], pc_col[sel]

                rr.log(
                    "world/pointcloud",
                    rr.Points3D(
                        positions=pts_ref.astype(np.float32), colors=pc_col, radii=0.004
                    ),
                )
        else:
            # Fallback to pinhole
            fx, fy = float(K[0, 0]), float(K[1, 1])
            us = np.arange(0, W_d, downsample_pc)
            vs = np.arange(0, H_d, downsample_pc)
            uu, vv = np.meshgrid(us, vs)
            dd = d_frame[vv, uu]
            dmask = (dd > depth_range[0]) & (dd < depth_range[1])
            n_valid = int(dmask.sum())

            if n_valid > 0:
                uf = uu[dmask].astype(np.float64)
                vf = vv[dmask].astype(np.float64)
                df = dd[dmask].astype(np.float64)
                pts_cam = np.stack(
                    [(uf - cx) / fx * df, (vf - cy) / fy * df, df], axis=1
                )
                pts_ref = transform_points(pts_cam, ref_from_cam)

                if rgb_np is not None and t < rgb_np.shape[0]:
                    H_rgb, W_rgb = rgb_np[t].shape[:2]
                    if H_rgb == H_d and W_rgb == W_d:
                        pc_col = rgb_np[t][vv[dmask], uu[dmask]]
                    else:
                        uu_r = (
                            (uu[dmask].astype(np.float64) * W_rgb / W_d)
                            .astype(int)
                            .clip(0, W_rgb - 1)
                        )
                        vv_r = (
                            (vv[dmask].astype(np.float64) * H_rgb / H_d)
                            .astype(int)
                            .clip(0, H_rgb - 1)
                        )
                        pc_col = rgb_np[t][vv_r, uu_r]
                else:
                    pc_col = np.full((n_valid, 3), 180, dtype=np.uint8)

                if n_valid > max_pc_points:
                    sel = np.random.RandomState(t).choice(
                        n_valid, max_pc_points, replace=False
                    )
                    pts_ref, pc_col = pts_ref[sel], pc_col[sel]

                rr.log(
                    "world/pointcloud",
                    rr.Points3D(
                        positions=pts_ref.astype(np.float32), colors=pc_col, radii=0.004
                    ),
                )

        pts_track = track_pos_ref[t]
        rr.log(
            "world/tracks/current",
            rr.Points3D(positions=pts_track, colors=traj_colors, radii=0.012),
        )

        if t > 0:
            strips_t, colors_t = [], []
            for j in range(len(traj_sel)):
                valid_mask_j = track_valid[: t + 1, j]
                if valid_mask_j.sum() < 2:
                    continue
                segs, seg = [], []
                for tt in range(t + 1):
                    if valid_mask_j[tt]:
                        seg.append(track_pos_ref[tt, j])
                    else:
                        if len(seg) >= 2:
                            segs.append(np.array(seg, dtype=np.float32))
                        seg = []
                if len(seg) >= 2:
                    segs.append(np.array(seg, dtype=np.float32))
                for s in segs:
                    strips_t.append(s)
                    colors_t.append(traj_dir_colors[j])
            if strips_t:
                rr.log(
                    "world/tracks/traj",
                    rr.LineStrips3D(strips=strips_t, colors=colors_t, radii=0.003),
                )

        rr.log(
            "world/camera_current",
            rr.Points3D(positions=[cam_positions[t]], colors=[[0, 0, 255]], radii=0.03),
        )

        if rgb_np is not None and t < rgb_np.shape[0]:
            rr.log("images/rgb", rr.Image(rgb_np[t]))

        uv_t = trajs_2d[t] * scale_uv
        uv_sel = uv_t[traj_sel]

        if visibs is not None and valids is not None:
            vis_t = visibs[t, traj_sel].astype(bool)
            val_t = valids[t, traj_sel].astype(bool)
            col_t = np.full((len(traj_sel), 3), [255, 0, 0], dtype=np.uint8)
            col_t[val_t & ~vis_t] = [255, 255, 0]
            col_t[val_t & vis_t] = [0, 255, 0]
        else:
            col_t = np.full((len(traj_sel), 3), [0, 255, 0], dtype=np.uint8)

        rr.log(
            "images/tracks",
            rr.Points2D(positions=uv_sel, colors=col_t, radii=2.5),
        )

    logging.info(f"Saved fisheye tracking debug to {output_path}")
    return output_path
