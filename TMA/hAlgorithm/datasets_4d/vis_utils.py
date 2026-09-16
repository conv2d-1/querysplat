import logging
import numpy as np


def compute_motion_colors(vectors):
    """Compute colors based on motion direction and magnitude (Any4D style).
    
    Args:
        vectors: (N, 3) motion vectors
        
    Returns:
        (N, 3) uint8 RGB colors
    """
    from matplotlib.colors import hsv_to_rgb

    n = vectors.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    mags = np.linalg.norm(vectors, axis=1)
    max_mag = mags.max() if mags.max() > 1e-6 else 1.0
    norm_vecs = vectors / (mags[:, np.newaxis] + 1e-8)
    hue = (np.arctan2(norm_vecs[:, 1], norm_vecs[:, 0]) + np.pi) / (2 * np.pi)
    norm_mag = np.clip(mags / max_mag, 0, 1)
    saturation = 0.3 + 0.7 * norm_mag
    value = 0.5 + 0.5 * norm_mag
    colors = hsv_to_rgb(np.stack([hue, saturation, value], axis=1))
    return (colors * 255).astype(np.uint8)


def transform_points(pts, transform_4x4):
    """Apply 4x4 transformation matrix to 3D points.
    
    Args:
        pts: (N, 3) points
        transform_4x4: (4, 4) transformation matrix
        
    Returns:
        (N, 3) transformed points
    """
    ones = np.ones((pts.shape[0], 1))
    pts_homo = np.concatenate([pts, ones], axis=1)
    transformed = (transform_4x4 @ pts_homo.T).T
    return transformed[:, :3]


def sample_colors_from_rgb(rgb_frame, trajs_2d_frame):
    """Sample RGB colors from image at 2D trajectory locations.
    
    Args:
        rgb_frame: (H, W, 3) uint8 image
        trajs_2d_frame: (N, 2) pixel coordinates
        
    Returns:
        (N, 3) uint8 RGB colors
    """
    H, W = rgb_frame.shape[:2]
    n_pts = trajs_2d_frame.shape[0]
    colors = np.zeros((n_pts, 3), dtype=np.uint8)

    for i, pt in enumerate(trajs_2d_frame):
        if not np.isfinite(pt[0]) or not np.isfinite(pt[1]):
            colors[i] = [128, 128, 128]
            continue
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < W and 0 <= y < H:
            colors[i] = rgb_frame[y, x, :3]
        else:
            colors[i] = [128, 128, 128]
    return colors


def denormalize_rgb(rgb_np):
    """Denormalize RGB array to uint8 [0, 255].
    
    Handles three cases:
    - [-1, 1] range (mean=127.5, std=127.5 normalization)
    - [0, 1] range
    - [0, 255] range (already uint8)
    
    Args:
        rgb_np: numpy array of RGB values
        
    Returns:
        uint8 numpy array in [0, 255]
    """
    rgb_min, rgb_max = rgb_np.min(), rgb_np.max()
    logging.info(f"  RGB range before denorm: [{rgb_min:.3f}, {rgb_max:.3f}]")

    if rgb_min < 0:
        rgb_np = ((rgb_np + 1) * 127.5).clip(0, 255).astype(np.uint8)
    elif rgb_max <= 1.0:
        rgb_np = (rgb_np * 255).clip(0, 255).astype(np.uint8)
    else:
        rgb_np = rgb_np.clip(0, 255).astype(np.uint8)
    return rgb_np


def prepare_rgb_for_vis(data, rgb_keys=('rgb', 'image', 'images', 'rgbs', 'img', 'imgs')):
    """Extract and prepare RGB data from a data dict for visualization.
    
    Handles format conversion (CHW -> HWC) and denormalization.
    
    Args:
        data: dict with possible RGB keys
        rgb_keys: tuple of key names to try
        
    Returns:
        (rgb_np, rgb_key) or (None, None) if not found.
        rgb_np is uint8 in HWC or THWC format.
    """
    rgb_key = None
    for key in rgb_keys:
        if key in data:
            rgb_key = key
            break

    if rgb_key is None:
        return None, None

    rgb = data[rgb_key]
    logging.info(f"  Found RGB data with key '{rgb_key}'")
    if hasattr(rgb, 'numpy'):
        rgb_np = rgb.numpy()
    else:
        rgb_np = np.array(rgb)

    # (T, C, H, W) -> (T, H, W, C)
    if len(rgb_np.shape) == 4:
        if rgb_np.shape[1] <= 4 and rgb_np.shape[1] < rgb_np.shape[2]:
            rgb_np = np.transpose(rgb_np, (0, 2, 3, 1))
            logging.info(f"  RGB transposed from (T,C,H,W) to (T,H,W,C): {rgb_np.shape}")
    # (C, H, W) -> (H, W, C)
    elif len(rgb_np.shape) == 3:
        if rgb_np.shape[0] <= 4 and rgb_np.shape[0] < rgb_np.shape[1]:
            rgb_np = np.transpose(rgb_np, (1, 2, 0))
            logging.info(f"  RGB transposed from (C,H,W) to (H,W,C): {rgb_np.shape}")

    rgb_np = denormalize_rgb(rgb_np)
    logging.info(f"  RGB shape: {rgb_np.shape}, range: [{rgb_np.min()}, {rgb_np.max()}]")
    return rgb_np, rgb_key


def save_ply(path, points, colors):
    """Save point cloud as ASCII PLY file.
    
    Args:
        path: output file path
        points: (N, 3) float XYZ
        colors: (N, 3) uint8 RGB
    """
    n_points = len(points)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n_points}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(path, 'w') as f:
        f.write(header)
        for i in range(n_points):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


def to_numpy(x):
    """Convert tensor or array-like to numpy."""
    if hasattr(x, 'numpy'):
        return x.numpy()
    return np.array(x)


def ensure_4x4(mat):
    """Ensure a 3x4 or 4x4 matrix is 4x4."""
    if mat.shape[0] == 3:
        mat_4x4 = np.eye(4)
        mat_4x4[:3, :] = mat
        return mat_4x4
    return mat


# ---------------------------------------------------------------------------
# High-level visualization routines
# ---------------------------------------------------------------------------

def visualize_data_batches(
    dataset,
    output_path="tracking_vis.rrd",
    num_samples=2,
):
    """
    Load data batches and visualize trajs_3d with Rerun.
    Generates an RRD file with 3D trajectories, camera positions, and motion analysis.
    
    All visualization is normalized to the first frame's camera coordinate system.
    
    Args:
        dataset: dataset instance (must support __getitem__ and __len__)
        output_path: Output path for the RRD file.
        num_samples: Number of samples to visualize.
    """
    try:
        import rerun as rr
    except ImportError:
        logging.error("Rerun is not installed. Install with: pip install rerun-sdk")
        return

    POINT_RADIUS = 0.015
    VIEW_COLORS = [
        [255, 105, 180],
        [100, 149, 237],
        [144, 238, 144],
        [255, 165, 0],
    ]

    rr.init(f"{dataset.name}_trajs3d", spawn=False)
    rr.save(output_path)

    num_samples = min(num_samples, len(dataset))
    logging.info(f"Loading {num_samples} samples for visualization...")

    for index in range(num_samples):
        logging.info(f"Loading sample {index}")
        data = dataset[index]

        shape_info = []
        logging.info(f"  Available keys: {list(data.keys())}")
        for key in ['trajs_2d', 'trajs_3d', 'valids', 'visibs', 'depth',
                     'extrinsics', 'intrinsics', 'rgb', 'image', 'images']:
            if key in data:
                shape_info.append(f"{key}: {data[key].shape}")
        if shape_info:
            logging.info(f"  Shapes: {', '.join(shape_info)}")

        rr.set_time_sequence("sample", index)

        # Normalization transform (first camera as reference)
        w2c_first = np.eye(4)
        if "extrinsics" in data:
            extrinsics = data['extrinsics']
            if len(extrinsics.shape) == 3 and extrinsics.shape[0] > 0:
                w2c_first = to_numpy(extrinsics[0])

        logging.info("  Normalizing to first camera frame")

        # Prepare RGB
        rgb_np, _ = prepare_rgb_for_vis(data)

        # --- trajs_3d ---
        if "trajs_3d" in data:
            trajs_3d = data['trajs_3d']
            if len(trajs_3d.shape) == 3:
                _vis_trajs3d_multiframe(
                    rr, index, trajs_3d, data, rgb_np, w2c_first,
                    VIEW_COLORS, POINT_RADIUS,
                )
            else:
                _vis_trajs3d_single(
                    rr, index, trajs_3d, data, rgb_np, w2c_first, POINT_RADIUS,
                )

        # --- RGB images ---
        if rgb_np is not None:
            logging.info(f"  Visualizing RGB: shape={rgb_np.shape}, dtype={rgb_np.dtype}")
            if len(rgb_np.shape) == 4:
                rr.log("reference_rgb", rr.Image(rgb_np[0]), static=True)
                for fi in range(1, rgb_np.shape[0]):
                    rr.set_time_sequence("frame", fi)
                    rr.log("target_rgb", rr.Image(rgb_np[fi]))
            elif len(rgb_np.shape) == 3:
                rr.log("reference_rgb", rr.Image(rgb_np))
        else:
            logging.warning("  No RGB data to visualize")

        # --- Cameras ---
        if "extrinsics" in data:
            _vis_cameras(rr, index, data, w2c_first)

    logging.info(f"Saved visualization to: {output_path}")
    logging.info(f"Open with: rerun {output_path}")
    return output_path


def _vis_trajs3d_multiframe(rr, index, trajs_3d, data, rgb_np, w2c_first,
                            view_colors, point_radius):
    """Visualize multi-frame trajs_3d."""
    num_frames = trajs_3d.shape[0]
    ref_pts = transform_points(to_numpy(trajs_3d[0]), w2c_first)

    # Reference colors
    ref_colors = np.array([view_colors[0]] * len(ref_pts), dtype=np.uint8)
    if rgb_np is not None and "trajs_2d" in data:
        trajs_2d_np = to_numpy(data['trajs_2d'])
        if len(rgb_np.shape) == 4 and len(trajs_2d_np.shape) == 3:
            ref_colors = sample_colors_from_rgb(rgb_np[0], trajs_2d_np[0])

    rr.log(
        f"world/sample_{index}/reference_points",
        rr.Points3D(positions=ref_pts, colors=ref_colors, radii=point_radius),
        static=True,
    )

    for fi in range(1, num_frames):
        rr.set_time_sequence("frame", fi)
        tgt_pts = transform_points(to_numpy(trajs_3d[fi]), w2c_first)

        tgt_colors = None
        if rgb_np is not None and "trajs_2d" in data:
            trajs_2d_np = to_numpy(data['trajs_2d'])
            if (len(rgb_np.shape) == 4 and len(trajs_2d_np.shape) == 3
                    and fi < rgb_np.shape[0]):
                tgt_colors = sample_colors_from_rgb(rgb_np[fi], trajs_2d_np[fi])

        motion = tgt_pts - ref_pts
        mag = np.linalg.norm(motion, axis=1)
        valid = mag > 1e-6
        if valid.sum() == 0:
            continue

        rv, tv, mv = ref_pts[valid], tgt_pts[valid], motion[valid]
        rc = ref_colors[valid] if isinstance(ref_colors, np.ndarray) else None
        tc = tgt_colors[valid] if tgt_colors is not None else None

        if len(rv) > 5000:
            idx = np.random.choice(len(rv), 5000, replace=False)
            rv, tv, mv = rv[idx], tv[idx], mv[idx]
            rc = rc[idx] if rc is not None else None
            tc = tc[idx] if tc is not None else None

        mc = compute_motion_colors(mv)

        rr.log(
            f"world/sample_{index}/trajectories/streaks",
            rr.LineStrips3D(strips=np.stack([rv, tv], axis=1), colors=mc, radii=0.005),
        )
        rr.log(
            f"world/sample_{index}/trajectories/arrows",
            rr.Arrows3D(origins=rv, vectors=mv, colors=mc),
        )
        rr.log(
            f"world/sample_{index}/target_points",
            rr.Points3D(positions=tv, colors=tc if tc is not None else mc,
                        radii=point_radius * 0.8),
        )

        # 2D flow images
        _vis_flow_images(rr, data, fi, ref_pts, tgt_pts, trajs_3d, rgb_np)


def _vis_trajs3d_single(rr, index, trajs_3d, data, rgb_np, w2c_first, point_radius):
    """Visualize single-frame trajs_3d."""
    pts = transform_points(to_numpy(trajs_3d), w2c_first)
    point_colors = [[255, 105, 180]] * len(pts)
    if rgb_np is not None and "trajs_2d" in data:
        trajs_2d_np = to_numpy(data['trajs_2d'])
        if len(rgb_np.shape) == 3 and len(trajs_2d_np.shape) == 2:
            point_colors = sample_colors_from_rgb(rgb_np, trajs_2d_np)
    rr.log(
        f"world/sample_{index}/trajs_3d",
        rr.Points3D(positions=pts, colors=point_colors, radii=point_radius),
    )


def _vis_flow_images(rr, data, frame_idx, ref_pts, tgt_pts, trajs_3d, rgb_np):
    """Render 2D flow magnitude and XYZ images."""
    if "trajs_2d" not in data:
        return
    trajs_2d = data['trajs_2d']
    if len(trajs_2d.shape) != 3 or trajs_2d.shape[0] <= frame_idx:
        return

    src_2d = to_numpy(trajs_2d[0])
    tgt_2d = to_numpy(trajs_2d[frame_idx])

    H, W = None, None
    if rgb_np is not None:
        if len(rgb_np.shape) == 4:
            H, W = rgb_np.shape[1], rgb_np.shape[2]
        elif len(rgb_np.shape) == 3:
            H, W = rgb_np.shape[0], rgb_np.shape[1]
    elif "intrinsics" in data:
        intrinsics = to_numpy(data['intrinsics'])
        if len(intrinsics.shape) == 3:
            cx, cy = intrinsics[0, 0, 2], intrinsics[0, 1, 2]
        else:
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        H, W = int(cy * 2), int(cx * 2)

    if H is None or W is None:
        return

    # Magnitude image
    flow_2d = tgt_2d - src_2d
    flow_mag_2d = np.linalg.norm(flow_2d, axis=1)
    mag_img = np.zeros((H, W), dtype=np.float32)
    for i, (pt, mag) in enumerate(zip(src_2d, flow_mag_2d)):
        if not np.isfinite(pt[0]) or not np.isfinite(pt[1]):
            continue
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < W and 0 <= y < H:
            mag_img[y, x] = max(mag_img[y, x], mag)
    if mag_img.max() > 0:
        rr.log("flow_magnitude", rr.Image((mag_img / mag_img.max() * 255).astype(np.uint8)))

    # Flow XYZ image
    motion_full = to_numpy(trajs_3d[frame_idx] - trajs_3d[0])
    flow_xyz_img = np.zeros((H, W, 3), dtype=np.uint8)
    for c in range(3):
        c_min, c_max = motion_full[:, c].min(), motion_full[:, c].max()
        if c_max - c_min > 1e-8:
            for i, pt in enumerate(src_2d):
                if not np.isfinite(pt[0]) or not np.isfinite(pt[1]):
                    continue
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < W and 0 <= y < H:
                    v_norm = (motion_full[i, c] - c_min) / (c_max - c_min)
                    flow_xyz_img[y, x, c] = int(v_norm * 255)
    rr.log("flow_xyz", rr.Image(flow_xyz_img))


def _vis_cameras(rr, index, data, w2c_first):
    """Visualize cameras normalized to first frame."""
    extrinsics = data['extrinsics']
    if len(extrinsics.shape) != 3:
        return

    num_views = extrinsics.shape[0]
    camera_positions = []

    for vi in range(num_views):
        w2c = to_numpy(extrinsics[vi])
        c2w = np.linalg.inv(w2c)
        c2w_norm = w2c_first @ c2w
        camera_positions.append(c2w_norm[:3, 3])

        rr.set_time_sequence("frame", vi)
        rr.log(
            f"world/sample_{index}/cameras/frame_{vi}",
            rr.Transform3D(translation=c2w_norm[:3, 3], mat3x3=c2w_norm[:3, :3]),
        )

        if "intrinsics" in data:
            intrinsics = data['intrinsics']
            if len(intrinsics.shape) == 3:
                K = to_numpy(intrinsics[vi])
            else:
                K = to_numpy(intrinsics)
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            rr.log(
                f"world/sample_{index}/cameras/frame_{vi}/pinhole",
                rr.Pinhole(
                    focal_length=[fx, fy],
                    principal_point=[cx, cy],
                    resolution=[int(cx * 2), int(cy * 2)],
                    image_plane_distance=0.2,
                ),
            )

    if len(camera_positions) > 1:
        cam_pos = np.array(camera_positions)
        rr.log(
            f"world/sample_{index}/camera_trajectory",
            rr.LineStrips3D(strips=[cam_pos], colors=[[255, 255, 0]], radii=0.01),
            static=True,
        )
        rr.log(
            f"world/sample_{index}/camera_start",
            rr.Points3D(positions=[cam_pos[0]], colors=[[0, 255, 0]], radii=0.03),
            static=True,
        )
        rr.log(
            f"world/sample_{index}/camera_end",
            rr.Points3D(positions=[cam_pos[-1]], colors=[[255, 0, 0]], radii=0.03),
            static=True,
        )


def generate_merged_pointcloud(
    dataset,
    output_path="merged_pointcloud.ply",
    sample_index=0,
    downsample=4,
    max_points_per_frame=100000,
    depth_min=0.01,
    depth_max=100.0,
):
    """
    Generate merged point cloud from multiple frames using depth maps.
    All points are transformed to the first frame's camera coordinate system.
    
    Args:
        dataset: dataset instance
        output_path: Output PLY file path
        sample_index: Index of sample to process
        downsample: Downsample factor for depth maps
        max_points_per_frame: Maximum points to keep per frame
        depth_min: Minimum valid depth value
        depth_max: Maximum valid depth value
        
    Returns:
        Output PLY file path or None on failure
    """
    logging.info(f"Generating merged point cloud for sample {sample_index}...")
    data = dataset[sample_index]
    logging.info(f"  Available keys: {list(data.keys())}")

    for req in ("depth", "extrinsics", "intrinsics"):
        if req not in data:
            logging.error(f"No {req} data available")
            return None

    depth = to_numpy(data['depth'])
    extrinsics = to_numpy(data['extrinsics'])
    intrinsics = to_numpy(data['intrinsics'])

    logging.info(f"  depth shape: {depth.shape}")
    logging.info(f"  extrinsics shape: {extrinsics.shape}")
    logging.info(f"  intrinsics shape: {intrinsics.shape}")

    if len(depth.shape) == 4 and depth.shape[1] == 1:
        depth = depth[:, 0]

    num_frames = depth.shape[0]

    # Prepare RGB
    rgb_np = None
    for rgb_key in ('rgb', 'image', 'images', 'rgbs'):
        if rgb_key in data:
            rgb_np = to_numpy(data[rgb_key])
            if len(rgb_np.shape) == 4 and rgb_np.shape[1] <= 4 and rgb_np.shape[1] < rgb_np.shape[2]:
                rgb_np = np.transpose(rgb_np, (0, 2, 3, 1))
            rgb_np = denormalize_rgb(rgb_np)
            break

    w2c_first = extrinsics[0] if len(extrinsics.shape) == 3 else extrinsics
    w2c_first_4x4 = ensure_4x4(w2c_first)

    all_points, all_colors = [], []

    for fi in range(num_frames):
        logging.info(f"  Processing frame {fi}/{num_frames}...")
        depth_frame = depth[fi]
        H, W = depth_frame.shape

        K = intrinsics[fi] if len(intrinsics.shape) == 3 else intrinsics
        w2c = extrinsics[fi] if len(extrinsics.shape) == 3 else extrinsics
        w2c_4x4 = ensure_4x4(w2c)
        c2w = np.linalg.inv(w2c_4x4)

        u = np.arange(0, W, downsample)
        v = np.arange(0, H, downsample)
        uu, vv = np.meshgrid(u, v)

        depth_sampled = depth_frame[vv, uu]
        valid = (depth_sampled > depth_min) & (depth_sampled < depth_max)

        uu_v = uu[valid].astype(np.float64)
        vv_v = vv[valid].astype(np.float64)
        d_v = depth_sampled[valid].astype(np.float64)

        if len(d_v) == 0:
            logging.warning(f"    No valid depth points in frame {fi}")
            continue

        if rgb_np is not None and fi < rgb_np.shape[0]:
            colors_v = rgb_np[fi][vv[valid], uu[valid]]
        else:
            colors_v = np.full((len(d_v), 3), 128, dtype=np.uint8)

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        pts_cam = np.stack([
            (uu_v - cx) / fx * d_v,
            (vv_v - cy) / fy * d_v,
            d_v,
        ], axis=1)

        pts_world = (c2w[:3, :3] @ pts_cam.T).T + c2w[:3, 3]
        pts_ref = (w2c_first_4x4[:3, :3] @ pts_world.T).T + w2c_first_4x4[:3, 3]

        if len(pts_ref) > max_points_per_frame:
            idx = np.random.choice(len(pts_ref), max_points_per_frame, replace=False)
            pts_ref = pts_ref[idx]
            colors_v = colors_v[idx]

        all_points.append(pts_ref.astype(np.float32))
        all_colors.append(colors_v)
        logging.info(f"    Added {len(pts_ref)} points")

    if not all_points:
        logging.error("No valid points generated")
        return None

    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)
    logging.info(f"Total points: {len(all_points)}")

    save_ply(output_path, all_points, all_colors)
    logging.info(f"Saved point cloud to: {output_path}")
    return output_path


def visualize_pointmap_vs_trajs3d(dataset, sample_idx=0,
                                   output_path="pointmap_vs_trajs3d.rrd"):
    """
    Visualize comparison between pointmap and trajs_3d using Rerun.
    
    Contents:
    - Gray points: pointmap point cloud (dense, from depth)
    - Blue points: pointmap sampled at trajs_2d locations
    - Red points: trajs_3d transformed to camera coordinates
    - Connection lines colored by distance (green=close, yellow=medium, red=far)
    
    Args:
        dataset: dataset instance
        sample_idx: sample index
        output_path: output RRD file path
    """
    try:
        import rerun as rr
    except ImportError:
        logging.error("Rerun is not installed. Install with: pip install rerun-sdk")
        return

    logging.info(f"\n{'='*60}")
    logging.info(f"Visualize pointmap vs trajs_3d (sample {sample_idx})")
    logging.info(f"{'='*60}")

    data = dataset[sample_idx]

    required_keys = ['trajs_2d', 'trajs_3d', 'extrinsics', 'pointmap', 'depth', 'intrinsics']
    for key in required_keys:
        if key not in data:
            logging.error(f"Missing data: {key}")
            logging.info(f"Available keys: {list(data.keys())}")
            return

    trajs_2d = to_numpy(data['trajs_2d'])
    trajs_3d = to_numpy(data['trajs_3d'])
    extrinsics = to_numpy(data['extrinsics'])
    pointmap = to_numpy(data['pointmap'])
    depth = to_numpy(data['depth'])
    intrinsics = to_numpy(data['intrinsics'])

    logging.info(f"trajs_2d: {trajs_2d.shape}")
    logging.info(f"trajs_3d: {trajs_3d.shape}")
    logging.info(f"extrinsics: {extrinsics.shape}")
    logging.info(f"pointmap: {pointmap.shape}")
    logging.info(f"depth: {depth.shape}")
    logging.info(f"intrinsics: {intrinsics.shape}")

    rr.init(f"{dataset.name}_pointmap_vs_trajs3d", spawn=False)
    rr.save(output_path)

    frame_idx = 0
    rr.set_time_sequence("frame", frame_idx)

    # 1. Pointmap point cloud
    pm_frame = pointmap[frame_idx]  # [3, H, W]
    H, W = pm_frame.shape[1], pm_frame.shape[2]
    pts_pm = pm_frame.reshape(3, -1).T
    valid = np.linalg.norm(pts_pm, axis=1) > 1e-6
    pts_pm_valid = pts_pm[valid]
    if len(pts_pm_valid) > 50000:
        pts_pm_valid = pts_pm_valid[np.random.choice(len(pts_pm_valid), 50000, replace=False)]

    logging.info(f"\nPointmap valid points: {len(pts_pm_valid)}")

    # 2. trajs_3d -> camera coords
    trajs_3d_frame = trajs_3d[frame_idx]
    w2c = ensure_4x4(extrinsics[frame_idx])
    R, T = w2c[:3, :3], w2c[:3, 3]
    trajs_3d_cam = (R @ trajs_3d_frame.T).T + T

    logging.info(f"trajs_3d points: {len(trajs_3d_cam)}")
    for axis, label in enumerate("XYZ"):
        logging.info(f"trajs_3d range {label}: [{trajs_3d_cam[:, axis].min():.2f}, {trajs_3d_cam[:, axis].max():.2f}]")
    for axis, label in enumerate("XYZ"):
        logging.info(f"pointmap range {label}: [{pts_pm_valid[:, axis].min():.2f}, {pts_pm_valid[:, axis].max():.2f}]")

    rr.log("world/pointmap", rr.Points3D(
        positions=pts_pm_valid, colors=[[128, 128, 128]] * len(pts_pm_valid), radii=0.01))
    rr.log("world/trajs_3d_transformed", rr.Points3D(
        positions=trajs_3d_cam, colors=[[255, 0, 0]] * len(trajs_3d_cam), radii=0.015))

    # 3. Connections and sampled pointmap points
    trajs_2d_frame = trajs_2d[frame_idx]
    meta = data.get('meta_data', {})
    origin_h = meta.get('origin_height', H)
    origin_w = meta.get('origin_width', W)
    if hasattr(origin_h, 'item'):
        origin_h = origin_h.item()
    if hasattr(origin_w, 'item'):
        origin_w = origin_w.item()

    scale_x, scale_y = W / origin_w, H / origin_h
    trajs_2d_scaled = trajs_2d_frame.copy()
    trajs_2d_scaled[:, 0] *= scale_x
    trajs_2d_scaled[:, 1] *= scale_y

    lines, line_colors, sampled_pts = [], [], []
    for i, (traj_cam, uv) in enumerate(zip(trajs_3d_cam, trajs_2d_scaled)):
        x, y = int(round(uv[0])), int(round(uv[1]))
        if 0 <= x < W and 0 <= y < H:
            pm_pt = pm_frame[:, y, x]
            if np.linalg.norm(pm_pt) > 1e-6:
                lines.append([traj_cam, pm_pt])
                dist = np.linalg.norm(traj_cam - pm_pt)
                if dist < 0.05:
                    line_colors.append([0, 255, 0])
                elif dist < 0.2:
                    line_colors.append([255, 255, 0])
                else:
                    line_colors.append([255, 0, 0])
                sampled_pts.append(pm_pt)

    if lines:
        logging.info(f"\nConnection lines: {len(lines)}")
        rr.log("world/connections", rr.LineStrips3D(
            strips=np.array(lines), colors=line_colors, radii=0.005))
        dists = [np.linalg.norm(l[0] - l[1]) for l in lines]
        logging.info(f"Distance stats: min={np.min(dists):.4f}, max={np.max(dists):.4f}, "
                     f"mean={np.mean(dists):.4f}, median={np.median(dists):.4f}")

    if sampled_pts:
        sampled_pts = np.array(sampled_pts)
        rr.log("world/pointmap_sampled", rr.Points3D(
            positions=sampled_pts, colors=[[0, 0, 255]] * len(sampled_pts), radii=0.012))
        logging.info(f"Sampled pointmap points: {len(sampled_pts)}")

    # Camera
    rr.log("world/camera", rr.Transform3D(translation=[0, 0, 0], mat3x3=np.eye(3)))
    K = intrinsics[frame_idx] if len(intrinsics.shape) == 3 else intrinsics
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rr.log("world/camera/pinhole", rr.Pinhole(
        focal_length=[fx, fy], principal_point=[cx, cy],
        resolution=[W, H], image_plane_distance=0.5))

    logging.info(f"\nVisualization legend:")
    logging.info(f"  Gray  = full pointmap cloud")
    logging.info(f"  Blue  = pointmap sampled at trajs_2d")
    logging.info(f"  Red   = trajs_3d in camera coords")
    logging.info(f"  Green lines  = distance < 0.05")
    logging.info(f"  Yellow lines = distance 0.05~0.2")
    logging.info(f"  Red lines    = distance > 0.2")
    logging.info(f"\nSaved to: {output_path}")
    return output_path


def visualize_tracking_debug(
    dataset,
    sample_idx=0,
    output_path="tracking_debug.rrd",
    downsample_pc=4,
    max_pc_points=50000,
    max_traj_vis=1000,
    depth_range=(0.01, 100.0),
):
    """
    Comprehensive multi-frame tracking visualization for dataset debugging.

    Layout (Rerun blueprint):
      Left  – Spatial3DView "3D World":
              * Per-frame RGB-colored point cloud (depth + K + extrinsics)
              * 3D trajectory line-strips (color = first-frame RGB at track point)
              * Highlighted current-frame track positions
              * Camera frustums + camera path
      Right – Spatial2DView "2D Tracking":
              * RGB frame
              * trajs_2d overlay colored by visibility / validity
                Green  = visible & valid
                Yellow = valid but occluded
                Red    = invalid

    All 3D content is normalised to the first camera's coordinate frame.

    Args:
        dataset:        Dataset supporting ``__getitem__`` / ``__len__``
        sample_idx:     Index of sample to visualise
        output_path:    ``.rrd`` output path
        downsample_pc:  Spatial downsample factor for depth → point-cloud
        max_pc_points:  Cap on points per frame after downsampling
        max_traj_vis:   Cap on trajectory points to draw (most-visible first)
        depth_range:    ``(min, max)`` valid depth values
    """
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError:
        logging.error("rerun-sdk not installed.  pip install rerun-sdk")
        return None

    logging.info(f"Loading sample {sample_idx} …")
    data = dataset[sample_idx]
    logging.info(f"Available keys: {list(data.keys())}")

    required = ["trajs_2d", "trajs_3d", "depth", "extrinsics", "intrinsics"]
    missing = [k for k in required if k not in data]
    if missing:
        logging.error(f"Missing required data: {missing}")
        return None

    # ------------------------------------------------------------------
    # Extract arrays
    # ------------------------------------------------------------------
    trajs_2d   = to_numpy(data["trajs_2d"])        # (T, N, 2)
    trajs_3d   = to_numpy(data["trajs_3d"])        # (T, N, 3)
    visibs     = to_numpy(data["visibs"]) if "visibs" in data else None
    valids     = to_numpy(data["valids"]) if "valids" in data else None
    depth_all  = to_numpy(data["depth"])            # (T, [1,] H, W)
    extrinsics = to_numpy(data["extrinsics"])       # (T, 4, 4)
    intrinsics = to_numpy(data["intrinsics"])       # (T, 3, 3) or (3, 3)

    rgb_np, _ = prepare_rgb_for_vis(data)           # (T, H, W, 3) uint8

    if len(depth_all.shape) == 4 and depth_all.shape[1] == 1:
        depth_all = depth_all[:, 0]

    T, N = trajs_3d.shape[:2]
    H_img = rgb_np.shape[1] if rgb_np is not None else depth_all.shape[1]
    W_img = rgb_np.shape[2] if rgb_np is not None else depth_all.shape[2]

    logging.info(f"T={T}  N={N}  image={H_img}x{W_img}  depth={depth_all.shape}")

    # trajs_2d lives in *original* resolution; images are in processed resolution
    meta = data.get("meta_data", {})
    def _s(v, default):
        if v is None:
            return default
        return v.item() if hasattr(v, "item") else v
    origin_h = _s(meta.get("origin_height"), H_img)
    origin_w = _s(meta.get("origin_width"), W_img)
    scale_uv = np.array([W_img / origin_w, H_img / origin_h], dtype=np.float64)

    # Reference = first camera
    w2c_ref = ensure_4x4(extrinsics[0])

    # ------------------------------------------------------------------
    # Sub-sample trajectory indices (keep most-visible points)
    # ------------------------------------------------------------------
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

    # Consistent per-point colour from first-frame RGB
    if rgb_np is not None:
        uv0_scaled = trajs_2d[0] * scale_uv
        traj_colors_all = sample_colors_from_rgb(rgb_np[0], uv0_scaled)
    else:
        traj_colors_all = np.random.RandomState(42).randint(50, 255, (N, 3)).astype(np.uint8)
    traj_colors = traj_colors_all[traj_sel]

    # ------------------------------------------------------------------
    # Init Rerun
    # ------------------------------------------------------------------
    rr.init("tracking_debug", spawn=False)
    blueprint = rrb.Horizontal(
        rrb.Spatial3DView(name="3D World", origin="world"),
        rrb.Spatial2DView(name="2D Tracking", origin="images"),
    )
    rr.send_blueprint(blueprint)
    rr.save(output_path)

    # ------------------------------------------------------------------
    # Pre-compute: per-track positions in ref frame + motion-direction colours
    # ------------------------------------------------------------------
    track_pos_ref = np.zeros((T, len(traj_sel), 3), dtype=np.float32)
    track_valid = np.ones((T, len(traj_sel)), dtype=bool)
    for t in range(T):
        track_pos_ref[t] = transform_points(trajs_3d[t, traj_sel], w2c_ref)
        if valids is not None:
            track_valid[t] = valids[t, traj_sel].astype(bool)

    # Net displacement (first valid → last valid) per track for colouring
    net_motions = np.zeros((len(traj_sel), 3), dtype=np.float32)
    for j in range(len(traj_sel)):
        valid_t = np.where(track_valid[:, j])[0]
        if len(valid_t) >= 2:
            net_motions[j] = track_pos_ref[valid_t[-1], j] - track_pos_ref[valid_t[0], j]
    traj_dir_colors = compute_motion_colors(net_motions)  # (M, 3) uint8

    # Log source points (frame 0) as static reference
    src_pts = track_pos_ref[0]
    rr.log(
        "world/tracks/source",
        rr.Points3D(
            positions=src_pts, colors=traj_colors, radii=0.010,
        ),
        static=True,
    )
    logging.info(f"Pre-computed {len(traj_sel)} track trajectories over {T} frames")

    # ------------------------------------------------------------------
    # STATIC: camera path
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # PER-FRAME data
    # ------------------------------------------------------------------
    for t in range(T):
        rr.set_time_sequence("frame", t)

        K = intrinsics[t] if len(intrinsics.shape) == 3 else intrinsics
        w2c_t = ensure_4x4(extrinsics[t])
        c2w_t = np.linalg.inv(w2c_t)
        ref_from_cam = w2c_ref @ c2w_t          # cam-t → ref
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        # ---- 3D coloured point cloud from depth ----
        d_frame = depth_all[t]
        H_d, W_d = d_frame.shape
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
                    uu_r = (uu[dmask].astype(np.float64) * W_rgb / W_d).astype(int).clip(0, W_rgb - 1)
                    vv_r = (vv[dmask].astype(np.float64) * H_rgb / H_d).astype(int).clip(0, H_rgb - 1)
                    pc_col = rgb_np[t][vv_r, uu_r]
            else:
                pc_col = np.full((n_valid, 3), 180, dtype=np.uint8)

            if n_valid > max_pc_points:
                sel = np.random.RandomState(t).choice(n_valid, max_pc_points, replace=False)
                pts_ref, pc_col = pts_ref[sel], pc_col[sel]

            rr.log(
                "world/pointcloud",
                rr.Points3D(
                    positions=pts_ref.astype(np.float32),
                    colors=pc_col,
                    radii=0.004,
                ),
            )

        # ---- 3D tracked points (current frame) ----
        pts_track = track_pos_ref[t]
        rr.log(
            "world/tracks/current",
            rr.Points3D(
                positions=pts_track,
                colors=traj_colors,
                radii=0.012,
            ),
        )

        # ---- 3D trajectories: growing from frame 0 → frame t ----
        if t > 0:
            strips_t, colors_t = [], []
            for j in range(len(traj_sel)):
                valid_mask = track_valid[: t + 1, j]
                valid_idx = np.where(valid_mask)[0]
                if len(valid_idx) < 2:
                    continue
                path = track_pos_ref[valid_idx, j]
                strips_t.append(path)
                colors_t.append(traj_dir_colors[j].tolist())
            if strips_t:
                rr.log(
                    "world/trajectories",
                    rr.LineStrips3D(
                        strips=strips_t, colors=colors_t, radii=0.003,
                    ),
                )

        # ---- 3D camera frustum ----
        rr.log(
            "world/camera",
            rr.Transform3D(
                translation=ref_from_cam[:3, 3].tolist(),
                mat3x3=ref_from_cam[:3, :3].tolist(),
            ),
        )
        rr.log(
            "world/camera/pinhole",
            rr.Pinhole(
                focal_length=[fx, fy],
                principal_point=[cx, cy],
                resolution=[int(W_d), int(H_d)],
                image_plane_distance=0.3,
            ),
        )

        # ---- 2D: RGB image ----
        if rgb_np is not None and t < rgb_np.shape[0]:
            rr.log("images", rr.Image(rgb_np[t]))

        # ---- 2D: trajs_2d overlay (green / yellow / red) ----
        uv = trajs_2d[t, traj_sel].copy()
        uv[:, 0] *= scale_uv[0]
        uv[:, 1] *= scale_uv[1]

        vis = visibs[t, traj_sel].astype(bool) if visibs is not None else np.ones(len(traj_sel), dtype=bool)
        val = valids[t, traj_sel].astype(bool) if valids is not None else np.ones(len(traj_sel), dtype=bool)

        c2d = np.empty((len(traj_sel), 4), dtype=np.uint8)
        both = val & vis
        occ  = val & ~vis
        inv  = ~val
        c2d[both] = [0, 255, 0, 255]       # visible + valid  → green
        c2d[occ]  = [255, 255, 0, 200]     # occluded         → yellow
        c2d[inv]  = [255, 0, 0, 100]       # invalid          → red

        rr.log("images/trajs_2d", rr.Points2D(positions=uv, colors=c2d, radii=1.5))

        logging.info(
            f"  frame {t}/{T}  pc={n_valid}  tracks={len(traj_sel)}  "
            f"vis={int(both.sum())} occ={int(occ.sum())} inv={int(inv.sum())}"
        )

    logging.info(f"\nSaved to: {output_path}")
    logging.info(f"Open with:  rerun {output_path}")
    logging.info(
        "\n2D Legend:  Green = visible+valid  |  Yellow = occluded  |  Red = invalid"
    )
    return output_path


def save_tracking_gif(
    dataset,
    sample_idx=0,
    output_path="tracking_debug.gif",
    downsample_pc=8,
    max_pc_points=5000,
    max_traj_vis=500,
    depth_range=(0.01, 100.0),
    fps=4,
    trail_len=8,
    rotate_3d=True,
):
    """
    Render 2D + 3D tracking visualisation as an animated GIF.

    Each GIF frame is a side-by-side figure:

      Left  – RGB image with 2D trajectory trails (motion-direction colour)
              and current-frame dots (green/yellow/red = visible/occluded/invalid).
      Right – 3D coloured point cloud, trajectory line strips
              (coloured by motion direction), current-frame track highlights,
              camera path with current camera marker.

    The 3D camera slowly rotates across frames for better spatial perception.

    Args:
        dataset / sample_idx / depth_range:
            Same semantics as :func:`visualize_tracking_debug`.
        output_path:    ``.gif`` output file
        downsample_pc:  Spatial downsample factor for depth → point cloud
        max_pc_points:  Cap on point-cloud points per frame
        max_traj_vis:   Cap on trajectories to draw
        fps:            GIF frame rate
        trail_len:      How many past frames to show as 2D trail
        rotate_3d:      Slowly rotate 3D viewpoint across frames
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from PIL import Image as PILImage
    import io

    logging.info(f"Generating tracking GIF for sample {sample_idx} ...")
    data = dataset[sample_idx]

    required = ["trajs_2d", "trajs_3d", "depth", "extrinsics", "intrinsics"]
    missing = [k for k in required if k not in data]
    if missing:
        logging.error(f"Missing required data: {missing}")
        return None

    # ------------------------------------------------------------------
    # Extract arrays
    # ------------------------------------------------------------------
    trajs_2d   = to_numpy(data["trajs_2d"])
    trajs_3d   = to_numpy(data["trajs_3d"])
    visibs     = to_numpy(data["visibs"]) if "visibs" in data else None
    valids     = to_numpy(data["valids"]) if "valids" in data else None
    depth_all  = to_numpy(data["depth"])
    extrinsics = to_numpy(data["extrinsics"])
    intrinsics = to_numpy(data["intrinsics"])
    rgb_np, _  = prepare_rgb_for_vis(data)

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

    # ------------------------------------------------------------------
    # Sub-sample trajectories
    # ------------------------------------------------------------------
    if N > max_traj_vis:
        rng = np.random.RandomState(0)
        if visibs is not None:
            score = visibs.astype(np.float64).sum(0) + rng.uniform(0, 0.01, N)
            traj_sel = np.sort(np.argsort(-score)[:max_traj_vis])
        else:
            traj_sel = np.sort(rng.choice(N, max_traj_vis, replace=False))
    else:
        traj_sel = np.arange(N)

    # ------------------------------------------------------------------
    # 3D trajectory strips + motion-direction colours
    # ------------------------------------------------------------------
    per_track = []  # (index_in_traj_sel, segments, net_motion)
    for j, pi in enumerate(traj_sel):
        segs, seg = [], []
        for t in range(T):
            ok = bool(valids[t, pi]) if valids is not None else True
            if not ok:
                if len(seg) >= 2:
                    segs.append(np.array(seg, dtype=np.float32))
                seg = []
                continue
            seg.append(transform_points(trajs_3d[t, pi:pi + 1], w2c_ref)[0])
        if len(seg) >= 2:
            segs.append(np.array(seg, dtype=np.float32))
        if segs:
            per_track.append((j, segs, segs[-1][-1] - segs[0][0]))

    # Direction colours for all tracks that have segments
    if per_track:
        net_vecs = np.array([m for _, _, m in per_track])
        dir_uint8 = compute_motion_colors(net_vecs)
        dir_float = dir_uint8 / 255.0
    else:
        dir_float = np.empty((0, 3))

    strips_pts, strips_col = [], []
    for idx, (_, segs, _) in enumerate(per_track):
        for s in segs:
            for k in range(len(s) - 1):
                strips_pts.append(s[k : k + 2])
                strips_col.append(dir_float[idx])

    # Per-traj_sel direction colour (default gray for tracks without segments)
    traj_dir = np.full((len(traj_sel), 3), 0.5)
    for idx, (j, _, _) in enumerate(per_track):
        traj_dir[j] = dir_float[idx]

    # ------------------------------------------------------------------
    # Camera positions in reference frame
    # ------------------------------------------------------------------
    cam_pos = np.zeros((T, 3), dtype=np.float32)
    for t in range(T):
        c2w = np.linalg.inv(ensure_4x4(extrinsics[t]))
        cam_pos[t] = (w2c_ref @ c2w)[:3, 3]

    # ------------------------------------------------------------------
    # Pre-compute per-frame point clouds
    # ------------------------------------------------------------------
    pc_frames = []
    for t in range(T):
        K = intrinsics[t] if len(intrinsics.shape) == 3 else intrinsics
        w2c_t = ensure_4x4(extrinsics[t])
        c2w_t = np.linalg.inv(w2c_t)
        rfc = w2c_ref @ c2w_t
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        d = depth_all[t]
        H_d, W_d = d.shape
        us, vs = np.arange(0, W_d, downsample_pc), np.arange(0, H_d, downsample_pc)
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
        pts_cam = np.stack([(uf - cx) / fx * df, (vf - cy) / fy * df, df], axis=1)
        pts = transform_points(pts_cam, rfc)

        if rgb_np is not None and t < rgb_np.shape[0]:
            Hr, Wr = rgb_np[t].shape[:2]
            if Hr == H_d and Wr == W_d:
                col = rgb_np[t][vv[dm], uu[dm]]
            else:
                ur = (uu[dm].astype(np.float64) * Wr / W_d).astype(int).clip(0, Wr - 1)
                vr = (vv[dm].astype(np.float64) * Hr / H_d).astype(int).clip(0, Hr - 1)
                col = rgb_np[t][vr, ur]
        else:
            col = np.full((len(pts), 3), 180, dtype=np.uint8)

        if len(pts) > max_pc_points:
            sel = np.random.RandomState(t).choice(len(pts), max_pc_points, replace=False)
            pts, col = pts[sel], col[sel]

        pc_frames.append((pts.astype(np.float32), col))

    # ------------------------------------------------------------------
    # 3D bounding box (consistent axis limits across all frames)
    # ------------------------------------------------------------------
    bbox_pts = [cam_pos]
    for pts, _ in pc_frames:
        if len(pts) > 0:
            bbox_pts.append(pts)
    for t in range(T):
        bbox_pts.append(
            transform_points(trajs_3d[t, traj_sel], w2c_ref).astype(np.float32)
        )
    all_bbox = np.concatenate(bbox_pts, axis=0)
    center = (all_bbox.min(0) + all_bbox.max(0)) / 2
    extent = (all_bbox.max(0) - all_bbox.min(0)).max() / 2 * 1.15

    # ------------------------------------------------------------------
    # Render each frame
    # ------------------------------------------------------------------
    frames_pil = []
    for t in range(T):
        logging.info(f"  Rendering frame {t}/{T} ...")
        fig = plt.figure(figsize=(18, 7))
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.94,
                            wspace=0.05)

        # ============== 2D panel ==============
        ax2 = fig.add_subplot(1, 2, 1)
        if rgb_np is not None and t < rgb_np.shape[0]:
            ax2.imshow(rgb_np[t])

        # -- 2D trajectory trails --
        t0 = max(0, t - trail_len + 1)
        trail_segs, trail_cols = [], []
        for j, pi in enumerate(traj_sel):
            uv_trail = trajs_2d[t0 : t + 1, pi] * scale_uv
            val_tr = (
                valids[t0 : t + 1, pi].astype(bool)
                if valids is not None
                else np.ones(t - t0 + 1, dtype=bool)
            )
            n_seg = len(uv_trail) - 1
            for k in range(n_seg):
                if val_tr[k] and val_tr[k + 1]:
                    trail_segs.append([uv_trail[k], uv_trail[k + 1]])
                    alpha = 0.2 + 0.8 * (k + 1) / max(n_seg, 1)
                    trail_cols.append(list(traj_dir[j]) + [alpha])

        if trail_segs:
            ax2.add_collection(
                LineCollection(trail_segs, colors=trail_cols, linewidths=0.7)
            )

        # -- Current 2D points (vis / valid colouring) --
        uv_cur = trajs_2d[t, traj_sel] * scale_uv
        vis = (
            visibs[t, traj_sel].astype(bool)
            if visibs is not None
            else np.ones(len(traj_sel), dtype=bool)
        )
        val = (
            valids[t, traj_sel].astype(bool)
            if valids is not None
            else np.ones(len(traj_sel), dtype=bool)
        )
        m_vis, m_occ, m_inv = val & vis, val & ~vis, ~val
        if m_vis.any():
            ax2.scatter(
                uv_cur[m_vis, 0], uv_cur[m_vis, 1],
                c="lime", s=4, zorder=3, linewidths=0,
            )
        if m_occ.any():
            ax2.scatter(
                uv_cur[m_occ, 0], uv_cur[m_occ, 1],
                c="yellow", s=4, zorder=3, linewidths=0,
            )
        if m_inv.any():
            ax2.scatter(
                uv_cur[m_inv, 0], uv_cur[m_inv, 1],
                c="red", s=2, alpha=0.4, zorder=3, linewidths=0,
            )

        ax2.set_xlim(0, W_img)
        ax2.set_ylim(H_img, 0)
        ax2.set_aspect("equal")
        ax2.axis("off")
        n_vis, n_occ, n_inv = int(m_vis.sum()), int(m_occ.sum()), int(m_inv.sum())
        ax2.set_title(
            f"2D Tracking — Frame {t}/{T - 1}    "
            f"vis={n_vis}  occ={n_occ}  inv={n_inv}",
            fontsize=10, fontweight="bold",
        )

        # ============== 3D panel ==============
        ax3 = fig.add_subplot(1, 2, 2, projection="3d")

        # -- point cloud --
        pts_pc, col_pc = pc_frames[t]
        if len(pts_pc) > 0:
            ax3.scatter(
                pts_pc[:, 0], pts_pc[:, 1], pts_pc[:, 2],
                c=col_pc / 255.0, s=0.3, alpha=0.15, depthshade=False,
            )

        # -- trajectory strips (batch via Line3DCollection) --
        if strips_pts:
            lc3 = Line3DCollection(
                strips_pts, colors=strips_col, linewidths=0.6, alpha=0.5,
            )
            ax3.add_collection3d(lc3)

        # -- current 3D track points --
        pts_t = transform_points(trajs_3d[t, traj_sel], w2c_ref)
        ax3.scatter(
            pts_t[:, 0], pts_t[:, 1], pts_t[:, 2],
            c=traj_dir, s=10, alpha=0.9, depthshade=False,
            edgecolors="k", linewidths=0.15,
        )

        # -- cameras --
        for ct in range(T):
            if ct == t:
                ax3.scatter(
                    *cam_pos[ct], c="lime", s=60, marker="^",
                    depthshade=False, zorder=5, edgecolors="k", linewidths=0.3,
                )
            else:
                ax3.scatter(
                    *cam_pos[ct], c="gray", s=12, marker="^",
                    depthshade=False, alpha=0.35,
                )
        if T > 1:
            ax3.plot(
                cam_pos[:, 0], cam_pos[:, 1], cam_pos[:, 2],
                c="gold", linewidth=1.5, alpha=0.6,
            )

        ax3.set_xlim(center[0] - extent, center[0] + extent)
        ax3.set_ylim(center[1] - extent, center[1] + extent)
        ax3.set_zlim(center[2] - extent, center[2] + extent)
        ax3.view_init(elev=25, azim=135)
        ax3.set_title(f"3D World — Frame {t}/{T - 1}", fontsize=10, fontweight="bold")
        ax3.xaxis.pane.fill = False
        ax3.yaxis.pane.fill = False
        ax3.zaxis.pane.fill = False
        ax3.grid(True, alpha=0.15)

        n_ticks = 5
        ax3.set_xticks(np.linspace(center[0] - extent, center[0] + extent, n_ticks))
        ax3.set_yticks(np.linspace(center[1] - extent, center[1] + extent, n_ticks))
        ax3.set_zticks(np.linspace(center[2] - extent, center[2] + extent, n_ticks))
        ax3.tick_params(labelsize=6)

        # -- rasterise to PIL --
        buf = io.BytesIO()
        fig.savefig(
            buf, format="png", dpi=120,
            facecolor="white", pad_inches=0,
        )
        buf.seek(0)
        frames_pil.append(PILImage.open(buf).copy())
        buf.close()
        plt.close(fig)

    # ------------------------------------------------------------------
    # Save GIF — normalise all frames to the same pixel size
    # ------------------------------------------------------------------
    if not frames_pil:
        logging.error("No frames rendered")
        return None

    target_size = frames_pil[0].size
    for i in range(1, len(frames_pil)):
        if frames_pil[i].size != target_size:
            frames_pil[i] = frames_pil[i].resize(target_size, PILImage.LANCZOS)

    duration_ms = int(1000 / fps)
    frames_pil[0].save(
        output_path,
        save_all=True,
        append_images=frames_pil[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    logging.info(f"Saved GIF ({T} frames, {fps} fps) → {output_path}")
    return output_path


def test_trajs3d_pointmap_consistency(dataset, sample_idx=0):
    """
    Test if trajs_3d and pointmap are consistent in scale.
    
    For frame 0:
    - Sample pointmap at trajs_2d[0] locations -> 3D position in camera 0 frame
    - Transform trajs_3d[0] using w2c[0] -> should give same 3D position
    """
    logging.info(f"\n{'='*60}")
    logging.info("Testing trajs_3d vs pointmap consistency")
    logging.info(f"{'='*60}")

    data = dataset[sample_idx]

    required_keys = ['trajs_2d', 'trajs_3d', 'extrinsics', 'pointmap', 'meta_data']
    for key in required_keys:
        if key not in data:
            logging.error(f"Missing key: {key}")
            logging.info(f"Available keys: {list(data.keys())}")
            return

    trajs_2d = to_numpy(data['trajs_2d'])
    trajs_3d = to_numpy(data['trajs_3d'])
    extrinsics = to_numpy(data['extrinsics'])
    pointmap = to_numpy(data['pointmap'])
    meta_data = data['meta_data']

    def _scalar(v):
        return v.item() if hasattr(v, 'item') else v

    origin_h = _scalar(meta_data.get('origin_height', None))
    origin_w = _scalar(meta_data.get('origin_width', None))
    input_h = _scalar(meta_data.get('input_height', None))
    input_w = _scalar(meta_data.get('input_width', None))
    depth_scale = _scalar(meta_data.get('depth_scale', getattr(dataset, 'depth_scale', 1.0)))

    logging.info(f"\ndepth_scale: {depth_scale}")
    logging.info(f"origin resolution: {origin_h}x{origin_w}")
    logging.info(f"input resolution: {input_h}x{input_w}")
    logging.info(f"\ntrajs_2d shape: {trajs_2d.shape}")
    logging.info(f"trajs_3d shape: {trajs_3d.shape}")
    logging.info(f"extrinsics shape: {extrinsics.shape}")
    logging.info(f"pointmap shape: {pointmap.shape}")

    trajs_2d_0 = trajs_2d[0]
    trajs_3d_0 = trajs_3d[0]
    w2c_0 = extrinsics[0]
    pm_0 = pointmap[0]
    H_pm, W_pm = pm_0.shape[1], pm_0.shape[2]

    logging.info(f"\nFrame 0: pointmap H={H_pm}, W={W_pm}")
    logging.info(f"trajs_2d[0] range (original): "
                 f"x=[{trajs_2d_0[:, 0].min():.1f}, {trajs_2d_0[:, 0].max():.1f}], "
                 f"y=[{trajs_2d_0[:, 1].min():.1f}, {trajs_2d_0[:, 1].max():.1f}]")

    if origin_w is not None and origin_h is not None:
        sx, sy = W_pm / origin_w, H_pm / origin_h
        logging.info(f"\nScaling trajs_2d: scale_x={sx:.4f}, scale_y={sy:.4f}")
        trajs_2d_0_sc = trajs_2d_0.copy()
        trajs_2d_0_sc[:, 0] *= sx
        trajs_2d_0_sc[:, 1] *= sy
    else:
        trajs_2d_0_sc = trajs_2d_0
        logging.warning("Cannot scale trajs_2d - missing origin resolution!")

    sampled, valid_idx = [], []
    for i, (x, y) in enumerate(trajs_2d_0_sc):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W_pm and 0 <= yi < H_pm:
            pt = pm_0[:, yi, xi]
            if np.abs(pt).max() > 1e-6:
                sampled.append(pt)
                valid_idx.append(i)
    sampled = np.array(sampled)
    valid_idx = np.array(valid_idx)

    logging.info(f"\nValid points after scaling: {len(valid_idx)} / {len(trajs_2d_0)}")
    if len(valid_idx) == 0:
        logging.error("No valid points to compare!")
        return

    t3d_valid = trajs_3d_0[valid_idx]

    def _test(label, pts):
        diff = np.abs(pts - sampled)
        ratio = pts / (sampled + 1e-8)
        logging.info(f"\n{'='*60}")
        logging.info(f"{label}")
        logging.info(f"{'='*60}")
        logging.info(f"Mean abs diff: {diff.mean():.6f}")
        for ax, name in enumerate("XYZ"):
            logging.info(f"Median ratio {name}: {np.median(ratio[:, ax]):.4f}")
        mr = np.median(ratio)
        logging.info(f"Overall median ratio: {mr:.4f}")
        return mr

    R, T = w2c_0[:3, :3], w2c_0[:3, 3]
    mr_world = _test("TEST 1: trajs_3d in WORLD coords + w2c", (R @ t3d_valid.T).T + T)
    mr_cam = _test("TEST 2: trajs_3d ALREADY in camera coords", t3d_valid)

    mr_scaled = None
    if depth_scale is not None and depth_scale != 1.0:
        mr_scaled = _test(f"TEST 3: trajs_3d / depth_scale ({depth_scale})", t3d_valid / depth_scale)

    # Conclusion
    results = [("WORLD coords + w2c", mr_world), ("CAMERA coords (direct)", mr_cam)]
    if mr_scaled is not None:
        results.append((f"CAMERA coords / depth_scale ({depth_scale})", mr_scaled))

    best_name, best_ratio = min(results, key=lambda x: abs(x[1] - 1.0))
    best_off = abs(best_ratio - 1.0)

    logging.info(f"\n{'='*60}")
    logging.info("CONCLUSION:")
    logging.info(f"{'='*60}")
    if best_off < 0.1:
        logging.info(f"[BEST MATCH] {best_name} with median ratio off by {best_off:.4f}")
    else:
        logging.warning(f"[NO GOOD MATCH] Best is {best_name} but still off by {best_off:.4f}")

    logging.info(f"\nSample point comparisons (first 5):")
    for i in range(min(5, len(valid_idx))):
        idx = valid_idx[i]
        u, v = trajs_2d_0_sc[idx]
        logging.info(f"  Point {idx} at scaled pixel ({u:.1f}, {v:.1f}):")
        logging.info(f"    pointmap:  {sampled[i]}")
        logging.info(f"    trajs_3d:  {t3d_valid[i]}")
        if depth_scale is not None and depth_scale != 1.0:
            logging.info(f"    trajs_3d/scale: {t3d_valid[i] / depth_scale}")