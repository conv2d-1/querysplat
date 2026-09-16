"""HaSim Fisheye tracking dataset.

This dataset handles HaBlender fisheye polynomial camera data with proper
distortion handling for static trajectory generation.

The fisheye model uses:
    θ = |k0 + k1*r + k2*r² + k3*r³ + k4*r⁴|

where r is physical radius on sensor (mm) and θ is incident angle (radians).
"""

import os
import sys

sys.path.append(os.getcwd())

import logging
import numpy as np

from hAlgorithm.datasets_4d_fisheye.base_fisheye_track_dataset import BaseFisheyeTrackDataset


class HaSimFisheyeTrackDataset(BaseFisheyeTrackDataset):
    """HaSim Fisheye tracking dataset.

    Handles HaBlender fisheye polynomial data with:
    - Automatic valid region cropping
    - Correct fisheye projection for static trajectories
    - w2c extrinsics (no inversion needed)

    The distortion coefficients and sensor size can be:
    1. Specified in config (fisheye_k, sensor_size)
    2. Loaded from HDF5 data (distor_k, sensor_size fields)
    """

    DEFAULT_FISHEYE_K = np.array(
        [-0.00288118, -0.76782615, -0.0900335, 0.12720956, -0.03704932]
    )
    DEFAULT_SENSOR_SIZE = [6.43, 4.87]  # mm

    def __init__(
        self,
        fisheye_k: list = None,
        sensor_size: list = None,
        **kwargs,
    ):
        if fisheye_k is None:
            fisheye_k = self.DEFAULT_FISHEYE_K.tolist()
            logging.info(f"Using default fisheye_k: {fisheye_k}")

        if sensor_size is None:
            sensor_size = self.DEFAULT_SENSOR_SIZE
            logging.info(f"Using default sensor_size: {sensor_size}")

        super().__init__(
            fisheye_k=fisheye_k,
            sensor_size=sensor_size,
            **kwargs,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    dataset = HaSimFisheyeTrackDataset(
        phase="test",
        name="HaSim_Fisheye",
        seed=0,
        data_root="/mnt/nasTeam2/AI/datasets/TMD",
        data_path="/mnt/nasTeam2/AI/datasets/TMD/Hasim_fisheye_4D_character_easy/hasim_4d_mf_all_with_tracking.json",
        fisheye_crop=True,
        track_points_nums=0,
        track_neg_ratio=0.0,
        static_traj_ratio=0.0,
        mf_to_mv=True,
        clip_maxlen=10,
        skip_static_threshold=0.0,
        clip_sampler=dict(
            type="hAlgorithm.datasets_mv.clip_sampler.random_sampler.RandomClipSampler",
            view_num=4,
            seed=None,
        ),
        train_transforms=[
            dict(
                type="hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio",
                max_size=518,
                patch_size=14,
            ),
            dict(
                type="hAlgorithm.datasets.transforms.metric3d_transforms.RGBCompresion",
                prob=0.1,
                compression=[0, 50],
            ),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[127.5] * 3,
                std=[127.5] * 3,
            ),
        ],
        test_transforms=[
            dict(
                type="hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio",
                max_size=518,
                patch_size=14,
            ),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[127.5] * 3,
                std=[127.5] * 3,
            ),
        ],
        min_depth=1e-3,
        max_depth=1000.0,
        with_pointmap=True,
        debug=True,
        normalize_cameras=False,
    )

    print(f"Dataset length: {len(dataset)}")

    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")

    if "fisheye_boundary_mask" in sample:
        mask = sample["fisheye_boundary_mask"]
        print(f"fisheye_boundary_mask shape: {mask.shape}, dtype: {mask.dtype}")
        print(f"fisheye_boundary_mask valid ratio: {mask.float().mean().item():.2%}")

    if "image" in sample:
        print(f"Image shape: {sample['image'].shape}")

    if "trajs_2d" in sample:
        print(f"trajs_2d shape: {sample['trajs_2d'].shape}")

    print("\n=== Testing with static_traj_ratio > 0 ===")
    dataset_with_static = HaSimFisheyeTrackDataset(
        phase="test",
        name="HaSim_Fisheye_Static",
        seed=0,
        data_root="/mnt/nasTeam2/AI/datasets/TMD",
        data_path="/mnt/nasTeam2/AI/datasets/TMD/Hasim_fisheye_4D_character_easy/hasim_4d_mf_all_with_tracking.json",
        fisheye_crop=True,
        track_points_nums=0,
        static_traj_ratio=1.0,
        mf_to_mv=True,
        clip_maxlen=10,
        skip_static_threshold=0.0,
        clip_sampler=dict(
            type="hAlgorithm.datasets_mv.clip_sampler.random_sampler.RandomClipSampler",
            view_num=4,
            seed=None,
        ),
        test_transforms=[
            dict(
                type="hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio",
                max_size=518,
                patch_size=14,
            ),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[127.5] * 3,
                std=[127.5] * 3,
            ),
        ],
        min_depth=1e-3,
        max_depth=1000.0,
        with_pointmap=True,
        debug=True,
        normalize_cameras=False,
    )

    sample_with_static = dataset_with_static[0]
    if "trajs_2d" in sample_with_static:
        print(f"trajs_2d shape (with static): {sample_with_static['trajs_2d'].shape}")

    dataset.visualize_tracking_debug(
        sample_idx=0,
        output_path="tracking_debug_hasim_fisheye.rrd",
        downsample_pc=4,
        max_pc_points=50000,
        max_traj_vis=1000,
    )

    dataset.save_tracking_gif(
        sample_idx=0,
        output_path="tracking_debug_hasim_fisheye.gif",
        fps=4,
        trail_len=8,
    )

    # === Merge multi-frame pointmaps into world coordinate and save as colored PLY ===
    print("\n=== Merging multi-frame pointmaps to world frame & saving PLY ===")
    import torch
    from hAlgorithm.datasets_4d.vis_utils import save_ply, to_numpy, ensure_4x4
    from hAlgorithm.datasets_4d_fisheye.utils import fisheye_unproject
    from hAlgorithm.datasets_4d_fisheye.vis_utils import compute_effective_sensor_size

    sample_for_ply = dataset[0]

    depth_all = to_numpy(sample_for_ply["depth"])       # (T, 1, H, W) or (T, H, W)
    extrinsics = to_numpy(sample_for_ply["extrinsics"]) # (T, 4, 4)
    intrinsics = to_numpy(sample_for_ply["intrinsics"]) # (T, 3, 3) or (3, 3)
    images = to_numpy(sample_for_ply["image"])           # (T, 3, H, W)

    if depth_all.ndim == 4:
        depth_all = depth_all.squeeze(1)  # (T, H, W)

    T = depth_all.shape[0]
    downsample = 4

    # Fisheye parameters
    fisheye_k = dataset.fisheye_k
    sensor_size = dataset.sensor_size
    original_image_size = dataset._get_original_image_size()

    # Compute effective sensor using pre-resize (post-crop) image size.
    # Resize changes pixel density but NOT the physical sensor extent covered,
    # so we must use the crop size (not the current resized size).
    if dataset.fisheye_crop and dataset._fisheye_crop_params is not None:
        crop_size = dataset._fisheye_crop_params["crop_size"]
        pre_resize_size = (crop_size, crop_size)
    else:
        pre_resize_size = original_image_size
    eff_sensor = compute_effective_sensor_size(
        sensor_size, original_image_size, pre_resize_size
    )
    sensor_w, sensor_h = eff_sensor
    print(f"  effective sensor: {sensor_w:.4f} x {sensor_h:.4f} mm "
          f"(pre-resize size: {pre_resize_size})")

    all_points = []
    all_colors = []

    for t in range(T):
        d = depth_all[t]  # (H, W)
        H, W = d.shape
        K = intrinsics[t] if intrinsics.ndim == 3 else intrinsics
        cx, cy = float(K[0, 2]), float(K[1, 2])

        # Downsampled pixel grid
        us = np.arange(0, W, downsample, dtype=np.float64) + 0.5
        vs = np.arange(0, H, downsample, dtype=np.float64) + 0.5
        uu, vv = np.meshgrid(us, vs, indexing="xy")
        dd = d[::downsample, ::downsample]

        # Fisheye unproject (radial depth convention)
        X, Y, Z = fisheye_unproject(
            uu.ravel(), vv.ravel(), dd.ravel().astype(np.float64),
            cx, cy, fisheye_k, sensor_w, sensor_h, W, H,
            depth_is_along_ray=True,
        )
        pts_cam = np.stack([X, Y, Z], axis=-1)  # (N, 3)

        # Camera to world
        w2c = ensure_4x4(extrinsics[t])
        c2w = np.linalg.inv(w2c)
        ones = np.ones((pts_cam.shape[0], 1), dtype=pts_cam.dtype)
        pts_homo = np.concatenate([pts_cam, ones], axis=1)
        pts_world = (c2w @ pts_homo.T).T[:, :3]

        # Colors: denormalize [-1, 1] -> [0, 255]
        img_t = images[t][:, ::downsample, ::downsample]
        img_t = ((img_t + 1.0) * 0.5 * 255.0).clip(0, 255)
        img_t = img_t.transpose(1, 2, 0).astype(np.uint8)
        colors = img_t.reshape(-1, 3)

        # Filter invalid points
        valid = np.isfinite(pts_world).all(axis=1)
        valid &= (np.abs(pts_cam[:, 2]) > 1e-3)
        valid &= (np.linalg.norm(pts_cam, axis=1) < 500.0)

        all_points.append(pts_world[valid])
        all_colors.append(colors[valid])

    merged_points = np.concatenate(all_points, axis=0)
    merged_colors = np.concatenate(all_colors, axis=0)

    max_points = 200000
    if merged_points.shape[0] > max_points:
        idx_sub = np.random.choice(merged_points.shape[0], max_points, replace=False)
        merged_points = merged_points[idx_sub]
        merged_colors = merged_colors[idx_sub]

    output_ply_path = "merged_pointmap_hasim_fisheye.ply"
    save_ply(output_ply_path, merged_points, merged_colors)
    print(f"Saved {merged_points.shape[0]} points -> {output_ply_path}")
