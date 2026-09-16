"""Waymo tracking dataset.

Extrinsics Convention
---------------------
Waymo JSON stores **w2c** (world-to-camera) directly.  No inversion needed.

World coordinate system (Waymo): +X forward, +Y left, +Z up.
Camera coordinate system (OpenCV): +X right, +Y down, +Z forward.

Depth Convention
----------------
``depth_plane_*.npy`` stores **radial / L2 range** (Euclidean distance from the
camera centre along the viewing ray), *not* OpenCV Z-plane depth.  On load we
convert to Z-depth via ``z = range / ||K^{-1}[u,v,1]||`` in :meth:`load_data`
so the batch ``depth`` tensor and ``pointmap`` share the same convention.
"""

import os
import sys

sys.path.append(os.getcwd())

import logging

import numpy as np
import torch

from hAlgorithm.datasets_4d.base_track_dataset import BaseTrackDataset


class WaymoTrackDataset(BaseTrackDataset):
    """Waymo tracking dataset (w2c extrinsics, sparse LiDAR range depth)."""

    @staticmethod
    def _intrinsics_to_matrix(intrinsics):
        """Build a 3×3 pinhole matrix from ``[fx, fy, cx, cy]`` or a 3×3 array."""
        if intrinsics is None:
            return None
        if isinstance(intrinsics, torch.Tensor):
            intrinsics = intrinsics.detach().cpu().numpy()
        intrinsics = np.asarray(intrinsics, dtype=np.float64)
        if intrinsics.shape == (3, 3):
            return intrinsics
        if intrinsics.size >= 4:
            fx, fy, cx, cy = intrinsics.reshape(-1)[:4]
            return np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
        raise ValueError(f"Unsupported intrinsics format: shape={intrinsics.shape}")

    @classmethod
    def _l2_depth_to_z_depth(cls, depth, intrinsics):
        """Convert per-pixel radial (L2) range to OpenCV Z-depth.

        Args:
            depth: (H, W) radial range in metres.
            intrinsics: ``[fx, fy, cx, cy]`` or 3×3 matrix.

        Returns:
            (H, W) float32 Z-depth map; invalid (≤0) entries stay zero.
        """
        if isinstance(depth, torch.Tensor):
            depth_np = depth.squeeze().detach().cpu().numpy()
        else:
            depth_np = np.asarray(depth)

        height, width = depth_np.shape
        K = cls._intrinsics_to_matrix(intrinsics)

        u, v = np.meshgrid(
            np.arange(width, dtype=np.float64) + 0.5,
            np.arange(height, dtype=np.float64) + 0.5,
            indexing="xy",
        )
        uv_homo = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3)
        directions = uv_homo @ np.linalg.inv(K).T
        ray_norm = np.linalg.norm(directions, axis=-1).reshape(height, width)

        z_depth = np.zeros_like(depth_np, dtype=np.float64)
        valid = depth_np > 0
        z_depth[valid] = depth_np[valid] / ray_norm[valid]
        return z_depth.astype(np.float32)

    def load_data(self, data_info):
        """Load frame data; convert stored L2 range depth to OpenCV Z-depth."""
        data_batch = super().load_data(data_info)
        if "depth_scale" in data_info:
            data_batch["depth_scale"] = data_info["depth_scale"]

        curr_depth = data_batch.get("curr_depth")
        curr_intrinsics = data_batch.get("curr_intrinsics")
        if curr_depth is not None and curr_intrinsics is not None:
            data_batch["curr_depth"] = self._l2_depth_to_z_depth(
                curr_depth, curr_intrinsics
            )

        curr_prompt = data_batch.get("curr_prompt")
        if curr_prompt is not None and curr_intrinsics is not None:
            data_batch["curr_prompt"] = self._l2_depth_to_z_depth(
                curr_prompt, curr_intrinsics
            )

        return data_batch

    def create_track_points(self, idx, data_batch_list, image_show_list):
        """Waymo has no dense depth; skip depth-based track creation."""
        return None, None, None


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    dataset = WaymoTrackDataset(
        phase="train",
        name="Waymo",
        seed=0,
        data_root="/mnt/nasTeam2/AI/datasets/TMD/Waymo",
        data_path="/mnt/nasTeam2/AI/datasets/TMD/Waymo/training_with_tracking.json",
        track_points_nums=512,
        track_neg_ratio=0.0,
        mf_to_mv=True,
        clip_maxlen=10,
        clip_sampler=dict(
            type="hAlgorithm.datasets_mv.clip_sampler.random_sampler.RandomClipSampler",
            view_num=10,
            seed=None,
            shuffle=False,
        ),
        train_transforms=[
            dict(type="hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio", max_size=518, patch_size=14),
            dict(type="hAlgorithm.datasets.transforms.metric3d_transforms.RGBCompresion", prob=0.1, compression=[0, 50]),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(type="hAlgorithm.datasets.transforms.transforms.Normalize", mean=[127.5]*3, std=[127.5]*3),
        ],
        test_transforms=[
            dict(type="hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio", max_size=518, patch_size=14),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(type="hAlgorithm.datasets.transforms.transforms.Normalize", mean=[127.5]*3, std=[127.5]*3),
        ],
        min_depth=1e-3,
        max_depth=100.0,
        with_pointmap=True,
        debug=True,
        normalize_cameras=False,
    )

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_waymo_vis")
    os.makedirs(out_dir, exist_ok=True)

    # Multi-frame pointmap fusion: cam_i -> world (c2w) -> reference cam0 (w2c0).
    dataset.save_colored_pointmap_ply(
        output_path=os.path.join(out_dir, "waymo_pointmap_multiframe_fused_cam0.ply"),
        sample_index=0,
        downsample=1,
        max_points_per_frame=80000,
        depth_min=0.05,
        depth_max=100.0,
        reference_frame=0,
    )
