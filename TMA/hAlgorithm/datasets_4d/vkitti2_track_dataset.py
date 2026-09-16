import os
import sys

sys.path.append(os.getcwd())

import logging

import numpy as np
import torch

from hAlgorithm.datasets_4d.base_track_dataset import BaseTrackDataset


class VKitti2TrackDataset(BaseTrackDataset):
    """Virtual KITTI 2 4D tracking dataset (w2c, Z-plane depth, 0-origin pixels)."""

    def __init__(self, static_traj_ratio: float = 0.0, **kwargs):
        super().__init__(static_traj_ratio=static_traj_ratio, **kwargs)

    # ------------------------------------------------------------------
    # Pointmap: use 0-origin pixel grid (no +0.5 offset) to match trajs_2d
    # ------------------------------------------------------------------

    def load_pointmap(self, depth, intrinsics):
        """Back-project Z-plane depth to 3D using 0-origin pixel centres.

        VKitti2 ``trajs_2d`` uses pixel-centres-at-integer (cx=(W-1)/2),
        so we use ``arange(W)`` instead of the base ``arange(W)+0.5``.
        """
        if isinstance(depth, torch.Tensor):
            depth = depth.squeeze(0).numpy()
        if isinstance(intrinsics, torch.Tensor):
            intrinsics = intrinsics.numpy()

        height, width = depth.shape

        u, v = np.meshgrid(
            np.arange(width, dtype=np.float64),
            np.arange(height, dtype=np.float64),
            indexing="xy",
        )
        uv_homo = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3)

        K_inv = np.linalg.inv(intrinsics.astype(np.float64))
        directions = uv_homo @ K_inv.T  # (H*W, 3)

        points = depth.reshape(-1, 1).astype(np.float64) * directions
        points = points.reshape(height, width, 3).transpose(2, 0, 1)

        return torch.from_numpy(points).float()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    dataset = VKitti2TrackDataset(
        phase="test",
        name="VKitti2",
        seed=0,
        data_root="/mnt/nasTeam2/AI/datasets/TMD",
        data_path="/mnt/nasTeam2/AI/datasets/TMD/VKitti2/vkitti2_with_tracking.json",
        track_points_nums=512,
        track_neg_ratio=0.0,
        mf_to_mv=True,
        clip_maxlen=24,
        clip_sampler=dict(
            type="hAlgorithm.datasets_mv.clip_sampler.random_sampler.RandomClipSampler",
            view_num=8,
            seed=None,
        ),
        train_transforms=[
            dict(type="hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio", max_size=518, patch_size=14),
            dict(type="hAlgorithm.datasets.transforms.metric3d_transforms.RGBCompresion", prob=0.1, compression=[0, 50]),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(type="hAlgorithm.datasets.transforms.transforms.Normalize", mean=[127.5] * 3, std=[127.5] * 3),
        ],
        test_transforms=[
            dict(type="hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio", max_size=518, patch_size=14),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(type="hAlgorithm.datasets.transforms.transforms.Normalize", mean=[127.5] * 3, std=[127.5] * 3),
        ],
        min_depth=1e-3,
        max_depth=200.0,
        with_pointmap=True,
        debug=True,
        normalize_cameras=False,
    )

    dataset.visualize_tracking_debug(
        sample_idx=0,
        output_path="tracking_debug_vkitti2.rrd",
        downsample_pc=4,
        max_pc_points=50000,
        max_traj_vis=1000,
        depth_range=(0.01, 200.0),  # VKitti2 driving scene: tens-to-hundreds of metres
    )

    dataset.save_tracking_gif(
        sample_idx=0,
        output_path="tracking_debug_vkitti2.gif",
        fps=4,
        trail_len=8,
        downsample_pc=4,
        max_pc_points=30000,
        depth_range=(0.01, 200.0),  # default 20 m clips out most of the scene
    )
