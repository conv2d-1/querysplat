"""Kubric-4D tracking dataset.

Extrinsics Convention
---------------------
The stored matrices are **w2c** – no inversion needed.

Depth Convention
----------------
``depth_plane_*.npy`` stores **OpenCV-style Z-plane depth** (metres along
the camera +Z axis), *not* per-pixel Euclidean / radial distance.  Values
are used as-is (no ``/ ‖ray‖`` scaling).

Pixel Convention
----------------
``trajs_2d`` uses integer pixel centres ``(0, 0)`` – **no** ``+0.5``
offset.  We match that in the overridden :meth:`load_pointmap`.
"""

import os
import sys

sys.path.append(os.getcwd())

import logging
import numpy as np
import torch

from hAlgorithm.datasets_4d.base_track_dataset import BaseTrackDataset


class Kubric4DTrackDataset(BaseTrackDataset):
    """Kubric-4D tracking dataset (w2c extrinsics, Z-plane depth, 0-origin pixels)."""

    def load_pointmap(self, depth, intrinsics):
        """Back-project Z-plane depth to 3D using 0-origin pixel centres.

        The only difference from the base class is the pixel grid: we use
        ``arange(W)`` instead of ``arange(W) + 0.5`` to match the Kubric4D
        ``trajs_2d`` convention.
        """
        if isinstance(depth, torch.Tensor):
            depth = depth.squeeze(0).numpy()
        if isinstance(intrinsics, torch.Tensor):
            intrinsics = intrinsics.numpy()

        height, width = depth.shape

        u, v = np.meshgrid(np.arange(width, dtype=np.float64),
                           np.arange(height, dtype=np.float64),
                           indexing="xy")
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

    dataset = Kubric4DTrackDataset(
        phase="test",
        name="Kubric4D",
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        data_path="/mnt/netdata/Team/AI/datasets/TMD/Kubric-4D/train_mf_with_tracking_2048.json",
        track_points_nums=512,
        track_neg_ratio=0.0,
        mf_to_mv=True,
        clip_maxlen=10,
        clip_sampler=dict(
            type="hAlgorithm.datasets_mv.clip_sampler.random_sampler.RandomClipSampler",
            view_num=4,
            seed=None,
        ),
        train_transforms=[
            dict(type="hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio", max_size=518, patch_size=14),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(type="hAlgorithm.datasets.transforms.transforms.Normalize", mean=[127.5]*3, std=[127.5]*3),
        ],
        test_transforms=[
            dict(type="hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio", max_size=518, patch_size=14),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(type="hAlgorithm.datasets.transforms.transforms.Normalize", mean=[127.5]*3, std=[127.5]*3),
        ],
        min_depth=1e-3,
        max_depth=1000.0,
        with_pointmap=True,
        debug=True,
        normalize_cameras=False,
    )

    dataset.visualize_pointmap_vs_trajs3d(
        sample_idx=0, output_path="pointmap_vs_trajs3d.rrd"
    )
