"""HaSim (HaBlender) tracking dataset.

Extrinsics Convention
---------------------
HaBlender JSON stores extrinsics in **w2c (world-to-camera)** format.
No inversion is needed.

Static Trajectory Generation
-----------------------------
When ``static_traj_ratio > 0``, synthetic static trajectories are sampled
from depth + segmentation masks in frame 0 and reprojected through every
frame.  This is handled by :class:`BaseTrackDataset._generate_static_trajs`.
"""

import os
import sys

sys.path.append(os.getcwd())

import logging

from hAlgorithm.datasets_4d.base_track_dataset import BaseTrackDataset
from hAlgorithm.datasets_4d.vis_utils import test_trajs3d_pointmap_consistency


class HaSimTrackDataset(BaseTrackDataset):
    """HaSim tracking dataset (w2c extrinsics, per-frame npy in frames_p dirs).

    All core logic (tracking loading, static-traj generation,
    skip-static filtering) lives in :class:`BaseTrackDataset`.
    """
    pass


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    dataset = HaSimTrackDataset(
        phase="test",
        name="HaSim",
        seed=0,
        data_root="/mnt/nasTeam2/AI/datasets/TMD",
        data_path="/mnt/nasTeam2/AI/datasets/TMD/Hasim_pinhole_4D/hasim_4d_mf_train_with_tracking.json",
        track_points_nums=512,
        track_neg_ratio=0.0,
        mf_to_mv=True,
        clip_maxlen=10,
        skip_static_threshold=0.02,
        static_traj_ratio=1.2,
        clip_sampler=dict(
            type="hAlgorithm.datasets_mv.clip_sampler.random_sampler.RandomClipSampler",
            view_num=4,
            seed=None,
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
        max_depth=1000.0,
        with_pointmap=True,
        debug=True,
        normalize_cameras=False,
    )

    test_trajs3d_pointmap_consistency(dataset, sample_idx=0)

    dataset.visualize_tracking_debug(
        sample_idx=0,
        output_path="tracking_debug_hasim.rrd",
        downsample_pc=4,
        max_pc_points=50000,
        max_traj_vis=1000,
    )

    dataset.save_tracking_gif(
        sample_idx=0,
        output_path="tracking_debug_hasim.gif",
        fps=4,
        trail_len=8,
    )
