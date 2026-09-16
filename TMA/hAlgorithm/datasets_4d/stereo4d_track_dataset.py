"""Stereo4D tracking dataset.

Stereo4D data is first converted to Hasim-compatible per-frame layout by
``convert_stereo4d_to_hasim_format.py``.

Extrinsics Convention
---------------------
The conversion script stores **w2c** 4×4.  No inversion needed.

Data Availability
-----------------
* Sparse depth maps derived from 3D track projections (saved as ``.npy``).
* No segmentation masks (``"sem": ""`` in JSON).
* Static trajectory generation is therefore **disabled** (``static_traj_ratio=0``).

:meth:`preprocess_data_info` normalises the empty ``sem`` string to ``None``
so the base-class loader skips that field.
"""

import os
import sys

sys.path.append(os.getcwd())

import logging

from hAlgorithm.datasets_4d.base_track_dataset import BaseTrackDataset


class Stereo4DTrackDataset(BaseTrackDataset):
    """Stereo4D tracking dataset (w2c extrinsics, sparse depth from tracks)."""

    def __init__(self, static_traj_ratio: float = 0.0, **kwargs):
        super().__init__(static_traj_ratio=static_traj_ratio, **kwargs)

    def preprocess_data_info(self, data_info):
        """Normalise empty sem path to None."""
        patched = dict(data_info)
        if patched.get("sem") == "":
            patched["sem"] = None
        return patched


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    dataset = Stereo4DTrackDataset(
        phase="test",
        name="Stereo4D",
        seed=0,
        data_root="/mnt/nasTeam2/AI/datasets/TMD",
        data_path="/mnt/nasTeam2/AI/datasets/TMD/Stereo4D/processed_data/stereo4d_test_mf_with_tracking.json",
        track_points_nums=512,
        track_neg_ratio=0.0,
        mf_to_mv=True,
        clip_maxlen=200,
        clip_sampler=dict(
            type="hAlgorithm.datasets_mv.clip_sampler.random_sampler.RandomClipSampler",
            view_num=20,
            shuffle=False,
            seed=42,
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

    dataset.visualize_tracking_debug(
        sample_idx=0,
        output_path="tracking_debug_stereo4d.rrd",
        downsample_pc=4,
        max_pc_points=50000,
        max_traj_vis=1000,
    )

    dataset.save_tracking_gif(
        sample_idx=0,
        output_path="tracking_debug_stereo4d.gif",
        fps=4,
        trail_len=8,
    )
