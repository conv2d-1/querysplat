"""DynamicReplica / DynamicStereo tracking dataset.

Extrinsics Convention
---------------------
JSON stores **c2w** (camera-to-world) in OpenCV convention.
:meth:`convert_extrinsics` inverts to w2c.

Depth Format
------------
Geometric PNG: uint16 bytes that encode float16 depth values.
Correct decoding: ``depth_png.view(np.float16).astype(np.float32)``.
Overridden via :meth:`load_depth`.

Coordinate System
-----------------
OpenCV convention: +X right, +Y down, +Z forward.
"""

import os
import sys

sys.path.append(os.getcwd())

import logging
import numpy as np
import cv2

from hAlgorithm.datasets_4d.base_track_dataset import BaseTrackDataset


class DynamicReplicaTrackDataset(BaseTrackDataset):
    """DynamicReplica tracking dataset (c2w extrinsics, float16 depth)."""

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def convert_extrinsics(self, extrinsics_raw):
        """c2w → w2c via matrix inversion."""
        return np.linalg.inv(extrinsics_raw)

    def load_depth(self, depth_path, image=None, const=None, dtype=np.float32):
        """Decode DynamicReplica geometric PNG (float16 in uint16 bytes)."""
        if not os.path.exists(depth_path):
            logging.warning(f"Depth file not found: {depth_path}")
            return None
        depth_png = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_png is None:
            logging.warning(f"Failed to load depth: {depth_path}")
            return None
        return depth_png.view(np.float16).astype(dtype)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    dataset = DynamicReplicaTrackDataset(
        phase="test",
        name="DynamicReplica",
        seed=0,
        data_root="/mnt/home/tcchen/workspace/TMA/",
        data_path="/mnt/home/tcchen/workspace/TMA/DynamicStereo/test_mf_stereo_with_trajs_opencv_c2w.json",
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

    dataset.visualize_data_batches(
        output_path="dynamicreplica_trajs3d.rrd",
        num_samples=5,
    )
