"""PointOdyssey tracking dataset.

Extrinsics Convention
---------------------
PointOdyssey JSON stores extrinsics in **c2w (camera-to-world)** format.
:meth:`convert_extrinsics` inverts them to w2c.

Tracking Data Layout
--------------------
NPY files live under ``<scene>/<subdir>/<frame:06d>.npy`` where *subdir*
is one of ``trajs_2d``, ``trajs_3d``, ``valids``, ``visibs``.
A fallback loader (:meth:`_load_tracking_npy`) constructs these paths
from ``scene`` + ``frame_id`` when the JSON does not embed explicit paths.
"""

import os
import sys

sys.path.append(os.getcwd())

import logging
import numpy as np

from hAlgorithm.datasets_4d.base_track_dataset import BaseTrackDataset
from hAlgorithm.datasets_4d.vis_utils import test_trajs3d_pointmap_consistency


class PointOdysseyTrackDataset(BaseTrackDataset):
    """PointOdyssey 4D tracking dataset (c2w extrinsics)."""

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def convert_extrinsics(self, extrinsics_raw):
        """c2w → w2c via matrix inversion."""
        return np.linalg.inv(extrinsics_raw)

    def load_tracking_data(self, data_batch, data_info):
        """Load tracking data; fall back to scene+frame_id path construction.

        The fallback via :meth:`_load_tracking_npy` is only attempted when the
        JSON does **not** provide ``trajs_2d`` as a file-path string – matching
        the original gating behaviour.
        """
        trajs_2d_ref = data_info.get("trajs_2d")

        # Primary path: delegate to base (loads from file-path strings)
        if isinstance(trajs_2d_ref, str):
            super().load_tracking_data(data_batch, data_info)
            return

        # Fallback: construct path from scene + frame_id
        scene = data_info.get("scene")
        frame_id = data_info.get("frame_id")
        if scene is None or frame_id is None:
            return

        track_data = self._load_tracking_npy(scene, frame_id)
        if track_data is None:
            return

        data_batch["curr_trajs_2d"] = track_data["trajs_2d"]
        for key in ("trajs_3d", "valids", "visibs"):
            if key in track_data:
                data_batch[f"curr_{key}"] = track_data[key]
        if "max_frames" in track_data and "max_frames" not in data_batch:
            data_batch["max_frames"] = track_data["max_frames"]

    # ------------------------------------------------------------------
    # Fallback path-based loader
    # ------------------------------------------------------------------

    def _load_tracking_npy(self, scene: str, frame_id: int):
        """Construct ``<scene>/trajs_2d/<frame:06d>.npy`` and load."""
        cache_key = f"{scene}_{frame_id}"
        if cache_key in self.track_cache:
            return self.track_cache[cache_key]

        try:
            frame_str = f"{frame_id:06d}"
            root = self.track_data_path or self.data_root
            scene_path = os.path.join(root, scene)
            trajs_2d_path = os.path.join(scene_path, "trajs_2d", f"{frame_str}.npy")

            if not os.path.exists(trajs_2d_path):
                if self.debug:
                    logging.warning(
                        f"Tracking not found: {trajs_2d_path}"
                    )
                return None

            track_data = {"trajs_2d": np.load(trajs_2d_path)}
            for subdir in ("trajs_3d", "valids", "visibs"):
                p = os.path.join(scene_path, subdir, f"{frame_str}.npy")
                if os.path.exists(p):
                    track_data[subdir] = np.load(p)

            trajs_2d_dir = os.path.join(scene_path, "trajs_2d")
            if os.path.exists(trajs_2d_dir):
                track_data["max_frames"] = len(
                    [f for f in os.listdir(trajs_2d_dir) if f.endswith(".npy")]
                )

            self.track_cache[cache_key] = track_data
            return track_data
        except Exception as e:
            logging.warning(
                f"Error loading tracking for scene={scene}, frame={frame_id}: {e}"
            )
            return None


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    dataset = PointOdysseyTrackDataset(
        phase="test",
        name="PointOdyssey",
        seed=0,
        data_root="/mnt/home/tcchen/workspace/TMA/",
        data_path="/mnt/home/tcchen/workspace/TMA/Kubric-4D/train_mf_with_tracking_2048.json",
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
        max_depth=1000.0,
        with_pointmap=True,
        debug=True,
        normalize_cameras=False,
    )

    test_trajs3d_pointmap_consistency(dataset, sample_idx=0)
    dataset.visualize_pointmap_vs_trajs3d(
        sample_idx=0, output_path="pointmap_vs_trajs3d.rrd"
    )
