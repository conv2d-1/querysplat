"""Base class for 4D tracking datasets.

All 4D tracking datasets (PointOdyssey, Kubric4D, HaSim, DynamicReplica,
Waymo, Stereo4D, ...) share the same multi-frame training pipeline:

1. Load RGB / depth from ``BaseDatasetMV``
2. Convert extrinsics to **w2c** (if the source stores c2w)
3. Load per-frame tracking data (trajs_2d, trajs_3d, visibs, valids)
4. Convert to tensors and assemble multi-frame batches
5. Optionally: generate static trajectories, skip all-static batches

Subclasses customize behaviour by overriding a small number of *hook
methods* (see the "Hook methods" section below).  Everything else –
collation, tensor conversion, multi-frame assembly, metadata – lives here
so that it is written once and shared.

Hook methods (override in subclasses)
--------------------------------------
* :meth:`preprocess_data_info` – patch ``data_info`` before loading
* :meth:`convert_extrinsics`   – convert raw extrinsics to w2c
* :meth:`load_tracking_data`   – load / parse tracking annotations
* :meth:`compute_max_frames`   – total frames in a scene
"""

import os
import sys

sys.path.append(os.getcwd())

import logging
import numpy as np
import torch

from hAlgorithm.datasets_mv.base_dataset import BaseDatasetMV
from hAlgorithm.datasets.dataloader.collate import pad_trajs_to_batch_max

from hAlgorithm.datasets_4d.vis_utils import (
    visualize_data_batches,
    generate_merged_pointcloud,
    visualize_pointmap_vs_trajs3d,
    visualize_tracking_debug,
    save_tracking_gif,
    save_ply,
    test_trajs3d_pointmap_consistency,
)


class BaseTrackDataset(BaseDatasetMV):
    """Shared base for every 4D tracking dataset.

    Parameters
    ----------
    track_data_path : str, optional
        Alternative root for tracking npy files.  Falls back to
        ``data_root`` when *None*.
    skip_static_threshold : float
        If > 0, training batches where *every* valid point moves less than
        this (metres) are skipped.  Recommended: 0.02.  0 = disabled.
    static_traj_ratio : float
        Ratio of synthetic static points to dynamic points (generated from
        depth + segmentation).  0 = disabled.
    static_traj_dynamic_seg_threshold : float
        Minimum fraction of dynamic trajs on a seg-ID to mark it dynamic.
    static_traj_depth_tolerance : float
        Relative depth tolerance for the occlusion test when generating
        static trajectories.
    """

    def __init__(
        self,
        track_data_path: str = None,
        skip_static_threshold: float = 0.0,
        static_traj_ratio: float = 0.0,
        static_traj_dynamic_seg_threshold: float = 0.05,
        static_traj_depth_tolerance: float = 0.05,
        **kwargs,
    ):
        self.track_data_path = track_data_path
        self.skip_static_threshold = skip_static_threshold
        self.static_traj_ratio = static_traj_ratio
        self.static_traj_dynamic_seg_threshold = static_traj_dynamic_seg_threshold
        self.static_traj_depth_tolerance = static_traj_depth_tolerance

        self.track_cache = {}
        self._scene_max_frames: dict[str, int] = {}

        super().__init__(**kwargs)

    # ==================================================================
    # Hook methods – override in subclasses
    # ==================================================================

    def preprocess_data_info(self, data_info: dict) -> dict:
        """Patch *data_info* before the base-class loader sees it.

        Default: identity (return as-is).
        Override e.g. to normalise empty strings to ``None``.
        """
        return data_info

    def convert_extrinsics(self, extrinsics_raw: np.ndarray) -> np.ndarray:
        """Convert raw extrinsics (as stored in JSON) to **w2c** 4×4.

        Default: identity – assumes the JSON already stores w2c.
        Override for c2w datasets::

            def convert_extrinsics(self, extrinsics_raw):
                return np.linalg.inv(extrinsics_raw)
        """
        return extrinsics_raw

    def load_tracking_data(self, data_batch: dict, data_info: dict) -> None:
        """Load tracking annotations into *data_batch* (mutated in-place).

        The default implementation handles two common storage formats:

        * **File-path strings** in ``data_info`` (PointOdyssey, HaSim, …)
        * **Inline arrays / lists** in ``data_info`` (DynamicReplica JSON)

        Subclasses may call ``super().load_tracking_data(…)`` and then add
        a fallback (e.g. scene + frame_id based path construction).
        """
        _TRACK_KEYS = ("trajs_2d", "trajs_3d", "valids", "visibs")

        trajs_2d_ref = data_info.get("trajs_2d")
        if trajs_2d_ref is None:
            return

        # Case 1: file-path strings
        if isinstance(trajs_2d_ref, str) and trajs_2d_ref:
            try:
                trajs_2d_path = os.path.join(self.data_root, trajs_2d_ref)
                if not os.path.exists(trajs_2d_path):
                    if self.debug:
                        logging.warning(
                            f"Tracking file not found: {trajs_2d_path}"
                        )
                    return

                data_batch["curr_trajs_2d"] = np.load(trajs_2d_path)
                for key in _TRACK_KEYS[1:]:
                    ref = data_info.get(key)
                    if isinstance(ref, str) and ref:
                        path = os.path.join(self.data_root, ref)
                        if os.path.exists(path):
                            data_batch[f"curr_{key}"] = np.load(path)
            except Exception as e:
                logging.error(f"Error loading tracking data: {e}")
            return

        # Case 2: inline data (list / ndarray)
        if not isinstance(trajs_2d_ref, str):
            data_batch["curr_trajs_2d"] = np.asarray(
                trajs_2d_ref, dtype=np.float32
            )
            for key in _TRACK_KEYS[1:]:
                ref = data_info.get(key)
                if ref is not None and not isinstance(ref, str):
                    dtype = (
                        np.float32 if key in ("trajs_3d",) else None
                    )
                    data_batch[f"curr_{key}"] = np.asarray(ref, dtype=dtype)

    def compute_max_frames(self, data_info: dict, data_batch: dict):
        """Return total frame count for the scene, or *None*.

        Default: look up ``_scene_max_frames`` populated during
        :meth:`load_mf_files`.
        """
        scene = data_info.get("scene")
        if scene and scene in self._scene_max_frames:
            return self._scene_max_frames[scene]
        return None

    # ==================================================================
    # Multi-frame file loading – capture scene→frame_count
    # ==================================================================

    def load_mf_files(self, datas):
        """Intercept the raw JSON dict to record per-scene frame counts.

        Uses ``max(frame_id) + 1`` instead of ``len(frames)`` so the
        time normalisation denominator is correct even when frame_ids
        are non-contiguous (e.g. subsampled videos).
        """
        for scene_name, scene_frames in datas.items():
            if isinstance(scene_frames, list) and scene_frames:
                max_fid = max(
                    (f.get("frame_id", 0) for f in scene_frames),
                    default=0,
                )
                self._scene_max_frames[scene_name] = max(max_fid + 1, len(scene_frames))
        return super().load_mf_files(datas)

    # ==================================================================
    # Data loading (template method)
    # ==================================================================

    def load_data(self, data_info):
        """Load a single frame's data (RGB, depth, tracking, extrinsics).

        Pipeline:
        1. :meth:`preprocess_data_info` – patch data_info
        2. ``super().load_data()``       – RGB, depth, intrinsics, …
        3. :meth:`convert_extrinsics`    – raw → w2c
        4. Load segmentation (if ``static_traj_ratio > 0``)
        5. :meth:`load_tracking_data`    – trajs, visibs, valids
        6. :meth:`compute_max_frames`    – total scene frames
        """
        data_info = self.preprocess_data_info(data_info)

        data_batch = super().load_data(data_info)

        # --- extrinsics conversion ---
        if data_batch.get("curr_extrinsics") is not None:
            raw = np.array(data_batch["curr_extrinsics"], dtype=np.float64)
            w2c = self.convert_extrinsics(raw)
            data_batch["curr_extrinsics"] = w2c.tolist()

        # --- segmentation / motion mask (for static-traj generation) ---
        if self.static_traj_ratio > 0:
            from PIL import Image as _PIL_Image

            sem_ref = data_info.get("sem")
            if isinstance(sem_ref, str) and sem_ref:
                try:
                    sem_path = os.path.join(self.data_root, sem_ref)
                    if os.path.exists(sem_path):
                        data_batch["curr_sem"] = np.array(_PIL_Image.open(sem_path))
                except Exception as e:
                    logging.warning(f"Failed to load segmentation: {e}")

            # motion_mask (uint8 PNG, 255 = dynamic object, 0 = background).
            # When present it supersedes the sem-based dynamic-ID detection in
            # _generate_static_trajs, giving a more accurate background region.
            mm_ref = data_info.get("motion_mask")
            if isinstance(mm_ref, str) and mm_ref:
                try:
                    mm_path = os.path.join(self.data_root, mm_ref)
                    if os.path.exists(mm_path):
                        data_batch["curr_motion_mask"] = (
                            np.array(_PIL_Image.open(mm_path)) > 128
                        )
                except Exception as e:
                    logging.warning(f"Failed to load motion mask: {e}")

        # --- tracking data ---
        self.load_tracking_data(data_batch, data_info)

        # --- max_frames ---
        if "max_frames" not in data_batch:
            mf = self.compute_max_frames(data_info, data_batch)
            if mf is not None:
                data_batch["max_frames"] = mf

        return data_batch

    # ==================================================================
    # Collation
    # ==================================================================

    @staticmethod
    def get_collate_fn(traj_keys=("trajs_2d", "trajs_3d", "valids", "visibs")):
        """Custom collate that pads variable-length trajectory tensors."""
        from torch.utils.data.dataloader import default_collate

        def collate_with_traj_padding(batch):
            if not batch:
                return {}

            result = {}
            traj_data = {k: [] for k in traj_keys}
            non_traj_batch = [{} for _ in batch]

            for i, sample in enumerate(batch):
                for k, v in sample.items():
                    if k in traj_keys and v is not None:
                        traj_data[k].append(v)
                    else:
                        non_traj_batch[i][k] = v

            if non_traj_batch[0]:
                result = default_collate(non_traj_batch)

            for k in traj_keys:
                if traj_data[k]:
                    padded, valid_mask = pad_trajs_to_batch_max(traj_data[k])
                    result[k] = padded
                    if k == "trajs_2d" and "traj_valid_mask" not in result:
                        result["traj_valid_mask"] = valid_mask

            return result

        return collate_with_traj_padding

    # ==================================================================
    # Per-frame tensor conversion
    # ==================================================================

    def get_data_for_trainval(
        self, idx, data_info=None, data_batch=None,
        transform_info=None, mf_debug=False,
    ):
        """Convert tracking arrays to tensors and stash in ``_tracking_data``.

        Tracking tensors are kept separate because ``stack_dicts_list``
        requires uniform keys across all frames; the variable-length
        trajectory dimension would break that.
        """
        data_dict = super().get_data_for_trainval(
            idx,
            data_info=data_info,
            data_batch=data_batch,
            transform_info=transform_info,
            mf_debug=mf_debug,
        )

        if data_batch is not None:
            tracking = {}
            if "curr_trajs_2d" in data_batch:
                tracking["trajs_2d"] = torch.from_numpy(
                    data_batch["curr_trajs_2d"]
                ).float()
            if "curr_trajs_3d" in data_batch:
                tracking["trajs_3d"] = torch.from_numpy(
                    data_batch["curr_trajs_3d"]
                ).float()
            if "curr_valids" in data_batch:
                tracking["valids"] = torch.from_numpy(
                    data_batch["curr_valids"]
                ).bool()
            if "curr_visibs" in data_batch:
                tracking["visibs"] = torch.from_numpy(
                    data_batch["curr_visibs"]
                ).bool()

            if tracking:
                data_dict["_tracking_data"] = tracking

            if "max_frames" in data_batch:
                data_dict.setdefault("meta_data", {})["max_frames"] = (
                    data_batch["max_frames"]
                )

        return data_dict

    # ==================================================================
    # Multi-frame assembly
    # ==================================================================

    def get_mf_data_for_trainval(self, idx):
        """Assemble a multi-frame training sample with tracking data."""

        # --- parse idx (may carry view_num / aspect_ratio) ---
        if isinstance(idx, (list, tuple)):
            idx, others = idx
            view_num = others.get("view_num")
            aspect_ratio = others.get("aspect_ratio")

            if self.phase == "train" and aspect_ratio is not None:
                update_flag = False
                for transform in self.data_transforms.transforms:
                    if hasattr(transform, "update_aspect_ratio"):
                        transform.update_aspect_ratio(
                            aspect_ratio=aspect_ratio
                        )
                        update_flag = True
                assert update_flag

            if self.phase == "train" and view_num is not None:
                base_view_num = self.clip_sampler.view_num
                max_view_num = self.clip_sampler.get_max_view_num()
                if max_view_num is not None:
                    view_num = min(view_num, max_view_num)
                self.clip_sampler.view_num = view_num

            mf_data = self.load_mf_data(idx)

            if self.phase == "train" and view_num is not None:
                self.clip_sampler.view_num = base_view_num
        else:
            mf_data = self.load_mf_data(idx)

        assert mf_data is not None

        # --- static trajectory injection (before transforms) ---
        if self.static_traj_ratio > 0:
            static_trajs = self._generate_static_trajs(mf_data)
            if static_trajs is not None:
                for i, st in enumerate(static_trajs):
                    fd = mf_data["data"][i]
                    for key in ("trajs_2d", "trajs_3d", "visibs", "valids"):
                        curr_key = f"curr_{key}"
                        if curr_key in fd and fd[curr_key] is not None:
                            fd[curr_key] = np.concatenate(
                                [fd[curr_key], st[key]], axis=0
                            )
                        else:
                            fd[curr_key] = st[key]

        # --- per-frame processing ---
        transform_info = {}
        mf_data_batch_list = []
        image_show_list = []
        tracking_data_list = []

        for i in range(len(mf_data["data"])):
            data_batch = self.get_data_for_trainval(
                idx,
                data_info=mf_data["info"][i],
                data_batch=mf_data["data"][i],
                transform_info=transform_info,
                mf_debug=self.debug,
            )

            # separate tracking data (variable-length) from stackable data
            if "_tracking_data" in data_batch:
                tracking_data_list.append(data_batch.pop("_tracking_data"))
            else:
                tracking_data_list.append(None)

            data_batch["meta_data"] = {
                "data_info": data_batch["meta_data"]["data_info"],
                "input_width": data_batch["meta_data"]["input_width"],
                "input_height": data_batch["meta_data"]["input_height"],
                "origin_width": data_batch["meta_data"]["origin_width"],
                "origin_height": data_batch["meta_data"]["origin_height"],
            }
            if "max_frames" in mf_data["data"][i]:
                data_batch["meta_data"]["max_frames"] = mf_data["data"][i][
                    "max_frames"
                ]

            if self.track_points_nums > 0 or self.with_sift_mask:
                image_show_list.append(transform_info.pop("image_show"))

            mf_data_batch_list.append(data_batch)

        # --- camera normalisation ---
        if self.normalize_cameras:
            self.normalize_cameras_base_first(mf_data_batch_list)

        if self.track_points_nums > 0:
            tracks, track_vis_masks, track_pos_masks = (
                self.create_track_points(
                    idx, mf_data_batch_list, image_show_list
                )
            )

        view_ids = [v["view_id"] for v in mf_data["info"]]
        same_view_id = all(vid == view_ids[0] for vid in view_ids[1:])

        # --- stack frames (uniform keys only) ---
        mf_data_batch = self.stack_dicts_list(mf_data_batch_list)

        # --- merge tracking data ---
        if any(td is not None for td in tracking_data_list):
            for tkey in ("trajs_2d", "trajs_3d", "valids", "visibs"):
                tensors = [
                    td.get(tkey)
                    for td in tracking_data_list
                    if td is not None and tkey in td
                ]
                if tensors:
                    mf_data_batch[tkey] = torch.stack(
                        tensors, dim=0
                    ).clone()

        # --- skip: no tracking GT ---
        if self.phase == "train" and "trajs_2d" not in mf_data_batch:
            raw_idx = idx[0] if isinstance(idx, (list, tuple)) else idx
            raise RuntimeError(
                f"[{self.name}] idx={raw_idx}: no tracking GT found, skipping"
            )

        # --- skip: all-static batch ---
        if (
            self.phase == "train"
            and self.skip_static_threshold > 0
            and "trajs_3d" in mf_data_batch
            and "visibs" in mf_data_batch
        ):
            trajs = mf_data_batch["trajs_3d"]      # [V, N, 3]
            vis = mf_data_batch["visibs"]            # [V, N]
            val = mf_data_batch.get("valids", vis)   # [V, N]
            has_dynamic = False
            for t in range(1, trajs.shape[0]):
                mag = torch.norm(trajs[t] - trajs[0], dim=-1)
                valid = vis[0].bool() & vis[t].bool() & val[0].bool() & val[t].bool()
                if valid.any() and (mag[valid] > self.skip_static_threshold).any():
                    has_dynamic = True
                    break
            if not has_dynamic:
                raw_idx = idx[0] if isinstance(idx, (list, tuple)) else idx
                raise RuntimeError(
                    f"[{self.name}] idx={raw_idx}: static batch, skipping"
                )

        # --- metadata ---
        mf_data_batch["meta_data"].update(
            {
                "name": self.name,
                "data_path": self.data_path,
                "data_root": self.data_root,
                "data_idx": idx,
                "depth_scale": self.depth_scale,
                "views": 1 if same_view_id else len(view_ids),
                "frames": len(view_ids) if same_view_id else 1,
            }
        )
        for k in ("input_width", "input_height", "origin_width", "origin_height"):
            mf_data_batch["meta_data"][k] = mf_data_batch["meta_data"][k][0]

        # --- clip boundaries for time normalization ---
        # Record the full clip's frame_id range (before clip_sampler
        # sub-selects view_num frames) so that pipelines can normalize
        # time relative to the clip window rather than the selected frames.
        if hasattr(self, "data_infos"):
            full_clip = self.data_infos[idx]
            if full_clip:
                clip_fids = [v.get("frame_id", 0) for v in full_clip]
                mf_data_batch["meta_data"]["clip_start_frame_id"] = min(clip_fids)
                mf_data_batch["meta_data"]["clip_end_frame_id"] = max(clip_fids)

        if self.novel_view_nums > 0:
            mf_data_batch["meta_data"]["novel_view_nums"] = self.novel_view_nums

        # --- track query points ---
        if self.track_points_nums > 0 and tracks is not None:
            mf_data_batch["track_query_points"] = tracks
            mf_data_batch["track_vis"] = track_vis_masks
            mf_data_batch["track_pos_masks"] = track_pos_masks
            if self.sample_query_uvt:
                from hAlgorithm.datasets_mv.utils.query_points import (
                    sample_query_points_uvt,
                )
                query_points_uvt = sample_query_points_uvt(
                    tracks[None], track_vis_masks[None]
                )[0]
                mf_data_batch["query_points_uvt"] = query_points_uvt

        # --- SIFT mask ---
        if self.with_sift_mask:
            (
                sift_mask,
                sift_track_points,
                sift_track_points_cam,
                sift_track_points_uv,
                sift_track_mask,
                sift_track_vis,
            ) = self.create_sift_mask(
                idx, mf_data_batch_list, image_show_list
            )
            mf_data_batch["sift_mask"] = sift_mask
            if self.sift_track_nums > 0:
                mf_data_batch["sift_track_points"] = sift_track_points
                mf_data_batch["sift_track_points_cam"] = sift_track_points_cam
                mf_data_batch["sift_track_points_uv"] = sift_track_points_uv
                mf_data_batch["sift_track_mask"] = sift_track_mask
                mf_data_batch["sift_track_vis"] = sift_track_vis

        for key, val in mf_data_batch.items():
            if isinstance(val, torch.Tensor):
                mf_data_batch[key] = val.contiguous()

        return mf_data_batch

    def get_mf_data_for_test(self, idx):
        return self.get_mf_data_for_trainval(idx)

    # ==================================================================
    # Static trajectory generation
    # ==================================================================

    @staticmethod
    def _motion_mask_to_dynamic_hw(mm, height: int, width: int):
        """Normalise ``curr_motion_mask`` to a (H, W) bool map (True = dynamic).

        Returns *None* when ``mm`` is missing or cannot be aligned to the
        depth grid (shape mismatch after squeezing simple layouts).
        """
        if mm is None:
            return None
        dyn = np.asarray(mm)
        if dyn.dtype != np.bool_ and dyn.dtype != bool:
            dyn = dyn > 128
        while dyn.ndim > 2:
            last = dyn.shape[-1]
            if last == 1:
                dyn = dyn[..., 0]
            elif last in (3, 4):
                dyn = dyn.any(axis=-1)
            else:
                logging.warning(
                    "motion_mask trailing dim %d not handled; expected 1, 3, or 4",
                    last,
                )
                return None
        if dyn.shape != (height, width):
            logging.warning(
                "motion_mask shape %s does not match depth (%d, %d); ignoring motion mask",
                tuple(dyn.shape),
                height,
                width,
            )
            return None
        return dyn.astype(bool, copy=False)

    def _build_static_traj_sampling_mask(
        self,
        data_0: dict,
        depth_0: np.ndarray,
        dyn_trajs_2d: np.ndarray,
    ):
        """Pixels where static trajectory anchors may be sampled (bool, H×W).

        Priority:
        1. ``curr_motion_mask`` when aligned — forbid pixels marked dynamic.
        2. Else legacy heuristic: ``curr_sem`` + dense dynamic seg-IDs from
           trajectory landing counts (requires ``curr_sem``).

        Returns *None* when no rule can be applied (e.g. no mask and no sem).
        """
        H, W = depth_0.shape[:2]
        depth_valid = depth_0 > self.min_depth

        mm_dyn = self._motion_mask_to_dynamic_hw(
            data_0.get("curr_motion_mask"), H, W
        )
        if mm_dyn is not None:
            return depth_valid & (~mm_dyn)

        sem_0 = data_0.get("curr_sem")
        if sem_0 is None:
            return None

        traj_px = dyn_trajs_2d.astype(np.int64)
        traj_px[:, 0] = np.clip(traj_px[:, 0], 0, W - 1)
        traj_px[:, 1] = np.clip(traj_px[:, 1], 0, H - 1)
        traj_seg = sem_0[traj_px[:, 1], traj_px[:, 0]]

        dynamic_seg_ids = set()
        for sid in np.unique(traj_seg):
            if (traj_seg == sid).sum() > self.static_traj_dynamic_seg_threshold * len(
                dyn_trajs_2d
            ):
                dynamic_seg_ids.add(int(sid))

        static_mask = depth_valid
        for sid in dynamic_seg_ids:
            static_mask &= sem_0 != sid
        return static_mask

    def _generate_static_trajs(self, mf_data):
        """Sample static 3D points from frame-0 depth + seg, reproject.

        Requires ``curr_sem`` and ``curr_depth`` in frame-0 data.
        Returns a list of per-frame dicts (trajs_2d, trajs_3d, visibs,
        valids) or *None* if generation is not possible.

        Sampling is deterministic: a local RNG seeded from scene name and
        frame id guarantees the same static points across runs, so that
        different algorithms are evaluated on an identical point set.
        """
        V = len(mf_data["data"])
        data_0 = mf_data["data"][0]

        dyn_trajs_2d = data_0.get("curr_trajs_2d")
        if dyn_trajs_2d is None or len(dyn_trajs_2d) == 0:
            return None

        num_static = int(len(dyn_trajs_2d) * self.static_traj_ratio)
        if num_static <= 0:
            return None

        sem_0 = data_0.get("curr_sem")
        depth_0 = data_0.get("curr_depth")
        if sem_0 is None or depth_0 is None:
            return None

        H, W = depth_0.shape
        intrinsics_0 = data_0["curr_intrinsics"]
        extrinsics_0 = np.array(data_0["curr_extrinsics"])

        # identify seg-IDs that carry enough dynamic trajs
        traj_px = dyn_trajs_2d.astype(np.int64)
        traj_px[:, 0] = np.clip(traj_px[:, 0], 0, W - 1)
        traj_px[:, 1] = np.clip(traj_px[:, 1], 0, H - 1)
        traj_seg = sem_0[traj_px[:, 1], traj_px[:, 0]]

        dynamic_seg_ids = set()
        for sid in np.unique(traj_seg):
            if (traj_seg == sid).sum() > self.static_traj_dynamic_seg_threshold * len(dyn_trajs_2d):
                dynamic_seg_ids.add(int(sid))

        # static mask: valid depth & not on a dynamic seg-ID
        static_mask = depth_0 > self.min_depth
        for sid in dynamic_seg_ids:
            static_mask &= sem_0 != sid

        static_yx = np.argwhere(static_mask)
        if len(static_yx) == 0:
            return None
        num_static = min(num_static, len(static_yx))

        # Deterministic seed from scene identity so the same static points
        # are selected across different algorithm runs.
        scene_name = mf_data["info"][0].get("scene", "") if mf_data.get("info") else ""
        frame_id = mf_data["info"][0].get("frame_id", 0) if mf_data.get("info") else 0
        seed = hash((scene_name, frame_id, num_static)) % (2**31)
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(static_yx), num_static, replace=False)
        sampled_y = static_yx[indices, 0]
        sampled_x = static_yx[indices, 1]

        # unproject to world
        fx, fy, cx, cy = intrinsics_0
        d = depth_0[sampled_y, sampled_x]
        x_cam = (sampled_x.astype(np.float64) - cx) * d / fx
        y_cam = (sampled_y.astype(np.float64) - cy) * d / fy
        z_cam = d.astype(np.float64)
        p_cam_h = np.stack(
            [x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=-1
        )

        E_c2w_0 = np.linalg.inv(extrinsics_0)
        p_world = (E_c2w_0 @ p_cam_h.T).T[:, :3]
        p_world_h = np.concatenate(
            [p_world, np.ones((num_static, 1), dtype=np.float64)], axis=-1
        )

        # project through every frame
        tol = self.static_traj_depth_tolerance
        result = []
        for t in range(V):
            dt = mf_data["data"][t]
            fx_t, fy_t, cx_t, cy_t = dt["curr_intrinsics"]
            E_w2c_t = np.array(dt["curr_extrinsics"], dtype=np.float64)
            depth_t = dt["curr_depth"]

            p_cam_t = (E_w2c_t @ p_world_h.T).T
            z_t = p_cam_t[:, 2]

            u_t = fx_t * p_cam_t[:, 0] / z_t + cx_t
            v_t = fy_t * p_cam_t[:, 1] / z_t + cy_t

            in_bounds = (
                (u_t >= 0) & (u_t < W) & (v_t >= 0) & (v_t < H) & (z_t > 0)
            )

            u_int = np.clip(np.round(u_t).astype(np.int64), 0, W - 1)
            v_int = np.clip(np.round(v_t).astype(np.int64), 0, H - 1)
            rendered_z = depth_t[v_int, u_int].astype(np.float64)
            depth_ok = np.abs(rendered_z - z_t) < tol * np.abs(z_t)

            result.append(
                {
                    "trajs_2d": np.stack([u_t, v_t], axis=-1).astype(np.float32),
                    "trajs_3d": p_world.astype(np.float32),
                    "visibs": in_bounds & depth_ok,
                    "valids": in_bounds & (z_t > 0),
                }
            )

        return result

    # ==================================================================
    # Utility helpers
    # ==================================================================

    def clear_track_cache(self):
        self.track_cache.clear()

    # ------------------------------------------------------------------
    # Visualisation wrappers (delegate to vis_utils)
    # ------------------------------------------------------------------

    def visualize_data_batches(self, output_path="tracking_vis.rrd",
                               num_samples=2):
        return visualize_data_batches(
            self, output_path=output_path, num_samples=num_samples
        )

    def generate_merged_pointcloud(
        self, output_path="merged_pointcloud.ply", sample_index=0,
        downsample=4, max_points_per_frame=100000,
        depth_min=0.01, depth_max=100.0,
    ):
        return generate_merged_pointcloud(
            self, output_path=output_path, sample_index=sample_index,
            downsample=downsample, max_points_per_frame=max_points_per_frame,
            depth_min=depth_min, depth_max=depth_max,
        )

    def visualize_pointmap_vs_trajs3d(self, sample_idx=0,
                                       output_path="pointmap_vs_trajs3d.rrd"):
        return visualize_pointmap_vs_trajs3d(
            self, sample_idx=sample_idx, output_path=output_path
        )

    def visualize_tracking_debug(self, sample_idx=0,
                                  output_path="tracking_debug.rrd", **kwargs):
        return visualize_tracking_debug(
            self, sample_idx=sample_idx, output_path=output_path, **kwargs
        )

    def save_tracking_gif(self, sample_idx=0,
                           output_path="tracking_debug.gif", **kwargs):
        return save_tracking_gif(
            self, sample_idx=sample_idx, output_path=output_path, **kwargs
        )
