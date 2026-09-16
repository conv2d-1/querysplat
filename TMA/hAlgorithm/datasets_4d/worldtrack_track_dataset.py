"""WorldTrack / Tapvid3d benchmark loader (mf_files JSON) for TMA inference."""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.append(os.getcwd())

from hAlgorithm.datasets_4d.base_track_dataset import BaseTrackDataset
from hAlgorithm.eval.worldtrack_json_loader import (
    frame0_query_indices_for_sequence,
    normalize_coord_convention,
    prefer_worldtrack_json_path,
    resolve_opend4rt_frame_count,
    resolve_trajs_2d_for_view,
    resolve_worldtrack_path,
)


class WorldTrackTrackDataset(BaseTrackDataset):
    """One dataset index = one sequence (first ``num_frames`` frames, single moving camera).

    Uses standard ``load_mf_files`` + ``mf_to_mv`` so ``getitem_multi_frame`` works like
    PointOdyssey / HaSim tracking datasets.

    Clip length and frame-0 queries follow Open-d4rt (see ``worldtrack_json_loader``).
    """

    def __init__(
        self,
        num_frames: int = 64,
        sequence_names: Optional[List[str]] = None,
        coord_convention: Optional[str] = None,
        dynamic_query_threshold: Optional[float] = None,
        **kwargs,
    ):
        self.num_frames = int(num_frames)
        self.sequence_names = sequence_names
        self.coord_convention = normalize_coord_convention(coord_convention)
        self.dynamic_query_threshold = (
            float(dynamic_query_threshold)
            if dynamic_query_threshold is not None
            else None
        )
        self._worldtrack_query_cache: Dict[str, np.ndarray] = {}
        kwargs.setdefault("mf_to_mv", True)
        super().__init__(**kwargs)
        self.data_path = str(
            prefer_worldtrack_json_path(self.data_path, data_root=self.data_root)
        )

    def convert_extrinsics(self, extrinsics_raw: np.ndarray) -> np.ndarray:
        """TapVid3D JSON stores w2c; SynthVerse / PO-style mf JSON stores c2w."""
        ext = np.asarray(extrinsics_raw, dtype=np.float64)
        if self.coord_convention == "pointodyssey":
            return np.linalg.inv(ext)
        return ext

    def get_data_infos(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            mf_files = json.load(f)["mf_files"]

        if self.sequence_names is not None:
            allow = set(self.sequence_names)
            mf_files = {k: v for k, v in mf_files.items() if k in allow}

        mf_trunc = {}
        for scene, frames in mf_files.items():
            clip_len = resolve_opend4rt_frame_count(
                self.num_frames,
                num_images=len(frames),
                num_track_frames=len(frames),
                num_visibility_frames=len(frames),
            )
            clip = []
            for frame in frames[:clip_len]:
                frame = dict(frame)
                frame_id = int(frame["frame_id"])
                views = []
                for view in frame["views"]:
                    view = dict(view)
                    if view.get("depth") is None:
                        view.pop("depth", None)
                    # Single moving camera: keep a constant view_id so mf_to_mv
                    # groups all frames into one temporal clip.
                    view["view_id"] = 0
                    view["frame_id"] = frame_id
                    views.append(view)
                frame["views"] = views
                clip.append(frame)
            mf_trunc[scene] = clip
        self.data_infos = self.load_mf_files(mf_trunc)
        return self.data_infos

    def preprocess_data_info(self, data_info: dict) -> dict:
        data_info = dict(data_info)
        if data_info.get("depth") is None:
            data_info.pop("depth", None)
        if "view_id" not in data_info and "frame_id" in data_info:
            data_info["view_id"] = data_info["frame_id"]
        for key in ("rgb", "trajs_3d", "valids", "visibs"):
            ref = data_info.get(key)
            if isinstance(ref, str):
                data_info[key] = str(
                    resolve_worldtrack_path(
                        ref,
                        data_root=self.data_root,
                        json_path=self.data_path,
                    )
                )
        if isinstance(data_info.get("trajs_2d"), str) or isinstance(
            data_info.get("trajs_2d_fixed"), str
        ):
            data_info["trajs_2d"] = str(
                resolve_trajs_2d_for_view(
                    data_info,
                    data_root=self.data_root,
                    json_path=self.data_path,
                )
            )
        return data_info

    def load_data(self, data_info):
        data_info = self.preprocess_data_info(data_info)
        frame_batch = super().load_data(data_info)
        return self._ensure_depth_from_trajs(frame_batch)

    def _frame0_query_indices_for_sequence(self, sequence_name: str) -> np.ndarray:
        """Identical to ``WorldTrackSequence.query_indices`` from the JSON loader."""
        if sequence_name not in self._worldtrack_query_cache:
            self._worldtrack_query_cache[sequence_name] = frame0_query_indices_for_sequence(
                self.data_path,
                sequence_name,
                data_root=self.data_root,
                num_frames=self.num_frames,
                coord_convention=self.coord_convention,
                dynamic_query_threshold=self.dynamic_query_threshold,
            )
        return self._worldtrack_query_cache[sequence_name]

    @staticmethod
    def _sequence_name_from_sample(sample: dict) -> str:
        meta = sample.get("meta_data", {})
        data_info = meta.get("data_info")
        if isinstance(data_info, list) and data_info:
            first = data_info[0]
            if isinstance(first, dict):
                scene = first.get("scene")
                if scene:
                    return str(scene)
        raise RuntimeError(
            "WorldTrackTrackDataset: cannot resolve sequence name from sample meta_data"
        )

    def _ensure_depth_from_trajs(self, frame_batch: dict) -> dict:
        """Synthesize a depth map when JSON has no ``depth`` path (TapVid3D).

        SynthVerse ships metric depth PNGs (``depth`` + ``depth_scale``); this fallback
        should not run there. When it does run under ``pointodyssey``, use camera-frame
        Z from world ``trajs_3d`` + w2c — not world Z (``trajs_3d[:, 2]``).
        """
        if frame_batch.get("curr_depth") is not None:
            return frame_batch
        trajs_3d = frame_batch.get("curr_trajs_3d")
        trajs_2d = frame_batch.get("curr_trajs_2d")
        if trajs_3d is None or trajs_2d is None:
            return frame_batch

        rgb = frame_batch.get("curr_rgb")
        if rgb is None:
            return frame_batch
        h, w = rgb.shape[:2]
        depth = np.full((h, w), np.nan, dtype=np.float32)
        uv = np.asarray(trajs_2d, dtype=np.float64)
        pts = np.asarray(trajs_3d, dtype=np.float64)
        if self.coord_convention == "pointodyssey" and frame_batch.get("curr_extrinsics") is not None:
            w2c = np.asarray(frame_batch["curr_extrinsics"], dtype=np.float64)
            hom = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
            z = (hom @ w2c.T)[:, 2]
        else:
            z = pts[:, 2]
        ui = np.clip(np.round(uv[:, 0]).astype(np.int32), 0, w - 1)
        vi = np.clip(np.round(uv[:, 1]).astype(np.int32), 0, h - 1)
        valid = np.isfinite(z) & np.isfinite(uv).all(axis=-1) & (z > 1e-6)
        depth[vi[valid], ui[valid]] = z[valid].astype(np.float32)
        fill = float(np.nanmedian(z[valid])) if np.any(valid) else 1.0
        depth[~np.isfinite(depth)] = fill
        frame_batch["curr_depth"] = depth
        return frame_batch

    @staticmethod
    def _strip_null_depth_from_meta(meta_data: dict) -> None:
        """``default_collate`` cannot stack ``None`` depth paths in ``meta_data``."""
        data_info = meta_data.get("data_info")
        if isinstance(data_info, list):
            for entry in data_info:
                if isinstance(entry, dict) and entry.get("depth") is None:
                    entry.pop("depth", None)
        elif isinstance(data_info, dict) and data_info.get("depth") is None:
            data_info.pop("depth", None)

    @staticmethod
    def _filter_sample_to_frame0_queries(
        sample: dict,
        query_indices: np.ndarray,
    ) -> dict:
        """Subset tracking tensors to WorldTrack frame-0 queries."""
        trajs_2d = sample.get("trajs_2d")
        trajs_3d = sample.get("trajs_3d")
        visibs = sample.get("visibs")
        valids = sample.get("valids")
        if trajs_2d is None or trajs_3d is None:
            return sample

        idx_q = np.asarray(query_indices, dtype=np.int64).reshape(-1)
        if idx_q.size == 0:
            return sample

        def _take(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x[:, idx_q].contiguous()
            return x[:, idx_q]

        sample["trajs_2d"] = _take(trajs_2d)
        sample["trajs_3d"] = _take(trajs_3d)
        if visibs is not None:
            sample["visibs"] = _take(visibs)
        if valids is not None:
            sample["valids"] = _take(valids)
        sample["worldtrack_num_queries"] = int(idx_q.size)
        sample["worldtrack_query_indices"] = idx_q.copy()
        return sample

    def get_mf_data_for_trainval(self, idx):
        sample = super().get_mf_data_for_trainval(idx)
        if isinstance(sample.get("meta_data"), dict):
            self._strip_null_depth_from_meta(sample["meta_data"])
        seq_name = self._sequence_name_from_sample(sample)
        query_indices = self._frame0_query_indices_for_sequence(seq_name)
        return self._filter_sample_to_frame0_queries(sample, query_indices)
