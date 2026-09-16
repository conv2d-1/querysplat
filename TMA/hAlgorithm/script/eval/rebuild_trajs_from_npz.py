#!/usr/bin/env python3
"""Rebuild Tapvid3d JSON ``trajs_3d`` / ``trajs_2d`` from official WorldTrack NPZ files.

NPZ is authoritative: ``tracks_XYZ`` (camera), ``fx_fy_cx_cy``, ``extrinsics_w2c``.
Overwrites on-disk npy paths referenced by the mf_files JSON (paths unchanged).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hAlgorithm.eval.worldtrack_json_loader import _project_cam_to_uv, resolve_worldtrack_path


def _resolve_npz_path(
    seq_name: str,
    *,
    data_root: Path,
    subset: str,
    npz_subdir: Optional[str] = None,
) -> Path:
    candidates: List[Path] = []
    if npz_subdir:
        candidates.append(data_root / npz_subdir / f"{seq_name}.npz")
    candidates.append(data_root / subset / f"{seq_name}.npz")
    candidates.append(data_root / f"{subset}_mini" / f"{seq_name}.npz")
    candidates.append(
        data_root / "po_mini-20260526T085635Z-3-001" / "po_mini" / f"{seq_name}.npz"
    )
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"No NPZ for {seq_name!r} under {data_root} (subset={subset})")


def rebuild_json(
    *,
    json_path: Path,
    data_root: Path,
    subset: str,
    npz_subdir: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    mf_files = payload.get("mf_files", payload)
    if not isinstance(mf_files, dict):
        raise ValueError(f"Invalid mf_files JSON: {json_path}")

    stats: Dict[str, Any] = {
        "json": str(json_path),
        "subset": subset,
        "dry_run": dry_run,
        "num_sequences": 0,
        "num_frames": 0,
        "num_views_written": 0,
        "num_missing_npz": 0,
        "num_frame_mismatch": 0,
    }

    for seq_name, frames in mf_files.items():
        stats["num_sequences"] += 1
        try:
            npz_path = _resolve_npz_path(
                seq_name, data_root=data_root, subset=subset, npz_subdir=npz_subdir
            )
        except FileNotFoundError:
            stats["num_missing_npz"] += 1
            continue

        pack = np.load(npz_path, allow_pickle=True)
        tracks_xyz = np.asarray(pack["tracks_XYZ"], dtype=np.float32)
        intrinsics = np.asarray(pack["fx_fy_cx_cy"], dtype=np.float64).reshape(-1)[:4]
        n_npz = int(tracks_xyz.shape[0])

        for frame in frames:
            stats["num_frames"] += 1
            frame_id = int(frame.get("frame_id", stats["num_frames"] - 1))
            if frame_id >= n_npz:
                stats["num_frame_mismatch"] += 1
                continue

            views = frame.get("views", [])
            if not views:
                continue
            view = views[0]
            xyz = tracks_xyz[frame_id]
            cam_in = np.asarray(view.get("cam_in", intrinsics), dtype=np.float64).reshape(-1)[:4]
            uv = _project_cam_to_uv(xyz.astype(np.float64), cam_in).astype(np.float32)

            rel3d = view.get("trajs_3d")
            rel2d = view.get("trajs_2d")
            if not isinstance(rel3d, str) or not isinstance(rel2d, str):
                continue

            p3d = resolve_worldtrack_path(rel3d, data_root=data_root, json_path=json_path)
            p2d = resolve_worldtrack_path(rel2d, data_root=data_root, json_path=json_path)

            if not dry_run:
                p3d.parent.mkdir(parents=True, exist_ok=True)
                p2d.parent.mkdir(parents=True, exist_ok=True)
                np.save(p3d, xyz.astype(np.float32))
                np.save(p2d, uv)
            stats["num_views_written"] += 1

    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild trajs_3d/trajs_2d from official NPZ.")
    p.add_argument("--json", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument(
        "--subset",
        required=True,
        help="Subset folder name, e.g. po_mini or ds_mini (for NPZ lookup).",
    )
    p.add_argument(
        "--npz-subdir",
        default=None,
        help="Optional NPZ directory under data-root, e.g. ds_mini.",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    npz_subdir = args.npz_subdir or args.subset
    stats = rebuild_json(
        json_path=args.json,
        data_root=args.data_root,
        subset=args.subset,
        npz_subdir=npz_subdir,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
