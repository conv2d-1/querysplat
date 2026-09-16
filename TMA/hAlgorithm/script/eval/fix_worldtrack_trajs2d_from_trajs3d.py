#!/usr/bin/env python3
"""Fix WorldTrack/Tapvid3d ``trajs_2d`` by pinhole projection.

Default (adt_mini / pstudio_mini): ``trajs_3d`` is already in the camera frame.

For po_mini / ds_mini, pass ``--world-coords-3d``: apply per-view ``extrinsics`` (w2c)
before projection. Optionally ``--convert-trajs3d-to-camera`` overwrites ``trajs_3d`` npy
with the camera-frame XYZ so TMA loaders match Open-d4rt.

Use ``--in-place`` to overwrite original ``trajs_2d`` npy files (JSON paths unchanged).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _resolve_path(rel_or_abs: str, *, data_root: Path, json_path: Path) -> Path:
    p = Path(rel_or_abs)
    if p.is_file():
        return p
    if p.is_absolute():
        return p
    candidates = [
        data_root / p,
        data_root / p.name,
        json_path.parent / p,
        json_path.parent.parent / p,
    ]
    rel = rel_or_abs.replace("\\", "/")
    if "Tapvid3d_mini_extracted/" in rel:
        suffix = rel.split("Tapvid3d_mini_extracted/")[-1]
        candidates.append(data_root / "Tapvid3d_mini_extracted" / suffix)
        candidates.append(data_root.parent / "Tapvid3d_mini_extracted" / suffix)
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"Could not resolve path {rel_or_abs!r} under data_root={data_root}")


def _world_to_camera(trajs_3d: np.ndarray, extrinsics_w2c: np.ndarray) -> np.ndarray:
    """Apply 4x4 world-to-camera extrinsic: X_cam = R @ X_world + t."""
    xyz = np.asarray(trajs_3d, dtype=np.float64)
    ext = np.asarray(extrinsics_w2c, dtype=np.float64).reshape(4, 4)
    rot = ext[:3, :3]
    trans = ext[:3, 3]
    return (rot @ xyz.T).T + trans


def _project_cam_to_uv(trajs_3d_cam: np.ndarray, cam_in: np.ndarray) -> np.ndarray:
    """Project camera-frame XYZ to pixel uv using pinhole intrinsics."""
    xyz = np.asarray(trajs_3d_cam, dtype=np.float64)
    cam = np.asarray(cam_in, dtype=np.float64).reshape(-1)
    if cam.size < 4:
        raise ValueError(f"cam_in must have 4 params (fx,fy,cx,cy); got shape {np.asarray(cam_in).shape}")
    fx, fy, cx, cy = cam[:4]
    z = xyz[:, 2]
    safe_z = np.where(np.abs(z) > 1e-8, z, np.nan)
    u = (xyz[:, 0] / safe_z) * fx + cx
    v = (xyz[:, 1] / safe_z) * fy + cy
    return np.stack([u, v], axis=-1).astype(np.float32)


def _default_out_rel_path(orig_rel: str, *, out_prefix: str) -> str:
    rel = orig_rel.replace("\\", "/")
    if "Tapvid3d_mini_extracted/" in rel:
        suffix = rel.split("Tapvid3d_mini_extracted/")[-1]
    else:
        suffix = rel
    return f"{out_prefix.rstrip('/')}/{suffix}"


def fix_json(
    *,
    json_path: Path,
    data_root: Path,
    out_root: Path,
    out_prefix: str,
    in_place: bool = False,
    backup_json: bool = True,
    dry_run: bool = False,
    world_coords_3d: bool = False,
    convert_trajs3d_to_camera: bool = False,
) -> Tuple[Path, Dict[str, Any]]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    mf_files = payload.get("mf_files", payload)
    if not isinstance(mf_files, dict):
        raise ValueError("Invalid mf_files JSON: expected dict at key 'mf_files'")

    stats: Dict[str, Any] = {
        "in_place": in_place,
        "dry_run": dry_run,
        "world_coords_3d": world_coords_3d,
        "convert_trajs3d_to_camera": convert_trajs3d_to_camera,
        "num_sequences": 0,
        "num_frames": 0,
        "num_views": 0,
        "num_trajs2d_written": 0,
        "num_trajs3d_converted": 0,
        "num_missing_trajs3d": 0,
        "num_missing_trajs2d_ref": 0,
        "num_missing_extrinsics": 0,
    }

    fixed = {"mf_files": {}}
    for seq_name, frames in mf_files.items():
        stats["num_sequences"] += 1
        fixed_frames = []
        for frame in frames:
            frame = dict(frame)
            views = []
            for view in frame.get("views", []):
                view = dict(view)
                rel3d = view.get("trajs_3d")
                if not isinstance(rel3d, str) or not rel3d:
                    stats["num_missing_trajs3d"] += 1
                    views.append(view)
                    continue

                p3d = _resolve_path(rel3d, data_root=data_root, json_path=json_path)
                trajs_3d = np.load(p3d)
                cam_in = np.asarray(view.get("cam_in", []), dtype=np.float64)

                trajs_3d_cam = np.asarray(trajs_3d, dtype=np.float64)
                if world_coords_3d:
                    ext = view.get("extrinsics")
                    if ext is None:
                        stats["num_missing_extrinsics"] += 1
                        views.append(view)
                        continue
                    trajs_3d_cam = _world_to_camera(trajs_3d_cam, np.asarray(ext, dtype=np.float64))

                uv = _project_cam_to_uv(trajs_3d_cam, cam_in)

                rel2d = view.get("trajs_2d")
                if in_place:
                    if not isinstance(rel2d, str) or not rel2d:
                        stats["num_missing_trajs2d_ref"] += 1
                        views.append(view)
                        continue
                    out_path_2d = _resolve_path(rel2d, data_root=data_root, json_path=json_path)
                else:
                    if isinstance(rel2d, str) and rel2d:
                        out_rel = _default_out_rel_path(rel2d, out_prefix=out_prefix)
                    else:
                        out_rel = _default_out_rel_path(
                            rel3d.replace("/trajs_3d/", "/trajs_2d/"), out_prefix=out_prefix
                        )
                    out_path_2d = out_root / out_rel
                    out_path_2d.parent.mkdir(parents=True, exist_ok=True)
                    view["trajs_2d_fixed"] = out_rel

                if not dry_run:
                    np.save(out_path_2d, uv)
                stats["num_trajs2d_written"] += 1

                if convert_trajs3d_to_camera and world_coords_3d:
                    if not dry_run:
                        np.save(p3d, trajs_3d_cam.astype(np.float32))
                    stats["num_trajs3d_converted"] += 1

                view.pop("trajs_2d_fixed", None)
                views.append(view)
                stats["num_views"] += 1
            frame["views"] = views
            fixed_frames.append(frame)
            stats["num_frames"] += 1
        fixed["mf_files"][seq_name] = fixed_frames

    if in_place:
        out_json = json_path.resolve()
        if backup_json and not dry_run:
            backup_path = out_json.with_suffix(out_json.suffix + ".bak")
            shutil.copy2(out_json, backup_path)
            stats["json_backup"] = str(backup_path)
    else:
        out_json = out_root / (json_path.stem + "_trajs2d_fixed.json")

    if not dry_run:
        out_json.write_text(json.dumps(fixed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stats["out_json"] = str(out_json)
    return out_json, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fix trajs_2d by projecting trajs_3d with cam_in.")
    p.add_argument("--json", required=True, type=Path, help="Input mf_files JSON.")
    p.add_argument("--data-root", required=True, type=Path, help="Tapvid3d_mini root directory.")
    p.add_argument(
        "--world-coords-3d",
        action="store_true",
        help="trajs_3d is world/ref; apply view extrinsics (w2c) before projection.",
    )
    p.add_argument(
        "--convert-trajs3d-to-camera",
        action="store_true",
        help="With --world-coords-3d, overwrite trajs_3d npy with camera-frame XYZ.",
    )
    p.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite original trajs_2d npy files (JSON paths unchanged).",
    )
    p.add_argument(
        "--no-backup-json",
        action="store_true",
        help="With --in-place, do not write <json>.bak before updating JSON.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Count files only; do not write npy or JSON.",
    )
    p.add_argument(
        "--out-root",
        default=Path("tmp/worldtrack_trajs2d_fixed"),
        type=Path,
        help="Output root when not using --in-place.",
    )
    p.add_argument(
        "--out-prefix",
        default="Tapvid3d_mini_extracted_trajs2d_fixed",
        help="Prefix for copied npys when not using --in-place.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.convert_trajs3d_to_camera and not args.world_coords_3d:
        raise SystemExit("--convert-trajs3d-to-camera requires --world-coords-3d")
    out_json, stats = fix_json(
        json_path=args.json,
        data_root=args.data_root,
        out_root=args.out_root,
        out_prefix=str(args.out_prefix),
        in_place=bool(args.in_place),
        backup_json=not args.no_backup_json,
        dry_run=bool(args.dry_run),
        world_coords_3d=bool(args.world_coords_3d),
        convert_trajs3d_to_camera=bool(args.convert_trajs3d_to_camera),
    )
    print(f"{'[dry-run] ' if args.dry_run else ''}JSON: {out_json}")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
