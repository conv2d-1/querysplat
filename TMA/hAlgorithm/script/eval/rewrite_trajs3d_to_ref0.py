#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def tracks_cam_to_ref0_world(tracks_xyz_cam: np.ndarray, extrinsics_w2c: np.ndarray) -> np.ndarray:
    """Camera-space tracks [T, N, 3] -> ref0 world [T, N, 3].

    Matches `hAlgorithm.modules.metrics.worldtrack_eval_metrics.tracks_cam_to_ref0_world`.
    Extrinsics are world-to-camera; frame 0 is the reference.
    """
    tracks_xyz_cam = np.asarray(tracks_xyz_cam, dtype=np.float64)
    extrinsics_w2c = np.asarray(extrinsics_w2c, dtype=np.float64)
    frame_count = int(tracks_xyz_cam.shape[0])
    first_inv = np.linalg.inv(extrinsics_w2c[0])
    extrinsics_n = np.asarray(
        [extr @ first_inv for extr in extrinsics_w2c[:frame_count]], dtype=np.float64
    )
    extrinsics_c2w = np.linalg.inv(extrinsics_n)
    tracks_xyz_world = np.empty_like(tracks_xyz_cam, dtype=np.float64)
    for frame_idx in range(frame_count):
        rot = extrinsics_c2w[frame_idx, :3, :3]
        trans = extrinsics_c2w[frame_idx, :3, 3]
        tracks_xyz_world[frame_idx] = (rot @ tracks_xyz_cam[frame_idx].T).T + trans
    return tracks_xyz_world


def _is_path_like(x: Any) -> bool:
    return isinstance(x, str) and x.strip() != ""


def _load_trajs3d(view: Dict[str, Any], *, json_path: Path) -> Tuple[np.ndarray, Path]:
    ref = view.get("trajs_3d")
    if not _is_path_like(ref):
        raise ValueError(f"{json_path}: view.trajs_3d is not a string path; refusing to overwrite.")
    p = Path(str(ref))
    if not p.is_absolute():
        p = (json_path.parent / p).resolve()
    if not p.is_file():
        # Try relative to json's parent.parent (matches loader default data_root)
        alt = (json_path.parent.parent / Path(str(ref))).resolve()
        if alt.is_file():
            p = alt
        else:
            raise FileNotFoundError(f"{json_path}: trajs_3d not found: {ref!r} (tried {p} and {alt})")
    arr = np.load(p)
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{json_path}: trajs_3d has shape {arr.shape}, expected [N,3]: {p}")
    return arr.astype(np.float64), p


def _load_extrinsics(view: Dict[str, Any], *, json_path: Path) -> np.ndarray:
    extr = view.get("extrinsics")
    if extr is None:
        raise KeyError(f"{json_path}: view has no 'extrinsics'")
    extr_arr = np.asarray(extr, dtype=np.float64)
    if extr_arr.shape != (4, 4):
        raise ValueError(f"{json_path}: extrinsics shape {extr_arr.shape}, expected (4,4)")
    return extr_arr


def rewrite_one_json(
    json_path: Path,
    *,
    dry_run: bool,
    backup: bool,
    backup_suffix: str,
    validate_samples: int,
) -> Dict[str, Any]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    mf = payload.get("mf_files", payload)
    if not isinstance(mf, dict):
        raise ValueError(f"Invalid mf_files JSON: {json_path}")

    sequences = sorted(mf.keys())
    changed_files = 0
    total_frames = 0
    total_sequences = 0
    max_abs_diff_sample = 0.0
    sampled = 0

    for seq_name in sequences:
        frames = mf[seq_name]
        if not isinstance(frames, list) or len(frames) == 0:
            continue
        total_sequences += 1

        trajs_cam: List[np.ndarray] = []
        extrs: List[np.ndarray] = []
        traj_paths: List[Path] = []

        for frame in frames:
            views = frame.get("views", [])
            if not isinstance(views, list) or len(views) != 1:
                raise ValueError(f"{json_path}: sequence {seq_name}: expected 1 view per frame")
            view = views[0]
            traj, traj_path = _load_trajs3d(view, json_path=json_path)
            extr = _load_extrinsics(view, json_path=json_path)
            trajs_cam.append(traj)
            extrs.append(extr)
            traj_paths.append(traj_path)

        tracks_cam = np.stack(trajs_cam, axis=0)  # [T,N,3]
        extrinsics_w2c = np.stack(extrs, axis=0)  # [T,4,4]
        tracks_ref0 = tracks_cam_to_ref0_world(tracks_cam, extrinsics_w2c)  # [T,N,3]

        # Optional small validation: re-run transform from cam and compare against ref0 we just computed.
        # This catches shape mishaps and NaN propagation issues without reading/writing again.
        if validate_samples > 0 and sampled < validate_samples:
            ref_check = tracks_cam_to_ref0_world(tracks_cam, extrinsics_w2c)
            diff = np.nanmax(np.abs(ref_check - tracks_ref0))
            if np.isfinite(diff):
                max_abs_diff_sample = max(max_abs_diff_sample, float(diff))
            sampled += 1

        for t, out_path in enumerate(traj_paths):
            total_frames += 1
            if dry_run:
                continue
            if backup:
                bak = Path(str(out_path) + backup_suffix)
                if not bak.exists():
                    shutil.copy2(out_path, bak)
            # np.save() appends ".npy" when the path does not end with ".npy".
            tmp = Path(str(out_path) + ".tmp.npy")
            np.save(str(tmp), np.asarray(tracks_ref0[t], dtype=np.float64))
            os.replace(tmp, out_path)
            changed_files += 1

    return {
        "json": str(json_path),
        "sequences_seen": int(total_sequences),
        "frames_seen": int(total_frames),
        "files_overwritten": int(changed_files),
        "validate_samples": int(validate_samples),
        "max_abs_diff_sample": float(max_abs_diff_sample),
        "dry_run": bool(dry_run),
        "backup": bool(backup),
        "backup_suffix": str(backup_suffix),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rewrite WorldTrack/Tapvid3d mini trajs_3d npys into frame0-ref coordinates (ref0)."
    )
    ap.add_argument(
        "json_paths",
        nargs="+",
        help="mf_files JSON(s) to process (each view.trajs_3d must be a string path to .npy).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Scan and compute but do not write files.")
    ap.add_argument("--no-backup", action="store_true", help="Do not create .bak backups.")
    ap.add_argument(
        "--backup-suffix",
        default=".bak",
        help="Backup suffix appended to original trajs_3d npy path (default: .bak).",
    )
    ap.add_argument(
        "--validate-samples",
        type=int,
        default=3,
        help="Validate up to this many sequences (recompute and compare max abs diff).",
    )
    args = ap.parse_args()

    reports = []
    for p in args.json_paths:
        jp = Path(p).expanduser().resolve()
        reports.append(
            rewrite_one_json(
                jp,
                dry_run=bool(args.dry_run),
                backup=not bool(args.no_backup),
                backup_suffix=str(args.backup_suffix),
                validate_samples=int(args.validate_samples),
            )
        )

    print(json.dumps({"reports": reports}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

