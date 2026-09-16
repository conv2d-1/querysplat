#!/usr/bin/env python3
"""Build SynthVerse mf JSON + npy with **dynamic queries only** (world-space motion gate).

Static background points (world ``trajs_3d`` displacement from frame 0 <= threshold) are
removed from per-frame ``trajs_3d`` / ``trajs_2d`` / ``valids`` / ``visibs`` arrays.

Example:
  python hAlgorithm/script/eval/convert_synthverse_dynamic_only.py \\
    --input-json /mnt/nasTeam2/AI/datasets/TMD/SynthVerse/converted/train_mf_with_tracking_subset50_64f_q430.json \\
    --data-root /mnt/nasTeam2/AI/datasets/TMD \\
    --output-json /mnt/nasTeam2/AI/datasets/TMD/SynthVerse/converted/train_mf_with_tracking_subset50_64f_dynamic.json \\
    --output-subdir converted_subset50_64f_dynamic \\
    --motion-threshold 0.01
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from hAlgorithm.eval.worldtrack_json_loader import (
    compute_dynamic_point_mask,
    list_sequences_in_json,
    resolve_worldtrack_path,
)


def _load_npy(data_root: Path, json_path: Path, ref: Any) -> np.ndarray:
    if isinstance(ref, str):
        return np.load(resolve_worldtrack_path(ref, data_root=data_root, json_path=json_path))
    return np.asarray(ref)


def _rel_path(path: Path, data_root: Path) -> str:
    p = path.resolve()
    root = data_root.resolve()
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def convert_sequence(
    *,
    seq_name: str,
    frames: List[dict],
    data_root: Path,
    input_json: Path,
    out_track_root: Path,
    motion_threshold: float,
    num_frames: int,
) -> Tuple[List[dict], int, int]:
    clip = frames[: min(int(num_frames), len(frames))]
    world_tracks = []
    for frame in clip:
        view = frame["views"][0]
        world_tracks.append(_load_npy(data_root, input_json, view["trajs_3d"]))
    world_stack = np.stack(world_tracks, axis=0)
    dyn_mask = compute_dynamic_point_mask(world_stack, motion_threshold)
    n_before = int(dyn_mask.size)
    n_after = int(dyn_mask.sum())
    if n_after == 0:
        return [], n_before, n_after

    seq_out = out_track_root / seq_name
    for sub in ("trajs_3d", "trajs_2d", "valids", "visibs"):
        (seq_out / sub).mkdir(parents=True, exist_ok=True)

    out_frames: List[dict] = []
    for t, frame in enumerate(clip):
        frame = dict(frame)
        view = dict(frame["views"][0])
        tr3 = _load_npy(data_root, input_json, view["trajs_3d"])[dyn_mask]
        tr2 = _load_npy(data_root, input_json, view["trajs_2d"])[dyn_mask]
        val = _load_npy(data_root, input_json, view["valids"])[dyn_mask]
        vis = _load_npy(data_root, input_json, view["visibs"])[dyn_mask]

        stem = Path(str(view["trajs_3d"])).name
        out_stem = f"{Path(stem).stem}.npy"
        for key, arr, sub in (
            ("trajs_3d", tr3, "trajs_3d"),
            ("trajs_2d", tr2, "trajs_2d"),
            ("valids", val, "valids"),
            ("visibs", vis, "visibs"),
        ):
            out_p = seq_out / sub / out_stem
            np.save(out_p, arr)
            view[key] = _rel_path(out_p, data_root)
        frame["views"] = [view]
        out_frames.append(frame)
    return out_frames, n_before, n_after


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert SynthVerse tracks to dynamic-only queries.")
    ap.add_argument(
        "--input-json",
        default="/mnt/nasTeam2/AI/datasets/TMD/SynthVerse/converted/train_mf_with_tracking_subset50_64f_q430.json",
    )
    ap.add_argument("--data-root", default="/mnt/nasTeam2/AI/datasets/TMD")
    ap.add_argument(
        "--output-json",
        default="/mnt/nasTeam2/AI/datasets/TMD/SynthVerse/converted/train_mf_with_tracking_subset50_64f_dynamic.json",
    )
    ap.add_argument(
        "--output-subdir",
        default="converted_subset50_64f_dynamic",
        help="Under SynthVerse/converted/ for filtered npy files.",
    )
    ap.add_argument("--motion-threshold", type=float, default=0.01)
    ap.add_argument("--num-frames", type=int, default=64)
    args = ap.parse_args()

    input_json = Path(args.input_json)
    data_root = Path(args.data_root)
    output_json = Path(args.output_json)
    out_track_root = data_root / "SynthVerse" / "converted" / args.output_subdir

    with open(input_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    mf_in = payload.get("mf_files", payload)

    mf_out: Dict[str, List[dict]] = {}
    stats = []
    skipped: List[str] = []
    total_q = 0
    for seq_name in list_sequences_in_json(input_json):
        frames, n_before, n_after = convert_sequence(
            seq_name=seq_name,
            frames=mf_in[seq_name],
            data_root=data_root,
            input_json=input_json,
            out_track_root=out_track_root,
            motion_threshold=float(args.motion_threshold),
            num_frames=int(args.num_frames),
        )
        if n_after == 0:
            skipped.append(seq_name)
            continue
        mf_out[seq_name] = frames
        total_q += n_after
        stats.append(
            {
                "sequence": seq_name,
                "tracks_before": n_before,
                "tracks_after": n_after,
                "static_removed": n_before - n_after,
            }
        )

    out_payload = dict(payload)
    out_payload["mf_files"] = mf_out
    out_payload["synthverse_dynamic_filter"] = {
        "source_json": str(input_json),
        "motion_threshold_m": float(args.motion_threshold),
        "coord_space": "world_trajs_3d",
        "num_sequences": len(mf_out),
        "total_queries": int(total_q),
        "skipped_zero_dynamic": skipped,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    stats_path = output_json.with_suffix(".stats.json")
    stats_path.write_text(
        json.dumps(
            {
                "output_json": str(output_json),
                "track_root": str(out_track_root),
                "total_queries": int(total_q),
                "sequences": stats,
                "skipped_zero_dynamic": skipped,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_json}")
    print(f"Wrote {stats_path}")
    print(f"sequences={len(mf_out)} total_queries={total_q} skipped={len(skipped)}")
    if skipped:
        print("skipped:", ", ".join(skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
