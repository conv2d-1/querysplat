#!/usr/bin/env python3
"""Visualize WorldTrack ground-truth 2D trajectories (all tracks, one MP4 per sequence)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hAlgorithm.eval.worldtrack_json_loader import (  # noqa: E402
    list_sequences_in_json,
    load_sequence_from_json,
    prefer_worldtrack_json_path,
)
from hAlgorithm.eval.worldtrack_vis import render_gt_tracks_2d_video  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render GT 2D trajectory videos for WorldTrack sequences.")
    p.add_argument(
        "--json",
        default="/mnt/nasTeam2/AI/datasets/TMD/Tapvid3d_mini/Tapvid3d_mini_extracted/pstudio_mini_test_mf_with_tracking.json",
        help="WorldTrack mf_files JSON path.",
    )
    p.add_argument(
        "--data-root",
        default="/mnt/nasTeam2/AI/datasets/TMD/Tapvid3d_mini",
        help="Dataset root for relative RGB / npy paths.",
    )
    p.add_argument(
        "--output-dir",
        default="tmp/vis_worldtrack_gt_pstudio",
        help="Output root; writes <output-dir>/<seq>/gt_tracks_2d.mp4.",
    )
    p.add_argument("--num-frames", type=int, default=64)
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--trace-frames", type=int, default=0, help="<=0 means full trajectory history.")
    p.add_argument("--seq-name", default="", help="Run only one sequence.")
    p.add_argument("--limit-seqs", type=int, default=0, help="<=0 means all sequences in JSON.")
    p.add_argument("--resume", action="store_true", help="Skip sequences with existing output mp4.")
    return p.parse_args()


def _select_sequences(json_path: Path, *, seq_name: str, limit_seqs: int) -> List[str]:
    names = list_sequences_in_json(json_path)
    if seq_name:
        if seq_name not in names:
            raise KeyError(f"Sequence {seq_name!r} not found in {json_path}")
        return [seq_name]
    if int(limit_seqs) > 0:
        return names[: int(limit_seqs)]
    return names


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    json_path = prefer_worldtrack_json_path(Path(args.json), data_root=Path(args.data_root))
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_names = _select_sequences(json_path, seq_name=args.seq_name, limit_seqs=args.limit_seqs)
    manifest: dict = {
        "json": str(json_path),
        "data_root": str(data_root),
        "num_frames": int(args.num_frames),
        "fps": float(args.fps),
        "trace_frames": int(args.trace_frames),
        "sequences": {},
    }

    for idx, seq_name in enumerate(seq_names, start=1):
        seq_out = output_dir / seq_name
        dst_mp4 = seq_out / "gt_tracks_2d.mp4"
        if args.resume and dst_mp4.is_file():
            logging.info("[%d/%d] skip existing %s", idx, len(seq_names), seq_name)
            manifest["sequences"][seq_name] = {"mp4": str(dst_mp4), "status": "skipped"}
            continue

        logging.info("[%d/%d] loading %s", idx, len(seq_names), seq_name)
        seq = load_sequence_from_json(
            json_path,
            seq_name,
            data_root=data_root,
            num_frames=int(args.num_frames),
            load_rgb=True,
        )
        seq_out.mkdir(parents=True, exist_ok=True)
        video_name = render_gt_tracks_2d_video(
            video_rgb=seq.video_rgb,
            tracks_uv_pixel=seq.tracks_uv,
            visibility_tq=seq.visibility,
            output_path=dst_mp4,
            trace_frames=int(args.trace_frames),
            fps=float(args.fps),
        )
        entry = {
            "mp4": str(dst_mp4),
            "video_name": video_name,
            "num_frames": int(seq.num_frames),
            "num_tracks": int(seq.num_tracks),
            "status": "ok",
        }
        manifest["sequences"][seq_name] = entry
        (seq_out / "gt_tracks_2d.meta.json").write_text(
            json.dumps(entry, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logging.info(
            "[%d/%d] wrote %s (T=%d Q=%d)",
            idx,
            len(seq_names),
            dst_mp4,
            seq.num_frames,
            seq.num_tracks,
        )

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    done = sum(1 for v in manifest["sequences"].values() if v.get("status") == "ok")
    skipped = sum(1 for v in manifest["sequences"].values() if v.get("status") == "skipped")
    logging.info("Finished: ok=%d skipped=%d total=%d -> %s", done, skipped, len(seq_names), output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
