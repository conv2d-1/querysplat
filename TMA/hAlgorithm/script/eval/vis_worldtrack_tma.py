#!/usr/bin/env python3
"""Re-render WorldTrack TMA visualization from a saved ``tracks.npz`` package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hAlgorithm.eval.worldtrack_vis import (  # noqa: E402
    export_tracks_debug_ply_package,
    visualize_tracks_package,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render WorldTrack track comparison videos.")
    p.add_argument(
        "package_dir",
        type=Path,
        help="Directory containing tracks.npz (e.g. output/adt_mini/seq_name).",
    )
    p.add_argument("--max-points", type=int, default=300)
    p.add_argument("--trace-frames", type=int, default=8)
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument(
        "--ply-only",
        action="store_true",
        help="Only export GT/Pred PLY (skip video rendering).",
    )
    p.add_argument(
        "--no-ply",
        action="store_true",
        help="Skip PLY export when rendering videos.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.ply_only:
        import json

        import numpy as np

        pack = np.load(args.package_dir / "tracks.npz", allow_pickle=False)
        metrics = {}
        mp = args.package_dir / "metrics.json"
        if mp.is_file():
            metrics = json.loads(mp.read_text(encoding="utf-8"))
        raw = pack["pred_tracks_ref0_raw"] if "pred_tracks_ref0_raw" in pack.files else None
        manifest = export_tracks_debug_ply_package(
            args.package_dir,
            gt_tracks_world=pack["gt_tracks_world"],
            pred_tracks_ref0=pack["pred_tracks_ref0"],
            pred_tracks_raw=raw,
            visibility_tq=pack["visibility"],
            scale_global=metrics.get("scale_global"),
            video_rgb=pack["video_rgb"],
            extrinsics_w2c=pack["extrinsics_w2c"],
            intrinsics=pack["intrinsics"],
        )
        print(f"Wrote PLY under {args.package_dir / 'vis'}:")
        print("  tracks_gt_imagecolor.ply              (GT only, video RGB)")
        print("  tracks_pred_aligned_imagecolor.ply    (Pred aligned, video RGB)")
        print("  tracks_pred_gt_aligned.ply            (same as pred aligned, video RGB)")
        if manifest.get("pred_raw"):
            print("  tracks_pred_raw_imagecolor.ply        (Pred before scale, video RGB)")
        print("  tracks_error_lines.ply                (error segments, video RGB endpoints)")
        print(json.dumps(manifest.get("aligned_stats", {}), indent=2))
        return 0

    paths = visualize_tracks_package(
        args.package_dir,
        max_points=args.max_points,
        trace_frames=args.trace_frames,
        fps=args.fps,
        export_ply=not args.no_ply,
    )
    print(f"Wrote videos under {args.package_dir / 'vis'}:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
