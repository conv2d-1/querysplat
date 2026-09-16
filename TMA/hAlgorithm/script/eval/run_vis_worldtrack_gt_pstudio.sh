#!/usr/bin/env bash
# Render GT 2D trajectory MP4s for all 50 pstudio scenes (first 64 frames).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/mnt/home/tcchen/miniforge3/envs/acc/bin/python}"
JSON="${JSON:-/mnt/nasTeam2/AI/datasets/TMD/Tapvid3d_mini/Tapvid3d_mini_extracted/pstudio_mini_test_mf_with_tracking.json}"
DATA_ROOT="${DATA_ROOT:-/mnt/nasTeam2/AI/datasets/TMD/Tapvid3d_mini}"
OUT_DIR="${OUT_DIR:-tmp/vis_worldtrack_gt_pstudio}"

"$PYTHON_BIN" hAlgorithm/script/eval/vis_worldtrack_gt.py \
  --json "$JSON" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUT_DIR" \
  --num-frames 64 \
  --fps 15 \
  --trace-frames 0 \
  --resume

echo "GT vis ready under: $REPO_ROOT/$OUT_DIR/<seq_name>/gt_tracks_2d.mp4"
