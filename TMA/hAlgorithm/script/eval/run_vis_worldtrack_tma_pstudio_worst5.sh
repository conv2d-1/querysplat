#!/usr/bin/env bash
# Visualize TMA on pstudio worst-5 scenes: MP4 + frame-0 GT/Pred PLY.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-acc}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]] && command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
  PYTHON_BIN="$(command -v python)"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

CONFIG="${CONFIG:-/mnt/home/tcchen/workspace/TMA/results_D4RT_v2/big_data/wfm_rgb_query_260526_bs1_4f_50k_hyp0.1_dl3dv0.1_waymo_kitti2_stereo4d_bigdata_v1_20260606-132621/wfm_rgb_query_260526_bs1_4f_50k_hyp0.1_dl3dv0.1_waymo_kitti2_stereo4d_bigdata_v1.py}"
CKPT="${CKPT:-/mnt/home/tcchen/workspace/TMA/results_D4RT_v2/big_data/wfm_rgb_query_260526_bs1_4f_50k_hyp0.1_dl3dv0.1_waymo_kitti2_stereo4d_bigdata_v1_20260606-132621/checkpoint/latest/ckpt.pth}"
JSON="${JSON:-/mnt/nasTeam2/AI/datasets/TMD/Tapvid3d_mini/Tapvid3d_mini_extracted/pstudio_mini_test_mf_with_tracking.json}"
OUT_DIR="${OUT_DIR:-tmp/vis_worldtrack_tma_pstudio_worst5}"
mkdir -p "$OUT_DIR"

WORST5=(
  boxes_27
  basketball_13
  boxes_17
  football_22
  boxes_11
)

for seq in "${WORST5[@]}"; do
  echo "========== TMA vis: ${seq} =========="
  "$PYTHON_BIN" hAlgorithm/script/eval/eval_worldtrack_tma.py \
    --config "$CONFIG" \
    --load-from "$CKPT" \
    --json "$JSON" \
    --subset-name pstudio_mini \
    --data-root "$(dirname "$(dirname "$JSON")")" \
    --output-dir "$OUT_DIR" \
    --num-frames 64 \
    --seq-name "$seq" \
    --pred-3d-source warp3d \
    --visualize \
    --vis-frame0-pointmap-ply \
    --vis-max-points 300 \
    --vis-fps 15 \
    --save-per-sequence
done

echo "Done. Outputs under: $REPO_ROOT/$OUT_DIR/pstudio_mini/<seq>/"
echo "  vis/tracks_2d_overlay.mp4"
echo "  vis/tracks_3d.mp4"
echo "  vis/frame0_gt_colored.ply"
echo "  vis/frame0_pred_dense_colored.ply"
echo "  vis/frame0_vis_sparse_gt_dense_pred.ply"
