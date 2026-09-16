#!/usr/bin/env bash
# Scene Flow: warp3d_delta -> pred_flow_ref0 + eval_worldtrack_tma_sceneflow.py
#   Metrics: avg_sf_global (tau), epe_sf_global (flow EPE). NOT Dynamic Points 3D EPE.
#   Protocol: docs/WORLDTRACK_SCENEFLOW_AND_DYNAMIC_POINTS_EVAL.md §2
#
# Smoke:  SUBSET=adt LIMIT_SEQS=1 VIS=1 bash run_eval_worldtrack_tma_sceneflow.sh
# Full:   SUBSET=po LIMIT_SEQS=0 VIS=0 bash run_eval_worldtrack_tma_sceneflow.sh
#
# Dynamic Points (warp3d 3D): use run_eval_worldtrack_tma.sh instead.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-acc}"
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

CONFIG="${CONFIG:-/mnt/home/tcchen/workspace/TMA/results_D4RT_v2/test/wfm/config.py}"
LOADFROM="${LOADFROM:-/mnt/home/tcchen/workspace/TMA/results_D4RT_v2/world_foundation_model/wfm_query_all_260513/wfm_rgb_query_lgct_260514_bs1_4f12f_150k_alldata_20260514-190932/checkpoint/iter_100000/ckpt.pth}"
DATA_ROOT="${DATA_ROOT:-/mnt/nasTeam2/AI/datasets/TMD/Tapvid3d_mini}"
OUTPUT_DIR="${OUTPUT_DIR:-tmp/eval_worldtrack_tma_sceneflow}"
NUM_FRAMES="${NUM_FRAMES:-64}"
LIMIT_SEQS="${LIMIT_SEQS:-1}"
SEQ_NAME="${SEQ_NAME:-}"
VIS="${VIS:-1}"
SUBSET="${SUBSET:-adt}"
COORD_CONVENTION="${COORD_CONVENTION:-opend4rt}"
# Prediction source (kept for CLI compatibility; scene flow eval uses warp3d_delta directly)
PRED_3D_SOURCE="${PRED_3D_SOURCE:-warp3d_delta}"

ADT_JSON="${DATA_ROOT}/Tapvid3d_mini_extracted/adt_mini_test_mf_with_tracking.json"
PSTUDIO_JSON="${DATA_ROOT}/Tapvid3d_mini_extracted/pstudio_mini_test_mf_with_tracking.json"
PO_JSON="${DATA_ROOT}/Tapvid3d_mini_extracted/po_mini_test_mf_with_tracking.json"
DS_JSON="${DATA_ROOT}/Tapvid3d_mini_extracted/ds_mini_test_mf_with_tracking.json"
SYNTHVERSE_JSON="${SYNTHVERSE_JSON:-/mnt/nasTeam2/AI/datasets/TMD/SynthVerse/converted/train_mf_with_tracking_subset50_64f_q430.json}"

ARGS=(--limit-seqs "$LIMIT_SEQS")
if [[ -n "$SEQ_NAME" ]]; then
  ARGS+=(--seq-name "$SEQ_NAME")
fi
if [[ "${SAVE_PER_SEQUENCE:-1}" != "0" ]]; then
  ARGS+=(--save-per-sequence)
fi
if [[ "$VIS" != "0" ]]; then
  ARGS+=(--visualize)
fi
if [[ "${RESUME:-0}" != "0" ]]; then
  ARGS+=(--resume)
fi
if [[ "${USE_WARP3D_DELTA:-0}" != "0" ]]; then
  ARGS+=(--use-warp3d-delta)
else
  ARGS+=(--pred-3d-source "$PRED_3D_SOURCE")
fi

JSON_ARGS=()
case "$SUBSET" in
  adt)
    JSON_ARGS+=(--json "$ADT_JSON" --subset-name adt_mini)
    ;;
  pstudio)
    JSON_ARGS+=(--json "$PSTUDIO_JSON" --subset-name pstudio_mini)
    ;;
  po)
    JSON_ARGS+=(--json "$PO_JSON" --subset-name po_mini)
    ;;
  ds)
    JSON_ARGS+=(--json "$DS_JSON" --subset-name ds_mini)
    ;;
  synthverse)
    DATA_ROOT="${DATA_ROOT:-/mnt/nasTeam2/AI/datasets/TMD}"
    COORD_CONVENTION="${COORD_CONVENTION:-pointodyssey}"
    JSON_ARGS+=(--json "$SYNTHVERSE_JSON" --subset-name synthverse_subset50)
    ;;
  all)
    JSON_ARGS+=(
      --json "$ADT_JSON" --subset-name adt_mini
      --json "$PSTUDIO_JSON" --subset-name pstudio_mini
    )
    ;;
  *)
    echo "Unknown SUBSET=$SUBSET (use adt, pstudio, po, ds, synthverse, or all)" >&2
    exit 1
    ;;
esac

python hAlgorithm/script/eval/eval_worldtrack_tma_sceneflow.py \
  --config "$CONFIG" \
  --load-from "$LOADFROM" \
  --data-root "$DATA_ROOT" \
  --coord-convention "$COORD_CONVENTION" \
  "${JSON_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --num-frames "$NUM_FRAMES" \
  "${ARGS[@]}"

