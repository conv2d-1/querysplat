#!/usr/bin/env bash
# TMA Dynamic Points (warp3d) — four subsets on GPUs 4–7.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

TAG="${TAG:-20260602_warp3d_dp}"
LOG_DIR="${LOG_DIR:-tmp/eval_logs_${TAG}}"
mkdir -p "$LOG_DIR"

GPUS=(4 5 6 7)
SUBSETS=(adt pstudio po ds)

for i in "${!SUBSETS[@]}"; do
  python3 hAlgorithm/script/auto_mem/clear_gpu.py "${GPUS[$i]}" 2>/dev/null || true
done

run_subset() {
  local gpu="$1"
  local subset="$2"
  local log="${LOG_DIR}/${subset}.log"
  echo "[$(date -Iseconds)] GPU${gpu} TMA warp3d ${subset}" | tee -a "${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  SUBSET="${subset}" \
  LIMIT_SEQS=0 \
  VIS=0 \
  SAVE_PER_SEQUENCE=1 \
  RESUME=0 \
  PRED_3D_SOURCE=warp3d \
  OUTPUT_DIR="tmp/eval_worldtrack_tma_warp3d_${TAG}_${subset}" \
  bash hAlgorithm/script/eval/run_eval_worldtrack_tma.sh >>"${log}" 2>&1
  echo "[$(date -Iseconds)] GPU${gpu} done ${subset}" | tee -a "${log}"
}

for i in "${!SUBSETS[@]}"; do
  run_subset "${GPUS[$i]}" "${SUBSETS[$i]}" &
done
wait
echo "TMA warp3d all subsets done. Logs: ${LOG_DIR}"
