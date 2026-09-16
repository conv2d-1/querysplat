#!/usr/bin/env bash
# Full WorldTrack eval (4 subsets) with warp3d (absolute) on GPUs 4-7 (one subset per GPU).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="${LOG_DIR:-tmp/eval_worldtrack_warp3d_full_logs}"
mkdir -p "$LOG_DIR"

GPUS=(4 5 6 7)
SUBSETS=(adt pstudio po ds)

for i in "${!SUBSETS[@]}"; do
  gpu="${GPUS[$i]}"
  python3 hAlgorithm/script/auto_mem/clear_gpu.py "$gpu" 2>/dev/null || true
done

run_subset() {
  local gpu="$1"
  local subset="$2"
  local log="${LOG_DIR}/${subset}.log"
  echo "[$(date -Iseconds)] GPU${gpu} start ${subset} (warp3d)" | tee -a "${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  SUBSET="${subset}" \
  LIMIT_SEQS=0 \
  VIS=0 \
  PRED_3D_SOURCE=warp3d \
  OUTPUT_DIR="tmp/eval_worldtrack_warp3d_full_${subset}" \
  bash hAlgorithm/script/eval/run_eval_worldtrack_tma.sh \
    >>"${log}" 2>&1
  echo "[$(date -Iseconds)] GPU${gpu} done ${subset}" | tee -a "${log}"
}

echo "Launching WorldTrack warp3d full eval (logs under ${LOG_DIR})" | tee "${LOG_DIR}/launcher.log"

for i in "${!SUBSETS[@]}"; do
  run_subset "${GPUS[$i]}" "${SUBSETS[$i]}" &
done

wait
echo "All four subsets finished." | tee -a "${LOG_DIR}/launcher.log"
