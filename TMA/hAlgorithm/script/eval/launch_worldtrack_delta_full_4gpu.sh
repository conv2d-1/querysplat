#!/usr/bin/env bash
# Full WorldTrack eval (4 subsets) with warp3d_delta + visualization on idle GPUs 5-7.
# GPU5: adt -> ds (serial); GPU6: pstudio; GPU7: po (parallel).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="${LOG_DIR:-tmp/eval_worldtrack_delta_full_logs}"
mkdir -p "$LOG_DIR"

run_subset() {
  local gpu="$1"
  local subset="$2"
  local log="${LOG_DIR}/${subset}.log"
  echo "[$(date -Iseconds)] GPU${gpu} start ${subset}" | tee -a "${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  SUBSET="${subset}" \
  LIMIT_SEQS=0 \
  VIS=1 \
  PRED_3D_SOURCE=warp3d_delta \
  OUTPUT_DIR="tmp/eval_worldtrack_delta_full_${subset}" \
  bash hAlgorithm/script/eval/run_eval_worldtrack_tma.sh \
    >>"${log}" 2>&1
  echo "[$(date -Iseconds)] GPU${gpu} done ${subset}" | tee -a "${log}"
}

echo "Launching WorldTrack delta full eval (logs under ${LOG_DIR})"

(
  run_subset 5 adt
  run_subset 5 ds
) &

run_subset 6 pstudio &
run_subset 7 po &

wait
echo "All four subsets finished."
