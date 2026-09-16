#!/usr/bin/env bash
# Launch remaining WorldTrack Dynamic Points baselines on 4 free GPUs.
# TMA + 4RC already done. VDPM runs last (slow). Query check after each subset.
set -eo pipefail

TAG="${TAG:-20260602_warp3d_dp}"
LOG_ROOT="${LOG_ROOT:-/mnt/home/tcchen/workspace/TMA-origin-dev/tmp/eval_logs_dp_${TAG}}"
mkdir -p "$LOG_ROOT"

# Reference total_queries (TMA warp3d_dp / sceneflow aligned)
declare -A REF_Q=(
  [adt_mini]=22187
  [pstudio_mini]=8720
  [po_mini]=53465
  [ds_mini]=45149
)

check_queries() {
  local method="$1" subset_key="$2" summary_json="$3"
  local sm="${subset_key}_mini"
  local ref="${REF_Q[$sm]:-}"
  if [[ -z "$ref" ]]; then
    echo "[check] unknown subset $sm"
    return 1
  fi
  if [[ ! -f "$summary_json" ]]; then
    echo "[check] MISSING $method $subset_key: $summary_json"
    return 1
  fi
  local got
  got=$(python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
s=d.get('subsets',{}).get(sys.argv[2], d)
q=int(s.get('total_queries',-1))
print(q)
" "$summary_json" "$sm")
  if [[ "$got" != "$ref" ]]; then
    echo "[check] FAIL $method $subset_key: queries=$got ref=$ref"
    return 1
  fi
  echo "[check] OK $method $subset_key queries=$got"
}

run_method_subsets() {
  local gpu="$1" method="$2" repo="$3" run_fn="$4"
  local log="${LOG_ROOT}/${method}.log"
  {
    echo "[$(date -Iseconds)] GPU${gpu} start ${method}"
    for subset in adt pstudio po ds; do
      echo "[$(date -Iseconds)] ${method} SUBSET=${subset}"
      (cd "${repo}" && CUDA_VISIBLE_DEVICES="${gpu}" SUBSET="${subset}" LIMIT_SEQS=0 \
        SAVE_PER_SEQUENCE=1 RESUME=1 \
        OUTPUT_DIR="tmp/eval_worldtrack_${method}_warp3d_${TAG}_${subset}" \
        bash ${run_fn}) || {
          echo "[$(date -Iseconds)] ERROR ${method} ${subset}"
          exit 1
        }
      local out="${repo}/tmp/eval_worldtrack_${method}_warp3d_${TAG}_${subset}/${subset}_mini/summary.json"
      if [[ ! -f "$out" ]]; then
        out="${repo}/tmp/eval_worldtrack_${method}_warp3d_${TAG}_${subset}/summary.json"
      fi
      check_queries "${method}" "${subset}" "$out" || exit 1
    done
    echo "[$(date -Iseconds)] GPU${gpu} done ${method}"
  } >>"$log" 2>&1
}

# Phase 1: four baselines in parallel (GPUs 3,6,7,4)
run_method_subsets 3 any4d /mnt/home/tcchen/workspace/Projects/Any4D \
  scripts/run_eval_worldtrack_any4d.sh &
PID_ANY4D=$!

run_method_subsets 6 spatrackerv2 /mnt/home/tcchen/workspace/Projects/SpaTrackerV2 \
  scripts/run_eval_worldtrack_spatrackerv2.sh &
PID_SPA=$!

run_method_subsets 7 st4rtrack /mnt/home/tcchen/workspace/Projects/St4RTrack \
  scripts/run_eval_worldtrack_st4rtrack.sh &
PID_ST4R=$!

run_method_subsets 4 traceanything /mnt/home/tcchen/workspace/TraceAnything \
  scripts/run_eval_worldtrack_traceanything.sh &
PID_TA=$!

wait "$PID_ANY4D" "$PID_SPA" "$PID_ST4R" "$PID_TA"
echo "[$(date -Iseconds)] Phase 1 baselines finished"

# Phase 2: VDPM last on GPU 3
run_method_subsets 3 vdpm /mnt/home/tcchen/workspace/Projects/vdpm \
  scripts/run_eval_worldtrack_vdpm.sh

echo "All Dynamic Points baselines done. Logs: ${LOG_ROOT}"
