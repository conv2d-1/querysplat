#!/usr/bin/env bash
# Remaining WorldTrack Dynamic Points: Any4D, SpaTrackerV2, TraceAnything, St4RTrack, VDPM (last).
set -eo pipefail

TAG="${TAG:-20260602_warp3d_dp}"
LOG_ROOT="${LOG_ROOT:-/mnt/home/tcchen/workspace/TMA-origin-dev/tmp/eval_logs_dp_${TAG}}"
mkdir -p "$LOG_ROOT"

declare -A REF_Q=([adt_mini]=22187 [pstudio_mini]=8720 [po_mini]=53465 [ds_mini]=45149)

check_q() {
  local method="$1" subset="$2" summary="$3"
  local sm="${subset}_mini" ref="${REF_Q[$sm]}"
  local got
  got=$(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));s=d.get('subsets',{}).get(sys.argv[2],d);print(int(s.get('total_queries',-1)))" "$summary" "$sm")
  if [[ "$got" != "$ref" ]]; then
    echo "[FAIL] $method $subset queries=$got ref=$ref path=$summary" >&2
    return 1
  fi
  echo "[OK] $method $subset queries=$got"
}

run_all_subsets() {
  local gpu="$1" name="$2" repo="$3" script="$4" extra="${5:-}"
  local log="${LOG_ROOT}/${name}.log"
  {
    echo "[$(date -Iseconds)] GPU${gpu} start ${name}"
    cd "$repo"
    for subset in adt pstudio po ds; do
      echo "[$(date -Iseconds)] ${name} SUBSET=${subset}"
      eval "CUDA_VISIBLE_DEVICES=${gpu} SUBSET=${subset} LIMIT_SEQS=0 SAVE_PER_SEQUENCE=1 RESUME=1 \
        OUTPUT_DIR=tmp/eval_worldtrack_${name}_warp3d_${TAG}_${subset} ${extra} \
        bash ${script}" || exit 1
      sj="tmp/eval_worldtrack_${name}_warp3d_${TAG}_${subset}/${subset}_mini/summary.json"
      check_q "$name" "$subset" "$sj" || exit 1
    done
    echo "[$(date -Iseconds)] GPU${gpu} done ${name}"
  } >>"$log" 2>&1
}

# Phase 1: four methods on GPUs 3/4/6/7
run_all_subsets 3 any4d /mnt/home/tcchen/workspace/Projects/Any4D scripts/run_eval_worldtrack_any4d.sh &
PID1=$!
run_all_subsets 4 traceanything /mnt/home/tcchen/workspace/TraceAnything scripts/run_eval_worldtrack_traceanything.sh &
PID2=$!
run_all_subsets 6 spatrackerv2 /mnt/home/tcchen/workspace/Projects/SpaTrackerV2 scripts/run_eval_worldtrack_spatrackerv2.sh &
PID3=$!
run_all_subsets 7 st4rtrack /mnt/home/tcchen/workspace/Projects/St4RTrack scripts/run_eval_worldtrack_st4rtrack.sh "EVAL_BATCH_SIZE=1" &
PID4=$!

wait "$PID1" "$PID2" "$PID3" "$PID4"
echo "[$(date -Iseconds)] Phase 1 done"

# Phase 2: VDPM last (slow)
run_all_subsets 3 vdpm /mnt/home/tcchen/workspace/Projects/vdpm scripts/run_eval_worldtrack_vdpm.sh
echo "All remaining DP baselines finished. Logs: ${LOG_ROOT}"
