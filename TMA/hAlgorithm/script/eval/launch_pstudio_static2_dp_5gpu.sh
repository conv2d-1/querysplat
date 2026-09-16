#!/usr/bin/env bash
# pstudio_static2 latest — Dynamic Points on 5 subsets, one GPU per subset.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

TAG="${TAG:-pstudio_static2_dp_20260617}"
LOG_DIR="${LOG_DIR:-tmp/eval_logs_${TAG}}"
mkdir -p "$LOG_DIR"

EXP_ROOT="${EXP_ROOT:-/mnt/home/tcchen/workspace/TMA-origin-dev/results/baseline_v4/wfm_rgb_query_260526_bs1_4f_50k_hyp0.1_dl3dv0.1_waymo_kitti2_stereo4d_bigdata_v1_finetune_pstudio_static2_20260616-193900}"
CONFIG="${CONFIG:-${EXP_ROOT}/wfm_rgb_query_260526_bs1_4f_50k_hyp0.1_dl3dv0.1_waymo_kitti2_stereo4d_bigdata_v1_finetune_pstudio_static2.py}"
LOADFROM="${LOADFROM:-${EXP_ROOT}/checkpoint/latest/ckpt.pth}"
OUTPUT_BASE="${OUTPUT_BASE:-tmp/eval_worldtrack_tma_${TAG}}"

GPUS=(${GPUS:-1 2 3 5 7})
SUBSETS=(adt pstudio po ds synthverse)

for gpu in "${GPUS[@]}"; do
  python3 hAlgorithm/script/auto_mem/clear_gpu.py "$gpu" 2>/dev/null || true
done

run_subset() {
  local gpu="$1"
  local subset="$2"
  local log="${LOG_DIR}/${subset}.log"
  echo "[$(date -Iseconds)] GPU${gpu} start ${subset}" | tee -a "${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  CONFIG="${CONFIG}" \
  LOADFROM="${LOADFROM}" \
  SUBSET="${subset}" \
  LIMIT_SEQS=0 VIS=0 SAVE_PER_SEQUENCE=1 RESUME=0 \
  PRED_3D_SOURCE=warp3d \
  OUTPUT_DIR="${OUTPUT_BASE}_${subset}" \
  bash hAlgorithm/script/eval/run_eval_worldtrack_tma.sh >>"${log}" 2>&1
  echo "[$(date -Iseconds)] GPU${gpu} done ${subset}" | tee -a "${log}"
}

merge_summaries() {
  python3 - <<'PY' "${OUTPUT_BASE}" "${TAG}" "${CONFIG}" "${LOADFROM}"
import json, sys
from pathlib import Path

out_base, tag, config, ckpt = sys.argv[1:5]
subset_map = {
    "adt": "adt_mini",
    "pstudio": "pstudio_mini",
    "po": "po_mini",
    "ds": "ds_mini",
    "synthverse": "synthverse_subset50_dynamic",
}
merged = {
    "inputs": {
        "model": "pstudio_static2",
        "config": config,
        "load_from": ckpt,
        "metric": "dynamic_points",
        "pred_3d_source": "warp3d",
        "tag": tag,
    },
    "subsets": {},
}
lines = [f"pstudio_static2 DP (5 subsets) tag={tag}", ""]
lines.append(f"{'Subset':<32} {'APD':>8} {'tau@0.1m':>10} {'EPE (m)':>12} {'Queries':>8}")
lines.append("-" * 74)
for subset, name in subset_map.items():
    root = Path(f"{out_base}_{subset}")
    p = root / "summary.json"
    if not p.is_file():
        p = root / name / "summary.json"
    if not p.is_file():
        lines.append(f"{name:<32} {'MISSING':>8}")
        continue
    data = json.loads(p.read_text(encoding="utf-8"))
    s = data.get("subsets", {}).get(name, data)
    merged["subsets"][name] = s
    apd = s.get("avg_pts_global", 0) * 100
    tau = s.get("tau_global", 0) * 100
    epe = s.get("epe_global", 0)
    q = s.get("total_queries", 0)
    lines.append(f"{name:<32} {apd:>7.2f}% {tau:>9.2f}% {epe:>12.4f} {q:>8}")

vals = [v for v in merged["subsets"].values() if v.get("total_queries")]
if vals:
    n = len(vals)
    agg_apd = sum(v.get("avg_pts_global", 0) for v in vals) / n * 100
    agg_tau = sum(v.get("tau_global", 0) for v in vals) / n * 100
    agg_epe = sum(v.get("epe_global", 0) for v in vals) / n
    lines.append("-" * 74)
    lines.append(f"{'5-subset average':<32} {agg_apd:>7.2f}% {agg_tau:>9.2f}% {agg_epe:>12.4f}")

out_dir = Path(out_base)
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "summary.json").write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
text = "\n".join(lines) + "\n"
(out_dir / "summary.txt").write_text(text, encoding="utf-8")
(out_dir / "summary_aggregate.txt").write_text(text, encoding="utf-8")
print(text, end="")
PY
}

echo "Launching pstudio_static2 DP on GPUs ${GPUS[*]} -> ${OUTPUT_BASE}" | tee "${LOG_DIR}/launcher.log"

for i in "${!SUBSETS[@]}"; do
  run_subset "${GPUS[$i]}" "${SUBSETS[$i]}" &
done
wait

merge_summaries | tee -a "${LOG_DIR}/launcher.log"
echo "Done: ${OUTPUT_BASE}/summary_aggregate.txt" | tee -a "${LOG_DIR}/launcher.log"
