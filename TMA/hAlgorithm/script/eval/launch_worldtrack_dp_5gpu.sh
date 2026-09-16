#!/usr/bin/env bash
# Dynamic Points (warp3d) on 5 subsets: adt / pstudio / po / ds / synthverse (dynamic-only).
# Override via env: CONFIG, LOADFROM, TAG, OUTPUT_BASE, LOG_DIR, GPUS.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

TAG="${TAG:-dp_5gpu}"
LOG_DIR="${LOG_DIR:-tmp/eval_logs_worldtrack_dp_${TAG}}"
mkdir -p "$LOG_DIR"

CONFIG="${CONFIG:?CONFIG is required}"
LOADFROM="${LOADFROM:?LOADFROM is required}"
OUTPUT_BASE="${OUTPUT_BASE:-tmp/eval_worldtrack_dp_${TAG}}"

GPUS=(${GPUS:-3 4 5})
SUBSETS=(adt pstudio po)

for gpu in "${GPUS[@]}"; do
  python3 hAlgorithm/script/auto_mem/clear_gpu.py "$gpu" 2>/dev/null || true
done

run_subset() {
  local gpu="$1"
  local subset="$2"
  local log="${LOG_DIR}/${subset}.log"
  echo "[$(date -Iseconds)] GPU${gpu} start ${subset} (warp3d DP)" | tee -a "${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  CONFIG="${CONFIG}" \
  LOADFROM="${LOADFROM}" \
  SUBSET="${subset}" \
  LIMIT_SEQS=0 \
  VIS=0 \
  SAVE_PER_SEQUENCE=1 \
  RESUME=0 \
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
        "config": config,
        "load_from": ckpt,
        "num_frames": 64,
        "pred_3d_source": "warp3d",
        "pred_field": "warp3d",
        "tag": tag,
        "metric": "dynamic_points",
    },
    "subsets": {},
}
lines = [f"WorldTrack DP (5 subsets) tag={tag}"]
for subset, name in subset_map.items():
    root = Path(f"{out_base}_{subset}")
    p = root / "summary.json"
    if not p.is_file():
        p = root / name / "summary.json"
    if not p.is_file():
        lines.append(f"{name}: MISSING")
        continue
    data = json.loads(p.read_text(encoding="utf-8"))
    if "subsets" in data and name in data["subsets"]:
        merged["subsets"][name] = data["subsets"][name]
    else:
        merged["subsets"][name] = data
    s = merged["subsets"][name]
    lines.append(
        f"{name}: APD(global)={s.get('avg_pts_global', float('nan')):.4f} "
        f"tau(global)={s.get('tau_global', float('nan')):.4f} "
        f"EPE(global)={s.get('epe_global', float('nan')):.4f} "
        f"queries={s.get('total_queries', 0)}"
    )

out_dir = Path(out_base)
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "summary.json").write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
vals = [v for v in merged["subsets"].values() if v.get("total_queries")]
if vals:
    n = len(vals)
    agg = {
        "avg_pts_global": sum(v.get("avg_pts_global", 0) for v in vals) / n,
        "tau_global": sum(v.get("tau_global", 0) for v in vals) / n,
        "epe_global": sum(v.get("epe_global", 0) for v in vals) / n,
    }
    text = (
        f"DP 5 subsets avg (tag={tag})\n"
        f"APD={agg['avg_pts_global']*100:.2f}% tau={agg['tau_global']*100:.2f}% "
        f"EPE={agg['epe_global']:.4f}m\n"
    )
    for name in merged["subsets"]:
        s = merged["subsets"][name]
        text += (
            f"{name}: APD={s.get('avg_pts_global', 0)*100:.2f}% "
            f"tau={s.get('tau_global', 0)*100:.2f}% "
            f"EPE={s.get('epe_global', 0):.4f}m "
            f"queries={s.get('total_queries', 0)}\n"
        )
    (out_dir / "summary_aggregate.txt").write_text(text, encoding="utf-8")
    print(text, end="")
PY
}

echo "Launching DP 5 subsets -> ${OUTPUT_BASE} (TAG=${TAG})" | tee "${LOG_DIR}/launcher.log"

WAVE1_GPU="${GPUS[0]:-3}"
WAVE1_GPU2="${GPUS[1]:-4}"
WAVE1_GPU3="${GPUS[2]:-5}"
WAVE2_GPU_DS="${GPUS[3]:-${GPUS[2]:-5}}"
WAVE2_GPU_SV="${GPUS[2]:-5}"

for i in "${!SUBSETS[@]}"; do
  run_subset "${GPUS[$i]}" "${SUBSETS[$i]}" &
done
wait

run_subset "${WAVE2_GPU_DS}" ds &
wait
run_subset "${WAVE2_GPU_SV}" synthverse &
wait

merge_summaries | tee -a "${LOG_DIR}/launcher.log"
echo "All five subsets finished. Merged: ${OUTPUT_BASE}/summary_aggregate.txt" | tee -a "${LOG_DIR}/launcher.log"
