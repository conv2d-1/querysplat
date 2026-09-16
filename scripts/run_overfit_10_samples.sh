#!/usr/bin/env bash
set -euo pipefail

# Run a 4DGS overfitting experiment on a small, isolated set of scenes.
# The source YAML files are never modified; filtered copies are generated per run.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMA_ROOT="${ROOT}/TMA"
ENV_PREFIX="${ENV_PREFIX:-${ROOT}/.envs/tma4dgs}"
PYTHON="${PYTHON:-${ENV_PREFIX}/bin/python}"
CONFIG="${CONFIG:-${ROOT}/4DGS/baseline_0903.py}"
CHECKPOINT="${CHECKPOINT:-/mnt/cfsdata/Team/AI/personal/chentiancheng/workspace/Projects/TMA/results/Q4RT1.12/latest/ckpt.pth}"
TRAIN_YAML="${TRAIN_YAML:-${ROOT}/4DGS/configs/local_train.yaml}"
VAL_YAML="${VAL_YAML:-${ROOT}/4DGS/configs/local_val.yaml}"
DATASET="${DATASET:-hasim_benchmark_running}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
SAMPLE_STRATEGY="${SAMPLE_STRATEGY:-first:${NUM_SAMPLES}}"
MAX_ITER="${MAX_ITER:-50000}"
SEED="${SEED:-2024}"
GPU="${GPU:-0}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/results}"
EXP="${EXP:-overfit_${DATASET}_${NUM_SAMPLES}samples}"
RUN_DIR="${RESULTS_ROOT}/${EXP}"
GENERATED_DIR="${RUN_DIR}/generated_data"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU}}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${TMA_ROOT}:${PYTHONPATH:-}"

die() { echo "ERROR: $*" >&2; exit 1; }
[[ -x "${PYTHON}" ]] || die "Python not found: ${PYTHON}"
[[ -f "${CONFIG}" ]] || die "Config not found: ${CONFIG}"
[[ -f "${TRAIN_YAML}" ]] || die "Train YAML not found: ${TRAIN_YAML}"
[[ -f "${VAL_YAML}" ]] || die "Val YAML not found: ${VAL_YAML}"
[[ -f "${CHECKPOINT}" ]] || die "Checkpoint not found: ${CHECKPOINT}"
[[ "${NUM_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || die "NUM_SAMPLES must be a positive integer"

mkdir -p "${GENERATED_DIR}"

# Keep exactly one named dataset and limit its multi-frame scenes to NUM_SAMPLES.
"${PYTHON}" - "${TRAIN_YAML}" "${VAL_YAML}" "${GENERATED_DIR}" "${DATASET}" "${SAMPLE_STRATEGY}" <<'PY'
import copy
import pathlib
import sys

import yaml

train_path, val_path, out_dir, wanted_name, strategy = sys.argv[1:]
out_dir = pathlib.Path(out_dir)

def make_filtered(src, dst):
    with open(src, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    key = "datasets_train" if "datasets_train" in doc else "datasets"
    datasets = doc.get(key, [])
    kept = []
    for entry in datasets:
        if entry is None:
            continue
        pipeline = entry.get("pipeline", entry)
        if isinstance(pipeline, str):
            raise SystemExit(f"{src}: pipeline for {wanted_name} is a file reference; use an expanded YAML")
        name = entry.get("name") or pipeline.get("name")
        if name == wanted_name:
            item = copy.deepcopy(entry)
            item_pipeline = item.get("pipeline", item)
            item_pipeline["mf_scene_sampling_strategy"] = strategy
            kept.append(item)
    if len(kept) != 1:
        available = sorted({(e.get("name") or e.get("pipeline", {}).get("name")) for e in datasets if isinstance(e, dict)})
        raise SystemExit(f"Dataset {wanted_name!r} not found exactly once in {src}; available={available}")
    doc[key] = kept
    with open(dst, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)

make_filtered(train_path, out_dir / "train.yaml")
make_filtered(val_path, out_dir / "val.yaml")
make_filtered(val_path, out_dir / "vis.yaml")
PY

echo "Running 4DGS overfit experiment"
echo "  dataset=${DATASET} samples=${NUM_SAMPLES} strategy=${SAMPLE_STRATEGY}"
echo "  max_iter=${MAX_ITER} gpu=${CUDA_VISIBLE_DEVICES} output=${RUN_DIR}"

cd "${TMA_ROOT}"
"${PYTHON}" hAlgorithm/script/train/train.py \
  --config "${CONFIG}" \
  --output_dir_full "${RUN_DIR}" \
  --exp "${EXP}" \
  --seed "${SEED}" \
  --resume None \
  --load_from "${CHECKPOINT}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --data.train="${GENERATED_DIR}/train.yaml" \
  --data.val="${GENERATED_DIR}/val.yaml" \
  --data.vis="${GENERATED_DIR}/vis.yaml" \
  --trainer.select_dataset="${DATASET}" \
  --trainer.select_val_dataset="${DATASET}" \
  --trainer.select_vis_dataset="${DATASET}" \
  --trainer.max_iter="${MAX_ITER}" \
  --trainer.num_workers="${NUM_WORKERS:-8}" \
  --trainer.batch_size="${BATCH_SIZE:-1}"
