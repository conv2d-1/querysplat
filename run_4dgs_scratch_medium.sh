#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMA_ROOT="${ROOT}/TMA"
D4GS_ROOT="${ROOT}/4DGS"
ENV_PREFIX="${ENV_PREFIX:-${ROOT}/.envs/tma4dgs}"
PYTHON="${PYTHON:-${ENV_PREFIX}/bin/python}"
CONFIG="${CONFIG:-${D4GS_ROOT}/scratch_4dgs.py}"
TRAIN_YAML="${TRAIN_YAML:-${D4GS_ROOT}/configs/medium_train.yaml}"
VAL_YAML="${VAL_YAML:-${D4GS_ROOT}/configs/medium_val.yaml}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/results}"

MAX_ITER="${MAX_ITER:-3600}"
NUM_WORKERS="${NUM_WORKERS:-1}"
LOGGING_STEP="${LOGGING_STEP:-1}"
SAVE_PERIOD="${SAVE_PERIOD:-600}"
VAL_PERIOD="${VAL_PERIOD:-1200}"
SEED="${SEED:-2024}"
RUN_NAME="${RUN_NAME:-}"
RUN_DIR="${RUN_DIR:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CC="${CC:-/usr/bin/gcc-11}"
export CXX="${CXX:-/usr/bin/g++-11}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-/usr/bin/g++-11}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export GSPLAT_CACHE_DIR="${GSPLAT_CACHE_DIR:-${ROOT}/.cache/gsplat}"
export TORCH_HOME="${TORCH_HOME:-${ROOT}/.cache/torch}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

usage() {
  cat <<'EOF'
Usage:
  ./run_4dgs_scratch_medium.sh check
  ./run_4dgs_scratch_medium.sh smoke
  ./run_4dgs_scratch_medium.sh train

The default recipe trains the complete 1.65B-parameter model from native
PyTorch/module initialization. It does not load a model/trainer checkpoint,
does not load a backbone pretrain, and disables LPIPS because LPIPS would load
a fixed VGG .pth for the perceptual loss.

Useful overrides:
  RUN_DIR, RUN_NAME, MAX_ITER, NUM_WORKERS, SAVE_PERIOD, VAL_PERIOD, SEED
  CUDA_VISIBLE_DEVICES, CONFIG, TRAIN_YAML, VAL_YAML, RESULTS_ROOT
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "file not found: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "directory not found: $1"
}

validate_integer() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be a non-negative integer: ${value}"
}

preflight() {
  require_file "${PYTHON}"
  require_file "${CONFIG}"
  require_file "${TRAIN_YAML}"
  require_file "${VAL_YAML}"
  require_dir "${TMA_ROOT}"
  validate_integer MAX_ITER "${MAX_ITER}"
  validate_integer NUM_WORKERS "${NUM_WORKERS}"
  validate_integer LOGGING_STEP "${LOGGING_STEP}"
  validate_integer SAVE_PERIOD "${SAVE_PERIOD}"
  validate_integer VAL_PERIOD "${VAL_PERIOD}"
  (( MAX_ITER > 0 )) || die "MAX_ITER must be greater than zero"
  mkdir -p "${RESULTS_ROOT}" "${GSPLAT_CACHE_DIR}" "${TORCH_HOME}"
}

check_scratch_contract() {
  "${PYTHON}" - "${CONFIG}" "${TRAIN_YAML}" "${VAL_YAML}" "${MAX_ITER}" <<'PY'
import os
import sys

import yaml

sys.path.insert(0, os.getcwd())
from hAlgorithm.utils import file2dict

config_path, train_yaml, val_yaml, max_iter = sys.argv[1:]
cfg = file2dict(config_path)
model_cfg = cfg["model"]["model"]
trainer_cfg = cfg["trainer"]
fuse_cfg = model_cfg["fuse_encoder"]

assert trainer_cfg.get("resume") is None, trainer_cfg.get("resume")
assert trainer_cfg.get("load_from") is None, trainer_cfg.get("load_from")
assert fuse_cfg.get("pretrain") is None, fuse_cfg.get("pretrain")
assert fuse_cfg.get("pretrained_pretrain") is None, fuse_cfg.get("pretrained_pretrain")
assert model_cfg.get("freeze_modules") in (None, []), model_cfg.get("freeze_modules")
assert cfg["model"]["sparse_dynamic_gaussian_render_loss"]["lpips_weight"] == 0.0

expected_train = {"pointodyssey", "kubric4d", "hasim", "hasim_character_medium"}
with open(train_yaml, "r", encoding="utf-8") as handle:
    train_doc = yaml.safe_load(handle)
train_names = {item["pipeline"]["name"] for item in train_doc["datasets"]}
assert train_names == expected_train, train_names

with open(val_yaml, "r", encoding="utf-8") as handle:
    val_doc = yaml.safe_load(handle)
val_names = {item["pipeline"]["name"] for item in val_doc["datasets"]}
assert val_names == {"hasim_benchmark_running"}, val_names

for item in train_doc["datasets"] + val_doc["datasets"]:
    paths = item["pipeline"].get("data_path")
    paths = paths if isinstance(paths, list) else [paths]
    for path in paths:
        assert os.path.isfile(path), path

print("scratch_contract=ok")
print("checkpoint_load=disabled")
print("backbone_pretrain=disabled")
print("lpips_pretrained_vgg=disabled")
print("backbone_trainable=yes")
print("train_datasets=" + ",".join(sorted(train_names)))
print("val_datasets=" + ",".join(sorted(val_names)))
print("max_iter=" + max_iter)
PY
}

run_train() {
  [[ ! -e "${RUN_DIR}/checkpoint/latest/ckpt.pth" ]] || \
    die "RUN_DIR already contains a checkpoint; choose a new RUN_DIR: ${RUN_DIR}"
  mkdir -p "${RUN_DIR}" "${RESULTS_ROOT}/scratch_medium"
  printf '%s\n' "${RUN_DIR}" > "${RESULTS_ROOT}/scratch_medium/LATEST_RUN"

  echo "run_dir=${RUN_DIR}"
  echo "max_iter=${MAX_ITER} save_period=${SAVE_PERIOD} val_period=${VAL_PERIOD}"
  echo "starting from native initialization; no checkpoint/pretrain/LPIPS weights"

  cd "${TMA_ROOT}"
  exec "${PYTHON}" hAlgorithm/script/train/train.py \
    --config "${CONFIG}" \
    --output_dir_full "${RUN_DIR}" \
    --seed "${SEED}" \
    --resume None \
    --load_from None \
    --mixed_precision bf16 \
    --data.train="${TRAIN_YAML}" \
    --data.val="${VAL_YAML}" \
    --data.vis="${VAL_YAML}" \
    --model.model.freeze_modules=None \
    --model.model.fuse_encoder.pretrain=None \
    --model.model.fuse_encoder.pretrained_pretrain=None \
    --model.sparse_dynamic_gaussian_render_loss.lpips_weight=0.0 \
    --trainer.resume=None \
    --trainer.load_from=None \
    --trainer.max_iter="${MAX_ITER}" \
    --trainer.num_workers="${NUM_WORKERS}" \
    --trainer.select_dataset=pointodyssey,kubric4d,hasim,hasim_character_medium \
    --trainer.select_val_dataset=hasim_benchmark_running \
    --trainer.select_vis_dataset=hasim_benchmark_running \
    --trainer.in_evaluation=False \
    --trainer.in_visualize=False \
    --trainer.logging_step="${LOGGING_STEP}" \
    --trainer.backup_period=0 \
    --trainer.save_period="${SAVE_PERIOD}" \
    --trainer.val_period="${VAL_PERIOD}" \
    --trainer.vis_period=0 \
    --trainer.test_num_workers=0
}

MODE="${1:-}"
case "${MODE}" in
  check)
    preflight
    cd "${TMA_ROOT}"
    check_scratch_contract
    "${PYTHON}" - <<'PY'
import torch
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()} gpus={torch.cuda.device_count()}")
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info(0)
    print(f"gpu={torch.cuda.get_device_name(0)} free_gib={free / 1024**3:.1f} total_gib={total / 1024**3:.1f}")
PY
    ;;
  smoke)
    MAX_ITER="${SMOKE_STEPS:-2}"
    SAVE_PERIOD=0
    VAL_PERIOD=0
    RUN_NAME="${RUN_NAME:-scratch_smoke_$(date +%Y%m%d-%H%M%S)}"
    RUN_DIR="${RUN_DIR:-${RESULTS_ROOT}/scratch_smoke/${RUN_NAME}}"
    preflight
    cd "${TMA_ROOT}"
    check_scratch_contract
    run_train
    ;;
  train)
    RUN_NAME="${RUN_NAME:-scratch_medium_$(date +%Y%m%d-%H%M%S)}"
    RUN_DIR="${RUN_DIR:-${RESULTS_ROOT}/scratch_medium/${RUN_NAME}}"
    preflight
    cd "${TMA_ROOT}"
    check_scratch_contract
    run_train
    ;;
  *)
    usage
    exit 2
    ;;
esac
