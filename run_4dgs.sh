#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMA_ROOT="${ROOT}/TMA"
D4GS_ROOT="${ROOT}/4DGS"
ENV_PREFIX="${ENV_PREFIX:-${ROOT}/.envs/tma4dgs}"
PYTHON="${PYTHON:-${ENV_PREFIX}/bin/python}"
CONFIG="${CONFIG:-${D4GS_ROOT}/local_4dgs.py}"
SMOKE_CONFIG="${SMOKE_CONFIG:-${D4GS_ROOT}/smoke_4dgs.py}"
CHECKPOINT="${CHECKPOINT:-${D4GS_ROOT}/checkpoint/latest/ckpt.pth}"
TEST_CHECKPOINT="${TEST_CHECKPOINT:-${D4GS_ROOT}/checkpoint/best/ckpt.pth}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/results}"
SMOKE_TRAIN_YAML="${SMOKE_TRAIN_YAML:-${D4GS_ROOT}/configs/smoke_train.yaml}"
SMOKE_VAL_YAML="${SMOKE_VAL_YAML:-${D4GS_ROOT}/configs/smoke_val.yaml}"
TRAIN_YAML="${TRAIN_YAML:-${D4GS_ROOT}/configs/local_train.yaml}"
VAL_YAML="${VAL_YAML:-${D4GS_ROOT}/configs/local_val.yaml}"
DEFAULT_INPUT="${DEFAULT_INPUT:-/mnt/cfsdata/Team/AI/datasets/TMD/DAVIS/JPEGImages/480p/bear}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CC="${CC:-/usr/bin/gcc-11}"
export CXX="${CXX:-/usr/bin/g++-11}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-/usr/bin/g++-11}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export GSPLAT_CACHE_DIR="${GSPLAT_CACHE_DIR:-${ROOT}/.cache/gsplat}"
export TORCH_HOME="${TORCH_HOME:-${ROOT}/.cache/torch}"

usage() {
  cat <<'EOF'
Usage:
  ./run_4dgs.sh check
  ./run_4dgs.sh infer [video-or-image-directory]
  ./run_4dgs.sh render [video-or-image-directory]
  ./run_4dgs.sh test
  ./run_4dgs.sh train-smoke
  ./run_4dgs.sh train
  ./run_4dgs.sh resume

Environment overrides:
  PYTHON, CONFIG, CHECKPOINT, TEST_CHECKPOINT, RESULTS_ROOT, CUDA_VISIBLE_DEVICES
  MAX_FRAMES (default 4), PROCESS_RES (default 280), QUERY_STRIDE (default 4)
  TRAIN_STEPS (default 2), MAX_ITER (train default 50000; resume default 50010)
  TEST_CHECKPOINT defaults to 4DGS/checkpoint/best/ckpt.pth
EOF
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file not found: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Required directory not found: $1" >&2
    exit 1
  fi
}

require_file "${PYTHON}"
require_file "${CONFIG}"
require_file "${CHECKPOINT}"
require_dir "${TMA_ROOT}"
mkdir -p "${RESULTS_ROOT}" "${GSPLAT_CACHE_DIR}"

MODE="${1:-}"
if [[ -z "${MODE}" ]]; then
  usage
  exit 2
fi
shift || true

cd "${TMA_ROOT}"

case "${MODE}" in
  check)
    "${PYTHON}" - "${CONFIG}" "${CHECKPOINT}" <<'PY'
import os
import sys

import accelerate
import gsplat
import torch
import yaml

from hAlgorithm.utils import file2dict

config_path, checkpoint_path = sys.argv[1:3]
cfg = file2dict(config_path)
assert cfg["model"]["model"]["sparse_gaussian_head"] is not None
assert cfg["model"]["model"]["fuse_encoder"]["pretrain"] is None

state = torch.load(
    checkpoint_path,
    map_location="meta",
    mmap=True,
    weights_only=True,
)
gs_keys = [key for key in state if "sparse_gaussian_head" in key]
assert len(state) == 906, len(state)
assert len(gs_keys) == 10, gs_keys

print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"accelerate={accelerate.__version__}")
print(f"gsplat={gsplat.__version__}")
print(f"yaml={yaml.__version__}")
print(f"config={config_path}")
print(f"checkpoint={checkpoint_path}")
print(f"checkpoint_keys={len(state)} gaussian_head_keys={len(gs_keys)}")
PY
    ;;

  infer|render)
    INPUT="${1:-${DEFAULT_INPUT}}"
    if [[ ! -e "${INPUT}" ]]; then
      echo "Input not found: ${INPUT}" >&2
      exit 1
    fi

    args=(
      --motion_config "${CONFIG}"
      --load_from "${CHECKPOINT}"
      --output_dir "${RESULTS_ROOT}/${MODE}"
      --max_frames "${MAX_FRAMES:-4}"
      --process_res "${PROCESS_RES:-280}"
      --frame_sampling debug_trajectory
      --use_amp
      --amp_dtype float16
    )

    if [[ -d "${INPUT}" ]]; then
      args+=(--sequence_dir "${INPUT}")
    else
      args+=(--video "${INPUT}")
    fi

    if [[ "${MODE}" == "render" ]]; then
      args+=(--per_pixel --export_web_viewer)
    else
      args+=(--query_stride "${QUERY_STRIDE:-4}" --no_export_web_viewer)
    fi

    "${PYTHON}" \
      hAlgorithm/script/infer/motion_head/wfm_video_gaussian_vis_infer.py \
      "${args[@]}"
    ;;

  test)
    # Use local_4dgs.py (max_size=504, view_num=4), same as the prior local_4dgs_* smoke_test runs.
    # smoke_4dgs.py is only for train-smoke (max_size=224, view_num=2).
    require_file "${CONFIG}"
    require_file "${SMOKE_VAL_YAML}"
    require_file "${TEST_CHECKPOINT}"
    "${PYTHON}" hAlgorithm/script/train/train.py \
      --config "${CONFIG}" \
      --output_dir "${RESULTS_ROOT}" \
      --exp smoke_test \
      --seed 2024 \
      --test \
      --test_vis \
      --test_data "${SMOKE_VAL_YAML}" \
      --load_from "${TEST_CHECKPOINT}" \
      --mixed_precision fp16 \
      --trainer.select_val_dataset=hasim_benchmark_running \
      --trainer.num_workers=0
    ;;

  train-smoke)
    require_file "${SMOKE_CONFIG}"
    require_file "${SMOKE_TRAIN_YAML}"
    require_file "${SMOKE_VAL_YAML}"
    "${PYTHON}" hAlgorithm/script/train/train.py \
      --config "${SMOKE_CONFIG}" \
      --output_dir "${RESULTS_ROOT}" \
      --exp smoke_train \
      --seed 2024 \
      --resume None \
      --load_from "${CHECKPOINT}" \
      --mixed_precision bf16 \
      --data.train="${SMOKE_TRAIN_YAML}" \
      --data.val="${SMOKE_VAL_YAML}" \
      --data.vis="${SMOKE_VAL_YAML}" \
      --trainer.max_iter="${TRAIN_STEPS:-2}" \
      --trainer.select_dataset=hasim_benchmark_running \
      --trainer.select_val_dataset=hasim_benchmark_running \
      --trainer.select_vis_dataset=hasim_benchmark_running \
      --trainer.num_workers=0 \
      --trainer.batch_size=1 \
      --trainer.eval_metrics=None \
      --trainer.backup_period=0 \
      --trainer.save_period=0 \
      --trainer.val_period=0 \
      --trainer.vis_period=0
    ;;

  train)
    require_file "${TRAIN_YAML}"
    require_file "${VAL_YAML}"
    "${PYTHON}" hAlgorithm/script/train/train.py \
      --config "${CONFIG}" \
      --output_dir "${RESULTS_ROOT}" \
      --exp dual_4dgs_finetune \
      --seed 2024 \
      --resume None \
      --load_from "${CHECKPOINT}" \
      --mixed_precision bf16 \
      --data.train="${TRAIN_YAML}" \
      --data.val="${VAL_YAML}" \
      --data.vis="${VAL_YAML}" \
      --trainer.max_iter="${MAX_ITER:-50000}"
    ;;

  resume)
    require_file "${TRAIN_YAML}"
    require_file "${VAL_YAML}"
    require_file "$(dirname "${CHECKPOINT}")/trainer.ckpt"
    "${PYTHON}" hAlgorithm/script/train/train.py \
      --config "${CONFIG}" \
      --output_dir "${RESULTS_ROOT}" \
      --exp dual_4dgs_resume \
      --seed 2024 \
      --resume "${CHECKPOINT}" \
      --load_from None \
      --mixed_precision bf16 \
      --data.train="${TRAIN_YAML}" \
      --data.val="${VAL_YAML}" \
      --data.vis="${VAL_YAML}" \
      --trainer.max_iter="${MAX_ITER:-50010}"
    ;;

  *)
    usage
    exit 2
    ;;
esac
