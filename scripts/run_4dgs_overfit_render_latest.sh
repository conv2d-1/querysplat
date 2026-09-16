#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMA_ROOT="${ROOT}/TMA"
ENV_PREFIX="${ENV_PREFIX:-${ROOT}/.envs/tma4dgs}"
PYTHON="${PYTHON:-${ENV_PREFIX}/bin/python}"
ACCELERATE="${ACCELERATE:-${ENV_PREFIX}/bin/accelerate}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${TMA_ROOT}/hAlgorithm/script/accelerate_config.yaml}"
TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-${ROOT}/results/overfit_hasim_benchmark_running_10samples}"
CHECKPOINT="${CHECKPOINT:-${TRAIN_RUN_DIR}/checkpoint/latest/ckpt.pth}"
CONFIG="${CONFIG:-${ROOT}/4DGS/baseline_0903.py}"
TEST_YAML="${TEST_YAML:-${TRAIN_RUN_DIR}/configs/val.yaml}"
RENDER_OUTPUT_DIR="${RENDER_OUTPUT_DIR:-${TRAIN_RUN_DIR}/render_latest}"
DATASET="${DATASET:-hasim_benchmark_running}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
SEED="${SEED:-2024}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${TMA_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TORCH_HOME="${TORCH_HOME:-${ROOT}/.cache/torch}"
export GSPLAT_CACHE_DIR="${GSPLAT_CACHE_DIR:-${ROOT}/.cache/gsplat}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"

die() { echo "ERROR: $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || die "文件不存在: $1"; }
positive_int() { [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 必须是正整数: $2"; }

usage() {
  cat <<'EOF'
用法:
  ./scripts/run_4dgs_overfit_render_latest.sh [check|render]

环境变量:
  TRAIN_RUN_DIR=/path/to/overfit_run
  CHECKPOINT=/path/to/checkpoint/latest/ckpt.pth
  TEST_YAML=/path/to/overfit_run/configs/val.yaml
  RENDER_OUTPUT_DIR=/path/to/render_latest
  CUDA_VISIBLE_DEVICES=0 NUM_PROCESSES=1 MIXED_PRECISION=bf16|fp16

云平台任务以前台方式运行，不要在命令末尾添加 &。
EOF
}

MODE="${1:-render}"
case "${MODE}" in
  check|render) shift || true ;;
  -h|--help|help) usage; exit 0 ;;
  *) die "模式必须是 check 或 render: ${MODE}" ;;
esac
(( $# == 0 )) || die "不支持位置参数，请使用环境变量配置"

positive_int NUM_PROCESSES "${NUM_PROCESSES}"
positive_int NUM_WORKERS "${NUM_WORKERS}"
require_file "${PYTHON}"
require_file "${CHECKPOINT}"
require_file "${CONFIG}"
require_file "${TEST_YAML}"
[[ -d "${TMA_ROOT}" ]] || die "TMA 目录不存在: ${TMA_ROOT}"
if (( NUM_PROCESSES > 1 )); then
  require_file "${ACCELERATE_CONFIG}"
  [[ -x "${ACCELERATE}" ]] || die "accelerate 不可用: ${ACCELERATE}"
fi

mkdir -p "${RENDER_OUTPUT_DIR}"

"${PYTHON}" - <<'PY'
import importlib
import sys
import numpy
import torch
print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"numpy={numpy.__version__}")
if tuple(map(int, numpy.__version__.split(".")[:2])) >= (2, 0):
    raise SystemExit("NumPy >=2 与 imgaug==0.4.0 不兼容")
importlib.import_module("imgaug")
print(f"cuda_available={torch.cuda.is_available()} devices={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA 不可用")
PY

if [[ "${MODE}" == "check" ]]; then
  echo "checkpoint=${CHECKPOINT}"
  echo "config=${CONFIG}"
  echo "test_yaml=${TEST_YAML}"
  echo "output_dir=${RENDER_OUTPUT_DIR}"
  echo "environment=ok"
  exit 0
fi

cat > "${RENDER_OUTPUT_DIR}/render_parameters.txt" <<EOF
checkpoint=${CHECKPOINT}
config=${CONFIG}
test_yaml=${TEST_YAML}
dataset=${DATASET}
num_processes=${NUM_PROCESSES}
num_workers=${NUM_WORKERS}
mixed_precision=${MIXED_PRECISION}
seed=${SEED}
output_dir=${RENDER_OUTPUT_DIR}
EOF

echo "开始 latest checkpoint 渲染测试（前台运行，云平台不要追加 &）"
echo "checkpoint=${CHECKPOINT}"
echo "test_yaml=${TEST_YAML}"
echo "output_dir=${RENDER_OUTPUT_DIR}"

train_args=(
  --config "${CONFIG}"
  --output_dir_full "${RENDER_OUTPUT_DIR}"
  --seed "${SEED}"
  --test
  --test_vis
  --save_outputs
  --test_data "${TEST_YAML}"
  --load_from "${CHECKPOINT}"
  --mixed_precision "${MIXED_PRECISION}"
  --trainer.select_val_dataset="${DATASET}"
  --trainer.num_workers="${NUM_WORKERS}"
  --trainer.test_num_workers="${NUM_WORKERS}"
)

cd "${TMA_ROOT}"
if (( NUM_PROCESSES == 1 )); then
  exec "${PYTHON}" hAlgorithm/script/train/train.py "${train_args[@]}"
fi

PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"
exec "${ACCELERATE}" launch   --config_file "${ACCELERATE_CONFIG}"   --num_machines 1   --num_processes "${NUM_PROCESSES}"   --main_process_port "${PORT}"   hAlgorithm/script/train/train.py "${train_args[@]}"
