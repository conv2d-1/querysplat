#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMA_ROOT="${ROOT}/TMA"
D4GS_ROOT="${ROOT}/4DGS"
ENV_PREFIX="${ENV_PREFIX:-${ROOT}/.envs/tma4dgs}"
PYTHON="${PYTHON:-${ENV_PREFIX}/bin/python}"
ACCELERATE="${ACCELERATE:-${ENV_PREFIX}/bin/accelerate}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${TMA_ROOT}/hAlgorithm/script/accelerate_config.yaml}"
CONFIG="${CONFIG:-${D4GS_ROOT}/baseline_0903.py}"
TRAIN_YAML="${TRAIN_YAML:-${D4GS_ROOT}/configs/local_train.yaml}"
VAL_YAML="${VAL_YAML:-${D4GS_ROOT}/configs/local_val.yaml}"
CHECKPOINT="${CHECKPOINT:-/mnt/cfsdata/Team/AI/personal/chentiancheng/workspace/Projects/TMA/results/Q4RT1.12/latest/ckpt.pth}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/results}"

DATASET="${DATASET:-hasim_benchmark_running}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
MAX_ITER="${MAX_ITER:-50000}"
NUM_PROCESSES_REQUESTED="${NUM_PROCESSES:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-2024}"
SAVE_PERIOD="${SAVE_PERIOD:-1000}"
VAL_PERIOD="${VAL_PERIOD:-2000}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
RUN_NAME="${RUN_NAME:-overfit_${DATASET}_${NUM_SAMPLES}samples}"
RUN_DIR="${RUN_DIR:-${RESULTS_ROOT}/${RUN_NAME}}"
GENERATED_DIR="${RUN_DIR}/generated_data"

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
nonnegative_int() { [[ "$2" =~ ^[0-9]+$ ]] || die "$1 必须是非负整数: $2"; }

usage() {
  cat <<'EOF'
用法:
  ./scripts/run_4dgs_overfit_10.sh [check|train]

环境变量:
  DATASET=hasim_benchmark_running  NUM_SAMPLES=10  MAX_ITER=50000
  CUDA_VISIBLE_DEVICES=0           NUM_PROCESSES=1  NUM_WORKERS=8
  RUN_DIR=/path/to/run             MIXED_PRECISION=bf16|fp16

云平台任务请以前台方式执行，不要在命令末尾添加 &。
EOF
}

MODE="${1:-train}"
case "${MODE}" in
  check|train) shift || true ;;
  -h|--help|help) usage; exit 0 ;;
  *) die "模式必须是 check 或 train: ${MODE}" ;;
esac
(( $# == 0 )) || die "不支持位置参数，请使用环境变量配置"

positive_int NUM_SAMPLES "${NUM_SAMPLES}"
positive_int MAX_ITER "${MAX_ITER}"
nonnegative_int NUM_WORKERS "${NUM_WORKERS}"
positive_int NUM_PROCESSES "${NUM_PROCESSES_REQUESTED}"
nonnegative_int SAVE_PERIOD "${SAVE_PERIOD}"
nonnegative_int VAL_PERIOD "${VAL_PERIOD}"
nonnegative_int SEED "${SEED}"

require_file "${PYTHON}"
require_file "${CONFIG}"
require_file "${TRAIN_YAML}"
require_file "${VAL_YAML}"
require_file "${CHECKPOINT}"
[[ -d "${TMA_ROOT}" ]] || die "TMA 目录不存在: ${TMA_ROOT}"
[[ -x "${ACCELERATE}" || "${NUM_PROCESSES_REQUESTED}" == 1 ]] || die "accelerate 不可用: ${ACCELERATE}"

mkdir -p "${RUN_DIR}" "${GENERATED_DIR}" "${TORCH_HOME}" "${GSPLAT_CACHE_DIR}"

check_environment() {
  "${PYTHON}" - <<'PY'
import importlib
import sys
import numpy
import torch
print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"numpy={numpy.__version__}")
if tuple(map(int, numpy.__version__.split('.')[:2])) >= (2, 0):
    raise SystemExit("NumPy >=2 与 imgaug==0.4.0 不兼容，请安装 numpy==1.26.4")
importlib.import_module("imgaug")
print("imgaug_import=ok")
print(f"cuda_available={torch.cuda.is_available()} devices={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA 不可用")
PY
  local visible
  visible="$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')"
  (( NUM_PROCESSES_REQUESTED <= visible )) || die "进程数超过可见 GPU 数"
}

generate_configs() {
  "${PYTHON}" - "${TRAIN_YAML}" "${VAL_YAML}" "${GENERATED_DIR}" "${DATASET}" "${NUM_SAMPLES}" <<'PY'
import copy
import pathlib
import sys
import yaml
train_path, val_path, out_dir, wanted, number = sys.argv[1:]
strategy = f"first:{number}"
out_dir = pathlib.Path(out_dir)
def filter_yaml(src, dst):
    with open(src, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    key = "datasets_train" if "datasets_train" in doc else "datasets"
    entries = doc.get(key, [])
    kept = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        pipeline = entry.get("pipeline", entry)
        if isinstance(pipeline, str):
            raise SystemExit(f"{src}: 不支持 pipeline 文件引用")
        name = entry.get("name") or pipeline.get("name")
        if name == wanted:
            item = copy.deepcopy(entry)
            item_pipeline = item.get("pipeline", item)
            item_pipeline["mf_scene_sampling_strategy"] = strategy
            kept.append(item)
    if len(kept) != 1:
        names = [e.get("name") or e.get("pipeline", {}).get("name") for e in entries if isinstance(e, dict)]
        raise SystemExit(f"{src}: 数据集 {wanted!r} 未唯一匹配，现有数据集: {names}")
    doc[key] = kept
    with open(dst, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)
filter_yaml(train_path, out_dir / "train.yaml")
filter_yaml(val_path, out_dir / "val.yaml")
filter_yaml(val_path, out_dir / "vis.yaml")
PY
}

write_manifest() {
  cat > "${RUN_DIR}/launch_parameters.txt" <<EOF
mode=train
dataset=${DATASET}
num_samples=${NUM_SAMPLES}
sampling_strategy=first:${NUM_SAMPLES}
config=${CONFIG}
train_yaml=${GENERATED_DIR}/train.yaml
val_yaml=${GENERATED_DIR}/val.yaml
checkpoint=${CHECKPOINT}
max_iter=${MAX_ITER}
num_processes=${NUM_PROCESSES_REQUESTED}
num_workers=${NUM_WORKERS}
mixed_precision=${MIXED_PRECISION}
seed=${SEED}
EOF
}

check_environment
if [[ "${MODE}" == "check" ]]; then
  echo "environment=ok"
  exit 0
fi

generate_configs
write_manifest
echo "run_dir=${RUN_DIR}"
echo "dataset=${DATASET} samples=${NUM_SAMPLES} max_iter=${MAX_ITER}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES} processes=${NUM_PROCESSES_REQUESTED}"
echo "开始训练（前台运行，云平台不要追加 &）"

train_args=(
  --config "${CONFIG}"
  --output_dir_full "${RUN_DIR}"
  --seed "${SEED}"
  --resume None
  --load_from "${CHECKPOINT}"
  --mixed_precision "${MIXED_PRECISION}"
  --data.train="${GENERATED_DIR}/train.yaml"
  --data.val="${GENERATED_DIR}/val.yaml"
  --data.vis="${GENERATED_DIR}/vis.yaml"
  --trainer.max_iter="${MAX_ITER}"
  --trainer.num_workers="${NUM_WORKERS}"
  --trainer.batch_size=1
  --trainer.select_dataset="${DATASET}"
  --trainer.select_val_dataset="${DATASET}"
  --trainer.select_vis_dataset="${DATASET}"
  --trainer.in_evaluation=False
  --trainer.in_visualize=False
  --trainer.backup_period=0
  --trainer.save_period="${SAVE_PERIOD}"
  --trainer.val_period="${VAL_PERIOD}"
  --trainer.vis_period=0
)

cd "${TMA_ROOT}"
if (( NUM_PROCESSES_REQUESTED == 1 )); then
  exec "${PYTHON}" hAlgorithm/script/train/train.py "${train_args[@]}"
fi
PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"
exec "${ACCELERATE}" launch --config_file "${ACCELERATE_CONFIG}" --num_machines 1 --num_processes "${NUM_PROCESSES_REQUESTED}" --main_process_port "${PORT}" hAlgorithm/script/train/train.py "${train_args[@]}"
