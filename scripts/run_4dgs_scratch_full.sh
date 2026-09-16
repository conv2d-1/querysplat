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
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/results}"

TRAIN_DATASETS="pointodyssey,kubric4d,dynamicreplica,cotracker3kubric,hasim,hasim_benchmark_running,hasim_character_medium,hasim_character_follow,hasim_character,hasim_character_easy,pstudio,hypersim,dl3dv"
VAL_DATASETS="pointodyssey,kubric4d,hasim,hasim_benchmark_running"

MAX_ITER="${MAX_ITER:-50000}"
NUM_PROCESSES_REQUESTED="${NUM_PROCESSES:-auto}"
NUM_WORKERS="${NUM_WORKERS:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LOGGING_STEP="${LOGGING_STEP:-10}"
SAVE_PERIOD="${SAVE_PERIOD:-1000}"
VAL_PERIOD="${VAL_PERIOD:-2000}"
SEED="${SEED:-2024}"
MIN_FREE_GPU_GIB="${MIN_FREE_GPU_GIB:-76}"
ALLOW_UNSUPPORTED_ENV="${ALLOW_UNSUPPORTED_ENV:-0}"
RESUME_FROM="${RESUME_FROM:-}"
RUN_DIR_WAS_SET="${RUN_DIR+x}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CC="${CC:-/usr/bin/gcc-11}"
export CXX="${CXX:-/usr/bin/g++-11}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-/usr/bin/g++-11}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export GSPLAT_CACHE_DIR="${GSPLAT_CACHE_DIR:-${ROOT}/.cache/gsplat}"
export TORCH_HOME="${TORCH_HOME:-${ROOT}/.cache/torch}"
export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DDP_GRADIENT_AS_BUCKET_VIEW="${DDP_GRADIENT_AS_BUCKET_VIEW:-1}"

usage() {
  cat <<'EOF'
用法:
  ./scripts/run_4dgs_scratch_full.sh check [--参数=值]
  ./scripts/run_4dgs_scratch_full.sh train [--参数=值]
  ./scripts/run_4dgs_scratch_full.sh resume --resume-from=PATH [--参数=值]
  ./scripts/run_4dgs_scratch_full.sh auto --run-dir=PATH [--参数=值]

参数:
  --run-dir=PATH --resume-from=PATH --max-iter=N
  --num-processes=N|auto --num-workers=N
  --gradient-accumulation-steps=N --logging-step=N
  --save-period=N --val-period=N --seed=N
  --min-free-gpu-gib=N --cuda-visible-devices=0,1,2,3,4,5,6,7

默认使用 13 个训练集、4 个验证集、504px、4 视角、bf16、
batch_size=1/GPU、50000 step。按 baseline_0903.py 从 Q4RT1.12
checkpoint warm-start，并使用其 encoder 预训练、冻结与 LPIPS 设置。
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

MODE="${1:-train}"
case "${MODE}" in
  check|train|resume|auto) shift || true ;;
  -h|--help|help) usage; exit 0 ;;
  *) die "模式必须是 check、train、resume 或 auto: ${MODE}" ;;
esac

while (( $# > 0 )); do
  case "$1" in
    --run-dir=*) RUN_DIR="${1#*=}"; RUN_DIR_WAS_SET=1 ;;
    --resume-from=*) RESUME_FROM="${1#*=}" ;;
    --max-iter=*) MAX_ITER="${1#*=}" ;;
    --num-processes=*) NUM_PROCESSES_REQUESTED="${1#*=}" ;;
    --num-workers=*) NUM_WORKERS="${1#*=}" ;;
    --gradient-accumulation-steps=*) GRADIENT_ACCUMULATION_STEPS="${1#*=}" ;;
    --logging-step=*) LOGGING_STEP="${1#*=}" ;;
    --save-period=*) SAVE_PERIOD="${1#*=}" ;;
    --val-period=*) VAL_PERIOD="${1#*=}" ;;
    --seed=*) SEED="${1#*=}" ;;
    --min-free-gpu-gib=*) MIN_FREE_GPU_GIB="${1#*=}" ;;
    --cuda-visible-devices=*) export CUDA_VISIBLE_DEVICES="${1#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) die "未知参数（请使用 --key=value）: $1" ;;
  esac
  shift
done

RUN_NAME="${RUN_NAME:-scratch_full_$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-${RESULTS_ROOT}/scratch_full/${RUN_NAME}}"

require_file() {
  [[ -f "$1" ]] || die "文件不存在: $1"
}

positive_int() {
  [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 必须是正整数: $2"
}

nonnegative_int() {
  [[ "$2" =~ ^[0-9]+$ ]] || die "$1 必须是非负整数: $2"
}

preflight() {
  [[ -x "${PYTHON}" ]] || die "项目 Python 不可用: ${PYTHON}"
  require_file "${CONFIG}"
  require_file "${TRAIN_YAML}"
  require_file "${VAL_YAML}"
  require_file "${ACCELERATE_CONFIG}"
  [[ -d "${TMA_ROOT}" ]] || die "TMA 目录不存在: ${TMA_ROOT}"
  [[ -d "${CUDA_HOME}" ]] || die "CUDA_HOME 不存在: ${CUDA_HOME}"
  [[ -x "${CC}" && -x "${CXX}" ]] || die "需要 gcc-11/g++-11"
  positive_int MAX_ITER "${MAX_ITER}"
  nonnegative_int NUM_WORKERS "${NUM_WORKERS}"
  positive_int GRADIENT_ACCUMULATION_STEPS "${GRADIENT_ACCUMULATION_STEPS}"
  positive_int LOGGING_STEP "${LOGGING_STEP}"
  nonnegative_int SAVE_PERIOD "${SAVE_PERIOD}"
  nonnegative_int VAL_PERIOD "${VAL_PERIOD}"
  nonnegative_int SEED "${SEED}"
  [[ "${MIN_FREE_GPU_GIB}" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
    die "MIN_FREE_GPU_GIB 必须是非负数"
  mkdir -p "${RESULTS_ROOT}/scratch_full" "${GSPLAT_CACHE_DIR}" "${TORCH_HOME}" "${HF_HOME}"
  ulimit -n 65535 2>/dev/null || true
}

detect_processes() {
  local visible
  visible="$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')"
  [[ "${visible}" =~ ^[1-9][0-9]*$ ]] || die "项目环境未发现 GPU"
  if [[ "${NUM_PROCESSES_REQUESTED}" == "auto" ]]; then
    NUM_PROCESSES="${visible}"
  else
    positive_int NUM_PROCESSES "${NUM_PROCESSES_REQUESTED}"
    NUM_PROCESSES="${NUM_PROCESSES_REQUESTED}"
    (( NUM_PROCESSES <= visible )) || die "进程数 ${NUM_PROCESSES} 超过可见 GPU 数 ${visible}"
  fi
  if (( NUM_PROCESSES > 1 )); then
    [[ -x "${ACCELERATE}" ]] || die "accelerate 不可用: ${ACCELERATE}"
  fi
}

check_contract() {
  (
    cd "${TMA_ROOT}"
    "${PYTHON}" - \
      "${CONFIG}" "${TRAIN_YAML}" "${VAL_YAML}" \
      "${TRAIN_DATASETS}" "${VAL_DATASETS}" <<'PY'
import os
import sys

import yaml
from hAlgorithm.utils import file2dict

config_path, train_yaml, val_yaml, train_csv, val_csv = sys.argv[1:]
cfg = file2dict(config_path)
model = cfg["model"]["model"]
fuse = model["fuse_encoder"]
trainer = cfg["trainer"]

assert trainer.get("resume") is None
load_from = trainer.get("load_from")
pretrain = fuse.get("pretrain")
pretrained_pretrain = fuse.get("pretrained_pretrain")
freeze_modules = model.get("freeze_modules") or []
lpips_weight = cfg["model"]["sparse_dynamic_gaussian_render_loss"].get("lpips_weight")

assert load_from is not None
assert pretrain is not None
assert "fuse_encoder" in freeze_modules
assert lpips_weight == 0.2

for path, value in (("trainer.load_from", load_from), ("model.model.fuse_encoder.pretrain", pretrain), ("model.model.fuse_encoder.pretrained_pretrain", pretrained_pretrain)):
    if value is not None:
        assert os.path.isfile(value), (path, value)

def check_yaml(path, expected_csv):
    with open(path, "r", encoding="utf-8") as handle:
        datasets = yaml.safe_load(handle)["datasets"]
    names = [item["pipeline"]["name"] for item in datasets]
    expected = expected_csv.split(",")
    assert len(names) == len(set(names)), names
    assert set(names) == set(expected), (names, expected)
    missing = []
    index_count = 0
    for item in datasets:
        pipeline = item["pipeline"]
        paths = pipeline.get("data_path")
        paths = paths if isinstance(paths, list) else [paths]
        for data_path in paths:
            index_count += 1
            if not data_path or not os.path.isfile(data_path):
                missing.append(data_path)
        data_root = pipeline.get("data_root")
        if data_root and not os.path.isdir(data_root):
            missing.append(data_root)
    assert not missing, missing
    return len(names), index_count

train_count, train_indices = check_yaml(train_yaml, train_csv)
val_count, val_indices = check_yaml(val_yaml, val_csv)
print("baseline_contract=ok")
print(f"trainer_load_from={load_from}")
print(f"backbone_pretrain={pretrain}")
print(f"frozen_modules={freeze_modules}")
print(f"lpips_weight={lpips_weight}")
print(f"train_datasets={train_count} train_indices={train_indices}")
print(f"val_datasets={val_count} val_indices={val_indices}")
PY
  )
}

check_resources() {
  local enforce_free="$1"
  "${PYTHON}" - \
    "${NUM_PROCESSES}" "${NUM_WORKERS}" "${GRADIENT_ACCUMULATION_STEPS}" \
    "${MIN_FREE_GPU_GIB}" "${enforce_free}" "${ALLOW_UNSUPPORTED_ENV}" <<'PY'
import sys

import accelerate
import gsplat
import numpy
import pytorch3d
import torch
import torchvision

processes, workers, accum = map(int, sys.argv[1:4])
min_free = float(sys.argv[4])
enforce_free = sys.argv[5] == "1"
allow_unsupported = sys.argv[6] == "1"
errors = []

if sys.version_info[:2] != (3, 10):
    errors.append(f"需要 Python 3.10，实际 {sys.version.split()[0]}")
if torch.__version__.split("+")[0] != "2.4.1":
    errors.append(f"需要 torch 2.4.1，实际 {torch.__version__}")
if torchvision.__version__.split("+")[0] != "0.19.1":
    errors.append(f"需要 torchvision 0.19.1，实际 {torchvision.__version__}")
if numpy.__version__ != "1.23.5":
    errors.append(f"需要 numpy 1.23.5，实际 {numpy.__version__}")
if gsplat.__version__ != "1.5.3+pt24cu124":
    errors.append(f"需要 gsplat 1.5.3+pt24cu124，实际 {gsplat.__version__}")
if torch.version.cuda != "12.4":
    errors.append(f"项目 torch 应为 cu124，实际 cu{torch.version.cuda}")
if not torch.cuda.is_available():
    errors.append("torch.cuda.is_available() 为 False")
if torch.cuda.device_count() < processes:
    errors.append(f"可见 GPU {torch.cuda.device_count()} 张，少于进程数 {processes}")

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__} torch_cuda={torch.version.cuda}")
print(f"torchvision={torchvision.__version__} accelerate={accelerate.__version__}")
print(f"numpy={numpy.__version__} gsplat={gsplat.__version__} pytorch3d={pytorch3d.__version__}")
print(
    f"processes={processes} workers_per_process={workers} "
    f"total_workers={processes * workers}"
)
print(
    f"micro_batch_per_gpu=1 gradient_accumulation={accum} "
    f"effective_global_batch={processes * accum}"
)

for index in range(torch.cuda.device_count()):
    prop = torch.cuda.get_device_properties(index)
    free, total = torch.cuda.mem_get_info(index)
    free_gib, total_gib = free / 1024**3, total / 1024**3
    print(
        f"gpu[{index}]={prop.name} capability={prop.major}.{prop.minor} "
        f"free={free_gib:.1f}GiB total={total_gib:.1f}GiB"
    )
    if index < processes:
        if prop.major < 8:
            errors.append(f"gpu[{index}] 不满足 bf16/sm80")
        if enforce_free and free_gib < min_free:
            errors.append(
                f"gpu[{index}] 空闲 {free_gib:.1f}GiB，低于要求 {min_free:.1f}GiB"
            )

if errors:
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if not allow_unsupported:
        raise SystemExit(1)
    print("WARNING: ALLOW_UNSUPPORTED_ENV=1，已忽略环境错误", file=sys.stderr)
PY
}

resolve_checkpoint() {
  local value="$1"
  if [[ -f "${value}" ]]; then
    printf '%s\n' "${value}"
  elif [[ -f "${value}/ckpt.pth" ]]; then
    printf '%s\n' "${value}/ckpt.pth"
  elif [[ -f "${value}/checkpoint/latest/ckpt.pth" ]]; then
    printf '%s\n' "${value}/checkpoint/latest/ckpt.pth"
  else
    die "无法解析 checkpoint: ${value}"
  fi
}

infer_run_dir() {
  local checkpoint_dir
  checkpoint_dir="$(dirname "$1")"
  if [[ "$(basename "$(dirname "${checkpoint_dir}")")" == "checkpoint" ]]; then
    dirname "$(dirname "${checkpoint_dir}")"
  else
    die "无法推断 RUN_DIR，请显式传 --run-dir"
  fi
}

checkpoint_iter() {
  "${PYTHON}" - "$1" <<'PY'
import os
import sys
import torch

path = os.path.join(os.path.dirname(sys.argv[1]), "trainer.ckpt")
print(int(torch.load(path, map_location="cpu", weights_only=False)["total_iter"]))
PY
}

write_manifest() {
  local kind="$1"
  local checkpoint="${2:-None}"
  local initialization="baseline_0903_warm_start"
  if [[ "${kind}" == "resume" ]]; then
    initialization="resume_full_training_state"
  fi
  mkdir -p "${RUN_DIR}"
  {
    printf 'mode=%s\n' "${kind}"
    printf 'run_dir=%s\n' "${RUN_DIR}"
    printf 'resume=%s\n' "${checkpoint}"
    printf 'config=%s\n' "${CONFIG}"
    printf 'train_yaml=%s\n' "${TRAIN_YAML}"
    printf 'val_yaml=%s\n' "${VAL_YAML}"
    printf 'max_iter=%s\n' "${MAX_ITER}"
    printf 'num_processes=%s\n' "${NUM_PROCESSES}"
    printf 'num_workers_per_process=%s\n' "${NUM_WORKERS}"
    printf 'gradient_accumulation_steps=%s\n' "${GRADIENT_ACCUMULATION_STEPS}"
    printf 'effective_global_batch=%s\n' \
      "$((NUM_PROCESSES * GRADIENT_ACCUMULATION_STEPS))"
    printf 'mixed_precision=bf16\n'
    printf 'seed=%s\n' "${SEED}"
    printf 'initialization=%s\n' "${initialization}"
  } > "${RUN_DIR}/launch_parameters.txt"
  printf '%s\n' "${RUN_DIR}" > "${RESULTS_ROOT}/scratch_full/LATEST_RUN"
}

launch_training() {
  local resume_checkpoint="${1:-}"
  local train_config="${CONFIG}"
  local kind="warm_start"
  local port
  local -a checkpoint_args

  if [[ -n "${resume_checkpoint}" ]]; then
    require_file "${resume_checkpoint}"
    require_file "$(dirname "${resume_checkpoint}")/trainer.ckpt"
    local completed
    completed="$(checkpoint_iter "${resume_checkpoint}")"
    (( MAX_ITER > completed )) || \
      die "MAX_ITER=${MAX_ITER} 必须大于 checkpoint total_iter=${completed}"
    if [[ -f "${RUN_DIR}/$(basename "${CONFIG}")" ]]; then
      train_config="${RUN_DIR}/$(basename "${CONFIG}")"
    fi
    checkpoint_args=(--resume "${resume_checkpoint}" --load_from None)
    kind="resume"
  else
    [[ ! -e "${RUN_DIR}/checkpoint/latest/ckpt.pth" ]] || \
      die "RUN_DIR 已有 checkpoint，请使用 auto/resume: ${RUN_DIR}"
    checkpoint_args=(--resume None)
  fi

  write_manifest "${kind}" "${resume_checkpoint:-None}"
  echo "run_dir=${RUN_DIR}"
  echo "mode=${kind} max_iter=${MAX_ITER} GPUs=${NUM_PROCESSES}"
  if [[ "${kind}" == "resume" ]]; then
    echo "initialization=resume_full_training_state checkpoint=${resume_checkpoint}"
  else
    echo "initialization=baseline_0903_warm_start (new modules random; configured checkpoint/pretrain preserved)"
  fi

  local -a train_args=(
    --config "${train_config}"
    --output_dir_full "${RUN_DIR}"
    --seed "${SEED}"
    "${checkpoint_args[@]}"
    --mixed_precision bf16
    --data.train="${TRAIN_YAML}"
    --data.val="${VAL_YAML}"
    --data.vis="${VAL_YAML}"
    --trainer.max_iter="${MAX_ITER}"
    --trainer.num_workers="${NUM_WORKERS}"
    --trainer.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}"
    --trainer.select_dataset="${TRAIN_DATASETS}"
    --trainer.select_val_dataset="${VAL_DATASETS}"
    --trainer.select_vis_dataset="${VAL_DATASETS}"
    --trainer.in_evaluation=False
    --trainer.in_visualize=False
    --trainer.logging_step="${LOGGING_STEP}"
    --trainer.backup_period=0
    --trainer.save_period="${SAVE_PERIOD}"
    --trainer.val_period="${VAL_PERIOD}"
    --trainer.vis_period=0
    --trainer.test_num_workers=2
    "--trainer.main_eval_metric=Camera|auc_1"
    --trainer.main_eval_metric_goal=maximize
    --trainer.tb_dataset_split=True
  )

  cd "${TMA_ROOT}"
  if (( NUM_PROCESSES == 1 )); then
    exec "${PYTHON}" hAlgorithm/script/train/train.py "${train_args[@]}"
  fi

  port="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"
  exec "${ACCELERATE}" launch \
    --config_file "${ACCELERATE_CONFIG}" \
    --num_machines 1 \
    --num_processes "${NUM_PROCESSES}" \
    --main_process_port "${port}" \
    hAlgorithm/script/train/train.py \
    "${train_args[@]}"
}

preflight
detect_processes
check_contract

case "${MODE}" in
  check)
    check_resources 0
    ;;
  train)
    check_resources 1
    launch_training
    ;;
  resume)
    if [[ -z "${RESUME_FROM}" ]]; then
      [[ -n "${RUN_DIR_WAS_SET}" ]] || \
        die "resume 需要 --resume-from=PATH 或固定 --run-dir=PATH"
      RESUME_FROM="${RUN_DIR}"
    fi
    resume_checkpoint="$(resolve_checkpoint "${RESUME_FROM}")"
    if [[ -z "${RUN_DIR_WAS_SET}" ]]; then
      RUN_DIR="$(infer_run_dir "${resume_checkpoint}")"
    fi
    check_resources 1
    launch_training "${resume_checkpoint}"
    ;;
  auto)
    [[ -n "${RUN_DIR_WAS_SET}" ]] || \
      die "auto 必须指定固定 --run-dir=PATH，才能在任务重启时续训"
    check_resources 1
    if [[ -f "${RUN_DIR}/checkpoint/latest/ckpt.pth" ]]; then
      launch_training "${RUN_DIR}/checkpoint/latest/ckpt.pth"
    else
      launch_training
    fi
    ;;
esac
