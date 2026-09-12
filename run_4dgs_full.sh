#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMA_ROOT="${ROOT}/TMA"
D4GS_ROOT="${ROOT}/4DGS"
ENV_PREFIX="${ENV_PREFIX:-${ROOT}/.envs/tma4dgs}"
PYTHON="${PYTHON:-${ENV_PREFIX}/bin/python}"
ACCELERATE="${ACCELERATE:-${ENV_PREFIX}/bin/accelerate}"
CONFIG="${CONFIG:-${D4GS_ROOT}/local_4dgs.py}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-${D4GS_ROOT}/checkpoint/latest/ckpt.pth}"
EVAL_CHECKPOINT="${EVAL_CHECKPOINT:-${D4GS_ROOT}/checkpoint/best/ckpt.pth}"
TRAIN_YAML="${TRAIN_YAML:-${D4GS_ROOT}/configs/local_train.yaml}"
VAL_YAML="${VAL_YAML:-${D4GS_ROOT}/configs/local_val.yaml}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/results}"

RUN_NAME="${RUN_NAME:-full_4dgs_$(date +%Y%m%d-%H%M%S)}"
RUN_DIR_WAS_SET="${RUN_DIR+x}"
RUN_DIR="${RUN_DIR:-${RESULTS_ROOT}/full_4dgs/${RUN_NAME}}"
GENERATED_EVAL_YAML="${GENERATED_EVAL_YAML:-${RUN_DIR}/configs/full_val_all_sequences.yaml}"

MAX_ITER="${MAX_ITER:-50000}"
LOGGING_STEP="${LOGGING_STEP:-10}"
TEST_NUM_WORKERS="${TEST_NUM_WORKERS:-2}"
EVAL_DATASETS="${EVAL_DATASETS:-pointodyssey,kubric4d,hasim,hasim_benchmark_running}"
NUM_PROCESSES_REQUESTED="${NUM_PROCESSES:-auto}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CC="${CC:-/usr/bin/gcc-11}"
export CXX="${CXX:-/usr/bin/g++-11}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-/usr/bin/g++-11}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export GSPLAT_CACHE_DIR="${GSPLAT_CACHE_DIR:-${ROOT}/.cache/gsplat}"
export TORCH_HOME="${TORCH_HOME:-${ROOT}/.cache/torch}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cpu_count="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
if (( cpu_count < 1 )); then
  cpu_count=1
fi
if (( cpu_count > 8 )); then
  default_num_workers=8
else
  default_num_workers="${cpu_count}"
fi
NUM_WORKERS="${NUM_WORKERS:-${default_num_workers}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${cpu_count}}"

usage() {
  cat <<'EOF'
用法:
  ./run_4dgs_full.sh check
  ./run_4dgs_full.sh train
  ./run_4dgs_full.sh resume <run-dir|ckpt.pth>
  ./run_4dgs_full.sh test [run-dir|ckpt.pth]
  ./run_4dgs_full.sh render [run-dir|ckpt.pth]
  ./run_4dgs_full.sh evaluate [run-dir|ckpt.pth]
  ./run_4dgs_full.sh all
  ./run_4dgs_full.sh media-render <video-or-image-directory> [...]

模式:
  train        原始全量训练配方，默认 50000 step；训练中沿用轻量协议验证。
  test         对四个验证集的全部场景/相机序列计算指标。
  render       对全部验证序列生成可视化，不重复计算指标。
  evaluate     一次前向同时完成全量指标和可视化，推荐。
  all          train 后自动选择 best（否则 latest）并执行 evaluate。
  media-render 对给定视频/图片目录使用全部帧，分块做稠密 4DGS 渲染。

常用环境变量:
  RUN_DIR, MAX_ITER, CUDA_VISIBLE_DEVICES, NUM_PROCESSES, NUM_WORKERS
  INIT_CHECKPOINT, EVAL_CHECKPOINT, EVAL_YAML, EVAL_DATASETS
  VAL_PERIOD, SAVE_PERIOD, VIS_PERIOD, BACKUP_PERIOD, SAVE_OUTPUTS
  PROCESS_RES (默认 504), INFER_CHUNK_SIZE (默认 4), NATIVE_RESOLUTION

提示:
  不设置 RUN_DIR 时，每次调用都会创建新的时间戳目录。分步训练和评测时，
  请复用同一个 RUN_DIR，或把训练目录/ckpt.pth 作为 test/render 参数传入。
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "文件不存在: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "目录不存在: $1"
}

common_preflight() {
  [[ "${MAX_ITER}" =~ ^[1-9][0-9]*$ ]] || die "MAX_ITER 必须是正整数"
  [[ "${NUM_WORKERS}" =~ ^[0-9]+$ ]] || die "NUM_WORKERS 必须是非负整数"
  [[ "${TEST_NUM_WORKERS}" =~ ^[0-9]+$ ]] || die "TEST_NUM_WORKERS 必须是非负整数"
  require_file "${PYTHON}"
  require_file "${CONFIG}"
  require_file "${TRAIN_YAML}"
  require_file "${VAL_YAML}"
  require_dir "${TMA_ROOT}"
  mkdir -p "${RESULTS_ROOT}" "${GSPLAT_CACHE_DIR}" "${TORCH_HOME}"
}

check_data_indices() {
  "${PYTHON}" - "${TRAIN_YAML}" "${VAL_YAML}" <<'PY'
import os
import sys

import yaml

missing = []
total = 0
for yaml_path in sys.argv[1:]:
    with open(yaml_path, "r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    for item in document.get("datasets", []):
        pipeline = item.get("pipeline", item)
        if not isinstance(pipeline, dict):
            pipeline = item
        paths = pipeline.get("data_path")
        paths = paths if isinstance(paths, list) else [paths]
        for path in paths:
            if path is None:
                continue
            total += 1
            if not os.path.isfile(path):
                missing.append(path)

if missing:
    print("缺失的数据索引文件:", file=sys.stderr)
    for path in missing:
        print(f"  {path}", file=sys.stderr)
    raise SystemExit(1)
print(f"数据索引检查通过: {total}/{total}")
PY
}

detect_num_processes() {
  local visible
  visible="$("${PYTHON}" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
  [[ "${visible}" =~ ^[0-9]+$ ]] || die "无法识别可见 GPU 数量: ${visible}"
  (( visible > 0 )) || die "PyTorch 未发现可用 GPU"

  if [[ "${NUM_PROCESSES_REQUESTED}" == "auto" ]]; then
    NUM_PROCESSES="${visible}"
  else
    NUM_PROCESSES="${NUM_PROCESSES_REQUESTED}"
    [[ "${NUM_PROCESSES}" =~ ^[1-9][0-9]*$ ]] || die "NUM_PROCESSES 必须是正整数或 auto"
    (( NUM_PROCESSES <= visible )) || die "请求 ${NUM_PROCESSES} 个进程，但只有 ${visible} 个可见 GPU"
  fi
}

print_resources() {
  "${PYTHON}" - "${NUM_PROCESSES}" "${NUM_WORKERS}" "${RUN_DIR}" <<'PY'
import sys

import torch

processes, workers, run_dir = sys.argv[1:]
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"visible_gpus={torch.cuda.device_count()} processes={processes} data_workers={workers}")
for index in range(torch.cuda.device_count()):
    prop = torch.cuda.get_device_properties(index)
    free, total = torch.cuda.mem_get_info(index)
    print(
        f"gpu[{index}]={prop.name} capability={prop.major}.{prop.minor} "
        f"free={free / 1024**3:.1f}GiB total={total / 1024**3:.1f}GiB"
    )
print(f"run_dir={run_dir}")
PY
}

make_full_eval_yaml() {
  if [[ -n "${EVAL_YAML:-}" ]]; then
    require_file "${EVAL_YAML}"
    FULL_EVAL_YAML="${EVAL_YAML}"
    return
  fi

  mkdir -p "$(dirname "${GENERATED_EVAL_YAML}")"
  "${PYTHON}" - "${VAL_YAML}" "${GENERATED_EVAL_YAML}" <<'PY'
import sys

import yaml

source, destination = sys.argv[1:3]
with open(source, "r", encoding="utf-8") as handle:
    document = yaml.safe_load(handle)

datasets = document.get("datasets", [])
for item in datasets:
    pipeline = item.get("pipeline")
    target = pipeline if isinstance(pipeline, dict) else item
    target["sampling_strategy"] = "all"
    target["mf_scene_sampling_strategy"] = "all"

with open(destination, "w", encoding="utf-8") as handle:
    yaml.safe_dump(document, handle, sort_keys=False, allow_unicode=True)
print(f"已生成全量验证配置（{len(datasets)} 个数据集）: {destination}")
PY
  FULL_EVAL_YAML="${GENERATED_EVAL_YAML}"
}

launch_train_entry() {
  if (( NUM_PROCESSES == 1 )); then
    "${PYTHON}" hAlgorithm/script/train/train.py "$@"
    return
  fi

  require_file "${ACCELERATE}"
  require_file "${TMA_ROOT}/hAlgorithm/script/accelerate_config.yaml"
  "${ACCELERATE}" launch \
    --config_file hAlgorithm/script/accelerate_config.yaml \
    --num_machines 1 \
    --num_processes "${NUM_PROCESSES}" \
    --main_process_port "$((29500 + RANDOM % 1000))" \
    hAlgorithm/script/train/train.py "$@"
}

period_overrides() {
  PERIOD_ARGS=()
  [[ -n "${VAL_PERIOD:-}" ]] && PERIOD_ARGS+=("--trainer.val_period=${VAL_PERIOD}")
  [[ -n "${SAVE_PERIOD:-}" ]] && PERIOD_ARGS+=("--trainer.save_period=${SAVE_PERIOD}")
  [[ -n "${VIS_PERIOD:-}" ]] && PERIOD_ARGS+=("--trainer.vis_period=${VIS_PERIOD}")
  [[ -n "${BACKUP_PERIOD:-}" ]] && PERIOD_ARGS+=("--trainer.backup_period=${BACKUP_PERIOD}")
}

run_train() {
  local output_dir="$1"
  local resume_checkpoint="${2:-}"
  local train_config="${CONFIG}"
  local -a checkpoint_args

  if [[ -n "${resume_checkpoint}" ]]; then
    require_file "${resume_checkpoint}"
    require_file "$(dirname "${resume_checkpoint}")/trainer.ckpt"
    if [[ -f "${output_dir}/$(basename "${CONFIG}")" ]]; then
      train_config="${output_dir}/$(basename "${CONFIG}")"
    fi
    local completed_iter
    completed_iter="$("${PYTHON}" - "${resume_checkpoint}" <<'PY'
import os
import sys
import torch
state = torch.load(
    os.path.join(os.path.dirname(sys.argv[1]), "trainer.ckpt"),
    map_location="cpu",
    weights_only=False,
)
print(int(state["total_iter"]))
PY
)"
    (( MAX_ITER > completed_iter )) || die \
      "MAX_ITER=${MAX_ITER} 必须大于 checkpoint 的 total_iter=${completed_iter}"
    checkpoint_args=(--resume "${resume_checkpoint}" --load_from None)
  else
    require_file "${INIT_CHECKPOINT}"
    [[ ! -e "${output_dir}/checkpoint/latest/ckpt.pth" ]] || die \
      "目标目录已有 checkpoint；请改用 resume 或更换 RUN_DIR: ${output_dir}"
    checkpoint_args=(--resume None --load_from "${INIT_CHECKPOINT}")
  fi

  period_overrides
  mkdir -p "${output_dir}"
  echo "开始训练: output=${output_dir}, max_iter=${MAX_ITER}, GPUs=${NUM_PROCESSES}"
  launch_train_entry \
    --config "${train_config}" \
    --output_dir_full "${output_dir}" \
    --seed 2024 \
    "${checkpoint_args[@]}" \
    --mixed_precision bf16 \
    --data.train="${TRAIN_YAML}" \
    --data.val="${VAL_YAML}" \
    --data.vis="${VAL_YAML}" \
    --trainer.max_iter="${MAX_ITER}" \
    --trainer.num_workers="${NUM_WORKERS}" \
    --trainer.logging_step="${LOGGING_STEP}" \
    "${PERIOD_ARGS[@]}"
}

resolve_checkpoint() {
  local candidate="$1"
  local preference="${2:-best}"

  if [[ -f "${candidate}" ]]; then
    printf '%s\n' "${candidate}"
  elif [[ -f "${candidate}/ckpt.pth" ]]; then
    printf '%s\n' "${candidate}/ckpt.pth"
  elif [[ -f "${candidate}/checkpoint/${preference}/ckpt.pth" ]]; then
    printf '%s\n' "${candidate}/checkpoint/${preference}/ckpt.pth"
  elif [[ -f "${candidate}/checkpoint/latest/ckpt.pth" ]]; then
    printf '%s\n' "${candidate}/checkpoint/latest/ckpt.pth"
  elif [[ -f "${candidate}/checkpoint/best/ckpt.pth" ]]; then
    printf '%s\n' "${candidate}/checkpoint/best/ckpt.pth"
  else
    die "无法从以下路径解析 checkpoint: ${candidate}"
  fi
}

run_dir_from_checkpoint() {
  local checkpoint="$1"
  local checkpoint_dir
  checkpoint_dir="$(dirname "${checkpoint}")"
  if [[ "$(basename "$(dirname "${checkpoint_dir}")")" == "checkpoint" ]]; then
    dirname "$(dirname "${checkpoint_dir}")"
  else
    die "请通过 RUN_DIR 指定恢复输出目录；无法从 checkpoint 推断: ${checkpoint}"
  fi
}

run_eval() {
  local kind="$1"
  local checkpoint="$2"
  local output_dir="$3"
  local -a mode_args save_args

  require_file "${checkpoint}"
  make_full_eval_yaml
  mkdir -p "${output_dir}"

  case "${kind}" in
    test)
      mode_args=()
      ;;
    render)
      mode_args=(--test_vis --trainer.eval_metrics=None)
      ;;
    evaluate)
      mode_args=(--test_vis)
      ;;
    *)
      die "未知评测模式: ${kind}"
      ;;
  esac

  save_args=()
  [[ "${SAVE_OUTPUTS:-0}" == "1" ]] && save_args+=(--save_outputs)

  echo "开始 ${kind}: checkpoint=${checkpoint}, output=${output_dir}"
  launch_train_entry \
    --config "${CONFIG}" \
    --output_dir_full "${output_dir}" \
    --seed 2024 \
    --test \
    --test_data "${FULL_EVAL_YAML}" \
    --load_from "${checkpoint}" \
    --mixed_precision fp16 \
    --trainer.select_val_dataset="${EVAL_DATASETS}" \
    --trainer.test_num_workers="${TEST_NUM_WORKERS}" \
    "${mode_args[@]}" \
    "${save_args[@]}"
}

run_media_render() {
  local checkpoint="$1"
  shift
  (( $# > 0 )) || die "media-render 至少需要一个视频或图片目录"
  require_file "${checkpoint}"

  local -a videos sequences args
  videos=()
  sequences=()
  for input in "$@"; do
    if [[ -d "${input}" ]]; then
      sequences+=("${input}")
    elif [[ -f "${input}" ]]; then
      videos+=("${input}")
    else
      die "输入不存在: ${input}"
    fi
  done

  local output_dir="${MEDIA_OUTPUT_DIR:-${RUN_DIR}/media_render}"
  mkdir -p "${output_dir}"
  args=(
    --motion_config "${CONFIG}"
    --load_from "${checkpoint}"
    --output_dir "${output_dir}"
    --all_frames
    --infer_chunk_size "${INFER_CHUNK_SIZE:-4}"
    --frame_sampling sequential
    --use_amp
    --amp_dtype float16
    --per_pixel
    --export_web_viewer
  )
  if [[ "${NATIVE_RESOLUTION:-0}" == "1" ]]; then
    args+=(--native_resolution)
  else
    args+=(--process_res "${PROCESS_RES:-504}")
  fi
  (( ${#videos[@]} == 0 )) || args+=(--videos "${videos[@]}")
  (( ${#sequences[@]} == 0 )) || args+=(--sequence_dirs "${sequences[@]}")

  "${PYTHON}" \
    hAlgorithm/script/infer/motion_head/wfm_video_gaussian_vis_infer.py \
    "${args[@]}"
}

MODE="${1:-}"
[[ -n "${MODE}" ]] || {
  usage
  exit 2
}
shift || true

case "${MODE}" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

common_preflight
detect_num_processes
cd "${TMA_ROOT}"

case "${MODE}" in
  check)
    require_file "${INIT_CHECKPOINT}"
    require_file "${EVAL_CHECKPOINT}"
    check_data_indices
    make_full_eval_yaml
    print_resources
    ;;

  train)
    (( $# == 0 )) || die "train 不接受位置参数"
    print_resources
    run_train "${RUN_DIR}"
    ;;

  resume)
    resume_source="${1:-${RESUME_FROM:-}}"
    [[ -n "${resume_source}" ]] || die "resume 需要 run-dir 或 ckpt.pth"
    resume_checkpoint="$(resolve_checkpoint "${resume_source}" latest)"
    if [[ -n "${RUN_DIR_WAS_SET}" ]]; then
      resume_run_dir="${RUN_DIR}"
    else
      resume_run_dir="$(run_dir_from_checkpoint "${resume_checkpoint}")"
    fi
    print_resources
    run_train "${resume_run_dir}" "${resume_checkpoint}"
    ;;

  test|render|evaluate)
    source="${1:-${EVAL_CHECKPOINT}}"
    checkpoint="$(resolve_checkpoint "${source}" best)"
    eval_output_root="${RUN_DIR}"
    if [[ -z "${RUN_DIR_WAS_SET}" && $# -gt 0 && -d "${source}/checkpoint" ]]; then
      eval_output_root="${source}"
    fi
    print_resources
    run_eval "${MODE}" "${checkpoint}" "${eval_output_root}/posttrain/${MODE}_full"
    ;;

  all|pipeline)
    (( $# == 0 )) || die "${MODE} 不接受位置参数"
    print_resources
    run_train "${RUN_DIR}"
    if [[ -f "${RUN_DIR}/checkpoint/best/ckpt.pth" ]]; then
      checkpoint="${RUN_DIR}/checkpoint/best/ckpt.pth"
    else
      checkpoint="${RUN_DIR}/checkpoint/latest/ckpt.pth"
    fi
    run_eval evaluate "${checkpoint}" "${RUN_DIR}/posttrain/evaluate_full"
    ;;

  media-render)
    print_resources
    checkpoint="$(resolve_checkpoint "${MEDIA_CHECKPOINT:-${EVAL_CHECKPOINT}}" best)"
    run_media_render "${checkpoint}" "$@"
    ;;

  *)
    usage
    exit 2
    ;;
esac
