#!/usr/bin/env bash

set -euo pipefail
set -v

ulimit -n 65535
export NCCL_TIMEOUT=3600

# Usage:
#   bash hAlgorithm/script/train_dist_v3.sh [CONFIG] [LOADFROM] [NNODES] [NUMPROCESSES] [RANK] [MASTER_ADDR] [MASTER_PORT]
#
# Example (v3 Envision main experiment):
#   bash hAlgorithm/script/train_dist_v3.sh
#
# Example (ablation — no depth loss):
#   bash hAlgorithm/script/train_dist_v3.sh \
#     hAlgorithm/configs/motion_head/baseline/wfm_rgb_query_dual_4dgs_perpixel_trainquality_v3_no_depth_loss.py
#
# Environment overrides:
#   CUDA_VISIBLE_DEVICES, OUTDIR, EXPNAME, RESUME, LOADFROM,
#   CUDA_HOME, CC, CXX, CUDAHOSTCXX, TORCH_CUDA_ARCH_LIST

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

export CUDA_HOME="${CUDA_HOME:-/mnt/home/tcchen/local/cuda-12.4}"
export CC="${CC:-/usr/bin/gcc-11}"
export CXX="${CXX:-/usr/bin/g++-11}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-/usr/bin/g++-11}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

DEFAULT_CONFIG="/mnt/home/tcchen/workspace/TMA-origin-dev/hAlgorithm/configs/motion_head/baseline/wfm_rgb_query_dual_4dgs_perpixel_trainquality_v3_envision.py"

CONFIG="${1:-$DEFAULT_CONFIG}"
DEFAULT_LOADFROM="None"
LOADFROM="${LOADFROM:-${2:-$DEFAULT_LOADFROM}}"
RESUME="${RESUME:-None}"

OUTDIR="${OUTDIR:-/mnt/home/tcchen/workspace/TMA-origin-dev/results}"
EXPNAME="${EXPNAME:-sparse_pair_4dgs_v2}"

ACCELERATE_CONFIG_FILE="hAlgorithm/script/accelerate_config.yaml"

default_ip="127.0.0.1"

NNODES=${3:-1}
NUMPROCESSES=${4:-1}
RANK=${5:-0}
MASTER_ADDR=${6:-$default_ip}
PORT=$((12345 + RANDOM))
MASTER_PORT=${7:-$PORT}

echo "CONFIG" "$CONFIG"
echo "LOADFROM" "$LOADFROM"
echo "RESUME" "$RESUME"
echo "OUTDIR" "$OUTDIR"
echo "EXPNAME" "$EXPNAME"
echo "NNODES" "$NNODES" "NUMPROCESSES" "$NUMPROCESSES" "RANK" "$RANK" \
  "MASTER_ADDR" "$MASTER_ADDR" "MASTER_PORT" "$MASTER_PORT"
echo "CUDA_HOME=$CUDA_HOME CC=$CC TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

accelerate launch \
    --config_file "$ACCELERATE_CONFIG_FILE" \
    --num_machines "$NNODES" \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    --num_processes "$NUMPROCESSES" \
    --machine_rank "$RANK" \
    hAlgorithm/script/train/train.py \
    --config "$CONFIG" \
    --output_dir "$OUTDIR" \
    --exp "$EXPNAME" \
    --seed 2024 \
    --resume "$RESUME" \
    --load_from "$LOADFROM" \
    --mixed_precision bf16
