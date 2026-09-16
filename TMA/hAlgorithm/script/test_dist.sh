#!/usr/bin/env bash

set -euo pipefail
set -v

# Usage:
#   bash hAlgorithm/script/test_dist.sh <config.py> <ckpt.pth> [NNODES] [NUMPROCESSES] [RANK] [MASTER_ADDR] [MASTER_PORT]
#
# Example:
#   bash hAlgorithm/script/test_dist.sh \
#     /mnt/home/tcchen/workspace/TMA-origin-dev/hAlgorithm/configs/motion_head/baseline_v4/wfm_rgb_query_260526_sparse_pair_4dgs_v1.py \
#     /path/to/checkpoint/iter_010000/ckpt.pth \
#     1 1
#
# Notes:
# - This script runs one test/eval pass and writes visualization to:
#     <OUTDIR>/<EXPNAME>/visualization/<iter_xxxxxx>/<dataset>/4dgs/...
# - Make sure your config enables `save_output_cfg.save_4dgs_results=True`.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

# # gsplat CUDA JIT: default gcc-13 fails on floatn.h; use gcc-11 + A100 arch.
export CUDA_HOME="${CUDA_HOME:-/mnt/home/tcchen/local/cuda-12.4}"
export CC="${CC:-/usr/bin/gcc-11}"
export CXX="${CXX:-/usr/bin/g++-11}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-/usr/bin/g++-11}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

DEFAULT_RUN_DIR="/mnt/home/tcchen/workspace/TMA-origin-dev/results/sparse_pair_4dgs_v2/wfm_rgb_query_dual_4dgs_perpixel_finetune_20260611-115145"
DEFAULT_CONFIG="${DEFAULT_RUN_DIR}/wfm_rgb_query_dual_4dgs_perpixel_finetune.py"
DEFAULT_CKPT="${DEFAULT_RUN_DIR}/checkpoint/latest/ckpt.pth"

CONFIG="${1:-$DEFAULT_CONFIG}"
LOADFROM="${2:-$DEFAULT_CKPT}"

# CONFIG="/mnt/home/tcchen/workspace/TMA-origin-dev/results/streaming/baseline_streaming_cross_attn_20260609-221408/baseline_streaming_cross_attn_backup.py"
# LOADFROM="/mnt/home/tcchen/workspace/TMA-origin-dev/results/streaming/baseline_streaming_cross_attn_20260609-221408/checkpoint/latest/ckpt.pth"


if [[ -z "$CONFIG" || -z "$LOADFROM" ]]; then
  echo "ERROR: missing args."
  echo "Usage: bash $0 <config.py> <ckpt.pth> [NNODES] [NUMPROCESSES] [RANK] [MASTER_ADDR] [MASTER_PORT]"
  exit 2
fi

OUTDIR="${OUTDIR:-/mnt/home/tcchen/workspace/TMA-origin-dev/results}"
EXPNAME="${EXPNAME:-test_hasim_benchmark_hard}"

ACCELERATE_CONFIG_FILE="hAlgorithm/script/accelerate_config.yaml"

default_ip="127.0.0.1"

NNODES=${3:-1}
NUMPROCESSES=${4:-1}
RANK=${5:-0}
MASTER_ADDR=${6:-$default_ip}
PORT=$((12345+$RANDOM)) # 每次launch自动避开上一次的port地址
MASTER_PORT=${7:-$PORT}

echo "NNODES" $NNODES "NUMPROCESSES" $NUMPROCESSES "RANK" $RANK "MASTER_ADDR" $MASTER_ADDR "MASTER_PORT" $MASTER_PORT

accelerate launch \
    --config_file $ACCELERATE_CONFIG_FILE \
    --num_machines $NNODES \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --num_processes $NUMPROCESSES \
    --machine_rank $RANK \
    hAlgorithm/script/train/train.py \
    --config $CONFIG \
    --output_dir $OUTDIR \
    --exp $EXPNAME \
    --seed 2024 \
    --test \
    --load_from $LOADFROM \
    --mixed_precision fp16 \
    --test_vis