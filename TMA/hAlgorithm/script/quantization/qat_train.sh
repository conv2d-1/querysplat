#!/usr/bin/env bash

set -e -v
ulimit -n 65535
export NCCL_TIMEOUT=3600
# export NCCL_BLOCKING_WAIT=1 
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

export CUDA_VISIBLE_DEVICES=0

CONFIG="results/prompt_pointmap_952_crop_250613_total_bs4_600k_20250616-095634/prompt_pointmap_952_crop_250613_total_bs4_600k.py"
LOADFROM="results/prompt_pointmap_952_crop_250613_total_bs4_600k_20250616-095634/quantization_20250624-102308/ckpt_quant.pth"

RESUME=None

EXPNAME="prompt_pointmap_v1.4_qat"

ACCELERATE_CONFIG_FILE="hAlgorithm/script/accelerate_config.yaml"
# ACCELERATE_CONFIG_FILE="hAlgorithm/script/accelerate_zero2_config.yaml"

default_ip="127.0.0.1"

NNODES=${1:-1} 
NUMPROCESSES=${2:-1}
RANK=${3:-0}
MASTER_ADDR=${4:-$default_ip}
PORT=$((12345+$RANDOM)) # 每次launch自动避开上一次的port地址
MASTER_PORT=${5:-$PORT}

echo "NNODES" $NNODES "NUMPROCESSES" $NUMPROCESSES "RANK" $RANK "MASTER_ADDR" $MASTER_ADDR "MASTER_PORT" $MASTER_PORT

accelerate launch \
    --config_file $ACCELERATE_CONFIG_FILE \
    --num_machines $NNODES \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --num_processes $NUMPROCESSES \
    --machine_rank $RANK \
    hAlgorithm/script/quantization/qat_train.py \
    --config $CONFIG \
    --exp $EXPNAME \
    --seed 2024 \
    --resume $RESUME \
    --load_from $LOADFROM \
    --mixed_precision fp16