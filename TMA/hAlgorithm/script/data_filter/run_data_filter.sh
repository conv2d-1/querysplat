#!/usr/bin/env bash

set -e -v
ulimit -n 65535
export NCCL_TIMEOUT=3600
# export NCCL_BLOCKING_WAIT=1 
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

export CUDA_VISIBLE_DEVICES=0

LOADFROM=None
RESUME=None

CONFIG="hAlgorithm/script/data_filter/config/prompt_pointmap_stage2_total_1epoch.py"
EXPNAME="data_filter/debug"

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
    hAlgorithm/script/data_filter/train.py \
    --config $CONFIG \
    --exp $EXPNAME \
    --seed 2024 \
    --mixed_precision fp16 \
    # --resume $RESUME \
    # --load_from $LOADFROM \
