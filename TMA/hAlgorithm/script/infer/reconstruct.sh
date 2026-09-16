#!/usr/bin/env bash

set -e -v

export CUDA_VISIBLE_DEVICES=7

OUTDIR="./results/"
LOADFROM=None

EXPNAME="reconstruct"

CONFIG="results/dense_prompt_pointmap_250211/prompt_pointmap_stage2_250211_sizex1.5_20250211-144058/prompt_pointmap_stage2_250211_sizex1.5.py"
LOADFROM="results/dense_prompt_pointmap_250211/prompt_pointmap_stage2_250211_sizex1.5_20250211-144058/checkpoint/latest/ckpt.pth"

DATA="hAlgorithm/script/infer/reconstruct.yaml"

ACCELERATE_CONFIG_FILE="hAlgorithm/script/accelerate_config.yaml"
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
    hAlgorithm/script/infer/reconstruct.py \
    --config $CONFIG \
    --output_dir $OUTDIR \
    --exp $EXPNAME \
    --seed 2024 \
    --load_from $LOADFROM \
    --data $DATA \
    --mixed_precision fp16 \
    # --test \
    # --vis \
    

