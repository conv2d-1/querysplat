#!/usr/bin/env bash

set -e -v

export CUDA_VISIBLE_DEVICES=4

OUTDIR="./results/"
LOADFROM=None

EXPNAME="app"

# CONFIG="results/dense_prompt_pointmap_250217/prompt_pointmap_stage2_250217_size1_interpout5_pprev3_20250217-194308/prompt_pointmap_stage2_250217_size1_interpout5_pprev3.py"
# LOADFROM="results/dense_prompt_pointmap_250217/prompt_pointmap_stage2_250217_size1_interpout5_pprev3_20250217-194308/checkpoint/latest/ckpt.pth"

CONFIG="results/dense_prompt_pointmap_250217/prompt_pointmap_stage2_250217_size1_res7_pprev3_wointerp_20250218-154409/prompt_pointmap_stage2_250217_size1_res7_pprev3_wointerp.py"
LOADFROM="results/dense_prompt_pointmap_250217/prompt_pointmap_stage2_250217_size1_res7_pprev3_wointerp_20250218-154409/checkpoint/latest/ckpt.pth"


DATA="hAlgorithm/script/infer/app.yaml"

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
    hAlgorithm/script/infer/app.py \
    --config $CONFIG \
    --output_dir $OUTDIR \
    --exp $EXPNAME \
    --seed 2024 \
    --load_from $LOADFROM \
    --data $DATA \
    --mixed_precision fp16 \
    # --server_name "172.20.11.65" \
    # --server_port 1116

