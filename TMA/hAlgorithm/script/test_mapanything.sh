#!/usr/bin/env bash

set -e -v

export CUDA_VISIBLE_DEVICES=0
python hAlgorithm/script/auto_mem/clear_gpu.py [$CUDA_VISIBLE_DEVICES]

OUTDIR=./results/
EXPNAME=debug

LOADFROM=None
CONFIG=hAlgorithm/configs/open/mapanything/mapanything_251009_4f.py

TESTCONFIG=debug.yaml

ACCELERATE_CONFIG_FILE=hAlgorithm/script/accelerate_config.yaml

default_ip="127.0.0.1"

NNODES=${1:-1}
NUMPROCESSES=${2:-1}
RANK=${3:-0}
MASTER_ADDR=${4:-$default_ip}
PORT=$((12345+$RANDOM)) # 每次launch自动避开上一次的port地址
MASTER_PORT=${5:-$PORT}

echo "NNODES" $NNODES "NUMPROCESSES" $NUMPROCESSES "RANK" $RANK "MASTER_ADDR" $MASTER_ADDR "MASTER_PORT" $MASTER_PORT

python hAlgorithm/script/train/infer.py \
    --config $CONFIG \
    --output_dir $OUTDIR \
    --exp $EXPNAME \
    --seed 2024 \
    --load_from $LOADFROM \
    --mixed_precision fp16 \
    --test \
    --trainer.test_num_workers 2 \
    --trainer.dist_test False \
    # --test_vis \  
    # --save_outputs \
    # --test_data $TESTCONFIG \
