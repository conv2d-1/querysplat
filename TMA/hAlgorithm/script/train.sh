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

CONFIG="hAlgorithm/configs/mv_v1.0/rc_250516_hyp_mvpdc_bs1_debug.py"
EXPNAME="prompt_pointmap_v1"

ACCELERATE_CONFIG_FILE="hAlgorithm/script/accelerate_config.yaml"
# ACCELERATE_CONFIG_FILE="hAlgorithm/script/accelerate_zero2_config.yaml"


python3 hAlgorithm/script/train/train_simple.py \
    --config $CONFIG \
    --exp $EXPNAME \
    --seed 2024 \
    --resume $RESUME \
    --load_from $LOADFROM \
    --mixed_precision fp16 \