#!/usr/bin/env bash

set -e -v

export CUDA_VISIBLE_DEVICES=0

OUTDIR="./results/"
LOADFROM=None

EXPNAME="test"
CONFIG="hAlgorithm/configs/prompt_pointmap_v1.1/prompt_pointmap_stage1_d30_250127_total_bs8.py"
# LOADFROM=""
# TESTCONFIG="hAlgorithm/configs/prompt_pointmap_v1.1/stage2_dataset_configs/base_vis_k1.yaml"
# ACCELERATE_CONFIG_FILE="hAlgorithm/script/accelerate_zero2_config.yaml"

ACCELERATE_CONFIG_FILE="hAlgorithm/script/accelerate_config.yaml"


python hAlgorithm/script/train/train_simple.py \
    --config $CONFIG \
    --output_dir $OUTDIR \
    --exp $EXPNAME \
    --seed 2024 \
    --test \
    --load_from $LOADFROM \
    --mixed_precision fp16 \
    # --test_vis \
    # --save_outputs \
    # --test_data $TESTCONFIG \