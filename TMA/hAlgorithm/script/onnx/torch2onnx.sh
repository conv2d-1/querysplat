#!/usr/bin/env bash

set -e -v

export CUDA_VISIBLE_DEVICES=0

OUTDIR="./results/"

CONFIG="/home/leixiao/data/TM/model_config/prompt_pointmap_stage2_250217.py"
LOADFROM="/mnt/personal/ts/projects/hAlgorithm/total_datas/prompt_pointmap_stage2_250205_total_dinol_bs8_20250207-203943/ckpt.pth"

python \
    hAlgorithm/script/onnx/prompt_pointmap_torch2onnx.py \
    --config $CONFIG \
    --output_dir $OUTDIR \
    --seed 2024 \
    # --load_from $LOADFROM \
    # --test \
    # --test_vis \
    # --save_outputs \
    # --test_data $TESTCONFIG \

