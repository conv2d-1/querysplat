#!/usr/bin/env bash

set -e -v

export CUDA_VISIBLE_DEVICES=0

OUTDIR="./results/"

CONFIG="/home/leixiao/data/TM/model_config/prompt_pointmap_stage2_250217.py"

PRETRAINED="/home/leixiao/workspace/TM/HaDL/results/quantization/prompt_pointmap_stage2_250217_pretrained_fp16_head_int8_seperated/pretrained.trt"
HEAD="/home/leixiao/workspace/TM/HaDL/results/quantization/prompt_pointmap_stage2_250217_pretrained_fp16_head_int8_seperated/quant_head.trt"

python \
    hAlgorithm/script/tensorrt/prompt_pointmap_seperate_trt.py \
    --pretrained $PRETRAINED \
    --head $HEAD \
    --config $CONFIG \
    --output_dir $OUTDIR \
    --seed 2024 \
    --vis_results \
    # --test \
    # --save_outputs \
    # --test_data $TESTCONFIG \

