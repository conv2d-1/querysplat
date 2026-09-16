#!/usr/bin/env bash

set -e -v

export CUDA_VISIBLE_DEVICES=0

CONFIG=${1:-"/root/disk2/TM/model_config/250220/prompt_pointmap_stage2_250220_oldpipe.py"}

ENGINE=${2:-"./results/quantization/prompt_pointmap_stage2_250220_oldpipe_int8_qat/trex_outputs/quant_model.onnx.engine"}

OUTDIR=${3:-"./results/"}

python \
    hAlgorithm/script/tensorrt/prompt_pointmap_v2_trt.py \
    --engine $ENGINE \
    --config $CONFIG \
    --output_dir $OUTDIR \
    --seed 2024 \
    --save_vis \
    --save_ds 50 \
    # --save_outputs \
    # --test_data $TESTCONFIG \

