#!/usr/bin/env bash

set -e -v

export CUDA_VISIBLE_DEVICES=0

CONFIG=${1:-"/mnt/naspersonal/lx/projects/TM/models/prompt_pointmap_952_250411_600k/prompt_pointmap_952_250411_600k.py"}

ONNX_PATH=${2:-"/mnt/naspersonal/lx/projects/TM/models/prompt_pointmap_952_250411_600k/ddim1_habitat_all_quant/quant_model.onnx"}

python hAlgorithm/script/quantization/analyse_tools/dequant_analyse.py \
    --config $CONFIG \
    --onnx_path $ONNX_PATH \