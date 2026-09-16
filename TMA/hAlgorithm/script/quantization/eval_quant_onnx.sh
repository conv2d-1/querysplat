#!/usr/bin/env bash

set -e -v

export CUDA_VISIBLE_DEVICES=0

CONFIG=${1:-"/mnt/naspersonal/lx/projects/TM/models/prompt_pointmap_952_250411_600k/prompt_pointmap_952_250411_600k.py"}

ONNX_PATH=${2:-"/mnt/naspersonal/lx/projects/TM/models/prompt_pointmap_952_250411_600k/QAT_speedup2/convert2onnx/quant_model.onnx"}

python hAlgorithm/script/quantization/eval_quant_onnx.py \
    --config $CONFIG \
    --onnx_path $ONNX_PATH \
    --vis_step -1 \