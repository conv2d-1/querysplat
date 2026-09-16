#!/usr/bin/env bash

set -e -v

export CUDA_VISIBLE_DEVICES=0

CONFIG=${1:-"/mnt/naspersonal/lx/projects/TM/models/prompt_pointmap_952_250411_600k/prompt_pointmap_952_250411_600k.py"}
CKPT_PATH=${2:-"/mnt/naspersonal/lx/projects/TM/models/prompt_pointmap_952_250411_600k/quantization/ckpt_quant.pth"}

python \
    hAlgorithm/script/quantization/convert2onnx.py \
    --config $CONFIG \
    --ckpt $CKPT_PATH

