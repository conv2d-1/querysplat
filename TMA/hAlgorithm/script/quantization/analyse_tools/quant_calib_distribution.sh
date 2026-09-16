#!/usr/bin/env bash

set -e -v

export CUDA_VISIBLE_DEVICES=0

CONFIG=${1:-"/mnt/naspersonal/lx/projects/TM/models/prompt_pointmap_952_250411_600k/prompt_pointmap_952_250411_600k.py"}
LOADFROM=${2:-"/mnt/naspersonal/lx/projects/TM/models/prompt_pointmap_952_250411_600k/ckpt.pth"}

python hAlgorithm/script/quantization/analyse_tools/quant_calib_distribution.py \
    --config $CONFIG \
    --load_from $LOADFROM \

