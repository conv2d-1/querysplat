#!/usr/bin/env bash

set -e -v

export CUDA_VISIBLE_DEVICES=0

CONFIG=${1:-"/mnt/naspersonal/lx/projects/TM/models/prompt_pointmap_952_250411_600k/prompt_pointmap_952_250411_600k.py"}
CKPT_PATH=${2:-"/mnt/naspersonal/lx/projects/TM/models/prompt_pointmap_952_250411_600k/ckpt.pth"}

OUTDIR=${3:-"None"}

python \
    hAlgorithm/script/quantization/quantization.py \
    --config $CONFIG \
    --output_dir $OUTDIR \
    --ckpt $CKPT_PATH \

