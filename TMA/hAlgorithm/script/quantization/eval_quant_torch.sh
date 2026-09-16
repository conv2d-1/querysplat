#!/usr/bin/env bash

set -e -v
ulimit -n 65535

export CUDA_VISIBLE_DEVICES=0

CKPT_PATH="/mnt/naspersonal/lx/projects/TM/models/prompt_pointmap_952_250411_600k/quantization/ckpt_quant.pth"

CONFIG="/mnt/naspersonal/lx/projects/TM/models/prompt_pointmap_952_250411_600k/prompt_pointmap_952_250411_600k.py"

python hAlgorithm/script/quantization/eval_quant_torch.py \
    --config $CONFIG \
    --ckpt $CKPT_PATH \
    --vis_step -1 \