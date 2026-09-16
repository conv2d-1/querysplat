#!/usr/bin/env bash

set -e -v

export CUDA_VISIBLE_DEVICES=0

CONFIG=${1:-"/mnt/naspersonal/lx/projects/TM/models/prompt_pointmap_952_250411_600k/prompt_pointmap_952_250411_600k.py"}

ENGINE=${2:-"/mnt/naspersonal/lx/projects/TM/models/prompt_pointmap_952_250411_600k/quantization/convert2onnx/trex_outputs/quant_model.onnx.engine"}

OUTDIR=${3:-"None"}

echo $ENGINE

python \
    hAlgorithm/script/quantization/eval_trt.py \
    --engine $ENGINE \
    --config $CONFIG \
    --output_dir $OUTDIR \
    --vis_step 50 \

