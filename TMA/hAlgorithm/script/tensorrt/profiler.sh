#!/usr/bin/env bash

set -e -v

export CUDA_VISIBLE_DEVICES=0

ENGINE="./results/torch2onnx/prompt_pointmap_stage2_250217_all_modules/prompt_pointmap_stage2_250217_trt10.trt"

python hAlgorithm/script/tensorrt/profiler.py $ENGINE \

