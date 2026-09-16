#!/usr/bin/env bash

set -e -v

export CUDA_VISIBLE_DEVICES=0

MODEL_DIR=${1:-"/mnt/naspersonal/lx/projects/TM/models/prompt_pointmap_952_250411_600k/quantization/convert2onnx/"}
ONNX_NAME=${2:-"quant_model.onnx"}

TREX_OUTPUT_DIR=${MODEL_DIR}/trex_outputs
mkdir -p ${TREX_OUTPUT_DIR}

ONNX_PATH=${MODEL_DIR}/${ONNX_NAME}

python3 hAlgorithm/script/quantization/trex_utils/process_engine.py ${ONNX_PATH} ${TREX_OUTPUT_DIR} fp16 #int8 best

ENGINE_PATH=${TREX_OUTPUT_DIR}/${ONNX_NAME}.engine
echo ${ENGINE_PATH}

python3 hAlgorithm/script/quantization/trex_profiler.py --engine=${ENGINE_PATH}

python3 hAlgorithm/script/tensorrt/profiler.py $ENGINE_PATH