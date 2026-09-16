#!/usr/bin/env bash
set -e -v

CONFIG="/mnt/naspersonal/lx/projects/TM/models/prompt_pointmap_952_250411_600k/QAT_most_quant_60w/prompt_pointmap_952_250411_600k_qat.py"
CKPT_PATH="/mnt/naspersonal/lx/projects/TM/models/prompt_pointmap_952_250411_600k/QAT_most_quant_60w/ckpt.pth"

BASE_DIR=$(dirname "$CONFIG")

bash hAlgorithm/script/quantization/convert2onnx.sh $CONFIG $CKPT_PATH
ONNX_FOLDER_PATTERN=$BASE_DIR/convert2onnx
ONNX_FOLDER=$(ls -td "$ONNX_FOLDER_PATTERN"*/ 2>/dev/null | head -n 1)
echo "Latest onnx directory: $ONNX_FOLDER"

ONNX_NAME="quant_model.onnx"
bash hAlgorithm/script/quantization/trex_profiler.sh $ONNX_FOLDER $ONNX_NAME

ENGINE_NAME=${ONNX_NAME}.engine
ENGINE_PATH=$ONNX_FOLDER/trex_outputs/$ENGINE_NAME
bash hAlgorithm/script/quantization/eval_trt.sh $CONFIG $ENGINE_PATH
