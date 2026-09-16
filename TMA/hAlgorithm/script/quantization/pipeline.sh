#!/usr/bin/env bash
set -e -v

CONFIG="results/debug/sdk/prompt_pointmap_952_dec4_head7_base_50k_20250714-151943/prompt_pointmap_952_dec4_head7_base_50k.py"
CKPT_PATH="results/debug/sdk/prompt_pointmap_952_dec4_head7_base_50k_20250714-151943/ckpt.pth"

# CONFIG_BASE=$(basename "$CONFIG" .py)

# echo "CONFIG_BASE : $CONFIG_BASE"

# #convert to onnx only
# bash hAlgorithm/script/quantization/convert2onnx.sh $CONFIG $CKPT_PATH

BASE_DIR=$(dirname "$CONFIG")

bash hAlgorithm/script/quantization/quantization.sh $CONFIG $CKPT_PATH
QUANT_FOLDER_PATTERN=$BASE_DIR/quantization
QAUNT_FOLDER=$(ls -td "$QUANT_FOLDER_PATTERN"*/ 2>/dev/null | head -n 1)
echo "Latest quantization directory: $QAUNT_FOLDER"

QUANT_CKPT_PATH=$QAUNT_FOLDER/ckpt_quant.pth
bash hAlgorithm/script/quantization/convert2onnx.sh $CONFIG $QUANT_CKPT_PATH
ONNX_FOLDER_PATTERN=$QAUNT_FOLDER/convert2onnx
ONNX_FOLDER=$(ls -td "$ONNX_FOLDER_PATTERN"*/ 2>/dev/null | head -n 1)
echo "Latest onnx directory: $ONNX_FOLDER"

# ONNX_NAME="quant_model.onnx"
# bash hAlgorithm/script/quantization/trex_profiler.sh $ONNX_FOLDER $ONNX_NAME

# ENGINE_NAME=${ONNX_NAME}.engine
# ENGINE_PATH=$ONNX_FOLDER/trex_outputs/$ENGINE_NAME
# bash hAlgorithm/script/quantization/eval_trt.sh $CONFIG $ENGINE_PATH
