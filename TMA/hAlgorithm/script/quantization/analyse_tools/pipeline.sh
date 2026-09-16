#!/usr/bin/env bash

set -e -v

export CUDA_VISIBLE_DEVICES=0

bash hAlgorithm/script/quantization/analyse_tools/layer_quant_analyse_pytorch.sh
bash hAlgorithm/script/quantization/analyse_tools/amax_searching.sh 