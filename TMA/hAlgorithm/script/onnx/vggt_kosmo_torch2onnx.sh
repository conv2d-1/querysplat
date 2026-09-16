#!/usr/bin/env bash

set -e -v

export CUDA_VISIBLE_DEVICES=3

OUTDIR="./results/"

ONNX_PATH="SDK/Normal/deploy/20251105/mvfr_rgb_normal_251028_bs1_1f4f_100k_vggtpre_trainall_840_nums4_20251031-101922/mvfr_rgb_normal_251028_bs1_1f4f_100k_vggtpre_trainall_840_nums4.onnx"

CONFIG="SDK/Normal/deploy/20251105/mvfr_rgb_normal_251028_bs1_1f4f_100k_vggtpre_trainall_840_nums4_20251031-101922/mvfr_rgb_normal_251028_bs1_1f4f_100k_vggtpre_trainall_840_nums4.py"
LOADFROM="SDK/Normal/MVFR/20251105/mvfr_rgb_normal_251028_bs1_1f4f_100k_vggtpre_trainall_840_nums4_20251031-101922/checkpoint/latest/ckpt.pth"

python \
    hAlgorithm/script/onnx/vggt_kosmo_torch2onnx.py \
    --config $CONFIG \
    --output_dir $OUTDIR \
    --seed 2024 \
    --load_from $LOADFROM \
    --fp16 \
    #--test --test_vis # --onnx $ONNX_PATH
    # --test \
    # --test_vis \
    # --save_outputs \
    # --test_data $TESTCONFIG \

