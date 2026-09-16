#!/usr/bin/env bash

set -e -v

export CUDA_VISIBLE_DEVICES=0

OUTDIR="./results/"


CONFIG="SDK/SV_Kosmo_v1.0/20251229/sv_da3b2_all_251207_bs4_100k_stage_kosmo_finetune_sky_svd_100k_20251225-103200/sv_da3b2_all_251207_bs4_100k_stage_kosmo_finetune_sky_svd_100k_backup.py"
LOADFROM="SDK/SV_Kosmo_v1.0/20251229/sv_da3b2_all_251207_bs4_100k_stage_kosmo_finetune_sky_svd_100k_20251225-103200/checkpoint/latest/ckpt.pth"


python \
    hAlgorithm/script/onnx/mvfr_sv_kosmo_torch2onnx.py \
    --config $CONFIG \
    --output_dir $OUTDIR \
    --seed 2024 \
    --load_from $LOADFROM \
    --fp16
    # --test \
    # --test_vis \
    # --save_outputs \
    # --test_data $TESTCONFIG \

