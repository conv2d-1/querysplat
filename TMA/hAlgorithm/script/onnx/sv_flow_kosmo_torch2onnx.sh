#!/usr/bin/env bash

set -e -v

export CUDA_VISIBLE_DEVICES=0

OUTDIR="./results/"


CONFIG="hAlgorithm/configs/custom/svflow/kosmo_flow/total_1114/svb_flow_worope_ppt_max_scale_840_bs2_4f_sky_svd_total_150k.py"
LOADFROM="results/2511/svflow_kosmo/total_1114/svb_flow_worope_ppt_max_scale_840_bs2_4f_sky_svd_total_150k_20251114-183524/checkpoint/best/ckpt.pth"


python \
    hAlgorithm/script/onnx/sv_flow_kosmo_torch2onnx.py \
    --config $CONFIG \
    --output_dir $OUTDIR \
    --seed 2024 \
    --load_from $LOADFROM \
    --fp16
    # --test \
    # --test_vis \
    # --save_outputs \
    # --test_data $TESTCONFIG \

