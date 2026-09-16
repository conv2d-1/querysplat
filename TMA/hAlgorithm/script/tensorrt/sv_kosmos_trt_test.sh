#!/usr/bin/env bash

set -e -v

export CUDA_VISIBLE_DEVICES=0

CONFIG=${1:-"/mnt/netdata/Team/AI/SDK/SV_Kosmo_v1.0/deploy/20251117/svb_ppt_all2_251110_840_bs4_normal_sky_300k_norm1_fp16/svb_ppt_all2_251110_840_bs4_normal_sky_300k_norm1.py"}
ENGINE=${2:-"/mnt/netdata/Team/AI/SDK/SV_Kosmo_v1.0/deploy/20251117/svb_ppt_all2_251110_840_bs4_normal_sky_300k_norm1_fp16/svb_ppt_all2_251110_840_bs4_normal_sky_300k_norm1.engine"}
ONNX="/mnt/netdata/Team/AI/SDK/SV_Kosmo_v1.0/deploy/20251117/svb_ppt_all2_251110_840_bs4_normal_sky_300k_norm1_fp16/svb_ppt_all2_251110_840_bs4_normal_sky_300k_norm1.onnx"

OUTDIR=${3:-"./results/"}

python \
    hAlgorithm/script/tensorrt/sv_kosmos_trt_test.py \
    --config $CONFIG \
    --output_dir $OUTDIR \
    --seed 2024 \
    --save_vis \
    --save_ds 50 \
    --engine $ENGINE \
    --onnx $ONNX\
    # --save_outputs \
    # --test_data $TESTCONFIG \

