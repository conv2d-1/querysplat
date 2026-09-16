#!/usr/bin/env bash

set -e -v

export CUDA_VISIBLE_DEVICES=0

OUTDIR="./results/"

# CONFIG="hAlgorithm/configs/custom/svflow/kosmo_flow/svs_flow_worope_ppt_max_scale_840_bs1_4f_syn_base_50k.py"
CONFIG="hAlgorithm/configs/custom/sv_kosmo/svb_ppt_all2_251110_840_bs4_normal_sky_300k_norm1.py"
LOADFROM="SDK/SV_Kosmo_v1.0/20251117/svb_ppt_all2_251110_840_bs4_normal_sky_300k_norm1_20251116-190733/checkpoint/best/ckpt.pth"

CONFIG="hAlgorithm/configs/custom/sv_kosmo/svs_ppt_all2_251110_840_bs4_normal_sky_100k_norm1.py"
LOADFROM="SDK/SV_Kosmo_v1.0/20251117/svs_ppt_all2_251110_840_bs4_normal_sky_100k_norm1_20251115-125244/checkpoint/latest/ckpt.pth"

python \
    hAlgorithm/script/onnx/sv_kosmo_torch2onnx.py \
    --config $CONFIG \
    --output_dir $OUTDIR \
    --seed 2024 \
    --load_from $LOADFROM \
    --fp16
    # --test \
    # --test_vis \
    # --save_outputs \
    # --test_data $TESTCONFIG \

