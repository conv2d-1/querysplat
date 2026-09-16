#!/usr/bin/env bash

set -e -v

export CUDA_VISIBLE_DEVICES=0

CONFIG=${1:-"/mnt/netdata/Team/AI/SDK/SV_Kosmo_v1.0/deploy/20251117/svb_ppt_all2_251110_840_bs4_normal_sky_300k_norm1_fp16/svb_ppt_all2_251110_840_bs4_normal_sky_300k_norm1.py"}
ENGINE=${2:-"/mnt/netdata/Team/AI/SDK/SV_Kosmo_v1.0/deploy/20251117/svb_ppt_all2_251110_840_bs4_normal_sky_300k_norm1_fp16/svb_ppt_all2_251110_840_bs4_normal_sky_300k_norm1.engine"}
ONNX="/mnt/netdata/Team/AI/SDK/SV_Kosmo_v1.0/deploy/20251117/svb_ppt_all2_251110_840_bs4_normal_sky_300k_norm1_fp16/svb_ppt_all2_251110_840_bs4_normal_sky_300k_norm1.onnx"

OUTDIR=${3:-"./results/"}
TESTCONFIG="/mnt/netdata/Team/AI/SDK/SV_Kosmo_v1.0/deploy/kosmo_share.yaml"
data_path="/mnt/nasTeam/Kosmo/json/kosmo/20251115_left_test_split.json"
python \
    hAlgorithm/script/tensorrt/sv_kosmos_trt_infer.py \
    --config $CONFIG \
    --output_dir $OUTDIR \
    --seed 2024 \
    --engine $ENGINE \
    --onnx $ONNX\
    --test_data $TESTCONFIG \
    --data.basic.data_path $data_path \
    --data.basic.mf_view_ids None \
    --trainer.test_num_workers 8 \
    --save_outputs \
    --data.basic.type "hAlgorithm.datasets_mv.kosmo_dataset.BaseDatasetMV" \
    # --debug