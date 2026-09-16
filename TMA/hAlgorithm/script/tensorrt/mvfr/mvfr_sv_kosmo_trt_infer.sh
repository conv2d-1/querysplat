#!/usr/bin/env bash

set -e -v

export CUDA_VISIBLE_DEVICES=0

CONFIG=${1:-"/mnt/netdata/Team/AI/SDK/SV_Kosmo_v1.0/deploy/20251229/sv_da3b2_all_251207_bs4_100k_stage_kosmo_finetune_sky_svd_100k/sv_da3b2_all_251207_bs4_100k_stage_kosmo_finetune_sky_svd_100k.py"}
ENGINE=${2:-"/mnt/netdata/Team/AI/SDK/SV_Kosmo_v1.0/deploy/20251229/sv_da3b2_all_251207_bs4_100k_stage_kosmo_finetune_sky_svd_100k/sv_da3b2_all_251207_bs4_100k_stage_kosmo_finetune_sky_svd_100k.engine"}
ONNX="/mnt/netdata/Team/AI/SDK/SV_Kosmo_v1.0/deploy/20251229/sv_da3b2_all_251207_bs4_100k_stage_kosmo_finetune_sky_svd_100k/sv_da3b2_all_251207_bs4_100k_stage_kosmo_finetune_sky_svd_100k.onnx"

OUTDIR=${3:-"./results/"}
TESTCONFIG="/mnt/netdata/Team/AI/SDK/SV_Kosmo_v1.0/deploy/kosmo_share.yaml"
data_path="/mnt/nasTeam/Kosmo/processed_data/kosmo/20251231_精品4/7764_D5/raw_data/data_hgaussian_v1.0.0_normal_sv_v251225_sfm_colmap_v1.0.0_mask_mos_v1.0.0_depth_sv_v251225@20251231100000.json"

python \
    hAlgorithm/script/tensorrt/mvfr/mvfr_sv_kosmo_trt_infer.py \
    --config $CONFIG \
    --output_dir $OUTDIR \
    --seed 2024 \
    --onnx $ONNX\
    --test_data $TESTCONFIG \
    --data.basic.data_path $data_path \
    --data.basic.mf_view_ids None \
    --trainer.test_num_workers 8 \
    --data.basic.type "hAlgorithm.datasets_mv.kosmo_dataset.BaseDatasetMV" --data.basic.kosmo_prompt_size 840 \
    --data.val_basic.mf_to_sf True \
    --engine $ENGINE \
    --save_outputs \
    # --debug --data.val_basic.sampling_strategy "first:5"