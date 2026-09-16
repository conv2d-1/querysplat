#!/usr/bin/env bash

set -e -v

export CUDA_VISIBLE_DEVICES=0

CONFIG=${1:-"/mnt/netdata/Team/AI/SDK/SV_Flow_Kosmo_v1.0/deploy/20251119/svb_flow_worope_ppt_max_scale_840_bs2_4f_sky_svd_total_150k_20251119-113513/svb_flow_worope_ppt_max_scale_840_bs2_4f_sky_svd_total_150k.py"}
ENGINE=${2:-"/mnt/netdata/Team/AI/SDK/SV_Flow_Kosmo_v1.0/deploy/20251119/svb_flow_worope_ppt_max_scale_840_bs2_4f_sky_svd_total_150k_20251119-113513/svb_flow_worope_ppt_max_scale_840_bs2_4f_sky_svd_total_150k.engine"}
ONNX="/mnt/netdata/Team/AI/SDK/SV_Flow_Kosmo_v1.0/deploy/20251119/svb_flow_worope_ppt_max_scale_840_bs2_4f_sky_svd_total_150k_20251119-113513/svb_flow_worope_ppt_max_scale_840_bs2_4f_sky_svd_total_150k.onnx"

OUTDIR=${3:-"./results/"}
TESTCONFIG="/mnt/netdata/Team/AI/SDK/SV_Kosmo_v1.0/deploy/kosmo_share.yaml"
data_path="/mnt/nasTeam/Kosmo/json/kosmo/20251115_left_test_split.json"
python \
    hAlgorithm/script/tensorrt/sv_flow_kosmos_trt_infer.py \
    --config $CONFIG \
    --output_dir $OUTDIR \
    --seed 2024 \
    --engine $ENGINE \
    --onnx $ONNX\
    --test_data $TESTCONFIG \
    --data.basic.data_path $data_path \
    --data.basic.mf_view_ids None --data.basic.mf_to_sf True \
    --trainer.test_num_workers 8 \
    --save_outputs \
    --data.basic.type "hAlgorithm.datasets_mv.kosmo_dataset.BaseDatasetMV" \
    # --debug