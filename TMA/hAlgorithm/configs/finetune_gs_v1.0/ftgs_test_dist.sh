#!/usr/bin/env bash

set -e -v

export CUDA_VISIBLE_DEVICES=0
python hAlgorithm/script/auto_mem/clear_gpu.py [$CUDA_VISIBLE_DEVICES]

OUTDIR="./results/"
LOADFROM=None

EXPNAME="mvs_test"
CONFIG="hAlgorithm/configs/finetune_gs_v1.0/250714_finetune2_gs_mv_ios_depth_dense5k.py"

LOADFROM="results_net/mvs3_250725_maskeval_debug_render/ftgs/0/250714_finetune2_gs_mv_ios_depth_dense5k_20250727-215920/checkpoint/latest/ckpt.pth"
CAM="results_net/mvs3_250725_maskeval_debug_render/ftgs/0/250714_finetune2_gs_mv_ios_depth_dense5k_20250727-215920/visualization/iter_015000/ios/cameras.json"
TESTCONFIG="results_net/mvs3_250725_maskeval_debug_render/ftgs/0/250714_finetune2_gs_mv_ios_depth_dense5k_20250727-215920/configs/gs_mv_ios.yaml"

ACCELERATE_CONFIG_FILE="hAlgorithm/script/accelerate_config.yaml"

default_ip="127.0.0.1"

NNODES=${1:-1}
NUMPROCESSES=${2:-1}
RANK=${3:-0}
MASTER_ADDR=${4:-$default_ip}
PORT=$((12345+$RANDOM)) # 每次launch自动避开上一次的port地址
MASTER_PORT=${5:-$PORT}

echo "NNODES" $NNODES "NUMPROCESSES" $NUMPROCESSES "RANK" $RANK "MASTER_ADDR" $MASTER_ADDR "MASTER_PORT" $MASTER_PORT

accelerate launch \
    --config_file $ACCELERATE_CONFIG_FILE \
    --num_machines $NNODES \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --num_processes $NUMPROCESSES \
    --machine_rank $RANK \
    hAlgorithm/script/train/train.py \
    --config $CONFIG \
    --output_dir $OUTDIR \
    --exp $EXPNAME \
    --seed 2024 \
    --test \
    --trainer.test_num_workers 10 \
    --test_data $TESTCONFIG \
    --data.basic None \
    --trainer.eval_metrics.debug True \
    --trainer.cam_pretrain $CAM \
    --load_from $LOADFROM \
    # --test_vis \
    # --model.model.gaussian_parameters.pretrain_ply $PRE \
    # --trainer.eval_metrics.debug True \
    
    

