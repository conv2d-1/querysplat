#!/usr/bin/env bash

set -e -v
ulimit -n 65535
export NCCL_TIMEOUT=3600
# export NCCL_BLOCKING_WAIT=1 
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

LOADFROM=None
RESUME="/mnt/home/tcchen/workspace/TMA-origin-dev/results/big_data/wfm_rgb_query_260526_bs1_4f_50k_hyp0.1_dl3dv0.1_waymo_kitti2_stereo4d_bigdata_v1_20260530-110658/checkpoint/iter_100000/ckpt.pth"

CONFIG="/mnt/home/tcchen/workspace/TMA-origin-dev/hAlgorithm/configs/motion_head/baseline_v4/wfm_rgb_query_260526_bs1_4f_50k_hyp0.1_dl3dv0.1_waymo_kitti2_stereo4d_bigdata_v1.py"
EXPNAME="big_data"

ACCELERATE_CONFIG_FILE="hAlgorithm/script/accelerate_config.yaml"
# ACCELERATE_CONFIG_FILE="hAlgorithm/script/accelerate_zero2_config.yaml"

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
    --exp $EXPNAME \
    --seed 2024 \
    --resume $RESUME \
    --load_from $LOADFROM \
    --mixed_precision bf16