#!/usr/bin/env bash

set -e -v
ulimit -n 65535
export NCCL_TIMEOUT=3600
# export NCCL_BLOCKING_WAIT=1 
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

export CUDA_VISIBLE_DEVICES=0

ACCELERATE_CONFIG_FILE="hAlgorithm/script/accelerate_config.yaml"

default_ip="127.0.0.1"

NNODES=${1:-1}
NUMPROCESSES=${2:-1}
RANK=${3:-0}
MASTER_ADDR=${4:-$default_ip}
PORT=$((12345+$RANDOM)) # 每次launch自动避开上一次的port地址
MASTER_PORT=${5:-$PORT}

echo "NNODES" $NNODES "NUMPROCESSES" $NUMPROCESSES "RANK" $RANK "MASTER_ADDR" $MASTER_ADDR "MASTER_PORT" $MASTER_PORT

#

OUTDIR="./results/"
EXPNAME="mvs"

# 第一步, MV 推理深度和点云
# CONFIG="results_net/vggt_250709/rc_vggt_250707_obj_infini_mvpdcg_bs1_8f_20250709-013716/rc_vggt_250707_obj_infini_mvpdcg_bs1_8f_backup.py"
CONFIG="results_net/vggt_250715/rc_vggt_250715_obj_infini_mvpdcg_bs1_8f_20250715-121756/rc_vggt_250715_obj_infini_mvpdcg_bs1_8f_backup.py"
LOADFROM="latest"

TESTCONFIG="hAlgorithm/configs/finetune_gs_v1.0/dataset_configs/mv_ios.yaml"

DATA="/mnt/netdata/Team/AI/datasets/TMD/iphone_data_v2/json/arkit_json/20250725.json"
META="/mnt/netdata/Team/AI/datasets/TMD/iphone_data_v2/json/arkit_json/20250725_metadata.json"
mf_scene_sampling_strategy=all

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
    --exp $EXPNAME/mv \
    --seed 2024 \
    --load_from $LOADFROM \
    --mixed_precision fp16 \
    --test \
    --trainer.test_num_workers 2 \
    --trainer.dist_test False \
    --trainer.select_val_dataset None \
    --test_data $TESTCONFIG \
    --only_save \
    --model.model.gaussian_head None \
    --model.model.mv_point_head None \
    --data.basic.data_path $DATA \
    --data.basic.meta_json $META \
    --data.basic.meta_json_split trainval \
    --data.basic.mf_scene_sampling_strategy $mf_scene_sampling_strategy \



# 第二步, FFGS 生成初始化 GS
TESTCONFIG="hAlgorithm/configs/finetune_gs_v1.0/dataset_configs/ffgs_ios.yaml"

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
    --exp $EXPNAME/ffgs \
    --seed 2024 \
    --load_from $LOADFROM \
    --mixed_precision fp16 \
    --test \
    --trainer.test_num_workers 10 \
    --trainer.dist_test False \
    --trainer.select_val_dataset None \
    --test_data $TESTCONFIG \
    --test_vis \
    --data.basic.data_path $DATA \
    --data.basic.meta_json $META \
    --data.basic.meta_json_split val \
    --data.basic.mf_scene_sampling_strategy $mf_scene_sampling_strategy \
    --model.save_glb_results False \
    --model.save_local2glb_results False \
    --model.save_filtered_results False \

