#!/usr/bin/env bash

set -e -v
ulimit -n 65535
export NCCL_TIMEOUT=3600
# export NCCL_BLOCKING_WAIT=1 
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

export CUDA_VISIBLE_DEVICES=0
# python hAlgorithm/script/auto_mem/clear_gpu.py [$CUDA_VISIBLE_DEVICES]

OUTDIR="./results/"
LOADFROM=None
RESUME=None

EXPNAME="mvs"

# CONFIG="hAlgorithm/configs/finetune_gs_v1.0/250714_finetune2_gs_mv_ios_dense5k.py"
CONFIG="hAlgorithm/configs/finetune_gs_v1.0/250714_finetune2_gs_mv_ios_depth_dense5k.py"

META="/mnt/netdata/Team/AI/datasets/TMD/iphone_data_v2/json/arkit_json/20250725_metadata.json"

EXPNAME="mvs3_250725_maskeval_debug_render"
DATA=results_net/mvs3_250725_maskeval/mv/rc_vggt_250715_obj_infini_mvpdcg_bs1_8f_backup_20250726-113830/outputs/iter_000001/ios/data_info_with_depth.json
PRE=results_net/mvs3_250725_maskeval/ffgs/rc_vggt_250715_obj_infini_mvpdcg_bs1_8f_backup_20250726-115146/visualization/iter_000001/ios/gaussians
# Index=(4 4)
Index=(0 31)

ACCELERATE_CONFIG_FILE="hAlgorithm/script/accelerate_config.yaml"

default_ip="127.0.0.1"

NNODES=${1:-1} 
NUMPROCESSES=${2:-1}
RANK=${3:-0}
MASTER_ADDR=${4:-$default_ip}
PORT=$((12345+$RANDOM)) # 每次launch自动避开上一次的port地址
MASTER_PORT=${5:-$PORT}

echo "NNODES" $NNODES "NUMPROCESSES" $NUMPROCESSES "RANK" $RANK "MASTER_ADDR" $MASTER_ADDR "MASTER_PORT" $MASTER_PORT


for i in $(eval echo "{${Index[0]}..${Index[1]}}")
do
    echo "Item $i"
    formatted_i=$(printf "%06d" $i)
    CURR_PRE="$PRE/gaussians_${formatted_i}.ply"
    echo $CURR_PRE

    accelerate launch \
        --config_file $ACCELERATE_CONFIG_FILE \
        --num_machines $NNODES \
        --main_process_ip $MASTER_ADDR \
        --main_process_port $MASTER_PORT \
        --num_processes $NUMPROCESSES \
        --machine_rank $RANK \
        hAlgorithm/script/train/train.py \
        --output_dir $OUTDIR \
        --config $CONFIG \
        --exp "$EXPNAME/ftgs/$i" \
        --seed 2024 \
        --data.basic.data_path $DATA \
        --data.basic.meta_json $META \
        --data.basic.mf_scene None \
        --data.basic.mf_scene_sampling_strategy index:$i \
        --model.model.gaussian_parameters.pretrain_ply $CURR_PRE \
        
done