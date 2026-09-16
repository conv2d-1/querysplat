#!/bin/bash
# MVFR Motion Network Inference with Scene Flow Visualization
# 
# MVFR Motion requires external camera parameters (intrinsics and extrinsics).
# This is NOT a pure image-based model like VDPM.
#
# Usage:
#   bash hAlgorithm/script/infer/motion_head/mvfr_motion_kosmo_infer.sh
#
# Before running:
#   1. Set the config path to your MVFR Motion model config
#   2. Set the data path to your input JSON file
#   3. Optionally adjust other parameters

export CUDA_VISIBLE_DEVICES=0

# ============ Configuration ============
# Path to the MVFR Motion model config file
config=/mnt/home/tcchen/workspace/TMA/hAlgorithm/configs/motion_head/motion_head_da3l_b1_8gpus.py

# Checkpoint: 'latest', 'best', or full path to checkpoint
# If not set, will use checkpoint path from config file
load_from="/mnt/home/tcchen/workspace/TMA/results/motion_head_da3l_b1_8gpus/motion_head_da3l_b1_8gpus_20260131-103119/checkpoint/latest/ckpt.pth"

# Path to input data JSON file (mf_files format)
data=/mnt/nasTeam/Kosmo/json/kosmo/20260107_8_天公2/9267_天山公园秋千.json

# Output directory
output_dir=./results/mvfr_motion_demo_v1

# Processing resolution (max dimension)
process_res=504

# Number of frames per batch
frames=20

# Specific camera view ID to use (e.g., 0, 1, 2, ...)
# Set to empty or remove --view_id flag to use all views
view_id=1

# Maximum number of frames to process (set to limit inference time)
nums=200

# Depth key in JSON for scale computation (e.g., 'depth', 'pred_depth', 'lidar_depth')
depth_key=lidar_depth

# ============ Run Inference ============
# NOTE: MVFRMotion requires external camera parameters (intrinsics and extrinsics)
python hAlgorithm/script/infer/motion_head/mvfr_motion_kosmo_infer.py \
    --config $config \
    --data $data \
    --output_dir $output_dir \
    --process_res $process_res \
    --frames $frames \
    --view_id $view_id \
    --nums $nums \
    --depth_key $depth_key \
    --save_motion_results \
    --save_motion_3d \
    --use_amp \
    --amp_dtype float16 \
    $([ -n "$load_from" ] && echo "--load_from $load_from")
    # --save_glb
    # --no_time

data=/mnt/nasTeam/Kosmo/json/kosmo/20260107_8_天公2/5187_天山公园儿童娱乐器材.json
output_dir=./results/mvfr_motion_demo_v2

python hAlgorithm/script/infer/motion_head/mvfr_motion_kosmo_infer.py \
    --config $config \
    --data $data \
    --output_dir $output_dir \
    --process_res $process_res \
    --frames $frames \
    --view_id $view_id \
    --nums $nums \
    --depth_key $depth_key \
    --save_motion_results \
    --save_motion_3d \
    --use_amp \
    --amp_dtype float16 \
    $([ -n "$load_from" ] && echo "--load_from $load_from")
    # --save_glb
    # --no_time

