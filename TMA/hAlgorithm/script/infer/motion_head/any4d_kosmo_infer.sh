#!/bin/bash
# Any4D Network Inference with Scene Flow Visualization
# 
# Usage:
#   bash hAlgorithm/script/infer/any4d_kosmo_infer.sh
#
# Before running:
#   1. Set the config path to your Any4D model config
#   2. Set the data path to your input JSON file
#   3. Optionally adjust other parameters

export CUDA_VISIBLE_DEVICES=0

# ============ Configuration ============
# Path to the Any4D model config file
config=/mnt/home/tcchen/workspace/TMA/hAlgorithm/configs/open/any4d/any4d_260127_4f.py

# Checkpoint: 'latest', 'best', or full path to checkpoint
# If not set, will use checkpoint path from config file
load_from=""

# Path to input data JSON file (mf_files format)
data=/mnt/nasTeam/Kosmo/json/kosmo/20260120_8/5206_龙之梦广场3楼25min.json

# Output directory
output_dir=./results/any4d_demo_v2

# Processing resolution (max dimension)
process_res=504

# Number of frames per batch
# Any4D typically uses 50 frames for evaluation
frames=10

# Specific camera view ID to use (e.g., 0, 1, 2, ...)
# Set to empty or remove --view_id flag to use all views
view_id=1

# Maximum number of frames to process (set to limit inference time)
nums=200

# Depth key in JSON for scale computation (e.g., 'depth', 'pred_depth', 'lidar_depth')
depth_key=lidar_depth

# ============ Run Inference ============
# NOTE: AMP with fp16/float16 is now supported after fixing linear solver precision handling
# Removed: --use_amp flag disabled by default to avoid mixed precision issues
# To enable AMP (faster inference): add --use_amp --amp_dtype float16
python hAlgorithm/script/infer/motion_head/any4d_kosmo_infer.py \
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