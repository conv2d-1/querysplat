#!/bin/bash
# VDPM Network Inference with Scene Flow Visualization
# 
# VDPM is a pure image-based algorithm that predicts depth, camera poses, and
# dynamic point maps from RGB video frames alone - no external camera parameters needed.
#
# Usage:
#   bash hAlgorithm/script/infer/motion_head/vdpm_kosmo_infer.sh
#
# Before running:
#   1. Set the config path to your VDPM model config
#   2. Set the data path to your input JSON file
#   3. Optionally adjust other parameters

export CUDA_VISIBLE_DEVICES=0

# ============ Configuration ============
# Path to the VDPM model config file
config=/mnt/home/tcchen/workspace/TMA/hAlgorithm/configs/open/vdpm/vdpm_260131_50f_eval.py

# Checkpoint: 'latest', 'best', or full path to checkpoint
# If not set, will use checkpoint path from config file
load_from=""

# Path to input data JSON file (mf_files format)
data=/mnt/nasTeam/Kosmo/json/kosmo/20260120_8/5206_龙之梦广场3楼25min.json

# Output directory
output_dir=./results/vdpm_demo_20f

# Processing resolution (max dimension)
process_res=504

# Number of frames per batch
# VDPM typically uses 50 frames for evaluation
frames=20

# Specific camera view ID to use (e.g., 0, 1, 2, ...)
# Set to empty or remove --view_id flag to use all views
view_id=1

# Maximum number of frames to process (set to limit inference time)
nums=200

# ============ Run Inference ============
# NOTE: VDPM is a pure image-based model - no external camera parameters needed
# The model predicts camera poses internally
python hAlgorithm/script/infer/motion_head/vdpm_kosmo_infer.py \
    --config $config \
    --data $data \
    --output_dir $output_dir \
    --process_res $process_res \
    --frames $frames \
    --view_id $view_id \
    --nums $nums \
    --save_motion_results \
    --save_motion_3d \
    --use_amp \
    --amp_dtype bfloat16 \
    $([ -n "$load_from" ] && echo "--load_from $load_from")
    # --save_glb
    # --no_time
