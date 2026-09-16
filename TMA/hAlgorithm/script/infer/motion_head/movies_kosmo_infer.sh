#!/bin/bash
# Movies Network Inference with Motion Visualization
# 
# Usage:
#   bash hAlgorithm/script/infer/movies.sh
#
# Before running:
#   1. Set the config path to your movies model config
#   2. Set the data path to your input JSON file
#   3. Optionally adjust other parameters

export CUDA_VISIBLE_DEVICES=0

# ============ Configuration ============
# Path to the movies model config file
config=/mnt/home/tcchen/workspace/TMA/results/expt_big_data_with_cam_loss_20260304/movies/movies_20260305-112237/movies_backup.py

# Checkpoint: 'latest', 'best', or full path to checkpoint
load_from=latest

# Processing resolution (max dimension)
process_res=518

# Number of frames per batch (determines temporal window)
# Now flexible - can differ from training view_num
frames=10

# Specific camera view ID to use (e.g., 0, 1, 2, ...)
# Set to empty or remove --view_id flag to use all views
view_id=1

# Maximum number of frames to process (set to limit inference time)
nums=200

# Depth key in JSON for scale computation (e.g., 'depth', 'pred_depth', 'lidar_depth')
depth_key=lidar_depth


data=/mnt/nasTeam/Kosmo/json/kosmo/20260120_8/5206_龙之梦广场3楼25min.json
# Output directory
output_dir=./results/movies_demo_260309/lzm

# ============ Run Inference ============
python hAlgorithm/script/infer/motion_head/movies_kosmo_infer.py \
    --config $config \
    --load_from $load_from \
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
    --amp_dtype float16
    # --no_time


data=/mnt/nasTeam/Kosmo/json/kosmo/20260107_8_天公2/9267_天山公园秋千.json
output_dir=./results/movies_demo_260309/tianshan_qiuqian

# ============ Run Inference ============
python hAlgorithm/script/infer/motion_head/movies_kosmo_infer.py \
    --config $config \
    --load_from $load_from \
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
    --amp_dtype float16
    # --no_time


data=/mnt/nasTeam/Kosmo/json/kosmo/20260107_8_天公2/5187_天山公园儿童娱乐器材.json
output_dir=./results/movies_demo_260309/tianshan_child

python hAlgorithm/script/infer/motion_head/movies_kosmo_infer.py \
    --config $config \
    --load_from $load_from \
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
    --amp_dtype float16