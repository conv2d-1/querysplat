#!/bin/bash
# DA3 + MotionHead4RC Pairwise Motion Mask Inference
#
# Runs consecutive-pair inference to produce per-frame motion masks.
# Each pair: frame_i (src) → frame_{i+1} (tgt), last frame reversed.
#
# Usage:
#   bash hAlgorithm/script/infer/motion_head/da3_4rc_motion_mask_infer.sh

export CUDA_VISIBLE_DEVICES=0

# ============ Configuration ============
# Path to the DA3 + 4RC decoder config
config=/mnt/home/tcchen/workspace/TMA/results_nasTeam2/expt_4rc_decoder/da3_with_4rc_decoder/da3_with_4rc_decoder_20260310-151423/da3_with_4rc_decoder_backup.py

# Checkpoint: 'latest', 'best', or full path
load_from=latest

# Processing resolution (max dimension, must be divisible by 14)
process_res=518

# Max frames to process (set to limit inference time; remove for all frames)
nums=200

# Camera view ID (set to specific view, or remove --view_id for all views)
view_id=1

# Depth key in JSON for scale computation
depth_key=lidar_depth

# L2 norm threshold for motion mask (tune for your scene)
motion_threshold=0.05

# Overlay transparency
overlay_alpha=0.5

# GIF frame rate
gif_fps=4.0

# Encoder chunk size: all frames in a chunk share one backbone pass;
# only the lightweight 4RC decoder runs per source frame.
window_size=20


# ============ Scene 1 ============
data=/mnt/nasTeam/Kosmo/json/kosmo/20260120_8/5206_龙之梦广场3楼25min.json
output_dir=./results/motion_mask_demo/lzm

python hAlgorithm/script/infer/motion_head/da3_4rc_motion_mask_infer.py \
    --config $config \
    --load_from $load_from \
    --data "$data" \
    --output_dir $output_dir \
    --process_res $process_res \
    --nums $nums \
    --view_id $view_id \
    --depth_key $depth_key \
    --motion_threshold $motion_threshold \
    --overlay_alpha $overlay_alpha \
    --gif_fps $gif_fps \
    --window_size $window_size \
    --save_gif \
    --use_amp \
    --amp_dtype float16


# ============ Scene 2 (uncomment to run) ============
# data=/mnt/nasTeam/Kosmo/json/kosmo/20260107_8_天公2/9267_天山公园秋千.json
# output_dir=./results/motion_mask_demo/tianshan_qiuqian
#
# python hAlgorithm/script/infer/motion_head/da3_4rc_motion_mask_infer.py \
#     --config $config \
#     --load_from $load_from \
#     --data "$data" \
#     --output_dir $output_dir \
#     --process_res $process_res \
#     --nums $nums \
#     --view_id $view_id \
#     --depth_key $depth_key \
#     --motion_threshold $motion_threshold \
#     --overlay_alpha $overlay_alpha \
#     --gif_fps $gif_fps \
#     --window_size $window_size \
#     --save_gif \
#     --use_amp \
#     --amp_dtype float16
