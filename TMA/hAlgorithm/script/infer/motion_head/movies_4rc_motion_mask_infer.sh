#!/bin/bash
# MOVIES + MotionHead4RC Pairwise Motion Mask Inference
#
# Runs consecutive-pair inference to produce per-frame motion masks.
# Each pair: frame_i (src) → frame_{i+1} (tgt), last frame reversed.
#
# Usage:
#   bash hAlgorithm/script/infer/motion_head/movies_4rc_motion_mask_infer.sh

export CUDA_VISIBLE_DEVICES=2

# ============ Configuration ============
# Path to the MOVIES + 4RC decoder config (point to your trained experiment)
config=/mnt/home/tcchen/workspace/TMA/results_nasTeam2/debug/movies_4rc_v5_stable_mtl/movies_4rc_v5_stable_mtl_20260316-105718/movies_4rc_v5_stable_mtl_backup.py

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

# Mask mode: 'adaptive' (recommended) or 'threshold' (original fixed-threshold)
mask_mode=adaptive

# [adaptive] Temporal sliding-window half-width (frames).
# Larger → more temporal smoothing, fewer flickering artifacts.
temporal_window=5

# [adaptive] Minimum connected-component area in pixels to keep.
# Smaller blobs (noise / background jitter) are discarded.
min_component_area=300

# [adaptive] Percentile used for per-frame normalization (default 95).
# A lower value makes the threshold stricter (fewer false positives).
norm_percentile=95

# [threshold mode only] Fixed L2 norm threshold — only used when mask_mode=threshold
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
output_dir=./results_nasTeam2/motion_mask_demo_v2/movies_lzm

python hAlgorithm/script/infer/motion_head/movies_4rc_motion_mask_infer.py \
    --config $config \
    --load_from $load_from \
    --data "$data" \
    --output_dir $output_dir \
    --process_res $process_res \
    --nums $nums \
    --view_id $view_id \
    --depth_key $depth_key \
    --mask_mode $mask_mode \
    --temporal_window $temporal_window \
    --min_component_area $min_component_area \
    --norm_percentile $norm_percentile \
    --motion_threshold $motion_threshold \
    --overlay_alpha $overlay_alpha \
    --gif_fps $gif_fps \
    --window_size $window_size \
    --save_gif \
    --use_amp \
    --amp_dtype float16
