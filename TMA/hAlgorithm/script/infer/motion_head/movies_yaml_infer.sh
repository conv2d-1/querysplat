#!/bin/bash
# Movies Network Inference with YAML Dataset Config
# 
# This script runs inference on datasets specified in YAML config files,
# similar to the evaluation pipeline but with visualization output.
#
# Usage:
#   bash hAlgorithm/script/infer/movies_yaml_infer.sh

export CUDA_VISIBLE_DEVICES=0

# ============ Configuration ============
# Path to the movies model config file
config=/mnt/home/tcchen/workspace/TMA/results/motion_head_baseline/movies_debug_20260115-205649/movies_debug.py

# Path to YAML dataset config file
# Options:
#   - val_any4d_50frames_v3.yaml (50 consecutive frames, Any4D aligned)
#   - val_any4d_50frames.yaml (50 random frames)
#   - val_motion_head_260117.yaml (original 5 frames)
yaml=hAlgorithm/configs/mv_v1.0/dataset_configs_2/val_any4d_50frames_v3.yaml

# Checkpoint: 'latest', 'best', or full path to checkpoint
load_from=latest

# Output directory
output_dir=./results/movies_yaml_infer

# Maximum number of samples per dataset (set to limit inference time)
nums=50

# Select specific datasets (comma-separated, or leave empty for all)
# Examples: "kubric4d", "pointodyssey,kubric4d", "dynamicreplica"
select_dataset="pointodyssey,"

# Number of dataloader workers
num_workers=4

# ============ Run Inference ============
python hAlgorithm/script/infer/motion_head/movies_yaml_infer.py \
    --config $config \
    --yaml $yaml \
    --load_from $load_from \
    --output_dir $output_dir \
    --nums $nums \
    --num_workers $num_workers \
    --save_motion_results \
    --save_motion_3d \
    --use_amp \
    --amp_dtype float16 \
    --use_yaml_sampler \
    ${select_dataset:+--select_dataset $select_dataset}
    # --no_time  # Uncomment to disable timestamp in output directory