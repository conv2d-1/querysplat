#!/usr/bin/env bash
# WFMQueryPipeline: dense motion mask inference via query-level motion_mask head.
#
# Unlike mv_query_pair2_motion_heatmap_infer.sh (warp3d_delta heatmap), this script
# reads per-query motion_mask logits from query_pair_decoder and visualizes binary masks.
#
# Usage:
#   bash hAlgorithm/script/infer/motion_head/mv_query_motion_mask_infer.sh
#
#   encoder_window=8 bash ...        # override window size
#   mask_logit_threshold=-1.0 bash ...   # lower → higher recall (prob≈0.27)
#   WINDOWS="4 8" bash .../mv_query_window_motion_mask_sweep.sh   # sweep windows

set -euo pipefail

infer_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$(cd "${infer_script_dir}/../../../.." && pwd)" || exit 1

: "${ACC_PYTHON:=/mnt/home/tcchen/miniforge3/envs/acc/bin/python}"
if [[ ! -x "${ACC_PYTHON}" ]]; then
    echo "ACC_PYTHON is not executable: ${ACC_PYTHON}" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ============ Model (wfm_rgb_query fisheye DA3-BASE + query motion_mask head) ============
MODEL_DIR="${MODEL_DIR:-/mnt/home/tcchen/workspace/TMA-origin-dev/results/baseline_v4/wfm_rgb_query_fisheye_da3_base_finetune_query_motion_mask_20260626-154247}"
CONFIG="${CONFIG:-${MODEL_DIR}/wfm_rgb_query_fisheye_da3_base_finetune_query_motion_mask_backup.py}"
LOADFROM="${LOADFROM:-${MODEL_DIR}/checkpoint/latest/ckpt.pth}"

# ============ Data (Kosmo 1816_室外魔方10min, fisheye only) ============
# cam8 / cam9 are fisheye in this scene; default cam8.
DATA_ROOT="${DATA_ROOT:-/mnt/nasTeam2/Kosmo/processed_data/kosmo2/20260527_0056/1816_室外魔方10min}"
data="${data:-${DATA_ROOT}/raw_data/data.json}"
view_id="${view_id:-8}"
nums="${nums:-200}"
sensor_size="6.43 4.87"
fisheye="${fisheye:-true}"
: "${sequence_dir:=}"
: "${sequence_dirs:=}"

output_dir="${output_dir:-./results/query_motion_mask_mofang}"
exp_name="${exp_name:-1816_室外魔方10min_fisheye_cam${view_id}_query_mask}"

# ============ Processing ============
process_res=840
max_frames="${max_frames:-${nums}}"
frame_sampling="${frame_sampling:-sequential}"
scale_mode=global
encoder_window="${encoder_window:-4}"
pair_schedule="${pair_schedule:-adjacent}"
dense_downsample=4
dense_query_batch_size=65536
mask_aggregation=max
# Dynamic if motion_mask logit >= threshold (sigmoid(0)=0.5). Lower → higher recall.
mask_logit_threshold="${mask_logit_threshold:--0.5}"

# ============ Precision ============
use_amp=true
amp_dtype=float16
no_time=true

# ============ Visualization ============
overlay_alpha=0.55
gif_fps=4.0

if [[ "${fisheye}" != "true" ]]; then
    echo "This script is configured for fisheye inference (cam8/cam9). Set fisheye=true." >&2
    exit 2
fi
if [[ "${view_id}" != "8" && "${view_id}" != "9" ]]; then
    echo "Fisheye view_id should be 8 or 9 for 1816_室外魔方10min (got view_id=${view_id})." >&2
    exit 2
fi

extra=()
if [[ "${use_amp}" == "true" ]]; then
    extra+=(--use_amp --amp_dtype "${amp_dtype}")
fi
if [[ "${fisheye}" == "true" ]]; then
    extra+=(--fisheye --sensor_size ${sensor_size})
fi
if [[ "${no_time}" == "true" ]]; then
    extra+=(--no_time)
fi

run_one() {
    local seq_path="$1"
    local out_root="$2"
    echo "======== query motion_mask | fisheye cam${view_id} | sequence=${seq_path:-JSON} → ${out_root} ========"
    cmd=(
        "${ACC_PYTHON}" hAlgorithm/script/infer/motion_head/mv_query_motion_mask_infer.py
        --config "${CONFIG}"
        --load_from "${LOADFROM}"
        --output_dir "${out_root}"
        --process_res "${process_res}"
        --max_frames "${max_frames}"
        --frame_sampling "${frame_sampling}"
        --scale_mode "${scale_mode}"
        --encoder_window "${encoder_window}"
        --pair_schedule "${pair_schedule}"
        --dense_downsample "${dense_downsample}"
        --dense_query_batch_size "${dense_query_batch_size}"
        --mask_aggregation "${mask_aggregation}"
        --mask_logit_threshold "${mask_logit_threshold}"
        --overlay_alpha "${overlay_alpha}"
        --gif_fps "${gif_fps}"
        --save_gif
    )
    if [[ -n "${seq_path}" ]]; then
        cmd+=(--sequence_dir "${seq_path}")
    elif [[ -n "${data}" ]]; then
        cmd+=(--data "${data}" --view_id "${view_id}" --nums "${nums}")
    fi
    cmd+=("${extra[@]}")
    "${cmd[@]}"
}

base_out="${output_dir}/${exp_name}"

if [[ -n "${sequence_dirs}" ]]; then
    read -r -a _seq_list <<< "${sequence_dirs}"
    for seq in "${_seq_list[@]}"; do
        [[ -n "${seq}" ]] || continue
        stem="$(basename "${seq}")"
        run_one "${seq}" "${base_out}/${stem}"
    done
elif [[ -n "${data}" ]]; then
    run_one "" "${base_out}"
elif [[ -n "${sequence_dir}" ]]; then
    stem="$(basename "${sequence_dir}")"
    run_one "${sequence_dir}" "${base_out}/${stem}"
else
    echo "Set sequence_dir, sequence_dirs, or data." >&2
    exit 2
fi

echo ""
echo "Motion mask outputs under:"
find "${base_out}" -path '*/motion_mask_overlay_win*.gif' -type f 2>/dev/null | sort || true
find "${base_out}" -path '*/motion_mask_overlay.gif' -type f 2>/dev/null | sort || true
find "${base_out}" -path '*/motion_masks/*.png' -type f 2>/dev/null | head -10 || true

unset infer_script_dir
