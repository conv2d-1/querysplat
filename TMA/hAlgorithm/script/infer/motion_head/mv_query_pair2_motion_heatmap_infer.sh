#!/usr/bin/env bash
# WFMQueryPipeline (MVQuery6): 2-frame-pair dense motion heatmap inference.
#
# Encodes frames in pairs (0,1), (2,3), ...; if odd frame count, adds overlapping
# pair (N-2, N-1). Each pair runs (src→tgt) and (tgt→src) pair_forward so every
# frame gets a motion heatmap.
#
# Usage:
#   bash hAlgorithm/script/infer/motion_head/mv_query_pair2_motion_heatmap_infer.sh
#
#   sequence_dir=/path/to/frames bash ...
#   data=/path/scene.json view_id=0 bash ...

set -euo pipefail

infer_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$(cd "${infer_script_dir}/../../../.." && pwd)" || exit 1

: "${ACC_PYTHON:=/mnt/home/tcchen/miniforge3/envs/acc/bin/python}"
if [[ ! -x "${ACC_PYTHON}" ]]; then
    echo "ACC_PYTHON is not executable: ${ACC_PYTHON}" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ============ Model (wfm_rgb_query DA3-BASE finetune) ============
MODEL_DIR="/mnt/home/tcchen/workspace/TMA-origin-dev/results/baseline_v4/wfm_rgb_query_260526_bs1_4f_50k_hyp0.1_dl3dv0.1_waymo_kitti2_stereo4d_bigdata_v1_da3_small_finetune_20260615-003020"
CONFIG="${CONFIG:-${MODEL_DIR}/wfm_rgb_query_260526_bs1_4f_50k_hyp0.1_dl3dv0.1_waymo_kitti2_stereo4d_bigdata_v1_da3_small_finetune_backup.py}"
LOADFROM="${LOADFROM:-${MODEL_DIR}/checkpoint/latest/ckpt.pth}"

# ============ Data (Kosmo LV大船, fisheye cam8) ============
data="${data:-/mnt/nasTeam2/Kosmo/processed_data/kosmo2/20260424_0015/3937_LV大船白天58min_0424_pointfilter/raw_data/3937_LV大船白天58min_0424_pointfilter.json}"
view_id="${view_id:-8}"
nums="${nums:-200}"
# Blender fisheye sensor size (mm) — HaSim training default
sensor_size="6.43 4.87"
fisheye=true
# RGB sequence mode: set sequence_dir and clear data="" to disable JSON
: "${sequence_dir:=}"
: "${sequence_dirs:=}"

output_dir="${output_dir:-./results/pair2_motion_heatmap}"
exp_name="${exp_name:-3937_LV大船_cam${view_id}}"

# ============ Processing ============
process_res=504
max_frames="${max_frames:-${nums}}"
frame_sampling="${frame_sampling:-sequential}"
scale_mode=global
encoder_window="${encoder_window:-4}"
pair_schedule="${pair_schedule:-adjacent}"
dense_downsample=4
dense_motion_threshold=0.02
dense_query_batch_size=65536
disp_aggregation=max

# ============ Precision ============
use_amp=true
amp_dtype=float16

# ============ Visualization (scheme C: global scale + global colormap) ============
heatmap_norm=global_log_percentile
heatmap_percentile=95
heatmap_gamma=1.2
heatmap_vmin_percentile=85
heatmap_colormap=turbo
heatmap_overlay_alpha=0.6
overlay_use_mask=true
gif_fps=4.0

extra=()
if [[ "${use_amp}" == "true" ]]; then
    extra+=(--use_amp --amp_dtype "${amp_dtype}")
fi
if [[ "${fisheye}" == "true" ]]; then
    extra+=(--fisheye --sensor_size ${sensor_size})
fi
if [[ "${no_time:-false}" == "true" ]]; then
    extra+=(--no_time)
fi

run_one() {
    local seq_path="$1"
    local out_root="$2"
    echo "======== pair2 sequence=${seq_path:-JSON} → ${out_root} ========"
    cmd=(
        "${ACC_PYTHON}" hAlgorithm/script/infer/motion_head/mv_query_pair2_motion_heatmap_infer.py
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
        --dense_motion_threshold "${dense_motion_threshold}"
        --dense_query_batch_size "${dense_query_batch_size}"
        --disp_aggregation "${disp_aggregation}"
        --heatmap_norm "${heatmap_norm}"
        --heatmap_percentile "${heatmap_percentile}"
        --heatmap_gamma "${heatmap_gamma}"
        --heatmap_vmin_percentile "${heatmap_vmin_percentile}"
        --heatmap_colormap "${heatmap_colormap}"
        --heatmap_overlay_alpha "${heatmap_overlay_alpha}"
        --gif_fps "${gif_fps}"
        --save_gif
    )
    if [[ "${overlay_use_mask}" == "true" ]]; then
        cmd+=(--overlay_use_mask)
    else
        cmd+=(--no_overlay_use_mask)
    fi
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
    # JSON mode (default: Kosmo LV大船). Override with data="" and sequence_dir=... for RGB folders.
    run_one "" "${base_out}"
elif [[ -n "${sequence_dir}" ]]; then
    stem="$(basename "${sequence_dir}")"
    run_one "${sequence_dir}" "${base_out}/${stem}"
else
    echo "Set sequence_dir, sequence_dirs, or data." >&2
    exit 2
fi

echo "Pair2 heatmap outputs under:"
find "${base_out}" -path '*/dense_heatmap_overlay_win*.gif' -type f 2>/dev/null | sort || true
find "${base_out}" -path '*/dense_heatmap_overlay.gif' -type f 2>/dev/null | sort || true
find "${base_out}" -path '*/dense_motion_heatmaps/*.png' -type f 2>/dev/null | head -10 || true

unset infer_script_dir
