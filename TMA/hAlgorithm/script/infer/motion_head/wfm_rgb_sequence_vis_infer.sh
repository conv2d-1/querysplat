#!/usr/bin/env bash
# WFM RGB sequence vis-only infer (no eval, no DA3, no 4DGS).
#
# Same vis path as test_dist.sh (--test --test_vis): WFMQueryPipeline.infer + visualize
# (motion/ + rerun_vis/).
#
# Default query_layout=center_dense_columns: middle 2/4 vertical bands dense_stride=2,
# outer bands sparse_stride=8. Override with query_layout=uniform query_stride=2 for
# a regular grid.
#
# Usage:
#   bash hAlgorithm/script/infer/motion_head/wfm_rgb_sequence_vis_infer.sh
#
#   video=/path/clip.mp4 bash ...
#   videos="/path/a.mp4 /path/b.mp4" bash ...
#   sequence_dirs="/path/bear /path/tennis" bash ...
#   max_frames=50 query_layout=uniform query_stride=2 bash ...

set -euo pipefail

infer_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "${infer_script_dir}/../../../.." && pwd)"
cd "${REPO_ROOT}" || exit 1

if [[ -z "${ACC_PYTHON:-}" ]]; then
    if [[ -x "/mnt/home/tcchen/miniforge3/envs/acc/bin/python" ]]; then
        ACC_PYTHON="/mnt/home/tcchen/miniforge3/envs/acc/bin/python"
    else
        ACC_PYTHON="$(command -v python3 || command -v python)"
    fi
fi
[[ -x "${ACC_PYTHON}" ]] || { echo "ACC_PYTHON is not executable: ${ACC_PYTHON}" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DEFAULT_RUN_DIR="${REPO_ROOT}/results/baseline_v4/wfm_rgb_query_260526_bs1_4f_50k_hyp0.1_dl3dv0.1_waymo_kitti2_stereo4d_bigdata_v1_finetune_psdutio_20260609-115200"
motion_config="${motion_config:-${DEFAULT_RUN_DIR}/wfm_rgb_query_260526_bs1_4f_50k_hyp0.1_dl3dv0.1_waymo_kitti2_stereo4d_bigdata_v1_finetune_psdutio_backup.py}"
motion_load_from="${motion_load_from:-${DEFAULT_RUN_DIR}/checkpoint/latest/ckpt.pth}"

output_dir="${output_dir:-${REPO_ROOT}/results/wfm_rgb_sequence_vis}"
exp_name="${exp_name:-rgb_sequence_vis}"

: "${video:=}"
: "${videos:=}"

# Default sequence (override with sequence_dirs for batch runs):
: "${sequence_dir:=/mnt/nasTeam2/AI/datasets/TMD/Syn4D/download/Syn4D_RGBD/bigoffice_v1/png/seq_000001_4}"
: "${sequence_dirs:=}"

: "${max_frames:=50}"
: "${frame_sampling:=debug_trajectory}"
: "${query_stride:=4}"
: "${query_layout:=center_dense_columns}"
: "${query_dense_stride:=2}"
: "${query_sparse_stride:=8}"
: "${use_amp:=true}"
: "${no_time:=false}"

extra=()
if [[ "${use_amp}" == "true" ]]; then
    extra+=(--use_amp --amp_dtype float16)
else
    extra+=(--no_amp)
fi
if [[ "${no_time}" == "true" ]]; then
    extra+=(--no_time)
fi
if [[ -n "${query_dense_band_indices:-}" ]]; then
    extra+=(--query_dense_band_indices "${query_dense_band_indices}")
fi

query_extra=(
    --query_stride "${query_stride}"
    --query_layout "${query_layout}"
    --query_dense_stride "${query_dense_stride}"
    --query_sparse_stride "${query_sparse_stride}"
)

run_one_sequence() {
    local seq_path="$1"
    local out_root="$2"
    echo "======== sequence=${seq_path} → ${out_root} ========"
    "${ACC_PYTHON}" hAlgorithm/script/infer/motion_head/wfm_rgb_sequence_vis_infer.py \
        --motion_config "${motion_config}" \
        --load_from "${motion_load_from}" \
        --sequence_dir "${seq_path}" \
        --output_dir "${out_root}" \
        --max_frames "${max_frames}" \
        --frame_sampling "${frame_sampling}" \
        "${query_extra[@]}" \
        "${extra[@]}"
}

run_one_video() {
    local vid_path="$1"
    local out_root="$2"
    echo "======== video=${vid_path} → ${out_root} ========"
    "${ACC_PYTHON}" hAlgorithm/script/infer/motion_head/wfm_rgb_sequence_vis_infer.py \
        --motion_config "${motion_config}" \
        --load_from "${motion_load_from}" \
        --video "${vid_path}" \
        --output_dir "${out_root}" \
        --max_frames "${max_frames}" \
        --frame_sampling "${frame_sampling}" \
        "${query_extra[@]}" \
        "${extra[@]}"
}

base_out="${output_dir}/${exp_name}"

echo "======== WFM RGB sequence vis infer ========"
echo "max_frames=${max_frames} (${frame_sampling})"
echo "query_layout=${query_layout}"
if [[ "${query_layout}" == "center_dense_columns" ]]; then
    echo "  dense_stride=${query_dense_stride} sparse_stride=${query_sparse_stride}"
    if [[ -n "${query_dense_band_indices:-}" ]]; then
        echo "  dense_band_indices=${query_dense_band_indices}"
    else
        echo "  dense_band_indices=1,2 (center half)"
    fi
else
    echo "  query_stride=${query_stride}"
fi
echo "output=${base_out}"
echo "=========================================="

if [[ -n "${videos}" ]]; then
    read -r -a _vid_list <<< "${videos}"
    for vid in "${_vid_list[@]}"; do
        [[ -n "${vid}" ]] || continue
        stem="$(basename "${vid}")"
        stem="${stem%.*}"
        run_one_video "${vid}" "${base_out}/${stem}"
    done
elif [[ -n "${video}" ]]; then
    stem="$(basename "${video}")"
    stem="${stem%.*}"
    run_one_video "${video}" "${base_out}/${stem}"
elif [[ -n "${sequence_dirs}" ]]; then
    read -r -a _seq_list <<< "${sequence_dirs}"
    for seq in "${_seq_list[@]}"; do
        [[ -n "${seq}" ]] || continue
        stem="$(basename "${seq}")"
        run_one_sequence "${seq}" "${base_out}/${stem}"
    done
elif [[ -n "${sequence_dir}" ]]; then
    stem="$(basename "${sequence_dir}")"
    run_one_sequence "${sequence_dir}" "${base_out}/${stem}"
else
    echo "Set video/videos or sequence_dir/sequence_dirs." >&2
    exit 2
fi

echo "Rerun / motion outputs under:"
find "${base_out}" -path '*/visualization/*/rerun_vis/*.rrd' -type f 2>/dev/null | sort || true
find "${base_out}" -path '*/visualization/*/motion/*' -name 'track3d_2d_pred_*.jpg' -type f 2>/dev/null | head -20 || true

unset infer_script_dir REPO_ROOT
