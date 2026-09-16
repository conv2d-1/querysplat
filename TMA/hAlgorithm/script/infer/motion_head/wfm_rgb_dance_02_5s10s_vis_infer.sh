#!/usr/bin/env bash
# WFM motion vis: test videos → one .rrd per clip (no chunking).
#
# Default videos:
#   - dance_02_5s_10s.mp4
#   - solo_04_13s_18s.mp4
#
# Default: all decoded frames, process_res=504, center_dense_columns
# (middle 2/4 vertical bands dense_stride=2, sides sparse_stride=8).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=6 bash hAlgorithm/script/infer/motion_head/wfm_rgb_dance_02_5s10s_vis_infer.sh
#
# Override video list (space-separated):
#   videos="/path/a.mp4 /path/b.mp4" bash ...
#
# Uniform 50-frame mode:
#   all_frames=false max_frames=50 native_res=true query_stride=2 bash ...

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
[[ -x "${ACC_PYTHON}" ]] || { echo "Set ACC_PYTHON=/path/to/python" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DEFAULT_RUN_DIR="${REPO_ROOT}/results/baseline_v4/wfm_rgb_query_260526_bs1_4f_50k_hyp0.1_dl3dv0.1_waymo_kitti2_stereo4d_bigdata_v1_finetune_psdutio_20260609-115200"
motion_config="${motion_config:-${DEFAULT_RUN_DIR}/wfm_rgb_query_260526_bs1_4f_50k_hyp0.1_dl3dv0.1_waymo_kitti2_stereo4d_bigdata_v1_finetune_psdutio_backup.py}"
motion_load_from="${motion_load_from:-${DEFAULT_RUN_DIR}/checkpoint/latest/ckpt.pth}"

video_root="${video_root:-/mnt/nasTeam2/AI/personal/ctc/workspace/test}"
: "${video:=}"
: "${videos:=${video_root}/dance_02_5s_10s.mp4 ${video_root}/solo_04_13s_18s.mp4}"

output_dir="${output_dir:-${REPO_ROOT}/results/wfm_rgb_sequence_vis}"
exp_name="${exp_name:-ctc_test_full121}"

: "${all_frames:=true}"
: "${max_frames:=50}"
: "${process_res:=504}"
: "${native_res:=false}"
: "${query_stride:=4}"
: "${query_layout:=center_dense_columns}"
: "${query_dense_stride:=2}"
: "${query_sparse_stride:=8}"
: "${frame_sampling:=sequential}"
: "${use_amp:=true}"

die() { echo "ERROR: $*" >&2; exit 1; }
[[ -f "${motion_config}" ]] || die "motion_config not found: ${motion_config}"
[[ -f "${motion_load_from}" ]] || die "checkpoint not found: ${motion_load_from}"

video_list=()
if [[ -n "${videos}" ]]; then
    read -r -a video_list <<< "${videos}"
elif [[ -n "${video}" ]]; then
    video_list=("${video}")
else
    die "Set video or videos."
fi
for v in "${video_list[@]}"; do
    [[ -f "${v}" ]] || die "video not found: ${v}"
done

extra=(--no_time --process_res "${process_res}")
if [[ "${use_amp}" == "true" ]]; then
    extra+=(--use_amp --amp_dtype float16)
else
    extra+=(--no_amp)
fi
if [[ "${native_res}" == "true" ]]; then
    extra+=(--native_resolution)
fi
if [[ "${all_frames}" == "true" ]]; then
    extra+=(--all_frames --frame_sampling sequential)
else
    extra+=(--max_frames "${max_frames}" --frame_sampling "${frame_sampling}")
fi
if [[ -n "${query_dense_band_indices:-}" ]]; then
    extra+=(--query_dense_band_indices "${query_dense_band_indices}")
fi

out_root="${output_dir}/${exp_name}"
mkdir -p "${out_root}"

echo "======== WFM test videos → single RRD per clip ========"
for v in "${video_list[@]}"; do echo "  ${v}"; done
if [[ "${all_frames}" == "true" ]]; then
    echo "frames=all (sequential decode, no temporal subsample)"
else
    echo "max_frames=${max_frames} (${frame_sampling})"
fi
echo "process_res=${process_res} native_res=${native_res} query_layout=${query_layout}"
if [[ "${query_layout}" == "center_dense_columns" ]]; then
    echo "  dense_stride=${query_dense_stride} sparse_stride=${query_sparse_stride}"
    if [[ -n "${query_dense_band_indices:-}" ]]; then
        echo "  dense_band_indices=${query_dense_band_indices}"
    else
        echo "  dense_band_indices=1,2 (default center half)"
    fi
else
    echo "  query_stride=${query_stride}"
fi
echo "output=${out_root}"
echo "======================================================="

if [[ ${#video_list[@]} -eq 1 ]]; then
  "${ACC_PYTHON}" hAlgorithm/script/infer/motion_head/wfm_rgb_sequence_vis_infer.py \
      --motion_config "${motion_config}" \
      --load_from "${motion_load_from}" \
      --video "${video_list[0]}" \
      --output_dir "${out_root}" \
      --query_stride "${query_stride}" \
      --query_layout "${query_layout}" \
      --query_dense_stride "${query_dense_stride}" \
      --query_sparse_stride "${query_sparse_stride}" \
      "${extra[@]}"
else
  "${ACC_PYTHON}" hAlgorithm/script/infer/motion_head/wfm_rgb_sequence_vis_infer.py \
      --motion_config "${motion_config}" \
      --load_from "${motion_load_from}" \
      --videos "${video_list[@]}" \
      --output_dir "${out_root}" \
      --query_stride "${query_stride}" \
      --query_layout "${query_layout}" \
      --query_dense_stride "${query_dense_stride}" \
      --query_sparse_stride "${query_sparse_stride}" \
      "${extra[@]}"
fi

echo ""
echo "RRD outputs:"
find "${out_root}" -path '*/rerun_vis/vis_dynamic_000000.rrd' -type f 2>/dev/null | sort || true

unset infer_script_dir REPO_ROOT
