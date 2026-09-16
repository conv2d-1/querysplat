#!/usr/bin/env bash
# DAVIS: per-frame (n,n) static (outside mask) accumulates over time + (0,t) ref0 dynamic → Rerun .rrd
#
# Default sequence: parkour (480p, 50 frames, debug_trajectory sampling).
#
# Usage:
#   bash hAlgorithm/script/infer/motion_head/wfm_rgb_davis_accum_static_vis_infer.sh
#
#   sequence=bear static_downsample=2 bash ...
#   sequences="parkour bear" bash ...

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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DAVIS_ROOT="${DAVIS_ROOT:-/mnt/netdata/Team/AI/datasets/VLM/DAVIS}"
resolution="${resolution:-480p}"

DEFAULT_RUN_DIR="${REPO_ROOT}/results/baseline_v4/wfm_rgb_query_260526_bs1_4f_50k_hyp0.1_dl3dv0.1_waymo_kitti2_stereo4d_bigdata_v1_finetune_psdutio_20260609-115200"
motion_config="${motion_config:-${DEFAULT_RUN_DIR}/wfm_rgb_query_260526_bs1_4f_50k_hyp0.1_dl3dv0.1_waymo_kitti2_stereo4d_bigdata_v1_finetune_psdutio_backup.py}"
motion_load_from="${motion_load_from:-${DEFAULT_RUN_DIR}/checkpoint/latest/ckpt.pth}"

output_dir="${output_dir:-${REPO_ROOT}/results/wfm_rgb_davis_accum_static_vis}"
exp_name="${exp_name:-davis_accum_static}"

: "${sequence:=parkour}"
: "${sequences:=}"

: "${max_frames:=50}"
: "${frame_sampling:=debug_trajectory}"
: "${static_downsample:=4}"
: "${dynamic_downsample:=1}"
: "${max_dynamic_queries:=0}"
: "${static_vis_step:=1}"
: "${max_per_frame_vis_points:=40000}"
: "${max_accum_vis_points:=250000}"
: "${max_traj_vis_points:=2048}"
: "${traj_vis_downsample:=1}"
: "${frustum_scale:=0.05}"
: "${frustum_line_radius:=0.003}"
: "${no_camera_frustums:=false}"
: "${export_rrd_only:=false}"
: "${process_res:=728}"
: "${use_amp:=true}"
: "${no_time:=true}"

extra=()
if [[ "${use_amp}" == "true" ]]; then
    extra+=(--use_amp --amp_dtype float16)
else
    extra+=(--no_amp)
fi
if [[ "${no_time}" == "true" ]]; then
    extra+=(--no_time)
fi

base_out="${output_dir}/${exp_name}"
mkdir -p "${base_out}"

cmd=(
    "${ACC_PYTHON}" hAlgorithm/script/infer/motion_head/wfm_rgb_davis_accum_static_vis_infer.py
    --motion_config "${motion_config}"
    --load_from "${motion_load_from}"
    --davis_root "${DAVIS_ROOT}"
    --resolution "${resolution}"
    --output_dir "${base_out}"
    --max_frames "${max_frames}"
    --frame_sampling "${frame_sampling}"
    --static_downsample "${static_downsample}"
    --dynamic_downsample "${dynamic_downsample}"
    --max_dynamic_queries "${max_dynamic_queries}"
    --static_vis_step "${static_vis_step}"
    --max_per_frame_vis_points "${max_per_frame_vis_points}"
    --max_accum_vis_points "${max_accum_vis_points}"
    --max_traj_vis_points "${max_traj_vis_points}"
    --traj_vis_downsample "${traj_vis_downsample}"
    --frustum_scale "${frustum_scale}"
    --frustum_line_radius "${frustum_line_radius}"
    --process_res "${process_res}"
    "${extra[@]}"
)

if [[ "${no_camera_frustums}" == "true" ]]; then
    cmd+=(--no_camera_frustums)
fi

if [[ "${export_rrd_only}" == "true" ]]; then
    cmd+=(--export_rrd_only)
fi

echo "======== DAVIS accum static (n,n) + dynamic (0,t) vis ========"
echo "davis_root=${DAVIS_ROOT} resolution=${resolution}"
echo "static_downsample=${static_downsample} dynamic_downsample=${dynamic_downsample}"
echo "max_frames=${max_frames} (${frame_sampling})"
echo "output=${base_out}"
echo "=============================================================="

if [[ -n "${sequences}" ]]; then
    read -r -a _seq_list <<< "${sequences}"
    cmd+=(--sequences "${_seq_list[@]}")
else
    cmd+=(--sequence "${sequence}")
fi

"${cmd[@]}"

echo ""
echo "RRD outputs:"
find "${base_out}" -path '*/rerun_vis/vis_accum_static_*.rrd' -type f 2>/dev/null | sort || true

unset infer_script_dir REPO_ROOT
