#!/usr/bin/env bash
# Re-run YOLO dynamic vis for clips using saved manual masks.
#
# Prerequisite:
#   mask_debug/<seq>_manual_ref_mask.png created by manual_person_mask_editor.py
#
# Usage:
#   bash hAlgorithm/script/infer/motion_head/manual_person_mask_rerun_infer.sh
#
#   clip_dirs="/path/clip0023 /path/clip0025" bash ...

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

BASE="${REPO_ROOT}/results/wfm_rgb_yolo_person_dynamic_vis/lalaland_sunset_dance_2m14/wfm_rgb_query_260526_bs1_4f_50k_hyp0.1_dl3dv0.1_waymo_kitti2_stereo4d_bigdata_v1_finetune_psdutio_backup"
DEFAULT_CLIP23="${BASE}/06_LaLaLand_sunset_dance_f004363_n050_clip0023"
DEFAULT_CLIP25="${BASE}/06_LaLaLand_sunset_dance_f004463_n050_clip0025"

output_dir="${output_dir:-${REPO_ROOT}/results/wfm_rgb_yolo_person_dynamic_vis/lalaland_sunset_dance_2m14}"
exp_name="${exp_name:-wfm_rgb_query_260526_bs1_4f_50k_hyp0.1_dl3dv0.1_waymo_kitti2_stereo4d_bigdata_v1_finetune_psdutio_backup}"

: "${clip_dirs:=${DEFAULT_CLIP23} ${DEFAULT_CLIP25}}"
: "${mask_source:=manual}"
: "${process_res:=728}"
: "${static_downsample:=2}"
: "${dynamic_downsample:=1}"
: "${dynamic_subpixel_factor:=2}"
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

read -r -a _clip_list <<< "${clip_dirs}"

cmd=(
    "${ACC_PYTHON}" hAlgorithm/script/infer/motion_head/wfm_rgb_yolo_person_dynamic_vis_infer.py
    --motion_config "${motion_config}"
    --load_from "${motion_load_from}"
    --output_dir "${base_out}"
    --process_res "${process_res}"
    --mask_source "${mask_source}"
    --rerun_clip_dirs "${_clip_list[@]}"
    --static_downsample "${static_downsample}"
    --dynamic_downsample "${dynamic_downsample}"
    --dynamic_subpixel_factor "${dynamic_subpixel_factor}"
    "${extra[@]}"
)

echo "======== Manual-mask rerun infer ========"
echo "mask_source=${mask_source}"
echo "static_downsample=${static_downsample} dynamic_subpixel_factor=${dynamic_subpixel_factor}"
echo "clip_dirs=${clip_dirs}"
echo "output=${base_out}"
echo "========================================"

"${cmd[@]}"

echo ""
echo "Updated RRD outputs:"
for d in "${_clip_list[@]}"; do
    find "${d}" -path '*/rerun_vis/vis_dynamic_*.rrd' -type f 2>/dev/null || true
done

unset infer_script_dir REPO_ROOT
