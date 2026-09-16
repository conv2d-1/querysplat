#!/usr/bin/env bash
# Video / RGB sequence: YOLO frame-0 person mask + (0,0) static + (0,t) dynamic warp3d → Rerun .rrd
#
# Same visualization path as wfm_rgb_davis_dynamic_vis_infer.sh, but mask from YOLO person
# detection instead of DAVIS Annotations.
#
# Usage:
#   bash hAlgorithm/script/infer/motion_head/wfm_rgb_yolo_person_dynamic_vis_infer.sh
#
#   video=/path/clip.mp4 bash ...
#   videos="a.mp4 b.mp4" bash ...
#   sequence_dir=/path/rgb_frames bash ...

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

output_dir="${output_dir:-${REPO_ROOT}/results/wfm_rgb_yolo_person_dynamic_vis}"
exp_name="${exp_name:-yolo_person_dynamic}"

: "${video:=}"
: "${videos:=}"
: "${sequence_dir:=}"
: "${sequence_dirs:=}"

: "${max_frames:=50}"
: "${frame_sampling:=debug_trajectory}"
: "${static_downsample:=4}"
: "${dynamic_downsample:=1}"
: "${max_dynamic_queries:=0}"
: "${static_vis_step:=1}"
: "${max_static_vis_points:=120000}"
: "${max_traj_vis_points:=2048}"
: "${traj_vis_downsample:=1}"
: "${min_arrow_length:=0.01}"
: "${export_rrd_only:=false}"
: "${process_res:=728}"
: "${use_amp:=true}"
: "${no_time:=true}"

: "${yolo_weights:=yolov8n-seg.pt}"
: "${yolo_conf:=0.25}"
: "${yolo_iou:=0.45}"
: "${yolo_imgsz:=}"

: "${video_start_time:=}"
: "${video_clip_frames:=}"

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
    "${ACC_PYTHON}" hAlgorithm/script/infer/motion_head/wfm_rgb_yolo_person_dynamic_vis_infer.py
    --motion_config "${motion_config}"
    --load_from "${motion_load_from}"
    --output_dir "${base_out}"
    --max_frames "${max_frames}"
    --frame_sampling "${frame_sampling}"
    --static_downsample "${static_downsample}"
    --dynamic_downsample "${dynamic_downsample}"
    --max_dynamic_queries "${max_dynamic_queries}"
    --static_vis_step "${static_vis_step}"
    --max_static_vis_points "${max_static_vis_points}"
    --max_traj_vis_points "${max_traj_vis_points}"
    --traj_vis_downsample "${traj_vis_downsample}"
    --min_arrow_length "${min_arrow_length}"
    --process_res "${process_res}"
    --yolo_weights "${yolo_weights}"
    --yolo_conf "${yolo_conf}"
    --yolo_iou "${yolo_iou}"
    "${extra[@]}"
)

if [[ -n "${yolo_imgsz}" ]]; then
    cmd+=(--yolo_imgsz "${yolo_imgsz}")
fi

if [[ -n "${video_clip_frames}" ]]; then
    cmd+=(--video_clip_frames "${video_clip_frames}")
    if [[ -n "${video_start_time}" ]]; then
        cmd+=(--video_start_time "${video_start_time}")
    fi
fi

if [[ "${export_rrd_only}" == "true" ]]; then
    cmd+=(--export_rrd_only)
fi

if [[ -n "${videos}" ]]; then
    read -r -a _vid_list <<< "${videos}"
    cmd+=(--videos "${_vid_list[@]}")
elif [[ -n "${video}" ]]; then
    cmd+=(--video "${video}")
elif [[ -n "${sequence_dirs}" ]]; then
    read -r -a _seq_list <<< "${sequence_dirs}"
    cmd+=(--sequence_dirs "${_seq_list[@]}")
elif [[ -n "${sequence_dir}" ]]; then
    cmd+=(--sequence_dir "${sequence_dir}")
else
    echo "Set video=, videos=, sequence_dir=, or sequence_dirs=" >&2
    exit 1
fi

echo "======== YOLO person dynamic warp3d vis ========"
echo "yolo_weights=${yolo_weights} conf=${yolo_conf} iou=${yolo_iou}"
echo "static_downsample=${static_downsample} dynamic_downsample=${dynamic_downsample}"
echo "max_frames=${max_frames} (${frame_sampling})"
if [[ -n "${video_clip_frames}" ]]; then
    echo "video_clip: start=${video_start_time:-0} frames=${video_clip_frames}"
fi
echo "output=${base_out}"
echo "================================================"

"${cmd[@]}"

echo ""
echo "RRD outputs:"
find "${base_out}" -path '*/rerun_vis/vis_dynamic_*.rrd' -type f 2>/dev/null | sort || true

unset infer_script_dir REPO_ROOT
