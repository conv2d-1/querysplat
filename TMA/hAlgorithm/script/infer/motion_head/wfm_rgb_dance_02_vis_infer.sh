#!/usr/bin/env bash
# WFM motion vis infer for dance_02 clips (5s_10s + 22s_28s).
#
# Runs wfm_rgb_sequence_vis_infer.py with video input on a machine that has a free GPU.
# Model loads once; both clips are processed in one Python invocation.
#
# Usage (from repo root, or any cwd — script cd's to repo root automatically):
#   bash hAlgorithm/script/infer/motion_head/wfm_rgb_dance_02_vis_infer.sh
#
# Pick GPU:
#   CUDA_VISIBLE_DEVICES=0 bash hAlgorithm/script/infer/motion_head/wfm_rgb_dance_02_vis_infer.sh
#
# Override paths / hyper-params:
#   motion_config=/path/to/config.py \
#   motion_load_from=/path/to/ckpt.pth \
#   video_root=/path/to/test \
#   query_stride=2 max_frames=30 \
#   bash hAlgorithm/script/infer/motion_head/wfm_rgb_dance_02_vis_infer.sh
#
# Outputs:
#   ${output_dir}/${exp_name}/visualization/iter_XXXXXX/dance_02_5s_10s/
#   ${output_dir}/${exp_name}/visualization/iter_XXXXXX/dance_02_22s_28s/
#     motion/000000/src000/track3d_2d_pred_tgt*.jpg
#     rerun_vis/vis_dynamic_000000.rrd

set -euo pipefail

infer_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "${infer_script_dir}/../../../.." && pwd)"
cd "${REPO_ROOT}" || exit 1

# ---------------------------------------------------------------------------
# Python (override if acc env is elsewhere on the target server)
# ---------------------------------------------------------------------------
if [[ -z "${ACC_PYTHON:-}" ]]; then
    if [[ -x "/mnt/home/tcchen/miniforge3/envs/acc/bin/python" ]]; then
        ACC_PYTHON="/mnt/home/tcchen/miniforge3/envs/acc/bin/python"
    else
        ACC_PYTHON="$(command -v python3 || command -v python)"
    fi
fi
if [[ ! -x "${ACC_PYTHON}" ]]; then
    echo "Python not found. Set ACC_PYTHON=/path/to/python" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---------------------------------------------------------------------------
# Model (baseline_v4 finetune checkpoint)
# ---------------------------------------------------------------------------
DEFAULT_RUN_DIR="${REPO_ROOT}/results/baseline_v4/wfm_rgb_query_260526_bs1_4f_50k_hyp0.1_dl3dv0.1_waymo_kitti2_stereo4d_bigdata_v1_finetune_psdutio_20260609-115200"
motion_config="${motion_config:-${DEFAULT_RUN_DIR}/wfm_rgb_query_260526_bs1_4f_50k_hyp0.1_dl3dv0.1_waymo_kitti2_stereo4d_bigdata_v1_finetune_psdutio_backup.py}"
motion_load_from="${motion_load_from:-${DEFAULT_RUN_DIR}/checkpoint/latest/ckpt.pth}"

# ---------------------------------------------------------------------------
# Input videos (dance_02 segments)
# ---------------------------------------------------------------------------
video_root="${video_root:-/mnt/nasTeam2/AI/personal/ctc/workspace/test}"
video_5s_10s="${video_5s_10s:-${video_root}/dance_02_5s_10s.mp4}"
video_22s_28s="${video_22s_28s:-${video_root}/dance_02_22s_28s.mp4}"

# ---------------------------------------------------------------------------
# Infer hyper-params
# ---------------------------------------------------------------------------
output_dir="${output_dir:-${REPO_ROOT}/results/wfm_rgb_sequence_vis}"
exp_name="${exp_name:-dance_02_test}"
: "${max_frames:=30}"
: "${query_stride:=2}"
: "${frame_sampling:=debug_trajectory}"
: "${use_amp:=true}"
: "${no_time:=true}"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
die() { echo "ERROR: $*" >&2; exit 1; }

[[ -f "${motion_config}" ]] || die "motion_config not found: ${motion_config}"
[[ -f "${motion_load_from}" ]] || die "checkpoint not found: ${motion_load_from}"
[[ -f "${video_5s_10s}" ]] || die "video not found: ${video_5s_10s}"
[[ -f "${video_22s_28s}" ]] || die "video not found: ${video_22s_28s}"

echo "======== WFM dance_02 motion vis infer ========"
echo "REPO_ROOT=${REPO_ROOT}"
echo "ACC_PYTHON=${ACC_PYTHON}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "motion_config=${motion_config}"
echo "motion_load_from=${motion_load_from}"
echo "videos:"
echo "  ${video_5s_10s}"
echo "  ${video_22s_28s}"
echo "output_dir=${output_dir}/${exp_name}"
echo "max_frames=${max_frames} query_stride=${query_stride} frame_sampling=${frame_sampling}"
echo "=============================================="

extra=()
if [[ "${use_amp}" == "true" ]]; then
    extra+=(--use_amp --amp_dtype float16)
else
    extra+=(--no_amp)
fi
if [[ "${no_time}" == "true" ]]; then
    extra+=(--no_time)
fi

out_root="${output_dir}/${exp_name}"
mkdir -p "${out_root}"

"${ACC_PYTHON}" hAlgorithm/script/infer/motion_head/wfm_rgb_sequence_vis_infer.py \
    --motion_config "${motion_config}" \
    --load_from "${motion_load_from}" \
    --videos "${video_5s_10s}" "${video_22s_28s}" \
    --output_dir "${out_root}" \
    --max_frames "${max_frames}" \
    --query_stride "${query_stride}" \
    --frame_sampling "${frame_sampling}" \
    "${extra[@]}"

echo ""
echo "Done. Key outputs:"
find "${out_root}" -path '*/visualization/*/rerun_vis/*.rrd' -type f 2>/dev/null | sort || true
find "${out_root}" -path '*/visualization/*/motion/*' -name 'track3d_2d_pred_tgt001.jpg' -type f 2>/dev/null | sort || true

echo ""
echo "View 3D (rerun):"
echo "  rerun ${out_root}/*/visualization/*/dance_02_5s_10s/rerun_vis/vis_dynamic_000000.rrd"

unset infer_script_dir REPO_ROOT
