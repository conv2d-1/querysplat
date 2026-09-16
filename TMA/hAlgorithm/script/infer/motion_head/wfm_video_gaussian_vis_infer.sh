#!/usr/bin/env bash
# WFM video Gaussian render vis-only infer (4DGS checkpoint, no eval, no motion vis).
#
# Outputs per video:
#   visualization/<iter_tag>/<video_stem>/4dgs/0/render_frames/*.jpg
#   visualization/<iter_tag>/<video_stem>/4dgs/0/render_*_grid_*.jpg
#   visualization/<iter_tag>/<video_stem>/4dgs/0/render_*.mp4
#   visualization/<iter_tag>/<video_stem>/web_viewer/  (PLY + manifest for WebGL)
#
# Usage:
#   bash hAlgorithm/script/infer/motion_head/wfm_video_gaussian_vis_infer.sh
#
#   video=/path/clip.mp4 bash ...
#   video=/path/clip.mp4 per_pixel=true bash ...
#   video=/path/clip.mp4 process_res=840 per_pixel=true bash ...
#   video=/path/clip.mp4 native_resolution=true per_pixel=true bash ...
#   videos="/path/a.mp4 /path/b.mp4" max_frames=30 per_pixel=true bash ...

set -euo pipefail

infer_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$(cd "${infer_script_dir}/../../../.." && pwd)" || exit 1

: "${ACC_PYTHON:=/mnt/home/tcchen/miniforge3/envs/acc/bin/python}"
if [[ ! -x "${ACC_PYTHON}" ]]; then
    echo "ACC_PYTHON is not executable: ${ACC_PYTHON}" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# gsplat CUDA JIT (FFGS path / web tooling): same as test_dist.sh
export CUDA_HOME="${CUDA_HOME:-/mnt/home/tcchen/local/cuda-12.4}"
export CC="${CC:-/usr/bin/gcc-11}"
export CXX="${CXX:-/usr/bin/g++-11}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-/usr/bin/g++-11}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

DEFAULT_4DGS_RUN_DIR="/mnt/home/tcchen/workspace/TMA-origin-dev/results/sparse_pair_4dgs_v2/wfm_rgb_query_dual_4dgs_perpixel_trainquality_v2_wild_norm_20260622-203044"
motion_config="${motion_config:-${DEFAULT_4DGS_RUN_DIR}/wfm_rgb_query_dual_4dgs_perpixel_trainquality_v2_wild_norm_backup.py}"
motion_load_from="${motion_load_from:-${DEFAULT_4DGS_RUN_DIR}/checkpoint/latest/ckpt.pth}"

output_dir="${output_dir:-./results/wfm_video_gaussian_vis}"
exp_name="${exp_name:-video_gaussian_vis}"

: "${video:=}"
: "${videos:=}"

: "${max_frames:=30}"
: "${query_stride:=4}"
: "${per_pixel:=true}"
: "${process_res:=728}"
: "${native_resolution:=false}"
: "${infer_gs_log_scale_bias:=}"
: "${infer_chunk_size:=0}"
: "${all_frames:=false}"
: "${frame_sampling:=debug_trajectory}"
: "${render_fps:=24}"
: "${use_amp:=true}"
: "${export_web_viewer:=true}"

extra=()
if [[ "${use_amp}" == "true" ]]; then
    extra+=(--use_amp --amp_dtype float16)
else
    extra+=(--no_amp)
fi
if [[ "${export_web_viewer}" == "true" ]]; then
    extra+=(--export_web_viewer)
else
    extra+=(--no_export_web_viewer)
fi
if [[ "${per_pixel}" == "true" ]]; then
    extra+=(--per_pixel)
else
    extra+=(--query_stride "${query_stride}")
fi
if [[ "${native_resolution}" == "true" ]]; then
    extra+=(--native_resolution)
elif [[ -n "${process_res}" ]]; then
    extra+=(--process_res "${process_res}")
fi
if [[ -n "${infer_gs_log_scale_bias}" ]]; then
    extra+=(--infer_gs_log_scale_bias "${infer_gs_log_scale_bias}")
fi
if [[ "${all_frames}" == "true" ]]; then
    extra+=(--all_frames)
fi
if [[ "${infer_chunk_size}" != "0" ]]; then
    extra+=(--infer_chunk_size "${infer_chunk_size}")
fi

run_one() {
    local vid_path="$1"
    local out_root="$2"
    echo "======== video=${vid_path} → ${out_root} ========"
    "${ACC_PYTHON}" hAlgorithm/script/infer/motion_head/wfm_video_gaussian_vis_infer.py \
        --motion_config "${motion_config}" \
        --load_from "${motion_load_from}" \
        --video "${vid_path}" \
        --output_dir "${out_root}" \
        --max_frames "${max_frames}" \
        --frame_sampling "${frame_sampling}" \
        --render_fps "${render_fps}" \
        "${extra[@]}"
}

base_out="${output_dir}/${exp_name}"

if [[ -n "${videos}" ]]; then
    read -r -a _vid_list <<< "${videos}"
    for vid in "${_vid_list[@]}"; do
        [[ -n "${vid}" ]] || continue
        stem="$(basename "${vid}")"
        stem="${stem%.*}"
        run_one "${vid}" "${base_out}/${stem}"
    done
elif [[ -n "${video}" ]]; then
    stem="$(basename "${video}")"
    stem="${stem%.*}"
    run_one "${video}" "${base_out}/${stem}"
else
    echo "Set video or videos env var." >&2
    exit 2
fi

echo "Gaussian render outputs:"
find "${base_out}" -path '*/4dgs/*/render_*.mp4' -type f 2>/dev/null | sort || true
find "${base_out}" -path '*/4dgs/*/render_rgb_grid_*.jpg' -type f 2>/dev/null | head -20 || true
echo "Web viewer exports:"
find "${base_out}" -path '*/web_viewer/manifest.json' -type f 2>/dev/null | sort || true

unset infer_script_dir
