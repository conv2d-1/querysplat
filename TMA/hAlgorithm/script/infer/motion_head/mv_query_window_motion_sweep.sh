#!/usr/bin/env bash
# Sweep encoder_window ∈ {2,4,8,16,50} with adjacent pair schedule on Kosmo LV大船 fisheye.
#
# Pair logic per chunk [s,e): encode (e-s) frames, then for each i in s..e-2:
#   pair_forward(i,i+1) → motion mask for frame i
#   pair_forward(i+1,i) → motion mask for frame i+1
#
# Usage:
#   bash hAlgorithm/script/infer/motion_head/mv_query_window_motion_sweep.sh
#
#   WINDOWS="4 8" bash ...   # subset

set -euo pipefail

infer_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$(cd "${infer_script_dir}/../../../.." && pwd)" || exit 1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

output_dir="${output_dir:-./results/window_motion_heatmap_pstudio}"
exp_base="${exp_base:-3937_LV大船_cam8_adjacent}"
no_time=true

MODEL_DIR="${MODEL_DIR:-/mnt/home/tcchen/workspace/TMA-origin-dev/results/baseline_v4/wfm_rgb_query_260526_bs1_4f_50k_hyp0.1_dl3dv0.1_waymo_kitti2_stereo4d_bigdata_v1_finetune_psdutio_20260609-115200}"
CONFIG="${CONFIG:-${MODEL_DIR}/wfm_rgb_query_260526_bs1_4f_50k_hyp0.1_dl3dv0.1_waymo_kitti2_stereo4d_bigdata_v1_finetune_psdutio_backup.py}"
LOADFROM="${LOADFROM:-${MODEL_DIR}/checkpoint/latest/ckpt.pth}"

WINDOWS="${WINDOWS:-2 4 8 16 50}"

for win in ${WINDOWS}; do
    echo ""
    echo "################################################################"
    echo "# da3_small | encoder_window=${win}  pair_schedule=adjacent"
    echo "################################################################"
    CONFIG="${CONFIG}" \
    LOADFROM="${LOADFROM}" \
    encoder_window="${win}" \
    pair_schedule=adjacent \
    output_dir="${output_dir}" \
    exp_name="${exp_base}_win${win}" \
    no_time="${no_time}" \
    bash hAlgorithm/script/infer/motion_head/mv_query_pair2_motion_heatmap_infer.sh
done

echo ""
echo "All window sweeps done. GIFs:"
find "${output_dir}/${exp_base}_win"* -path '*/dense_heatmap_overlay_win*.gif' -type f 2>/dev/null | sort || true

unset infer_script_dir
