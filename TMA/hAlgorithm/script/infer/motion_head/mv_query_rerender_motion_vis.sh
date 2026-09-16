#!/usr/bin/env bash
# Re-render motion heatmaps / overlays from existing dense_motion_npy (no GPU inference).
#
# Usage:
#   vis_from=/path/to/seq_output bash hAlgorithm/script/infer/motion_head/mv_query_rerender_motion_vis.sh
#
# Kosmo fisheye example (win8):
#   vis_from=./results/window_motion_heatmap/3937_LV大船_cam8_adjacent_win8/.../3937_LV大船..._cam8 \
#   bash hAlgorithm/script/infer/motion_head/mv_query_rerender_motion_vis.sh

set -euo pipefail

infer_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$(cd "${infer_script_dir}/../../../.." && pwd)" || exit 1

: "${ACC_PYTHON:=/mnt/home/tcchen/miniforge3/envs/acc/bin/python}"
if [[ ! -x "${ACC_PYTHON}" ]]; then
    echo "ACC_PYTHON is not executable: ${ACC_PYTHON}" >&2
    exit 1
fi

: "${vis_from:?Set vis_from to sequence output dir with dense_motion_npy/}"

# Same JSON / RGB source as original inference
data="${data:-/mnt/nasTeam2/Kosmo/processed_data/kosmo2/20260424_0015/3937_LV大船白天58min_0424_pointfilter/raw_data/3937_LV大船白天58min_0424_pointfilter.json}"
view_id="${view_id:-8}"
nums="${nums:-200}"
process_res="${process_res:-504}"
fisheye="${fisheye:-true}"
sensor_size="6.43 4.87"

vis_to="${vis_to:-${vis_from}_vis}"

heatmap_norm="${heatmap_norm:-global_log_percentile}"
heatmap_percentile="${heatmap_percentile:-95}"
heatmap_gamma="${heatmap_gamma:-1.2}"
heatmap_vmin_percentile="${heatmap_vmin_percentile:-85}"
heatmap_colormap="${heatmap_colormap:-turbo}"
heatmap_overlay_alpha="${heatmap_overlay_alpha:-0.6}"
dense_motion_threshold="${dense_motion_threshold:-0.02}"
overlay_use_mask="${overlay_use_mask:-true}"
gif_fps="${gif_fps:-4.0}"

extra=()
if [[ "${fisheye}" == "true" ]]; then
    extra+=(--fisheye --sensor_size ${sensor_size})
fi
if [[ "${overlay_use_mask}" == "true" ]]; then
    extra+=(--overlay_use_mask)
else
    extra+=(--no_overlay_use_mask)
fi

cmd=(
    "${ACC_PYTHON}" hAlgorithm/script/infer/motion_head/mv_query_pair2_motion_heatmap_infer.py
    --vis_only
    --vis_from "${vis_from}"
    --vis_to "${vis_to}"
    --data "${data}"
    --view_id "${view_id}"
    --nums "${nums}"
    --process_res "${process_res}"
    --dense_motion_threshold "${dense_motion_threshold}"
    --heatmap_norm "${heatmap_norm}"
    --heatmap_percentile "${heatmap_percentile}"
    --heatmap_gamma "${heatmap_gamma}"
    --heatmap_vmin_percentile "${heatmap_vmin_percentile}"
    --heatmap_colormap "${heatmap_colormap}"
    --heatmap_overlay_alpha "${heatmap_overlay_alpha}"
    --gif_fps "${gif_fps}"
    --save_gif
    --output_dir /tmp/motion_vis_only_unused
)
cmd+=("${extra[@]}")

echo "======== vis_only ${vis_from} → ${vis_to} ========"
"${cmd[@]}"

echo "GIF:"
find "${vis_to}" -name 'dense_heatmap_overlay*.gif' 2>/dev/null | head -1

unset infer_script_dir
