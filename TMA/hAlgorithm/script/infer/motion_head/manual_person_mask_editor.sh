#!/usr/bin/env bash
# Launch Gradio manual person-mask editor for existing clip outputs.
#
# Usage:
#   bash hAlgorithm/script/infer/motion_head/manual_person_mask_editor.sh
#
#   clip_dirs="/path/clip0023 /path/clip0025" port=7860 bash ...
#   share=true bash ...   # optional public Gradio link

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

BASE="${REPO_ROOT}/results/wfm_rgb_yolo_person_dynamic_vis/lalaland_sunset_dance_2m14/wfm_rgb_query_260526_bs1_4f_50k_hyp0.1_dl3dv0.1_waymo_kitti2_stereo4d_bigdata_v1_finetune_psdutio_backup"

DEFAULT_CLIP23="${BASE}/06_LaLaLand_sunset_dance_f004363_n050_clip0023"
DEFAULT_CLIP25="${BASE}/06_LaLaLand_sunset_dance_f004463_n050_clip0025"

: "${clip_dirs:=${DEFAULT_CLIP23} ${DEFAULT_CLIP25}}"
: "${host:=127.0.0.1}"
: "${port:=7860}"
: "${share:=false}"

cmd=(
    "${ACC_PYTHON}" hAlgorithm/script/infer/motion_head/manual_person_mask_editor.py
    --host "${host}"
    --port "${port}"
)

read -r -a _clip_list <<< "${clip_dirs}"
cmd+=(--clip_dirs "${_clip_list[@]}")

echo "======== Manual person mask editor ========"
echo "clips=${clip_dirs}"
echo "url=http://127.0.0.1:${port}  (SSH: ssh -L ${port}:127.0.0.1:${port} <user>@<host>)"
echo "==========================================="

exec "${cmd[@]}"
