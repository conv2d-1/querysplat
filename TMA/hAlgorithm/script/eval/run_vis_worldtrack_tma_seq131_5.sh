#!/usr/bin/env bash
# Re-render TMA vis for adt/Apartment_release_decoration_seq131_5 from tracks.npz
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-acc}"
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

PKG="${PKG:-tmp/eval_worldtrack_full_vis_adt/adt_mini/Apartment_release_decoration_seq131_5}"
OUT_COPY="${OUT_COPY:-tmp/vis_worldtrack_tma_Apartment_release_decoration_seq131_5}"

python hAlgorithm/script/eval/vis_worldtrack_tma.py "$PKG" --max-points 300 --trace-frames 8 --fps 15

mkdir -p "$OUT_COPY"
rsync -a "$PKG/" "$OUT_COPY/"
echo "TMA vis ready: $OUT_COPY/vis/"
