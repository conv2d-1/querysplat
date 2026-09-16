#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${ENV_PREFIX:-${ROOT}/.envs/tma4dgs}"
CONDA_BIN="${CONDA_BIN:-/mnt/cfsdata/Team/AI/personal/zhangweiqi/miniconda3/bin/conda}"
export TORCH_HOME="${TORCH_HOME:-${ROOT}/.cache/torch}"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "Conda not found: ${CONDA_BIN}" >&2
  exit 1
fi

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  # Use only conda-forge so the setup does not depend on Anaconda default-channel ToS.
  "${CONDA_BIN}" create \
    --prefix "${ENV_PREFIX}" \
    --override-channels \
    --channel conda-forge \
    python=3.10 pip -y
fi

PYTHON="${ENV_PREFIX}/bin/python"
# imagecorruptions 1.1.2 still imports pkg_resources, which setuptools 81+
# no longer provides.
"${PYTHON}" -m pip install --upgrade pip wheel "setuptools<81"

# Install packages that also appear as Torch dependencies from the fast default
# mirror first. Otherwise PyTorch's wheel index serves them very slowly.
"${PYTHON}" -m pip install \
  numpy==1.26.4 \
  Pillow \
  filelock \
  "typing-extensions>=4.8.0" \
  sympy \
  networkx \
  jinja2 \
  fsspec \
  ninja

# Match the versions recorded by the original TMA 4DGS experiment.
"${PYTHON}" -m pip install \
  torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu124

if ! "${PYTHON}" -c "import gsplat; assert gsplat.__version__ == '1.5.3+pt24cu124'" >/dev/null 2>&1; then
  "${PYTHON}" -m pip install \
    "https://github.com/nerfstudio-project/gsplat/releases/download/v1.5.3/gsplat-1.5.3+pt24cu124-cp310-cp310-linux_x86_64.whl"
fi

if ! "${PYTHON}" -c "import fused_ssim" >/dev/null 2>&1; then
  CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}" \
  TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}" \
  CC="${CC:-/usr/bin/gcc-11}" \
  CXX="${CXX:-/usr/bin/g++-11}" \
  CUDAHOSTCXX="${CUDAHOSTCXX:-/usr/bin/g++-11}" \
    "${PYTHON}" -m pip install --no-build-isolation \
      "git+https://github.com/rahul-goel/fused-ssim.git"
fi

if ! "${PYTHON}" -c "import pytorch3d" >/dev/null 2>&1; then
  CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}" \
  TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}" \
  CC="${CC:-/usr/bin/gcc-11}" \
  CXX="${CXX:-/usr/bin/g++-11}" \
  CUDAHOSTCXX="${CUDAHOSTCXX:-/usr/bin/g++-11}" \
  MAX_JOBS="${MAX_JOBS:-8}" \
    "${PYTHON}" -m pip install --no-build-isolation \
      "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7"
fi

"${PYTHON}" -m pip install \
  accelerate==1.0.1 \
  diffusers==0.31.0 \
  hydra-core \
  PyYAML==6.0.2 \
  numpy==1.26.4 \
  easydict \
  einops \
  Pillow \
  h5py \
  opencv-python \
  open3d==0.19.0 \
  pycolmap==3.10.0 \
  tqdm \
  tensorboard \
  torch-geometric==2.6.1 \
  matplotlib \
  scipy \
  trimesh \
  imageio \
  imagecorruptions==1.1.2 \
  imgaug==0.4.0 \
  moviepy \
  scikit-image \
  jaxtyping \
  beartype \
  bitsandbytes==0.44.1 \
  plyfile \
  dacite \
  kornia \
  lpips \
  tabulate

# imgaug==0.4.0 uses np.sctypes, removed in NumPy 2. Pin NumPy after the
# remaining packages because modern scipy/opencv wheels may otherwise upgrade it.
"${PYTHON}" -m pip install --force-reinstall "numpy==1.26.4"

"${PYTHON}" - <<'PY'
from torchvision.models import VGG16_Weights, vgg16

# LPIPS constructs a pretrained VGG16 trunk at runtime. Cache it inside the
# workspace so inference and training do not depend on a later network fetch.
vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
print("VGG16 cache ready")
PY

"${PYTHON}" - <<'PY'
import accelerate
import fused_ssim
import gsplat
import pytorch3d
import torch
import yaml

from gsplat.rendering import rasterization

print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"accelerate={accelerate.__version__}")
print(f"fused_ssim={fused_ssim.__file__}")
print(f"gsplat={gsplat.__version__}")
print(f"pytorch3d={pytorch3d.__version__}")
print(f"yaml={yaml.__version__}")
print(f"rasterization={rasterization.__name__}")
PY

echo "Environment ready: ${ENV_PREFIX}"
