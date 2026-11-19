#!/usr/bin/env bash
# Build the two runtime envs for EgoHandWorld on remote92 (uv venvs, pinned to the versions validated on server 45).
# Usage: bash scripts/setup_envs.sh [hawor|egohand|all]   (logs to $B/envs/setup_*.log)
set -euo pipefail
B=/data/wenwang
P=/home/wangwen/code/EgoHandWorld
UV=$HOME/.local/bin/uv
export UV_CACHE_DIR=$B/cache/uv UV_PYTHON_INSTALL_DIR=$B/cache/uv-python UV_HTTP_TIMEOUT=300
MIRROR=https://mirrors.aliyun.com/pypi/simple
MOCAP_WHL=/data/yuanshuai/python_codes/mocap/deployment
WHAT=${1:-all}

build_hawor() {
  local E=$B/envs/hawor
  [ -x $E/bin/python ] || { $UV python install 3.10; $UV venv -q -p 3.10 $E; }
  local I="$UV pip install -p $E/bin/python -i $MIRROR --index-strategy unsafe-best-match"
  # torch 2.1.0 (PyPI build = cu121) + its nvidia libs: reuse the wheels already on disk in the mocap deployment
  # (92's internet link is ~1 MB/s); identical to torch 2.1.0+cu121 used on server 45.
  $I nvidia-nvtx-cu12==12.1.105 nvidia-nvjitlink-cu12   # tiny; not among the mocap wheels
  $UV pip install -p $E/bin/python --no-index --find-links $MOCAP_WHL torch==2.1.0 torchvision==0.16.0 triton==2.1.0
  $I "setuptools<70" wheel ninja
  $I pytorch3d==0.7.5 --find-links https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/download.html
  $UV pip install -p $E/bin/python --no-index --find-links $MOCAP_WHL torch-scatter==2.1.2+pt21cu121
  $I pip   # chumpy's setup.py imports pip
  $I --no-build-isolation $P/third_party/chumpy
  # everything else exactly as frozen on 45
  grep -v -E '^(torch|torchvision|triton|pytorch3d|torch-scatter|chumpy|mmcv|setuptools|pip|wheel)[=@ ]' $P/scripts/freeze_hawor.txt > /tmp/ehw_req_hawor.txt
  $I -r /tmp/ehw_req_hawor.txt
  MMCV_WITH_OPS=0 $I --no-build-isolation mmcv==1.3.9
  # DROID-SLAM + lietorch CUDA extensions for RTX 4090 (sm_89), CUDA 12.1 toolkit (matches torch cu121)
  cd $P/third_party/hawor/thirdparty/DROID-SLAM
  cp setup.py.bak setup.py
  sed -i -E '/compute_(60|61|70|75|80),code/d; s/compute_86,code=sm_86/compute_89,code=sm_89/' setup.py
  rm -rf build dist *.egg-info
  CUDA_HOME=/usr/local/cuda-12.1 PATH=/usr/local/cuda-12.1/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=16 \
    $E/bin/python setup.py install
  rm -rf build
  cd /tmp && $E/bin/python -c "import torch,droid_backends,lietorch,mmcv,pytorch3d,torch_scatter,ultralytics;print('hawor env ok',torch.__version__)"
}

build_egohand() {
  local E=$B/envs/egohand
  [ -x $E/bin/python ] || { $UV python install 3.12; $UV venv -q -p 3.12 $E; }
  local I="$UV pip install -p $E/bin/python -i $MIRROR --index-strategy unsafe-best-match"
  grep -v -E '^(dinov3)[=@ ]' $P/scripts/freeze_egohand.txt > /tmp/ehw_req_egohand.txt
  $I --extra-index-url https://download.pytorch.org/whl/cu124 -r /tmp/ehw_req_egohand.txt
  $UV pip install -p $E/bin/python --no-deps "dinov3 @ https://github.com/facebookresearch/dinov3/archive/5aa7e93ab42ae3526291f6b4e3fcd22637e03326.zip"  # archive URL: git here goes through a dead proxy in ~/.gitconfig
  cd /tmp && $E/bin/python -c "import torch,numpy,warp,soma,onnxruntime;from dinov3.models.vision_transformer import DinoVisionTransformer;print('egohand env ok',torch.__version__,numpy.__version__)"
}

case $WHAT in
  hawor) build_hawor ;;
  egohand) build_egohand ;;
  all) build_hawor; build_egohand ;;
esac
