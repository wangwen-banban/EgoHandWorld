#!/usr/bin/env bash
# Build the ViPE env used by camera_backend=vipe (what was actually done on remote92, 2026-11-09).
set -euo pipefail
B=/data/wenwang; SRC=$HOME/code/third_party_src/vipe; E=$B/envs/vipe; UV=$HOME/.local/bin/uv
M=https://mirrors.aliyun.com/pypi/simple
export UV_CACHE_DIR=$B/cache/uv UV_HTTP_TIMEOUT=300
# 1. source, pinned commit
[ -d $SRC ] || git clone https://github.com/nv-tlabs/vipe.git $SRC
git -C $SRC checkout ead2e008bc055310121736a49ff306bba50d10dc
# 2. Eigen 3.4.0 headers: gitlab is unreachable from 92, so download it elsewhere and unpack Eigen/ to csrc/include/eigen3/
[ -d $SRC/csrc/include/eigen3/Eigen ] || { echo "put eigen-3.4.0/Eigen at $SRC/csrc/include/eigen3/Eigen first"; exit 1; }
# 3. env = copy of the egohand env (Python 3.12, torch 2.6.0+cu124; driver 550 supports CUDA <= 12.4)
[ -x $E/bin/python ] || { cp -a $B/envs/egohand $E; grep -rlI "$B/envs/egohand" $E/bin $E/pyvenv.cfg | xargs sed -i "s#$B/envs/egohand#$E#g"; }
# 4. runtime deps from pyproject, minus gtsam-develop (its pinned dev build is gone from PyPI; only pose_only_long uses it)
$UV pip install -p $E/bin/python -i $M --index-strategy unsafe-best-match pip click einops gdown huggingface_hub hydra-core \
  "imageio[ffmpeg]" kornia matplotlib ninja omegaconf "opencv-python<5" "OpenEXR<3.3.0" "pydantic>=2,<3" pillow python-pycg \
  ray rerun-sdk safetensors scipy timm tqdm "transformers>=4,<5" viser "setuptools>=77" wheel
# 5. CUDA extension for RTX 4090 with the system CUDA 12.1 toolkit (uv rejects the match-runtime build config, so pip)
cd $SRC && CUDA_HOME=/usr/local/cuda-12.1 PATH=/usr/local/cuda-12.1/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=16 \
  $E/bin/python -m pip install --no-build-isolation --no-deps -e .
cd /tmp && $E/bin/python -c "import torch, vipe_ext, vipe; print('vipe ok')"
# 6. weights (92 cannot reach Google Drive or huggingface.co; they were downloaded on the Mac and streamed over):
#    $B/checkpoints/vipe_torchhub/hub/{geocalib/pinhole.tar, moge2/moge-2-vitl.pt, checkpoints/groundingdino_swint_ogc.pth,
#      sam/sam_vit_b_01ec64.pth, aot/R50_DeAOTL_PRE_YTB_DAV.pth, droid_slam/droid.pth -> hawor_external/droid.pth}
#    $B/checkpoints/vipe_hf: bert-base-uncased (HF_ENDPOINT=https://hf-mirror.com snapshot_download)
