#!/usr/bin/env bash
# Download / copy all EgoHandWorld weights into $B/checkpoints and verify sha256 against the copies validated on server 45.
# Usage: bash scripts/fetch_weights.sh
set -uo pipefail
B=/data/wenwang
C=$B/checkpoints
MOCAP=/data/yuanshuai/python_codes/mocap/checkpoints
HF=https://hf-mirror.com
SOMA_REV=32f0ab41a0db0f2710d6542a435f90ad452c3b73
mkdir -p $C/egohand/soma_x/MHR $C/egohand/soma_x/MANO $C/egohand/detector $C/hawor/checkpoints $C/hawor_external $C/metric3d $C/mano

get() {  # url dest
  local url=$1 dst=$2
  for i in 1 2 3 4 5; do
    curl -fL --retry 3 -C - -s -S -o "$dst.part" "$url" && mv "$dst.part" "$dst" && return 0
    echo "retry $i: $url"; sleep 5
  done
  echo "FAILED $url"; return 1
}

H=ThunderVVV/HaWoR
get $HF/$H/resolve/main/hawor/checkpoints/hawor.ckpt        $C/hawor/checkpoints/hawor.ckpt
get $HF/$H/resolve/main/hawor/model_config.yaml             $C/hawor/model_config.yaml
get $HF/$H/resolve/main/external/detector.pt                $C/hawor_external/detector.pt
get $HF/$H/resolve/main/external/droid.pth                  $C/hawor_external/droid.pth
get $HF/$H/resolve/main/external/metric_depth_vit_large_800k.pth $C/metric3d/metric_depth_vit_large_800k.pth

for f in SOMAHand.npz SOMA_neutral.npz SOMA_template_rig.usda SOMA_procedural_transforms.json LICENSE \
         MHR/mhr_model_lod1.pt MHR/base_body_lod1.obj MHR/SOMA_wrap_lod1.obj \
         MANO/base_hand_left.obj MANO/base_hand_right.obj MANO/SOMA_wrap_left.obj MANO/SOMA_wrap_right.obj; do
  get $HF/nvidia/SOMA-X/resolve/$SOMA_REV/$f $C/egohand/soma_x/$f
done
get $HF/nvidia/EgoHand-v1/resolve/main/egohand_s060000.ckpt $C/egohand/egohand_s060000.ckpt

# local copies from the mocap project (read-only use)
cp -n $MOCAP/body_models/mano/MANO_RIGHT.pkl $MOCAP/body_models/mano/MANO_LEFT.pkl $C/mano/
[ -d $C/unidepth-v2-vitl14 ] || cp -r $MOCAP/unidepth-v2-vitl14 $C/
# hand_detector.onnx (RT-DETR) is not on hf-mirror: relayed from server 45 into $C/egohand/detector/

cd $C && sha256sum -c <<'EOF'
4d1cc43853c190d6f2c10d9b6295c73109f0faf9ef41ac817a2b31d94b4823f2  hawor/checkpoints/hawor.ckpt
edfe12dc14ce371d698da722b59acfed5b4a38a7f8f5116cbc1fce459a07dd2d  hawor/model_config.yaml
5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033  hawor_external/detector.pt
46476ef64cde45a97504910d6f3de2eef7b398ec1c6e4e668815c29076024526  hawor_external/droid.pth
15328ffc42b528b95f188687418f6f03b3f123eb34ccdbd686c112abbea6d972  metric3d/metric_depth_vit_large_800k.pth
233e9225c1da05c6a3dd98107b9f027f2366e3b972bbe67e0d164e5d28619fff  egohand/soma_x/SOMAHand.npz
5829af4a5333abdba5ad449360b868faacee99d395971e62f35c0e7169432b63  egohand/soma_x/SOMA_neutral.npz
a952cb7bdf4ccd801e355d53122d395b985b69d0d9181439edad5081bb171856  egohand/soma_x/SOMA_template_rig.usda
9853e19ab51625ffaa7d9e7ccf0e443272c2740837604e6bacd66604e8edf509  egohand/soma_x/SOMA_procedural_transforms.json
d1a7d615ab8eff4de143b1456f46dabf232f54daf0fcf9a70442bb6f637a9e95  egohand/soma_x/LICENSE
e8bbb3f3681d76737635c10a9e6f6a058ad3d9be5a75547f84a578cea572e916  egohand/soma_x/MHR/base_body_lod1.obj
352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc  egohand/soma_x/MHR/mhr_model_lod1.pt
9ab7a2328d5b026c17ef1d09ed692ffeb68f96d3fc74970cba72e667dbe0f728  egohand/soma_x/MHR/SOMA_wrap_lod1.obj
fd8b895d18e38cad9f082e4e7f8c66534c70253eedebdf77c5cd4987fcad54b8  egohand/soma_x/MANO/base_hand_left.obj
9d83807a5852cf35a85da421e32d6168de6e33c2ce6f42dd531aecc88febc3b6  egohand/soma_x/MANO/base_hand_right.obj
e198a37e9d33e32176d6325d47d48c153b5429dce5d248769ac03959b9c28aaa  egohand/soma_x/MANO/SOMA_wrap_left.obj
5c5f8dc18bd8386aacee78dfa4437f3cd1cfbad066ae3772d2f666db1d047771  egohand/soma_x/MANO/SOMA_wrap_right.obj
f940ecb6e3b73d5bd9e562cff232550e1c072f6c063eb4f705951122b15d8a75  egohand/egohand_s060000.ckpt
c4022f7083f2ca7c78b2b3d595abbab52debd32b09d372b16923a801f0ea6a30  mano/MANO_LEFT.pkl
45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767  mano/MANO_RIGHT.pkl
85d36428cc1ecf8832ca2d48f7aa04df0659cf94dde358352fb3cf651a319810  egohand/detector/hand_detector.onnx
EOF
echo "fetch_weights done"
