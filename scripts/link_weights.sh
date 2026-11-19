#!/usr/bin/env bash
# Point the relative weight paths that HaWoR's official code expects at $B/checkpoints (run once after fetch_weights.sh).
set -euo pipefail
B=/data/wenwang
P=/home/wangwen/code/EgoHandWorld
C=$B/checkpoints
H=$P/third_party/hawor
ln -sfn $C $P/checkpoints
mkdir -p $H/weights $H/_DATA/data/mano $H/_DATA/data_left/mano_left $H/thirdparty/Metric3D/weights
ln -sfn $C/hawor $H/weights/hawor                    # weights/hawor/checkpoints/hawor.ckpt, weights/hawor/model_config.yaml
ln -sfn $C/hawor_external $H/weights/external        # weights/external/{detector.pt,droid.pth}
ln -sfn $C/metric3d/metric_depth_vit_large_800k.pth $H/thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth
ln -sfn $C/mano/MANO_RIGHT.pkl $H/_DATA/data/mano/MANO_RIGHT.pkl
ln -sfn $C/mano/MANO_LEFT.pkl $H/_DATA/data_left/mano_left/MANO_LEFT.pkl
ls -l $P/checkpoints $H/weights $H/_DATA/data/mano $H/_DATA/data_left/mano_left $H/thirdparty/Metric3D/weights
