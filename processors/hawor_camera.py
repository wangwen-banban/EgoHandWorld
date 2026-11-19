"""Stage 2, optional backend (camera_backend: hawor; hawor env, GPU): camera trajectory from HaWoR's official
pipeline, unchanged. Needs given intrinsics.

detect_track_video (WiLoR YOLO hand boxes) -> hawor_motion_estimation (HaWoR; renders the hand masks that SLAM
needs) -> hawor_slam (masked DROID-SLAM + Metric3D metric scale). HaWoR's infiller is not run.

Writes <vdir>/camera.npz:
  R_c2w (T,3,3), t_c2w (T,3)  SLAM world = first camera frame, metres (scale already applied)
  scale, focal, center
and <vdir>/hawor_cam.npz: HaWoR camera-space MANO per hand (by-product, NaN where the hand was not tracked):
  {hand}_valid (T,), {hand}_rot_aa (T,3), {hand}_pose_aa (T,45), {hand}_trans (T,3), {hand}_betas (T,10),
  {hand}_joints_cam (T,21,3)
"""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from processors.common import HAWOR_DIR, HANDS, write_intrinsics  # noqa: E402

os.chdir(HAWOR_DIR)
sys.path.insert(0, str(HAWOR_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scripts.scripts_test_video.detect_track_video import detect_track_video  # noqa: E402
from scripts.scripts_test_video.hawor_video import hawor_motion_estimation  # noqa: E402
from scripts.scripts_test_video.hawor_slam import hawor_slam  # noqa: E402
from hawor.utils.process import run_mano, run_mano_left  # noqa: E402
from hawor.utils.rotation import rotation_matrix_to_angle_axis  # noqa: E402
from lib.eval_utils.custom_utils import load_slam_cam  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vdir", required=True)
    ap.add_argument("--video_id", required=True)
    ap.add_argument("--config", default=None)  # accepted for a uniform stage interface
    a = ap.parse_args()
    vdir = Path(a.vdir).resolve()
    intr = json.load(open(vdir / "given_intrinsics.json"))  # preprocess refuses to run the hawor backend without them
    args = SimpleNamespace(video_path=str(vdir / f"{a.video_id}.mp4"), img_focal=float(intr["fx"]), input_type="file",
                           checkpoint="./weights/hawor/checkpoints/hawor.ckpt",
                           infiller_weight="./weights/hawor/checkpoints/infiller.pt", vis_mode="world")
    start, end, seq_folder, imgfiles = detect_track_video(args)
    T = len(imgfiles)
    chunks, focal = hawor_motion_estimation(args, start, end, seq_folder)
    slam = f"{seq_folder}/SLAM/hawor_slam_w_scale_{start}_{end}.npz"
    if not os.path.exists(slam):
        hawor_slam(args, start, end)
    _, _, R_c2w, t_c2w = load_slam_cam(slam)
    s = np.load(slam)
    np.savez(vdir / "camera.npz", R_c2w=R_c2w.numpy().astype(np.float64), t_c2w=t_c2w.numpy().astype(np.float64),
             scale=float(s["scale"]), focal=float(focal), center=np.asarray(s["img_center"], float), num_frames=T,
             backend="hawor")
    write_intrinsics(vdir, intr, intr["source"])

    out = {}
    for h, hand in enumerate(HANDS):
        valid = np.zeros(T, bool)
        rot, trans = np.full((T, 3), np.nan), np.full((T, 3), np.nan)
        pose, betas = np.full((T, 45), np.nan), np.full((T, 10), np.nan)
        joints = np.full((T, 21, 3), np.nan)
        for ck in chunks.get(h, []):
            d = json.load(open(f"{seq_folder}/cam_space/{h}/{ck[0]}_{ck[-1]}.json"))
            d = {k: torch.tensor(x) for k, x in d.items()}
            r = rotation_matrix_to_angle_axis(d["init_root_orient"])[0]
            p = rotation_matrix_to_angle_axis(d["init_hand_pose"])[0].reshape(len(ck), 45)
            f = run_mano_left if h == 0 else run_mano
            j = f(d["init_trans"][0][None], r[None], p[None], betas=d["init_betas"][0][None])["joints"][0]
            valid[ck], rot[ck], pose[ck] = True, r.numpy(), p.numpy()
            trans[ck], betas[ck], joints[ck] = d["init_trans"][0].numpy(), d["init_betas"][0].numpy(), j.cpu().numpy()
        out.update({f"{hand}_valid": valid, f"{hand}_rot_aa": rot, f"{hand}_pose_aa": pose,
                    f"{hand}_trans": trans, f"{hand}_betas": betas, f"{hand}_joints_cam": joints})
    np.savez_compressed(vdir / "hawor_cam.npz", **out)
    print(f"camera done: T={T} scale={float(s['scale']):.3f} focal={focal} "
          f"hawor tracked L/R={out['left_valid'].sum()}/{out['right_valid'].sum()}", flush=True)


if __name__ == "__main__":
    main()
