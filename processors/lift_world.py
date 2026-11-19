"""Stage 4 (any env, CPU): lift EgoHand camera-space hands into the SLAM world frame and write the final result.

p_cam   = joints (wrist-local) + cam_t                       (EgoHand, per frame)
p_world = R_c2w(t) @ p_cam + t_c2w(t)                         (HaWoR SLAM, metric)
World frame = first camera frame of the video. No gravity alignment, no infilling.
valid = EgoHand predicted the hand AND the raw detector confirms it (score >= validity.score_thr), or the frame sits in
a gap of <= validity.max_gap frames between confirmed frames; a hand with < validity.min_frames confirmed frames in
the whole video is dropped. Invalid frames are NaN.

Writes <vdir>/result.npz and <vdir>/summary.json (field list in README.md).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from processors.common import HANDS, SOMA25_TO_OPENPOSE21, load_config  # noqa: E402


def load_egohand(vdir, video_id, hand, T):
    eg = vdir / "egohand" / video_id
    out = dict(valid=np.zeros(T, bool), joints_cam_soma25=np.full((T, 25, 3), np.nan),
               hand_poses=np.full((T, 25, 3, 3), np.nan, np.float32), identity_coeffs=np.full((T, 5), np.nan, np.float32),
               global_scale=np.full(T, np.nan), scale_params=np.full((T, 26), np.nan, np.float32),
               cam_t=np.full((T, 3), np.nan), kp2d_presence=np.full((T, 25), np.nan, np.float32),
               bbox_cxcys=np.full((T, 3), np.nan, np.float32))
    f = eg / f"{hand}_results.npz"
    if f.exists():
        d = np.load(f, allow_pickle=True)
        if len(d["frame_names"]):
            idx = np.array([int(re.findall(r"\d+", str(n))[-1]) for n in d["frame_names"]])
            ok = (idx >= 0) & (idx < T)
            i = idx[ok]
            out["valid"][i] = True
            out["joints_cam_soma25"][i] = (d["joints"] + d["cam_t"][:, None, :])[ok]
            for k_out, k_in in (("hand_poses", "hand_poses"), ("identity_coeffs", "identity_coeffs"),
                                ("global_scale", "global_scale"), ("scale_params", "scale_params"),
                                ("cam_t", "cam_t"), ("kp2d_presence", "pred_kp2d_crop_presence")):
                out[k_out][i] = d[k_in][ok]
    bb = eg / f"{video_id}_{hand}_bbox.json"
    if bb.exists():
        for name, v in json.load(open(bb)).items():
            t = int(name)
            if 0 <= t < T and v is not None:
                out["bbox_cxcys"][t] = v[:3]
    return out


def validity(pred, score, vcfg):
    """pred: EgoHand has a prediction; score: raw detector score per frame -> (valid, confirmed)."""
    conf = pred & (score >= vcfg["score_thr"])
    if conf.sum() < vcfg["min_frames"]:
        return np.zeros_like(pred), conf
    valid = conf.copy()
    idx = np.flatnonzero(conf)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= vcfg["max_gap"] + 1:
            valid[a + 1:b] = pred[a + 1:b]
    return valid, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vdir", required=True)
    ap.add_argument("--video_id", required=True)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    vcfg = load_config(a.config)["validity"]
    vdir = Path(a.vdir).resolve()
    cam = np.load(vdir / "camera.npz")
    R, t = cam["R_c2w"], cam["t_c2w"]
    T = len(R)
    intr = json.load(open(vdir / "intrinsics.json"))
    info = json.load(open(vdir / "video_info.json"))
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], float)
    res = dict(fps=info["fps"], K=K, image_size=np.array([intr["width"], intr["height"]]),
               R_c2w=R, t_c2w=t, slam_scale=float(cam["scale"]))
    backend = str(cam["backend"]) if "backend" in cam.files else "hawor"
    summary = dict(video_id=a.video_id, num_frames=T, fps=info["fps"], intrinsics_source=intr["source"],
                   camera_backend=backend, slam_scale=float(cam["scale"]), validity=vcfg, hands={})
    vk = vdir / "vipe_intrinsics.json"
    if vk.exists():
        summary["vipe_fx"] = round(json.load(open(vk))["fx"], 1)
    hw = np.load(vdir / "hawor_cam.npz") if (vdir / "hawor_cam.npz").exists() else None
    sc = np.load(vdir / "hand_scores.npz")
    for hand in HANDS:
        e = load_egohand(vdir, a.video_id, hand, T)
        pred = e.pop("valid")
        score = sc[f"{hand}_score"][:T] if len(sc[f"{hand}_score"]) >= T else np.pad(sc[f"{hand}_score"], (0, T - len(sc[f"{hand}_score"])))
        valid, conf = validity(pred, score, vcfg)
        pc25 = np.where(valid[:, None, None], e.pop("joints_cam_soma25"), np.nan)
        pw25 = np.einsum("tij,tkj->tki", R, pc25) + t[:, None]
        res[f"{hand}_valid"] = valid
        res[f"{hand}_det_score"] = score.astype(np.float32)
        res[f"{hand}_joints_world"] = pw25[:, SOMA25_TO_OPENPOSE21].astype(np.float32)
        res[f"{hand}_joints_world_soma25"] = pw25.astype(np.float32)
        res[f"{hand}_joints_cam"] = pc25[:, SOMA25_TO_OPENPOSE21].astype(np.float32)
        for k, v in e.items():  # SOMA params: NaN where not valid
            vv = v.astype(np.float32 if v.dtype.kind == "f" else v.dtype).copy()
            vv[~valid] = np.nan
            res[f"{hand}_soma_{k}"] = vv
        hs = dict(valid_frames=int(valid.sum()), coverage=round(float(valid.mean()), 3),
                  egohand_frames=int(pred.sum()), detector_confirmed=int(conf.sum()))
        if hw is not None:
            for k in ("valid", "rot_aa", "pose_aa", "trans", "betas", "joints_cam"):
                res[f"{hand}_hawor_mano_{k}"] = hw[f"{hand}_{k}"]
            hs["hawor_tracked_frames"] = int(hw[f"{hand}_valid"].sum())
        summary["hands"][hand] = hs
    np.savez_compressed(vdir / "result.npz", **res)
    json.dump(summary, open(vdir / "summary.json", "w"), indent=1)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
