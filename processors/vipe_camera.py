"""Stage 2, default backend (hawor env drives a ViPE subprocess, GPU): camera intrinsics + metric trajectory.

Runs NVIDIA ViPE's pose_only pipeline unchanged on the fps-normalised video: GeoCalib intrinsics init, DROID-style
bundle adjustment that also optimises the focal length, MoGe-2 keyframe depth for metric scale. It never uses the
hands, so hand-free videos are fine.

Writes <vdir>/camera.npz (R_c2w, t_c2w in metres, ViPE world = first camera frame, OpenCV axes), vipe_intrinsics.json,
and the final intrinsics.json + egohand_intrinsics.json: the given intrinsics if there are any, else ViPE's.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from processors.common import load_config, write_intrinsics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vdir", required=True)
    ap.add_argument("--video_id", required=True)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    v = cfg["vipe"]
    vdir = Path(a.vdir).resolve()
    out = vdir / "vipe"
    shutil.rmtree(out, ignore_errors=True)
    env = dict(os.environ, TORCH_HOME=v["torch_home"], HF_HOME=v["hf_home"], HF_HUB_OFFLINE="1")
    cmd = [cfg["python"]["vipe"], "run.py", f"pipeline={v['pipeline']}", "streams=raw_mp4_stream",
           f"streams.base_path={vdir / (a.video_id + '.mp4')}", f"streams.frame_end={v['max_frames']}",
           f"pipeline.output.path={out}", "pipeline.output.save_artifacts=true", "pipeline.output.save_viz=false"]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=v["src"], env=env, check=True)

    info = json.load(open(vdir / "video_info.json"))
    pose = np.load(out / "pose" / f"{a.video_id}.npz")
    kk = np.load(out / "intrinsics" / f"{a.video_id}.npz")
    T = info["frames"]
    if len(pose["inds"]) != T or not (pose["inds"] == np.arange(T)).all():
        raise RuntimeError(f"ViPE returned poses for {len(pose['inds'])} of {T} frames")
    C = pose["data"].astype(np.float64)                      # (T,4,4) OpenCV cam2world, metric
    fx, fy, cx, cy = kk["data"].mean(0).tolist()
    vk = dict(fx=fx, fy=fy, cx=cx, cy=cy, width=info["width"], height=info["height"], source="vipe")
    json.dump(vk, open(vdir / "vipe_intrinsics.json", "w"), indent=1)
    np.savez(vdir / "camera.npz", R_c2w=C[:, :3, :3], t_c2w=C[:, :3, 3], scale=1.0, focal=fx,
             center=np.array([cx, cy]), num_frames=T, backend="vipe")

    given = vdir / "given_intrinsics.json"
    if given.exists():
        g = json.load(open(given))
        rel = abs(fx - g["fx"]) / g["fx"]
        msg = f"given fx {g['fx']:.1f} vs ViPE fx {fx:.1f} ({rel:+.1%})"
        if rel > v["focal_mismatch_warn"]:
            print(f"WARNING: {msg} -- check the given intrinsics (wrong resolution or device?)", flush=True)
        else:
            print(msg, flush=True)
        write_intrinsics(vdir, g, g["source"])
    else:
        write_intrinsics(vdir, vk, "vipe")
    print(f"camera done (vipe): T={T} fx={fx:.1f} path length {np.linalg.norm(np.diff(C[:, :3, 3], axis=0), axis=1).sum():.3f} m",
          flush=True)


if __name__ == "__main__":
    main()
