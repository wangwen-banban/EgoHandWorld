"""Process one ego video: preprocess -> camera (ViPE by default, or HaWoR) -> egohand -> lift.

Each stage runs as its own subprocess in its own env (frees GPU memory between models), logs to <vdir>/logs/<stage>.log
and is recorded in <vdir>/manifest.json. Re-running skips stages already marked done (use --force to redo all).

  python tools/run_video.py --video /abs/clip.mp4 [--intrinsics K.json] [--run_dir DIR] [--video_id ID] [--gpu 4]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from processors.common import load_config  # noqa: E402

CAMERA = {  # camera_backend -> (stage name, env, module, outputs)
    "vipe": ("vipe_camera", "hawor", "processors/vipe_camera.py", ["camera.npz", "intrinsics.json"]),
    "hawor": ("hawor_camera", "hawor", "processors/hawor_camera.py", ["camera.npz", "hawor_cam.npz", "intrinsics.json"]),
}


def stages(cfg):  # name, env, module, outputs that mark it done
    return [("preprocess", "hawor", "processors/preprocess.py", ["video_info.json"]),
            CAMERA[cfg.get("camera_backend", "vipe")],
            ("egohand", "egohand", "processors/egohand_infer.py", ["hand_scores.npz"]),
            ("lift", "hawor", "processors/lift_world.py", ["result.npz", "summary.json"])]


def run(video, vdir, video_id, cfg, cfg_path=None, intrinsics=None, gpu=None, force=False):
    vdir = Path(vdir)
    (vdir / "logs").mkdir(parents=True, exist_ok=True)
    mpath = vdir / "manifest.json"
    m = json.loads(mpath.read_text()) if mpath.exists() and not force else {}
    m.update(video=str(video), video_id=video_id, intrinsics=intrinsics, gpu=gpu, status="running", stages=m.get("stages", {}))
    mpath.write_text(json.dumps(m, indent=1))
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    m["camera_backend"] = cfg.get("camera_backend", "vipe")
    for name, env_name, module, outputs in stages(cfg):
        if m["stages"].get(name, {}).get("status") == "done" and all((vdir / o).exists() for o in outputs):
            continue
        cmd = [cfg["python"][env_name], str(ROOT / module), "--vdir", str(vdir), "--video_id", video_id]
        if name == "preprocess":
            cmd += ["--video", str(video)] + (["--intrinsics", str(intrinsics)] if intrinsics else [])
        if cfg_path:
            cmd += ["--config", str(cfg_path)]
        t0 = time.time()
        print(f"[{video_id}] {name} ...", flush=True)
        with open(vdir / "logs" / f"{name}.log", "w") as log:
            code = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        m["stages"][name] = dict(status="done" if code == 0 else "failed", seconds=round(time.time() - t0, 1))
        if code:
            m.update(status="failed", failed_stage=name)
            mpath.write_text(json.dumps(m, indent=1))
            raise RuntimeError(f"{video_id}: stage {name} failed (exit {code}); see {vdir / 'logs' / (name + '.log')}")
        mpath.write_text(json.dumps(m, indent=1))
    m["status"] = "complete"
    mpath.write_text(json.dumps(m, indent=1))
    print(f"[{video_id}] complete: {vdir / 'result.npz'}", flush=True)
    return vdir / "result.npz"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--intrinsics", default=None, help="json with fx,fy,cx,cy (or EgoHand PINHOLE format)")
    ap.add_argument("--run_dir", default=None, help="default: <data_root>/work/<timestamp>")
    ap.add_argument("--video_id", default=None, help="default: video file stem")
    ap.add_argument("--gpu", default=None, help="sets CUDA_VISIBLE_DEVICES for the GPU stages")
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    video = Path(a.video).expanduser().resolve()
    if not video.is_file():
        ap.error(f"video not found: {video}")
    run_dir = Path(a.run_dir or Path(cfg["data_root"]) / "work" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    vid = a.video_id or video.stem
    run(video, run_dir / vid, vid, cfg, a.config, a.intrinsics, a.gpu, a.force)


if __name__ == "__main__":
    main()
