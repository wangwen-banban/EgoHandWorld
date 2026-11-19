"""Stage 3 (egohand env, GPU): NVIDIA EgoHand per-frame camera-space SOMA-Hand, official inference script.

Same options as the EgoDex evaluation on server 45: RT-DETR hand detector (ONNX, CPU onnxruntime) with tracking and
box smoothing, perspective cropping, bf16, no mesh export. Results: <vdir>/egohand/<video_id>/{left,right}_results.npz

Then re-scores every emitted hand box with the raw RT-DETR detector -> <vdir>/hand_scores.npz:
  {hand}_score (T,)  best raw detection score whose box centre lies inside EgoHand's box for that frame (0 if none).
EgoHand's tracker starts tracks at 0.15 and interpolates all gaps, so without this a hand-free video gets a full
"hand" track; lift_world.py uses these scores to decide which frames are real.
"""
import json
import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from processors.common import EGOHAND_DIR, load_config  # noqa: E402

DETECTOR_YAML = """detector_backend: rtdetr
rtdetr_config:
  weights: {onnx}
  bbox_smoothing: true
  bbox_size_scale: 0.8
  bbox_size_alpha_grow: 0.12
  bbox_size_alpha_shrink: 0.06
  bbox_max_size_change: 0.01
tracking: true
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vdir", required=True)
    ap.add_argument("--video_id", required=True)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    ck = Path(cfg["data_root"]) / "checkpoints" / "egohand"
    vdir = Path(a.vdir).resolve()
    det_cfg = vdir / "egohand_detector.yaml"
    det_cfg.write_text(DETECTOR_YAML.format(onnx=ck / "detector" / "hand_detector.onnx"))
    e = cfg["egohand"]
    cmd = [sys.executable, "inference/scripts/run_egohand_inference_pipeline.py",
           "--input", str(vdir / f"{a.video_id}.mp4"),
           "--egohand_ckpt", str(ck / "egohand_s060000.ckpt"),
           "--soma_hand_assets", str(ck / "soma_x"),
           "--bbox_detector_config", str(det_cfg),
           "--intrinsics_json", str(vdir / "egohand_intrinsics.json"),
           "--dtype", e["dtype"], "--no_mesh", "--output_dir", str(vdir / "egohand")]
    if e.get("perspective_cropping", True):
        cmd.append("--perspective_cropping")
    cmd += ["--draw_skeleton"] if e.get("save_videos") else ["--no_save_videos"]
    env = dict(os.environ, PYTHONPATH=str(EGOHAND_DIR), PYOPENGL_PLATFORM="egl", HF_HUB_OFFLINE="1")
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=EGOHAND_DIR, env=env, check=True)
    score_hands(vdir, a.video_id, ck / "detector" / "hand_detector.onnx")


def score_hands(vdir, video_id, onnx, batch=8):
    import cv2
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(EGOHAND_DIR))
    from inference.hand_detector_2d.rtdetr.onnx_model import HandDetector
    det = HandDetector(str(onnx), score_thr=0.05)
    cap = cv2.VideoCapture(str(vdir / f"{video_id}.mp4"))
    dets = []
    while True:
        frames = []
        for _ in range(batch):
            ok, f = cap.read()
            if not ok:
                break
            frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        if not frames:
            break
        dets += det.detect_batch(frames)
    T = len(dets)
    out = {}
    for hand in ("left", "right"):
        score = np.zeros(T, np.float32)
        bb = vdir / "egohand" / video_id / f"{video_id}_{hand}_bbox.json"
        boxes = json.load(open(bb)) if bb.exists() else {}
        for name, b in boxes.items():
            t = int(name)
            if b is None or not 0 <= t < T:
                continue
            cx, cy, size = b[:3]
            inside = [d.score for d in dets[t] if abs((d.xyxy[0] + d.xyxy[2]) / 2 - cx) <= size / 2
                      and abs((d.xyxy[1] + d.xyxy[3]) / 2 - cy) <= size / 2]
            score[t] = max(inside, default=0.0)
        out[f"{hand}_score"] = score
        print(f"{hand}: {len(boxes)} EgoHand boxes, {(score >= 0.5).sum()} frames with a detector box >= 0.5", flush=True)
    np.savez(vdir / "hand_scores.npz", **out)


if __name__ == "__main__":
    main()
