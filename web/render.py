"""Build the web view of one finished EgoHandWorld run (hawor env: numpy + cv2 + ffmpeg).

  python web/render.py --vdir <run>/<video_id> --out <task>/view

Writes into --out:
  input.mp4    the fps-normalised input, downscaled to <= 960 px wide, H.264 + faststart
  overlay.mp4  same frames with both hands' 21 joints projected with K (only frames judged valid)
  poster.jpg   a middle frame of overlay.mp4
  web.json     per-frame camera poses + world joints for the browser 3D viewer
  result.npz, summary.json  copied from the run for download
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]
COLORS = {"left": (40, 150, 255), "right": (255, 140, 50)}   # BGR: left orange, right blue
MAX_W = 960


def encode(frames_dir, fps, out):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames_dir / "%06d.jpg"),
                    "-c:v", "libx264", "-crf", "23", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(out)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vdir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    vdir, out = Path(a.vdir), Path(a.out)
    vid = vdir.name
    out.mkdir(parents=True, exist_ok=True)
    r = np.load(vdir / "result.npz")
    summary = json.load(open(vdir / "summary.json"))
    fps = float(r["fps"])
    K = r["K"].astype(np.float64)
    T = len(r["R_c2w"])
    W, H = [int(x) for x in r["image_size"]]
    s = min(1.0, MAX_W / W)
    w2, h2 = int(round(W * s / 2) * 2), int(round(H * s / 2) * 2)

    tmp_in, tmp_ov = out / "_in", out / "_ov"
    for d in (tmp_in, tmp_ov):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir()
    cap = cv2.VideoCapture(str(vdir / f"{vid}.mp4"))
    lw = max(2, int(round(3 * w2 / 960)))
    t = 0
    while t < T:
        ok, frame = cap.read()
        if not ok:
            break
        img = cv2.resize(frame, (w2, h2), interpolation=cv2.INTER_AREA) if s < 1 else frame
        cv2.imwrite(str(tmp_in / f"{t:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        ov = img.copy()
        for hand in ("left", "right"):
            if not r[f"{hand}_valid"][t]:
                continue
            p = r[f"{hand}_joints_cam"][t].astype(np.float64)
            if not np.isfinite(p).all() or (p[:, 2] <= 1e-3).any():
                continue
            uv = (p[:, :2] / p[:, 2:3]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
            uv = (uv * s).round().astype(int)
            col = COLORS[hand]
            for i, j in BONES:
                cv2.line(ov, tuple(uv[i]), tuple(uv[j]), col, lw, cv2.LINE_AA)
            for k, q in enumerate(uv):
                cv2.circle(ov, tuple(q), lw + (2 if k == 0 else 1), (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(ov, tuple(q), lw + (1 if k == 0 else 0), col, -1, cv2.LINE_AA)
            cv2.putText(ov, "L" if hand == "left" else "R", tuple(uv[0] + [8, 22]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8 * w2 / 960, col, 2, cv2.LINE_AA)
        cv2.imwrite(str(tmp_ov / f"{t:06d}.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 92])
        t += 1
    n = t
    encode(tmp_in, fps, out / "input.mp4")
    encode(tmp_ov, fps, out / "overlay.mp4")
    shutil.copy(tmp_ov / f"{n // 2:06d}.jpg", out / "poster.jpg")
    shutil.rmtree(tmp_in)
    shutil.rmtree(tmp_ov)

    def q(x):
        return np.round(np.nan_to_num(x.astype(np.float64), nan=0.0), 4).tolist()
    cam = np.concatenate([r["R_c2w"].reshape(T, 9), r["t_c2w"]], axis=1)   # row-major R then t
    web = dict(fps=fps, num_frames=n, width=W, height=H, K=q(K), cam=q(cam[:n]), summary=summary, hands={})
    for hand in ("left", "right"):
        v = r[f"{hand}_valid"][:n]
        web["hands"][hand] = dict(valid=v.astype(int).tolist(), joints=q(r[f"{hand}_joints_world"][:n]),
                                  det_score=np.round(r[f"{hand}_det_score"][:n], 3).tolist())
    (out / "web.json").write_text(json.dumps(web, separators=(",", ":")))
    shutil.copy(vdir / "result.npz", out / "result.npz")
    shutil.copy(vdir / "summary.json", out / "summary.json")
    print(f"view ready: {n} frames, {w2}x{h2}, web.json {(out / 'web.json').stat().st_size / 1e6:.2f} MB", flush=True)


if __name__ == "__main__":
    main()
