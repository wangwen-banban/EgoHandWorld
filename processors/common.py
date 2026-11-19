"""Shared paths, config loading and small helpers (numpy/pyyaml only, so every env can import it)."""
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HAWOR_DIR = ROOT / "third_party" / "hawor"
EGOHAND_DIR = ROOT / "third_party" / "egohand"

# EgoHand SOMA-Hand 25 joints -> OpenPose/MANO-style 21 joints (drop the 4 finger metacarpals 5/10/15/20).
SOMA25_TO_OPENPOSE21 = [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24]
HANDS = ("left", "right")


def load_config(path=None):
    import yaml
    with open(path or ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def video_dir(run_dir, video_id):
    return Path(run_dir) / video_id


def read_intrinsics(path):
    """Accept {fx,fy,cx,cy[,width,height]} or EgoHand's PINHOLE json; return dict fx,fy,cx,cy,width,height."""
    d = json.load(open(path))
    if "camera_matrix" in d:
        K = np.asarray(d["camera_matrix"]["matrix_3x3"], float)
        return dict(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
                    width=d.get("image_width"), height=d.get("image_height"))
    return {k: d.get(k) for k in ("fx", "fy", "cx", "cy", "width", "height")}


def write_intrinsics(vdir, intr, source):
    """intrinsics.json (ours) + egohand_intrinsics.json (EgoHand PINHOLE format)."""
    intr = {k: float(intr[k]) for k in ("fx", "fy", "cx", "cy", "width", "height")}
    json.dump(dict(intr, source=source), open(Path(vdir) / "intrinsics.json", "w"), indent=1)
    K = [[intr["fx"], 0.0, intr["cx"]], [0.0, intr["fy"], intr["cy"]], [0.0, 0.0, 1.0]]
    json.dump({"model": "PINHOLE", "image_width": int(intr["width"]), "image_height": int(intr["height"]),
               "camera_matrix": {"matrix_3x3": K}, "distortion_coefficients": []},
              open(Path(vdir) / "egohand_intrinsics.json", "w"), indent=1)


def probe_video(path):
    import subprocess
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets", "-of", "json",
                          str(path)], capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return dict(width=int(s["width"]), height=int(s["height"]), fps=float(num) / float(den),
                frames=int(s.get("nb_read_packets", 0)))
