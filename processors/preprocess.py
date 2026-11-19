"""Stage 1 (hawor env, CPU): re-encode to a fixed fps and record any given intrinsics.

Writes into <vdir>/:  <video_id>.mp4 (fps-normalised, the only video later stages read), video_info.json,
given_intrinsics.json (only when --intrinsics is passed). The final intrinsics.json is written by the camera stage.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from processors.common import load_config, probe_video, read_intrinsics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--vdir", required=True)
    ap.add_argument("--video_id", required=True)
    ap.add_argument("--intrinsics", default=None)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    vdir = Path(a.vdir)
    vdir.mkdir(parents=True, exist_ok=True)
    out = vdir / f"{a.video_id}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.video, "-an", "-vf", f"fps={cfg['fps']}",
                    "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", str(out)], check=True)
    info = probe_video(out)
    json.dump(info, open(vdir / "video_info.json", "w"), indent=1)
    print("normalised video:", info, flush=True)

    given = vdir / "given_intrinsics.json"
    given.unlink(missing_ok=True)
    if a.intrinsics:
        intr = read_intrinsics(a.intrinsics)
        orig = probe_video(a.video)
        if intr.get("width") and (intr["width"], intr["height"]) != (orig["width"], orig["height"]):
            raise ValueError(f"intrinsics are for {intr['width']}x{intr['height']}, video is {orig['width']}x{orig['height']}")
        intr.update(width=info["width"], height=info["height"], source=f"given:{a.intrinsics}")
        json.dump(intr, open(given, "w"), indent=1)
        print("given intrinsics:", intr, flush=True)
    elif cfg.get("camera_backend", "vipe") == "hawor":
        raise SystemExit("camera_backend=hawor needs --intrinsics (HaWoR cannot estimate them); "
                         "use camera_backend=vipe to estimate intrinsics.")


if __name__ == "__main__":
    main()
