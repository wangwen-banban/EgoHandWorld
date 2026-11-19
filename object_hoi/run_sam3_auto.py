"""Automated 2-pass SAM 3 video segmentation wrapper (no GUI clicks)."""
import argparse
import glob
import os
import subprocess
import numpy as np
from PIL import Image

DAID_RECON = os.path.expanduser("~/code/HandReconstruction/do-as-i-do/reconstruction")
SAM3_PY = "/data/wenwang/HandReconstruction/envs/daid_sam3/bin/python"
SAM3_SCRIPT = f"{DAID_RECON}/scripts/run_sam3_video.py"


def mask_area(path: str) -> int:
    if not os.path.exists(path):
        return 0
    arr = np.array(Image.open(path).convert("L"))
    return int((arr > 128).sum())


def find_masks(mask_root: str, name: str):
    return sorted(glob.glob(f"{mask_root}/frame_*_masks/{name}.png"))


def run_cmd(cmd, gpu=None):
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=DAID_RECON, env=env)


def infer_num_frames(video_dir: str, num_frames: int | None) -> int:
    if num_frames is not None:
        return num_frames
    frames = glob.glob(f"{video_dir}/all_frames/[0-9][0-9][0-9][0-9][0-9][0-9].png")
    if frames:
        return len(frames)
    return 150


def run_auto(video_dir: str, prompt: str, output_name: str, fallback_prompts=(), num_frames: int | None = None, gpu=None):
    video_mp4 = f"{video_dir}/color.mp4"
    mask_root = f"{video_dir}/video_segmentation/masks"
    prompts = [prompt] + list(fallback_prompts)
    T = infer_num_frames(video_dir, num_frames)
    scan_frames = []
    for fidx in (0, T // 3, (2 * T) // 3):
        if 0 <= fidx < T and fidx not in scan_frames:
            scan_frames.append(fidx)

    found_frames = []
    # Scan frame 0 across all prompts first, then T//3, then 2*T//3
    for fidx in scan_frames:
        for p in prompts:
            cmd = [
                SAM3_PY, SAM3_SCRIPT,
                "--video", video_mp4,
                "--text", p,
                "--frame_idx", str(fidx),
                "--obj_id", output_name,
            ]
            run_cmd(cmd, gpu=gpu)
            files = find_masks(mask_root, output_name)
            areas = [(f, mask_area(f)) for f in files]
            valid = [(f, a) for f, a in areas if a >= 80]
            if valid:
                found_frames = valid
                print(f"[SAM3 auto] frame={fidx} prompt={p!r} -> {len(valid)}/{T} non-empty masks", flush=True)
                break
            print(f"[SAM3 auto] frame={fidx} prompt={p!r} yielded 0 valid masks, trying next...", flush=True)
        if found_frames:
            break

    if not found_frames:
        raise RuntimeError(f"SAM 3 failed all prompts {prompts} at frames {scan_frames} for {output_name} in {video_dir}")

    f0_path = f"{mask_root}/frame_000000_masks/{output_name}.png"
    need_pass2 = mask_area(f0_path) < 80 or len(found_frames) < int(0.95 * T)
    if need_pass2:
        best_file, best_area = max(found_frames, key=lambda x: x[1])
        fidx = int(os.path.basename(os.path.dirname(best_file)).split("_")[1])
        arr = np.array(Image.open(best_file).convert("L"))
        ys, xs = np.where(arr > 128)
        cx = int(np.median(xs))
        cy = int(np.median(ys))
        print(f"[SAM3 auto] Pass 2 point prompt ({cx},{cy}) at frame {fidx} (area={best_area})", flush=True)
        cmd2 = [
            SAM3_PY, SAM3_SCRIPT,
            "--video", video_mp4,
            "--points", f"{cx},{cy}", "--point_labels", "1",
            "--frame_idx", str(fidx),
            "--obj_id", output_name,
        ]
        run_cmd(cmd2, gpu=gpu)
        files2 = find_masks(mask_root, output_name)
        valid2 = [f for f in files2 if mask_area(f) >= 80]
        print(f"[SAM3 auto] After Pass 2: {len(valid2)}/{T} non-empty masks, f0 area={mask_area(f0_path)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--output_name", required=True)
    ap.add_argument("--fallback", nargs="*", default=[])
    ap.add_argument("--num_frames", type=int, default=None)
    ap.add_argument("--gpu", default=None)
    args = ap.parse_args()
    run_auto(args.video_dir, args.prompt, args.output_name, args.fallback, args.num_frames, args.gpu)
