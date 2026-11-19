"""Run many videos on several GPUs (default: 4,5,6,7 from configs/default.yaml), one worker per GPU.

Input list: one video per line, optionally followed by an intrinsics json:  /abs/a.mp4 [/abs/a_K.json]
If no json is given and <video>.json / <video stem>_intrinsics.json exists next to the video, it is used.
Before each video a worker waits until its GPU has >= min_free_mb free (other users share these cards).
Finished videos (manifest status complete) are skipped, so the same command can be re-run to resume.

  python tools/launch_batch.py --list videos.txt --run_dir /data/wenwang/work/batch_x [--gpus 4,5,6,7]
"""
import argparse
import json
import multiprocessing as mp
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from processors.common import load_config  # noqa: E402
from tools.run_video import run  # noqa: E402


def free_mb(gpu):
    out = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free",
                                   "--format=csv,noheader,nounits"], text=True)
    return int(out.strip())


def worker(gpu, queue, cfg, cfg_path, run_dir, results):
    while True:
        try:
            item = queue.get_nowait()
        except Exception:
            return
        video, intr, vid = item
        while free_mb(gpu) < cfg["min_free_mb"]:
            print(f"[gpu {gpu}] waiting for {cfg['min_free_mb']} MiB free (now {free_mb(gpu)})", flush=True)
            time.sleep(60)
        t0 = time.time()
        try:
            run(video, Path(run_dir) / vid, vid, cfg, cfg_path, intr, gpu)
            results[vid] = dict(status="complete", gpu=gpu, seconds=round(time.time() - t0, 1))
        except Exception as e:
            traceback.print_exc()
            results[vid] = dict(status="failed", gpu=gpu, error=str(e)[:500])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--gpus", default=None, help="comma list, overrides config")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    gpus = [int(g) for g in a.gpus.split(",")] if a.gpus else cfg["gpus"]
    run_dir = Path(a.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    queue, todo = mp.Queue(), 0
    for line in open(a.list):
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        video = Path(parts[0]).resolve()
        intr = parts[1] if len(parts) > 1 else None
        if intr is None:
            for cand in (video.with_suffix(".json"), video.with_name(video.stem + "_intrinsics.json")):
                if cand.exists():
                    intr = str(cand)
                    break
        vid = video.stem
        mf = run_dir / vid / "manifest.json"
        if mf.exists() and json.loads(mf.read_text()).get("status") == "complete":
            continue
        queue.put((video, intr, vid))
        todo += 1
    print(f"{todo} videos to process on GPUs {gpus} -> {run_dir}", flush=True)
    with mp.Manager() as mgr:
        results = mgr.dict()
        procs = [mp.Process(target=worker, args=(g, queue, cfg, a.config, run_dir, results)) for g in gpus]
        for p in procs:
            p.start()
            time.sleep(5)  # stagger model loading
        for p in procs:
            p.join()
        res = dict(results)
    prev = run_dir / "batch_status.json"
    allres = json.loads(prev.read_text()) if prev.exists() else {}
    allres.update(res)
    prev.write_text(json.dumps(allres, indent=1))
    n_fail = sum(r["status"] != "complete" for r in res.values())
    print(f"done: {len(res) - n_fail} complete, {n_fail} failed; status in {prev}", flush=True)


if __name__ == "__main__":
    main()
