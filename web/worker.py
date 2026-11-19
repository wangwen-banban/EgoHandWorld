"""Single background worker: takes queued tasks in order, picks a free GPU, runs the pipeline, builds the view.

Runs as its own process (started by web/manage.py). One task at a time; GPUs are shared with other users, so before
each task it waits until some allowed GPU has >= min_free_mb free.
"""
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, ROOT, SETTINGS, db, update  # noqa: E402

lock = (DATA / "worker.lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)          # refuse to start twice
active = None


def stop(signum, frame):
    if active and active.poll() is None:
        os.killpg(active.pid, signal.SIGTERM)
        try:
            active.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(active.pid, signal.SIGKILL)
    raise SystemExit(0)


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
with db() as c:
    c.execute("UPDATE tasks SET status='failed', stage='failed', error='服务重启，任务中断，请重新提交' WHERE status='running'")


def pick_gpu():
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                                   text=True)
    allowed = None if SETTINGS["gpus"] == "all" else set(str(SETTINGS["gpus"]).split(","))
    best = None
    for line in rows.strip().splitlines():
        idx, free = (x.strip() for x in line.split(","))
        if (allowed is None or idx in allowed) and int(free) >= SETTINGS["min_free_mb"]:
            if best is None or int(free) > best[1]:
                best = (idx, int(free))
    return best[0] if best else None


def run_step(cmd, log_path, task_id, folder, deadline, env):
    global active
    with open(log_path, "a") as log:
        log.write("\n$ " + " ".join(map(str, cmd)) + "\n")
        log.flush()
        active = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        while active.poll() is None:
            (DATA / "worker.heartbeat").touch()
            mf = folder / "run" / "clip" / "manifest.json"
            if mf.exists():
                try:
                    m = json.loads(mf.read_text())
                    st = m.get("stages", {})
                    cam = "hawor_camera" if m.get("camera_backend") == "hawor" else "vipe_camera"
                    running = [k for k in ("preprocess", cam, "egohand", "lift") if k not in st]
                    if running:
                        update(task_id, stage=running[0])
                except (json.JSONDecodeError, OSError):
                    pass
            if time.time() > deadline:
                os.killpg(active.pid, signal.SIGTERM)
                active.wait()
                raise RuntimeError("处理超时")
            time.sleep(2)
        code = active.returncode
        active = None
    return code


def last_error(log_path):
    lines = [l.strip() for l in open(log_path, errors="replace") if l.strip()]
    for l in reversed(lines):
        if "Error" in l or "error" in l or "WARNING" in l or "failed" in l:
            return l[-300:]
    return lines[-1][-300:] if lines else ""


while True:
    (DATA / "worker.heartbeat").touch()
    with db() as c:
        row = c.execute("SELECT * FROM tasks WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
    if not row:
        time.sleep(2)
        continue
    gpu = pick_gpu()
    if gpu is None:
        update(row["id"], stage="queued", error="等待空闲显卡")
        time.sleep(10)
        continue
    tid, folder = row["id"], DATA / row["id"]
    log_path = folder / "pipeline.log"
    update(tid, status="running", stage="preprocess", gpu=gpu, error="", started=time.time())
    env = dict(os.environ, PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES=gpu)
    deadline = time.time() + SETTINGS["task_timeout_s"]
    try:
        video = next(folder.glob("input.*"))
        cmd = [SETTINGS["hawor_python"], "tools/run_video.py", "--video", str(video), "--run_dir", str(folder / "run"),
               "--video_id", "clip", "--gpu", gpu]
        if row["intrinsics"]:
            kf = folder / "intrinsics.json"
            kf.write_text(row["intrinsics"])
            cmd += ["--intrinsics", str(kf)]
        if run_step(cmd, log_path, tid, folder, deadline, env):
            raise RuntimeError("处理失败：" + last_error(log_path))
        update(tid, stage="render")
        if run_step([SETTINGS["hawor_python"], "web/render.py", "--vdir", str(folder / "run" / "clip"),
                     "--out", str(folder / "view")], log_path, tid, folder, deadline, env):
            raise RuntimeError("生成可视化失败：" + last_error(log_path))
        summary = json.loads((folder / "view" / "summary.json").read_text())
        update(tid, status="complete", stage="complete", summary=json.dumps(summary), finished=time.time())
    except Exception as e:  # noqa: BLE001 - every failure is reported on the task
        update(tid, status="failed", stage="failed", error=str(e)[:500], finished=time.time())
