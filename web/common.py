"""Shared paths, settings and the SQLite task table for the EgoHandWorld web service (stdlib only)."""
import json
import os
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                      # ~/code/EgoHandWorld
DATA = Path(os.environ.get("EHW_WEB_DATA", "/data/wenwang/web")).resolve()
DATA.mkdir(parents=True, exist_ok=True)

SETTINGS = dict(
    port=5100,
    bind="127.0.0.1",          # 0.0.0.0 to accept connections on the LAN IP (keep access.json login on)
    max_upload_bytes=2 * 1024 ** 3,
    min_seconds=1.0,
    max_seconds=180.0,          # longer videos are not validated yet (ViPE pose_only keyframe buffer)
    min_free_mb=16000,          # a GPU is used only if it has this much free memory
    gpus="all",                 # "all" or a comma list such as "0,3,4,7"
    task_timeout_s=7200,
    hawor_python="/data/wenwang/envs/hawor/bin/python",
)
_override = DATA / "settings.json"
if _override.exists():
    SETTINGS.update(json.loads(_override.read_text()))

STAGE_LABELS = {
    "queued": "排队中", "preprocess": "预处理视频", "vipe_camera": "估计相机内参与轨迹", "hawor_camera": "估计相机轨迹",
    "egohand": "估计手部", "lift": "变换到世界坐标", "render": "生成可视化", "complete": "完成", "failed": "失败",
}
# files a viewer may fetch for a task
PUBLIC_FILES = {"overlay.mp4", "input.mp4", "web.json", "result.npz", "summary.json", "poster.jpg"}


def db():
    c = sqlite3.connect(DATA / "tasks.sqlite", timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("""CREATE TABLE IF NOT EXISTS tasks (
        id TEXT PRIMARY KEY, name TEXT, created REAL, status TEXT, stage TEXT, gpu TEXT,
        intrinsics TEXT, error TEXT DEFAULT '', summary TEXT DEFAULT '{}', started REAL, finished REAL)""")
    return c


def update(task_id, **fields):
    with db() as c:
        c.execute("UPDATE tasks SET " + ",".join(k + "=?" for k in fields) + " WHERE id=?", [*fields.values(), task_id])


def public(row):
    d = dict(row)
    d["summary"] = json.loads(d["summary"] or "{}")
    d["intrinsics"] = json.loads(d["intrinsics"]) if d["intrinsics"] else None
    d["stage_label"] = STAGE_LABELS.get(d["stage"], d["stage"])
    if d.get("started"):
        d["elapsed_s"] = round((d["finished"] or time.time()) - d["started"], 1)
    return d


def cases():
    f = DATA / "cases.json"
    return json.loads(f.read_text()) if f.exists() else []
