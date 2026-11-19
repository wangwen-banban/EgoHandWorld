"""EgoHandWorld web service: upload a video, follow its progress, view 2D overlay + 3D world result, browse cases.

Listens on 127.0.0.1 only (see manage.py). Access control: requests must use an allowed Host (localhost, or the name
in DATA/public-host); if DATA/access.json exists, HTTP Basic auth is required (werkzeug password hash).
Task ids are random; the index page lists only cases and the tasks this browser submitted (kept in localStorage).
"""
import json
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from werkzeug.security import check_password_hash

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, PUBLIC_FILES, SETTINGS, cases, db, public  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = SETTINGS["max_upload_bytes"]
AUTH_FILE, PUBLIC_HOST_FILE = DATA / "access.json", DATA / "public-host"


@app.before_request
def access_control():
    allowed = {"127.0.0.1", "localhost"}
    if PUBLIC_HOST_FILE.exists():
        allowed.update(h.strip() for h in PUBLIC_HOST_FILE.read_text().splitlines() if h.strip())  # one host per line
    if urlsplit("http://" + request.host).hostname not in allowed:
        abort(403)
    if request.method == "POST" and request.headers.get("Origin"):
        if urlsplit(request.headers["Origin"]).netloc != request.host:
            abort(403)
    if AUTH_FILE.exists():
        cfg = json.loads(AUTH_FILE.read_text())
        auth = request.authorization
        if not (auth and auth.username == cfg["username"] and check_password_hash(cfg["password_hash"], auth.password or "")):
            return jsonify(error="请登录"), 401, {"WWW-Authenticate": 'Basic realm="EgoHandWorld", charset="UTF-8"'}
    return None


@app.after_request
def headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    if resp.mimetype == "application/json":
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.errorhandler(413)
def too_large(_):
    return jsonify(error=f"文件超过 {SETTINGS['max_upload_bytes'] // 1024 ** 2} MB 上限"), 413


def find_task(task_id):
    if not task_id.replace("-", "").isalnum():
        abort(404)
    with db() as c:
        row = c.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    return row


@app.get("/")
def index():
    return render_template("index.html", settings=SETTINGS, cases=cases())


@app.get("/t/<task_id>")
def task_page(task_id):
    case = next((c for c in cases() if c["id"] == task_id), None)
    if case is None and find_task(task_id) is None:
        abort(404)
    return render_template("task.html", task_id=task_id, case=case)


@app.get("/api/health")
def health():
    hb = DATA / "worker.heartbeat"
    with db() as c:
        queued = c.execute("SELECT COUNT(*) FROM tasks WHERE status IN ('queued','running')").fetchone()[0]
    return jsonify(worker_alive=hb.exists() and time.time() - hb.stat().st_mtime < 30, queue=queued,
                   max_seconds=SETTINGS["max_seconds"], max_upload_mb=SETTINGS["max_upload_bytes"] // 1024 ** 2)


@app.get("/api/cases")
def api_cases():
    return jsonify(cases())


@app.get("/api/tasks/<task_id>")
def api_task(task_id):
    case = next((c for c in cases() if c["id"] == task_id), None)
    if case is not None:
        return jsonify(dict(id=task_id, name=case["title"], status="complete", stage="complete", stage_label="完成",
                            summary=json.loads((DATA / task_id / "view" / "summary.json").read_text()), case=case))
    row = find_task(task_id)
    if row is None:
        abort(404)
    d = public(row)
    if d["status"] == "queued":
        with db() as c:
            d["queue_position"] = c.execute("SELECT COUNT(*) FROM tasks WHERE status='queued' AND created<?",
                                            (d["created"],)).fetchone()[0] + 1
    return jsonify(d)


@app.post("/api/tasks")
def create_task():
    f = request.files.get("video")
    if not f or not f.filename:
        return jsonify(error="请选择视频文件"), 400
    intr = None
    vals = [request.form.get(k, "").strip() for k in ("fx", "fy", "cx", "cy")]
    if any(vals):
        try:
            fx, fy, cx, cy = (float(v) for v in vals)
            assert fx > 0 and fy > 0
        except (ValueError, AssertionError):
            return jsonify(error="内参需要同时填写 fx、fy、cx、cy 四个正数，或全部留空"), 400
    task_id = uuid.uuid4().hex
    folder = DATA / task_id
    folder.mkdir()
    ext = Path(f.filename).suffix.lower()[:8] or ".mp4"
    path = folder / f"input{ext if ext.replace('.', '').isalnum() else '.mp4'}"
    try:
        f.save(path)
        p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height:format=duration", "-of", "json", str(path)],
                           capture_output=True, text=True, timeout=30)
        info = json.loads(p.stdout) if p.returncode == 0 else {}
        if not info.get("streams"):
            raise ValueError("文件不是可读取的视频")
        dur = float(info.get("format", {}).get("duration", 0))
        if not SETTINGS["min_seconds"] <= dur <= SETTINGS["max_seconds"]:
            raise ValueError(f"视频时长 {dur:.1f} 秒，目前支持 {SETTINGS['min_seconds']:g}–{SETTINGS['max_seconds']:g} 秒")
        st = info["streams"][0]
        if any(vals):
            intr = json.dumps(dict(fx=fx, fy=fy, cx=cx, cy=cy, width=st["width"], height=st["height"]))
        with db() as c:
            c.execute("INSERT INTO tasks(id,name,created,status,stage,intrinsics) VALUES(?,?,?,?,?,?)",
                      (task_id, f.filename[:200], time.time(), "queued", "queued", intr))
    except (ValueError, subprocess.TimeoutExpired) as e:
        shutil.rmtree(folder)
        return jsonify(error=str(e) if isinstance(e, ValueError) else "视频读取超时"), 400
    except Exception:
        shutil.rmtree(folder)
        raise
    return jsonify(id=task_id, status="queued"), 202


@app.get("/files/<task_id>/<name>")
def files(task_id, name):
    if name not in PUBLIC_FILES:
        abort(404)
    is_case = any(c["id"] == task_id for c in cases())
    if not is_case:
        row = find_task(task_id)
        if row is None or row["status"] != "complete":
            abort(404)
    return send_from_directory(DATA / task_id / "view", name, as_attachment=name in ("result.npz", "summary.json"),
                               conditional=True, max_age=3600)
