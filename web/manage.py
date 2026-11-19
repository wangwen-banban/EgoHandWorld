"""Start / stop / status of the web server (gunicorn on 127.0.0.1:<port>) and the background worker.

  /data/wenwang/envs/web/bin/python web/manage.py start|stop|status
Not started on boot: after a server reboot run `start` again. Stopping interrupts a running task (it is marked failed
when the worker comes back); queued tasks stay queued.
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, ROOT, SETTINGS  # noqa: E402

WEB_ENV = Path(sys.executable).parent


def alive(pid, marker):
    try:
        return marker in Path(f"/proc/{pid}/cmdline").read_text()
    except OSError:
        return False


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    if action not in ("start", "stop", "status", "restart"):
        raise SystemExit(__doc__)
    if action == "restart":
        sys.argv[1] = "stop"; main(); sys.argv[1] = "start"; main(); return
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    procs = {
        "worker": ([sys.executable, str(ROOT / "web" / "worker.py")], "web/worker.py"),
        "server": ([str(WEB_ENV / "gunicorn"), "--chdir", str(ROOT / "web"), "--bind", f"{SETTINGS['bind']}:{SETTINGS['port']}",
                    "--workers", "2", "--threads", "8", "--timeout", "600", "app:app"], "app:app"),
    }
    for name, (cmd, marker) in procs.items():
        pidf = DATA / f"{name}.pid"
        pid = int(pidf.read_text()) if pidf.exists() else 0
        running = alive(pid, marker)
        if action == "start" and not running:
            with open(DATA / f"{name}.log", "a") as log:
                p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            pidf.write_text(str(p.pid))
            time.sleep(1.5)
            if p.poll() is not None:
                raise SystemExit(f"{name} failed to start; see {DATA / (name + '.log')}")
            print(name, "started", p.pid)
        elif action == "stop" and running:
            os.killpg(pid, signal.SIGTERM)
            for _ in range(60):
                if not alive(pid, marker):
                    break
                time.sleep(0.5)
            pidf.unlink(missing_ok=True)
            print(name, "stopped")
        else:
            print(name, "running" if running else "stopped")
    if action in ("start", "status"):
        print(f"http://127.0.0.1:{SETTINGS['port']}  (from your computer: ssh -N -L 15100:127.0.0.1:{SETTINGS['port']} remote92,"
              f" then open http://127.0.0.1:15100)")


if __name__ == "__main__":
    main()
