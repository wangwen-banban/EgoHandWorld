"""Turn a finished EgoHandWorld run into an example case shown on the web front page (hawor env).

  python web/import_case.py --vdir /data/wenwang/work/smoke_v2/egodex-pour-12 --id case-pour \
      --title "倒水" --desc "..." [--source "EgoDex test"] [--intrinsics "GT"]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, ROOT, cases  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vdir", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--desc", default="")
    ap.add_argument("--source", default="")
    ap.add_argument("--intrinsics", default="")
    a = ap.parse_args()
    if not a.id.replace("-", "").isalnum():
        raise SystemExit("id: letters, digits and '-' only")
    subprocess.run([sys.executable, str(ROOT / "web" / "render.py"), "--vdir", a.vdir, "--out", str(DATA / a.id / "view")],
                   check=True)
    summary = json.loads((DATA / a.id / "view" / "summary.json").read_text())
    entry = dict(id=a.id, title=a.title, desc=a.desc, source=a.source, intrinsics=a.intrinsics,
                 frames=summary["num_frames"], fps=summary["fps"],
                 hands={h: v["valid_frames"] for h, v in summary["hands"].items()})
    allc = [c for c in cases() if c["id"] != a.id] + [entry]
    (DATA / "cases.json").write_text(json.dumps(allc, ensure_ascii=False, indent=1))
    print("case", a.id, "imported:", entry)


if __name__ == "__main__":
    main()
