"""Regression check on the EgoDex smoke clips: compare result.npz against EgoDex GT, same protocol as server 45.

World anchoring (no scale): A = c2w_gt[0] @ inv(S0), S0 = SLAM c2w of frame 0; p_eval = A @ p_world.
Metrics on frames with an EgoHand prediction (mm): W-MPJPE, WA-MPJPE (one Sim3 over the clip), PA-MPJPE, AccErr.
Prints them next to the server-45 numbers ("egohand+hawor_slam") from eval/metrics_independent.json.

  python tools/eval_egodex.py --run_dir <work/run> [--data /data/wenwang/data/egodex_smoke]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def procrustes(X, Y, scale=True):
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Xc.T @ Yc)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = (U @ D @ Vt).T
    s = (S * np.diag(D)).sum() / (Xc ** 2).sum() if scale else 1.0
    return s * Xc @ R.T + my


def metrics(P, G):
    w = np.linalg.norm(P - G, axis=-1).mean() * 1000
    wa = np.linalg.norm(procrustes(P.reshape(-1, 3), G.reshape(-1, 3)).reshape(P.shape) - G, axis=-1).mean() * 1000
    pa = np.mean([np.linalg.norm(procrustes(p, g) - g, axis=-1).mean() for p, g in zip(P, G)]) * 1000
    acc = np.linalg.norm((P[2:] - 2 * P[1:-1] + P[:-2]) - (G[2:] - 2 * G[1:-1] + G[:-2]), axis=-1).mean() * 900
    return {"W-MPJPE": w, "WA-MPJPE": wa, "PA-MPJPE": pa, "AccErr": acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data", default="/data/wenwang/data/egodex_smoke")
    a = ap.parse_args()
    D = Path(a.data)
    ref = json.load(open(D / "eval" / "metrics_independent.json"))
    out, rows = {}, []
    for line in open(D / "clip_id_map.tsv"):
        _, clip = line.split()
        f = Path(a.run_dir) / clip / "result.npz"
        if not f.exists():
            print("missing", f)
            continue
        r = np.load(f)
        g = np.load(D / "gt" / f"{clip}.npz")
        c2w_gt = np.linalg.inv(g["w2c"].astype(np.float64))
        S0 = np.eye(4); S0[:3, :3] = r["R_c2w"][0]; S0[:3, 3] = r["t_c2w"][0]
        A = c2w_gt[0] @ np.linalg.inv(S0)
        T = min(len(r["R_c2w"]), len(g["frames"]))
        for hi, hand in enumerate(("left", "right")):
            v = r[f"{hand}_valid"][:T]
            if v.sum() < 3:
                continue
            P = r[f"{hand}_joints_world"][:T].astype(np.float64) @ A[:3, :3].T + A[:3, 3]
            G = g["joints_world"][hi][:T].astype(np.float64)
            m = metrics(P[v], G[v])
            old = ref.get(clip, {}).get(hand, {}).get("egohand+hawor_slam", {})
            out.setdefault(clip, {})[hand] = dict(ours=m, server45=old, n=int(v.sum()), T=T)
            rows.append((clip[7:35], hand, int(v.sum()), T, m, old))
    print(f"{'clip':28s} hand  n/T      | W    WA   PA   (92)    | W    WA   PA   (45)")
    for c, h, n, T, m, o in rows:
        f = lambda d: " ".join(f"{d.get(k, float('nan')):4.0f}" for k in ("W-MPJPE", "WA-MPJPE", "PA-MPJPE"))
        print(f"{c:28s} {h:5s} {n:3d}/{T:<4d} | {f(m)}        | {f(o)}")
    Path(a.run_dir, "eval_egodex.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
