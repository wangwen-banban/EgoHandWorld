"""Convert EgoHand {left,right}_results.npz into HaWoR-compatible all_hand_meshes.npz."""
import argparse
import re
import numpy as np

SOMA25_TO_OPENPOSE21 = [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24]
SOMA_PATH = "/data/wenwang/HandReconstruction/models/egohand/soma_x/SOMAHand.npz"


def convert(egohand_color_dir: str, out_path: str, num_frames: int | None = None, soma_path: str = SOMA_PATH):
    soma = np.load(soma_path, allow_pickle=True)
    res = {}
    inferred_T = 0
    raw = {}
    for side in ("left", "right"):
        p = f"{egohand_color_dir}/{side}_results.npz"
        d = np.load(p, allow_pickle=True)
        raw[side] = d
        nf = int(d["num_frames"]) if "num_frames" in d else 0
        inferred_T = max(inferred_T, nf)
        if len(d["frame_names"]) > 0:
            idxs = [int(re.findall(r"\d+", str(n))[-1]) for n in d["frame_names"]]
            inferred_T = max(inferred_T, max(idxs) + 1)
    T = num_frames or inferred_T
    for side in ("left", "right"):
        d = raw[side]
        n_verts = soma[f"{side}_vert_ids"].shape[0]
        verts = np.zeros((T, n_verts, 3), dtype=np.float32)
        joints = np.zeros((T, 21, 3), dtype=np.float32)
        valid = np.zeros((T,), dtype=bool)
        if len(d["frame_names"]) > 0:
            idxs = np.array([int(re.findall(r"\d+", str(n))[-1]) for n in d["frame_names"]], dtype=int)
            ok = (idxs >= 0) & (idxs < T)
            v_cam = d["vertices"][ok] + d["cam_t"][ok, None, :]
            j_cam = (d["joints"][ok] + d["cam_t"][ok, None, :])[:, SOMA25_TO_OPENPOSE21]
            verts[idxs[ok]] = v_cam.astype(np.float32)
            joints[idxs[ok]] = j_cam.astype(np.float32)
            valid[idxs[ok]] = True
        res[f"{side}_vertices"] = verts
        res[f"{side}_faces"] = soma[f"{side}_triangles"].astype(np.int32)
        res[f"{side}_joints"] = joints
        res[f"{side}_valid"] = valid
    np.savez_compressed(out_path, **res)
    print(f"Saved EgoHand HaWoR-compat NPZ ({T} frames): {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--egohand_color_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--num_frames", type=int, default=None)
    args = ap.parse_args()
    convert(args.egohand_color_dir, args.output, args.num_frames)
