"""Route 3: SAM 3D mesh + FoundationPose 6DoF tracking using calibrated smoothed MoGe depth (or custom intrinsics)."""
import argparse
import glob
import json
import os
import sys

for _i, _a in enumerate(sys.argv):
    if _a == "--gpu" and _i + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[_i + 1]
        break

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation as Rot

FP_ROOT = os.environ.get("FP_ROOT", os.path.expanduser("~/code/HandReconstruction/FoundationPose"))
FP_PYDEPS = os.path.join(FP_ROOT, ".pydeps")
FP_MYCPP = os.path.join(FP_ROOT, "mycpp", "build")
DAID_SCRIPTS = os.environ.get("DAID_SCRIPTS", os.path.expanduser("~/code/HandReconstruction/do-as-i-do/reconstruction/scripts"))
sys.path.append(FP_PYDEPS)
for p in (FP_MYCPP, FP_ROOT, DAID_SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor, dr, set_seed
from optimize_translation_scale import raycast_first_hits


def load_init_poses(init_json: str, T: int):
    with open(init_json) as f:
        lay = json.load(f)
    s0 = float(lay.get("translation_scale_optimization", {}).get("mesh_scale", 1.0))
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], T, axis=0)
    seen = np.zeros(T, dtype=bool)
    for o in lay["objects"]:
        fi = int(o["frame_idx"])
        if 0 <= fi < T:
            lts = o["local_to_scene"]
            if "translation_scale_optimization" not in lay:
                s0 = float(lts["scale"][0])
            qw = lts["quat_wxyz_camera_frame"]
            poses[fi, :3, :3] = Rot.from_quat([qw[1], qw[2], qw[3], qw[0]]).as_matrix().astype(np.float32)
            poses[fi, :3, 3] = np.array(lts["translation_camera_frame"], dtype=np.float32)
            seen[fi] = True
    for t in range(T):
        if not seen[t] and seen.any():
            nf = np.where(seen)[0][int(np.argmin(np.abs(np.where(seen)[0] - t)))]
            poses[t] = poses[nf]
    return s0, poses


def load_raw_scale_from_sam3d(ref_layout_json: str, obj: str) -> float:
    with open(ref_layout_json) as f:
        lay = json.load(f)
    for o in lay.get("objects", []):
        if o.get("mesh_obj", "") == f"{obj}.obj":
            s = o["local_to_scene"]["scale"]
            return float(s[0] if isinstance(s, list) else s)
    raise ValueError(f"{obj}.obj not found in {ref_layout_json}")


def compute_hand_pointmap_k(pms, K0, hand_data, hand_side, mask_dir, H_full, W_full):
    T = len(pms)
    h_faces = hand_data[f"{hand_side}_faces"]
    rng = np.random.default_rng(42)
    ks = np.full(T, np.nan, dtype=np.float32)
    fx, fy, cx, cy = float(K0[0, 0]), float(K0[1, 1]), float(K0[0, 2]), float(K0[1, 2])
    for t in range(T):
        mp = f"{mask_dir}/frame_{t:06d}_masks/{hand_side}_hand_0.png"
        if not os.path.exists(mp):
            continue
        hm = np.array(Image.open(mp).convert("L")) > 128
        if hm.sum() < 20:
            continue
        hi = min(t, len(hand_data[f"{hand_side}_vertices"]) - 1)
        hv = hand_data[f"{hand_side}_vertices"][hi]
        hits, hu, hv_px = raycast_first_hits(hv, h_faces, hm, fx, fy, cx, cy, max_rays=300, rng=rng)
        if len(hits) >= 10:
            z_real = float(hits[:, 2].mean())
            z_pm = float(pms[t][hv_px, hu, 2].mean())
            if z_pm > 0.05:
                ks[t] = z_real / z_pm
    ok = np.isfinite(ks)
    k_med = float(np.clip(np.median(ks[ok]), 0.25, 2.5)) if ok.any() else 1.0
    ks = np.where(ok, np.clip(ks, 0.25, 2.5), k_med).astype(np.float32)
    return ks, float(ks[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--obj", required=True)
    ap.add_argument("--hand_source", default="hawor", choices=["hawor", "egohand"])
    ap.add_argument("--anchor_hand", default="right", choices=["right", "left"])
    ap.add_argument("--init_mode", default="seeded", choices=["seeded", "register"])
    ap.add_argument("--intrinsics_txt", default="", help="Optional 3x3 calibrated intrinsics txt")
    ap.add_argument("--intrinsics_json", default="", help="Optional ViPE/EgoHandWorld intrinsics JSON (fx,fy,cx,cy or 3x3 K)")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--max_side", type=int, default=960)
    ap.add_argument("--est_refine_iter", type=int, default=5)
    ap.add_argument("--track_refine_iter", type=int, default=3)
    ap.add_argument("--hand_npz", default="", help="Optional override for hand meshes npz")
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    set_seed(0)

    wdir = args.work_dir
    obj = args.obj
    tdir = f"{wdir}/tracking_{obj}"
    os.makedirs(tdir, exist_ok=True)
    init_json = f"{tdir}/layout_cam_opt_{args.hand_source}.json"
    mesh_path = f"{wdir}/ref_obj_masks_000000/{obj}/{obj}.obj"

    frame_files = sorted(glob.glob(f"{wdir}/all_frames/[0-9][0-9][0-9][0-9][0-9][0-9].png"))
    T = len(frame_files)
    H_full, W_full = np.array(Image.open(frame_files[0])).shape[:2]
    ds_scale = min(1.0, float(args.max_side) / max(H_full, W_full))
    Hd, Wd = int(round(H_full * ds_scale)), int(round(W_full * ds_scale))

    K0 = np.load(f"{wdir}/all_frames/000000_intrinsics.npy").astype(np.float32)
    if args.intrinsics_json and os.path.exists(args.intrinsics_json):
        with open(args.intrinsics_json) as f:
            kd = json.load(f)
        if "K" in kd:
            K_use = np.asarray(kd["K"], dtype=np.float32).reshape(3, 3)
        else:
            K_use = np.array(
                [[float(kd["fx"]), 0.0, float(kd["cx"])], [0.0, float(kd["fy"]), float(kd["cy"])], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
    elif args.intrinsics_txt and os.path.exists(args.intrinsics_txt):
        K_use = np.loadtxt(args.intrinsics_txt).astype(np.float32)
    else:
        K_use = K0.copy()
    pms = [np.load(f"{wdir}/all_frames/{t:06d}_pointmap.npy").astype(np.float32) for t in range(T)]
    masks = []
    for t in range(T):
        mp = f"{wdir}/video_segmentation/masks/frame_{t:06d}_masks/{obj}.png"
        masks.append((np.array(Image.open(mp).convert("L")) > 128) if os.path.exists(mp) else np.zeros((H_full, W_full), dtype=bool))

    hand_npz_path = args.hand_npz or (
        f"{wdir}/egohand/all_hand_meshes_egohand.npz"
        if args.hand_source == "egohand"
        else f"{wdir}/color/all_hand_meshes.npz"
    )
    hand_data = np.load(hand_npz_path)

    has_opt_layout = os.path.exists(init_json)
    if has_opt_layout:
        s0, init_poses = load_init_poses(init_json, T)
        z_smooth = cv2.GaussianBlur(init_poses[:, 2:3, 3], (1, 9), 0).ravel() if T >= 9 else init_poses[:, 2, 3]
        pm_ratios = np.full(T, np.nan, dtype=np.float32)
        for t in range(T):
            m = masks[t]
            if m.sum() >= 32 and z_smooth[t] > 0.05:
                z_pm = pms[t][m, 2]
                ok = np.isfinite(z_pm) & (z_pm > 0.1) & (z_pm < 5.0)
                if ok.sum() >= 16:
                    pm_ratios[t] = float(z_smooth[t]) / float(np.median(z_pm[ok]))
        valid_r = np.isfinite(pm_ratios)
        k_med = float(np.clip(np.median(pm_ratios[valid_r]), 0.25, 2.5)) if valid_r.any() else 1.0
        k_pm_t = np.where(valid_r, np.clip(pm_ratios, 0.25, 2.5), k_med).astype(np.float32)
    else:
        # Direct hand-anchored scaling from frame-0 SAM 3D mesh without running Fast-SAM3D
        s_raw = load_raw_scale_from_sam3d(f"{wdir}/ref_obj_masks_000000/layout.json", obj)
        k_pm_t, k0_hand = compute_hand_pointmap_k(pms, K_use, hand_data, args.anchor_hand, f"{wdir}/video_segmentation/masks", H_full, W_full)
        s0 = s_raw * k0_hand
        init_poses = None
        if args.init_mode == "seeded":
            args.init_mode = "register"

    if T >= 11:
        k_pm_t = cv2.GaussianBlur(k_pm_t[:, None], (1, 11), 0).ravel()

    if args.intrinsics_txt and os.path.exists(args.intrinsics_txt):
        focal_ratio = float(K0[0, 0] / K_use[0, 0])
        s_mesh = s0 * focal_ratio
        if init_poses is not None:
            init_poses[:, :2, 3] *= focal_ratio
    else:
        s_mesh = s0
    K_ds = K_use.copy()
    K_ds[:2, :] *= ds_scale

    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32) * float(s_mesh)
    mesh.fix_normals()

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        glctx=glctx,
        debug=0,
        debug_dir=f"/tmp/fp_dbg_{obj}",
    )

    poses_out = np.zeros((T, 4, 4), dtype=np.float32)
    for t in range(T):
        rgb = np.array(Image.open(frame_files[t]).convert("RGB").resize((Wd, Hd), Image.Resampling.BILINEAR), dtype=np.uint8)
        d = pms[t][..., 2] * float(k_pm_t[t])
        d = np.where(np.isfinite(d) & (d > 0.1) & (d < 5.0), d, 0.0).astype(np.float32)
        if ds_scale < 0.999:
            d = cv2.resize(d, (Wd, Hd), interpolation=cv2.INTER_NEAREST)
        if t == 0:
            if args.init_mode == "register" or init_poses is None:
                m0 = cv2.resize(masks[0].astype(np.uint8), (Wd, Hd), interpolation=cv2.INTER_NEAREST) > 0
                pose = est.register(K=K_ds, rgb=rgb, depth=d, ob_mask=m0, iteration=args.est_refine_iter)
            else:
                tf_c_inv = torch.linalg.inv(est.get_tf_to_centered_mesh())
                p0 = torch.tensor(init_poses[0], dtype=torch.float32, device="cuda") @ tf_c_inv
                est.pose_last = p0[None]
                pose = est.track_one(rgb=rgb, depth=d, K=K_ds, iteration=args.est_refine_iter)
        else:
            pose = est.track_one(rgb=rgb, depth=d, K=K_ds, iteration=args.track_refine_iter)
        poses_out[t] = pose

    out_json = args.out_json or f"{tdir}/layout_fp_{args.init_mode}_{args.hand_source}.json"
    out_layout = {
        "translation_scale_optimization": {
            "mesh_scale": float(s_mesh),
            "init_scale": float(s0),
            "route": f"route3_foundationpose_{args.init_mode}",
            "hand_source": args.hand_source,
            "calibrated_intrinsics": bool(args.intrinsics_txt),
            "used_fast_sam3d": has_opt_layout,
        },
        "objects": [],
    }
    for t in range(T):
        q_xyzw = Rot.from_matrix(poses_out[t, :3, :3]).as_quat()
        q_wxyz = [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]
        out_layout["objects"].append({
            "frame_idx": int(t),
            "local_to_scene": {
                "scale": [float(s_mesh), float(s_mesh), float(s_mesh)],
                "quat_wxyz_camera_frame": q_wxyz,
                "translation_camera_frame": [float(x) for x in poses_out[t, :3, 3]],
            },
        })
    with open(out_json, "w") as f:
        json.dump(out_layout, f, indent=2)
    print(f"[Route3 FoundationPose {obj}] DONE ({args.init_mode}, s_mesh={s_mesh:.4f}, fast_sam3d={has_opt_layout}, T={T}) -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
