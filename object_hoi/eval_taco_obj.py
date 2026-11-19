"""Quantitative evaluation on TACO_subset: hands (HaWoR / EgoHand), object mesh & 6DoF pose, HOI contact/penetration."""
import argparse
import json
import os
import pickle
import numpy as np
import torch
import smplx
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

TACO_BASE = "/data/wenwang/HandReconstruction/data/TACO_subset"
MANO_DIR = "/data/wenwang/mano_models"
MANO16_TO_16 = [0, 13, 14, 15, 1, 2, 3, 4, 5, 6, 10, 11, 12, 7, 8, 9]
MANO_TIPS_VID = [744, 320, 443, 554, 671]


def chamfer_and_fscore(p: np.ndarray, q: np.ndarray, thresholds_mm=(5.0, 10.0, 20.0)):
    """Symmetric L2 surface distance (mm) and F-scores between two point clouds (m)."""
    tp = cKDTree(q)
    tq = cKDTree(p)
    d_p2q, _ = tp.query(p, k=1)
    d_q2p, _ = tq.query(q, k=1)
    d_p2q_mm = d_p2q * 1000.0
    d_q2p_mm = d_q2p * 1000.0
    cd_mm = 0.5 * (d_p2q_mm.mean() + d_q2p_mm.mean())
    fscores = {}
    for thr in thresholds_mm:
        prec = float((d_p2q_mm < thr).mean())
        rec = float((d_q2p_mm < thr).mean())
        f = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        fscores[f"f@{int(thr)}mm"] = f * 100.0
    return float(cd_mm), fscores


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Fit dst ~ s * src @ R.T + t."""
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    cs = src - mu_s
    cd = dst - mu_d
    cov = (cd.T @ cs) / src.shape[0]
    U, S, Vt = np.linalg.svd(cov)
    d = np.ones(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[-1] = -1
    R = U @ np.diag(d) @ Vt
    if with_scale:
        var_s = (cs ** 2).sum() / src.shape[0]
        s = float((S * d).sum() / max(var_s, 1e-12))
    else:
        s = 1.0
    t = mu_d - s * (R @ mu_s)
    return s, R, t


def icp_align(src: np.ndarray, dst: np.ndarray, with_scale: bool = False, iters: int = 40):
    """Multi-start ICP (Principal axes + 24 octahedral rotations) from src -> dst."""
    # Center both
    src_c = src.mean(0)
    dst_c = dst.mean(0)
    s0 = src - src_c
    d0 = dst - dst_c
    init_scale = float(np.sqrt((d0 ** 2).sum(1).mean()) / max(np.sqrt((s0 ** 2).sum(1).mean()), 1e-9)) if with_scale else 1.0

    # 24 cube/octahedral rotations + PCA alignment
    rots = [np.eye(3)]
    for R_cand in Rot.create_group("O").as_matrix():
        rots.append(R_cand)

    tree_dst = cKDTree(dst)
    best_cd = 1e9
    best_tf = (init_scale, np.eye(3), dst_c - init_scale * src_c)

    # Subsample for fast multi-start
    idx_sub = np.linspace(0, len(src) - 1, min(1024, len(src)), dtype=int)
    src_sub = src[idx_sub]

    # Quick coarse scoring of 24+1 rotations
    cands = []
    for R_init in rots:
        pts_t = (src_sub - src_c) @ R_init.T * init_scale + dst_c
        dists, _ = tree_dst.query(pts_t, k=1)
        cands.append((float(dists.mean()), R_init))
    cands.sort(key=lambda x: x[0])

    # Run full ICP on top-4 starts
    for _, R_init in cands[:4]:
        s = init_scale
        R = R_init.copy()
        t = dst_c - s * (R @ src_c)
        for _ in range(iters):
            pts_t = s * (src @ R.T) + t
            dists, nn_idx = tree_dst.query(pts_t, k=1)
            # Robust trimming at 90th percentile after first few iters
            thr = np.percentile(dists, 90)
            keep = dists <= max(thr, 1e-4)
            if keep.sum() < 20:
                keep = np.ones(len(dists), dtype=bool)
            s_d, R_d, t_d = umeyama(pts_t[keep], dst[nn_idx[keep]], with_scale=with_scale)
            s = s * s_d
            R = R_d @ R
            t = s_d * (R_d @ t) + t_d
        pts_final = s * (src @ R.T) + t
        cd, _ = chamfer_and_fscore(pts_final, dst, ())
        if cd < best_cd:
            best_cd = cd
            best_tf = (s, R, t)
    return best_tf, best_cd


def load_taco_gt_hands(seq: str):
    """Return left & right GT MANO vertices (T,778,3) & OpenPose21 joints (T,21,3) in camera frame."""
    E = np.load(f"{TACO_BASE}/Egocentric_Camera_Parameters/{seq}/egocentric_frame_extrinsic.npy")
    T = len(E)
    out = {}
    for side, is_r in (("left", False), ("right", True)):
        with open(f"{TACO_BASE}/Hand_Poses/{seq}/{side}_hand.pkl", "rb") as f:
            hp = pickle.load(f)
        with open(f"{TACO_BASE}/Hand_Poses/{seq}/{side}_hand_shape.pkl", "rb") as f:
            hs = pickle.load(f)["hand_shape"]
        pose = torch.stack([hp[f"{t+1:05d}"]["hand_pose"] for t in range(T)]).float()
        trans = torch.stack([hp[f"{t+1:05d}"]["hand_trans"] for t in range(T)]).float()
        betas = hs.float().unsqueeze(0).repeat(T, 1)
        mano = smplx.MANO(MANO_DIR, is_rhand=is_r, use_pca=False, flat_hand_mean=True)
        with torch.no_grad():
            J0 = mano(betas=betas[:1], global_orient=torch.zeros(1, 3), hand_pose=torch.zeros(1, 45), transl=torch.zeros(1, 3)).joints[0, 0]
            res = mano(betas=betas, global_orient=pose[:, :3], hand_pose=pose[:, 3:], transl=trans - J0)
        v_cam = (E[:, None, :3, :3] @ res.vertices.numpy()[..., None]).squeeze(-1) + E[:, None, :3, 3]
        j16_cam = (E[:, None, :3, :3] @ res.joints.numpy()[..., None]).squeeze(-1) + E[:, None, :3, 3]
        j21_cam = np.zeros((T, 21, 3), dtype=np.float32)
        j21_cam[:, [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]] = j16_cam[:, MANO16_TO_16]
        j21_cam[:, [4, 8, 12, 16, 20]] = v_cam[:, MANO_TIPS_VID]
        out[side] = {"verts_cam": v_cam.astype(np.float32), "joints_cam": j21_cam, "faces": mano.faces.astype(np.int32)}
    return E, out


def eval_hand_npz(gt_hands: dict, pred_npz_path: str):
    pred = np.load(pred_npz_path)
    metrics = {}
    for side in ("left", "right"):
        if f"{side}_joints" not in pred:
            continue
        pj = pred[f"{side}_joints"]
        gj = gt_hands[side]["joints_cam"]
        T = min(len(pj), len(gj))
        valid = pred[f"{side}_valid"][:T].astype(bool) if f"{side}_valid" in pred else np.isfinite(pj[:T]).all((1, 2))
        if valid.sum() == 0:
            continue
        p = pj[:T][valid]
        g = gj[:T][valid]
        wrist_err = np.linalg.norm(p[:, 0] - g[:, 0], axis=-1) * 1000.0
        dz = (p[:, 0, 2] - g[:, 0, 2]) * 1000.0
        mpjpe = np.linalg.norm(p - g, axis=-1).mean(axis=-1) * 1000.0
        ra = np.linalg.norm((p - p[:, 0:1]) - (g - g[:, 0:1]), axis=-1).mean(axis=-1) * 1000.0
        pa_list = []
        for pf, gf in zip(p, g):
            s, R, t = umeyama(pf, gf, with_scale=True)
            pa_list.append(np.linalg.norm((s * (pf @ R.T) + t) - gf, axis=-1).mean() * 1000.0)
        m = {
            "n_valid": int(valid.sum()),
            "n_total": int(T),
            "wrist_err_mm": float(wrist_err.mean()),
            "depth_dz_mm": float(dz.mean()),
            "depth_abs_dz_mm": float(np.abs(dz).mean()),
            "mpjpe_mm": float(mpjpe.mean()),
            "ra_mpjpe_mm": float(ra.mean()),
            "pa_mpjpe_mm": float(np.mean(pa_list)),
        }
        if f"{side}_vertices" in pred and pred[f"{side}_vertices"].shape[1] == 778:
            pv = pred[f"{side}_vertices"][:T][valid]
            gv = gt_hands[side]["verts_cam"][:T][valid]
            m["mpvpe_mm"] = float((np.linalg.norm(pv - gv, axis=-1).mean(axis=-1) * 1000.0).mean())
        metrics[side] = m
    return metrics


def eval_object(seq: str, gt_obj_name: str, pred_mesh_path: str, pred_layout_json: str, use_opt_scale: bool = True, hand_npz_path: str | None = None, gt_hands: dict | None = None):
    E = np.load(f"{TACO_BASE}/Egocentric_Camera_Parameters/{seq}/egocentric_frame_extrinsic.npy")
    P_gt = np.load(f"{TACO_BASE}/Object_Poses/{seq}/{gt_obj_name}.npy") # (T, 4, 4) obj2world
    obj_id = gt_obj_name.split("_")[-1]
    gt_mesh = trimesh.load(f"{TACO_BASE}/object_models_released/{obj_id}_cm.obj", force="mesh", process=False)
    gt_mesh.vertices = gt_mesh.vertices * 0.01  # cm -> m

    pred_mesh = trimesh.load(pred_mesh_path, force="mesh", process=False)
    if isinstance(pred_mesh, trimesh.Scene):
        pred_mesh = pred_mesh.dump(concatenate=True)

    with open(pred_layout_json) as f:
        layout = json.load(f)

    opt_info = layout.get("translation_scale_optimization", {})
    opt_scale = opt_info.get("mesh_scale", None)

    # Sample deterministic surface points + normals
    rng_state = np.random.get_state()
    np.random.seed(42)
    pts_gt, fidx_gt = trimesh.sample.sample_surface(gt_mesh, 4096)
    norms_gt = gt_mesh.face_normals[fidx_gt]
    pts_pred_raw, fidx_pred = trimesh.sample.sample_surface(pred_mesh, 4096)
    norms_pred = pred_mesh.face_normals[fidx_pred]
    np.random.set_state(rng_state)

    # Collect frame poses from layout
    frames_pred = {}
    scales = []
    for obj in layout["objects"]:
        fidx = obj.get("frame_idx", None)
        if fidx is None:
            continue
        lts = obj["local_to_scene"]
        s = float(opt_scale) if (use_opt_scale and opt_scale is not None) else float(lts["scale"][0])
        q_wxyz = lts["quat_wxyz_camera_frame"]
        R_cam = Rot.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
        t_cam = np.array(lts["translation_camera_frame"], dtype=np.float64)
        frames_pred[int(fidx)] = (s, R_cam, t_cam)
        scales.append(s)

    mean_scale = float(np.mean(scales)) if scales else 1.0
    gt_diag = float(np.linalg.norm(gt_mesh.bounds[1] - gt_mesh.bounds[0]))
    pred_diag = float(np.linalg.norm(pred_mesh.bounds[1] - pred_mesh.bounds[0]) * mean_scale)

    # Geometry alignment (Sim3 = pure shape, SE3 = metric shape+scale)
    (_, _, _), sim3_cd = icp_align(pts_pred_raw, pts_gt, with_scale=True)
    (s_s3, R_s3, t_s3), _ = icp_align(pts_pred_raw, pts_gt, with_scale=True)
    _, sim3_f = chamfer_and_fscore(s_s3 * (pts_pred_raw @ R_s3.T) + t_s3, pts_gt, (5.0, 10.0, 20.0))

    (s_se3, R_se3, t_se3), se3_cd = icp_align(pts_pred_raw * mean_scale, pts_gt, with_scale=False)
    _, se3_f = chamfer_and_fscore((pts_pred_raw * mean_scale) @ R_se3.T + t_se3, pts_gt, (5.0, 10.0, 20.0))

    # Per-frame camera-space evaluation
    cam_cds, f10s, f20s, centr_errs, dzs = [], [], [], [], []
    R_pred_list, R_gt_list, valid_fidxs = [], [], []
    hand_pen_pred, hand_min_pred, hand_pen_gt, hand_min_gt = [], [], [], []

    hand_pred = np.load(hand_npz_path) if hand_npz_path else None

    for fidx in sorted(frames_pred.keys()):
        if fidx >= len(E):
            continue
        T_gt_cam = E[fidx] @ P_gt[fidx]
        R_gt = T_gt_cam[:3, :3]
        t_gt = T_gt_cam[:3, 3]
        P_gt_c = pts_gt @ R_gt.T + t_gt

        s, R_p, t_p = frames_pred[fidx]
        P_pred_c = (pts_pred_raw * s) @ R_p.T + t_p
        cd, fs = chamfer_and_fscore(P_pred_c, P_gt_c, (10.0, 20.0))
        cam_cds.append(cd)
        f10s.append(fs["f@10mm"])
        f20s.append(fs["f@20mm"])
        centr_errs.append(float(np.linalg.norm(P_pred_c.mean(0) - P_gt_c.mean(0)) * 1000.0))
        dzs.append(float((P_pred_c.mean(0)[2] - P_gt_c.mean(0)[2]) * 1000.0))
        R_pred_list.append(R_p)
        R_gt_list.append(R_gt)
        valid_fidxs.append(fidx)

        # Hand-object interaction check (closest surface dist & inward normal penetration)
        if gt_hands is not None:
            # GT hand vs Pred object & GT object
            V_h_gt = np.concatenate([gt_hands[s_k]["verts_cam"][fidx] for s_k in ("left", "right")], axis=0)
            # Use Pred hand if given, else GT hand
            if hand_pred is not None:
                vs = []
                for s_k in ("left", "right"):
                    if f"{s_k}_vertices" in hand_pred and (f"{s_k}_valid" not in hand_pred or hand_pred[f"{s_k}_valid"][fidx]):
                        vs.append(hand_pred[f"{s_k}_vertices"][fidx])
                V_h_pred = np.concatenate(vs, axis=0) if vs else V_h_gt
            else:
                V_h_pred = V_h_gt

            N_pred_c = norms_pred @ R_p.T
            tree_p = cKDTree(P_pred_c)
            d_hp, idx_hp = tree_p.query(V_h_pred, k=1)
            signed_hp = np.sum((V_h_pred - P_pred_c[idx_hp]) * N_pred_c[idx_hp], axis=-1)
            pen_hp = np.maximum(-signed_hp, 0.0) * 1000.0
            hand_min_pred.append(float(d_hp.min() * 1000.0))
            hand_pen_pred.append(float(pen_hp.max()))

            N_gt_c = norms_gt @ R_gt.T
            tree_g = cKDTree(P_gt_c)
            d_hg, idx_hg = tree_g.query(V_h_gt, k=1)
            signed_hg = np.sum((V_h_gt - P_gt_c[idx_hg]) * N_gt_c[idx_hg], axis=-1)
            pen_hg = np.maximum(-signed_hg, 0.0) * 1000.0
            hand_min_gt.append(float(d_hg.min() * 1000.0))
            hand_pen_gt.append(float(pen_hg.max()))

    # Relative rotation drift vs ref frame (canonical-frame free!)
    rot_drifts = []
    if len(R_pred_list) > 1:
        R_p0 = R_pred_list[0]
        R_g0 = R_gt_list[0]
        for Rp, Rg in zip(R_pred_list[1:], R_gt_list[1:]):
            dRp = Rp @ R_p0.T
            dRg = Rg @ R_g0.T
            cos_a = np.clip((np.trace(dRp.T @ dRg) - 1.0) * 0.5, -1.0, 1.0)
            rot_drifts.append(float(np.degrees(np.arccos(cos_a))))

    res = {
        "gt_obj": gt_obj_name,
        "n_frames_pred": len(valid_fidxs),
        "n_frames_gt": len(E),
        "gt_diag_mm": gt_diag * 1000.0,
        "pred_diag_mm": pred_diag * 1000.0,
        "scale_ratio": float(pred_diag / max(gt_diag, 1e-6)),
        "shape_sim3_cd_mm": float(sim3_cd),
        "shape_sim3_f5": sim3_f["f@5mm"],
        "shape_sim3_f10": sim3_f["f@10mm"],
        "shape_se3_cd_mm": float(se3_cd),
        "shape_se3_f10": se3_f["f@10mm"],
        "cam_cd_mm": float(np.mean(cam_cds)) if cam_cds else None,
        "cam_f10": float(np.mean(f10s)) if f10s else None,
        "cam_f20": float(np.mean(f20s)) if f20s else None,
        "centroid_err_mm": float(np.mean(centr_errs)) if centr_errs else None,
        "depth_dz_mm": float(np.mean(dzs)) if dzs else None,
        "depth_abs_dz_mm": float(np.abs(dzs).mean()) if dzs else None,
        "rel_rot_drift_deg": float(np.mean(rot_drifts)) if rot_drifts else None,
    }
    if hand_min_pred:
        res["pred_hand_obj_min_dist_mm"] = float(np.mean(hand_min_pred))
        res["pred_hand_obj_max_pen_mm"] = float(np.mean(hand_pen_pred))
        res["gt_hand_obj_min_dist_mm"] = float(np.mean(hand_min_gt))
        res["gt_hand_obj_max_pen_mm"] = float(np.mean(hand_pen_gt))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--hawor_npz", default=None)
    ap.add_argument("--egohand_npz", default=None)
    ap.add_argument("--obj_spec", action="append", default=[], help="gt_name:pred_mesh:layout_json")
    ap.add_argument("--hand_for_obj", default=None, help="NPZ for HOI check")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    E, gt_hands = load_taco_gt_hands(args.seq)
    report = {"seq": args.seq, "hands": {}, "objects": {}}
    if args.hawor_npz and os.path.exists(args.hawor_npz):
        report["hands"]["hawor"] = eval_hand_npz(gt_hands, args.hawor_npz)
    if args.egohand_npz and os.path.exists(args.egohand_npz):
        report["hands"]["egohand"] = eval_hand_npz(gt_hands, args.egohand_npz)
    for spec in args.obj_spec:
        gt_name, pred_mesh, layout_json = spec.split(":")
        report["objects"][gt_name] = eval_object(
            args.seq, gt_name, pred_mesh, layout_json,
            hand_npz_path=args.hand_for_obj or args.hawor_npz, gt_hands=gt_hands,
        )
    print(json.dumps(report, indent=2))
    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
