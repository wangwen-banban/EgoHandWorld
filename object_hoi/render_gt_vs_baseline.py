import os, json, subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import cv2
import trimesh
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as Rot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import (
    PerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRasterizer,
    SoftPhongShader, PointLights, TexturesVertex, TexturesUV, BlendParams
)

TACO_BASE = "/data/wenwang/HandReconstruction/data/TACO_subset"
WORK_BASE = "/data/wenwang/HandReconstruction/work/daid_taco"
OUT_DIR   = f"{WORK_BASE}/final_vis/gt_vs_baseline"
os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda:0")

def load_trimesh_textured(path, color_rgb=(0.35, 0.72, 0.88)):
    tm = trimesh.load(path, process=False)
    if isinstance(tm, trimesh.Scene):
        tm = tm.dump(concatenate=True)
    v = torch.tensor(np.asarray(tm.vertices, np.float32), device=device)
    f = torch.tensor(np.asarray(tm.faces, np.int64), device=device)
    tex = None
    try:
        vis = tm.visual
        if hasattr(vis, "uv") and vis.uv is not None and hasattr(vis, "material") and getattr(vis.material, "image", None) is not None:
            im = np.asarray(vis.material.image.convert("RGB"), np.float32) / 255.0
            uv = np.asarray(vis.uv, np.float32)
            tex = TexturesUV(maps=torch.tensor(im[None], device=device), faces_uvs=f[None], verts_uvs=torch.tensor(uv[None], device=device))
    except Exception:
        tex = None
    if tex is None:
        if hasattr(tm.visual, "vertex_colors") and tm.visual.vertex_colors is not None and len(tm.visual.vertex_colors) == len(v):
            vc = np.asarray(tm.visual.vertex_colors[:, :3], np.float32) / 255.0
            if vc.std() < 0.02:
                vc = np.tile(np.array(color_rgb, np.float32), (len(v), 1))
        else:
            vc = np.tile(np.array(color_rgb, np.float32), (len(v), 1))
        tex = TexturesVertex(verts_features=torch.tensor(vc[None], device=device))
    return v, f, tex

_RC = {}
def get_rend(K_np, H, W):
    sx, sy = W / 1920.0, H / 1080.0
    key = (round(float(K_np[0,0]*sx), 2), round(float(K_np[0,2]*sx), 2), H, W)
    if key in _RC: return _RC[key]
    cams = PerspectiveCameras(
        focal_length=torch.tensor([[float(K_np[0,0]*sx), float(K_np[1,1]*sy)]], dtype=torch.float32, device=device),
        principal_point=torch.tensor([[float(K_np[0,2]*sx), float(K_np[1,2]*sy)]], dtype=torch.float32, device=device),
        R=torch.diag(torch.tensor([-1.0, -1.0, 1.0], device=device))[None],
        T=torch.zeros((1, 3), device=device),
        in_ndc=False,
        image_size=torch.tensor([[H, W]], dtype=torch.float32, device=device),
        device=device,
    )
    rs = RasterizationSettings(image_size=(H, W), blur_radius=0.0, faces_per_pixel=1, bin_size=0)
    lights = PointLights(device=device, location=[[0.0, -0.8, -0.5]], ambient_color=((0.55,0.55,0.55),), diffuse_color=((0.45,0.45,0.45),), specular_color=((0.15,0.15,0.15),))
    r = MeshRenderer(rasterizer=MeshRasterizer(cameras=cams, raster_settings=rs), shader=SoftPhongShader(device=device, cameras=cams, lights=lights, blend_params=BlendParams(background_color=(0.0,0.0,0.0))))
    _RC[key] = r
    return r

def render_overlay(bg, items, K_render, H, W):
    meshes_list = [Meshes(verts=vc[None], faces=fc[None], textures=tex) for vc, fc, tex in items]
    if not meshes_list:
        return bg.copy()
    if not all(isinstance(m.textures, TexturesUV) for m in meshes_list):
        norm = []
        for (vc, fc, tex), m in zip(items, meshes_list):
            if isinstance(tex, TexturesUV):
                uv = tex.verts_uvs_padded()[0]; mp = tex.maps_padded()[0]
                Hm, Wm = mp.shape[:2]
                px = torch.clamp((uv[:,0]*Wm).long(), 0, Wm-1)
                py = torch.clamp(((1.0 - uv[:,1])*Hm).long(), 0, Hm-1)
                norm.append(Meshes(verts=vc[None], faces=fc[None], textures=TexturesVertex(verts_features=mp[py, px][None])))
            else:
                norm.append(m)
        meshes_list = norm
    scene = join_meshes_as_scene(meshes_list)
    rend = get_rend(K_render, H, W)
    frags = rend.rasterizer(scene)
    img = rend.shader(frags, scene)[0].detach().cpu().numpy()
    fg = np.clip(img[..., :3] * 255.0, 0, 255).astype(np.float32)
    mask = (frags.pix_to_face[0, ..., 0].detach().cpu().numpy() >= 0)[..., None].astype(np.float32) * 0.80
    return np.clip(bg.astype(np.float32) * (1.0 - mask) + fg * mask, 0, 255).astype(np.uint8)

def load_Rt(jpath):
    lay = json.load(open(jpath))
    s = float(lay.get("translation_scale_optimization", {}).get("mesh_scale", 1.0))
    objs = sorted(lay["objects"], key=lambda o: o["frame_idx"])
    R = np.stack([Rot.from_quat([o["local_to_scene"]["quat_wxyz_camera_frame"][1], o["local_to_scene"]["quat_wxyz_camera_frame"][2], o["local_to_scene"]["quat_wxyz_camera_frame"][3], o["local_to_scene"]["quat_wxyz_camera_frame"][0]]).as_matrix() for o in objs]).astype(np.float32)
    t = np.array([o["local_to_scene"]["translation_camera_frame"] for o in objs], np.float32)
    return s, R, t

def put_label(img, text, color=(255, 220, 60)):
    out = img.copy()
    w = min(img.shape[1], int(len(text)*10.5 + 18))
    cv2.rectangle(out, (0, 0), (w, 30), (14, 17, 22), -1)
    cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
    return out

def render_case_2x2(sdir, seq_str, objs_spec, gt_npz_path, out_name, fps=20):
    wdir = f"{WORK_BASE}/{sdir}"
    gt_c = np.load(gt_npz_path)
    K_gt = gt_c["K_gt"].astype(np.float32); E_gt = gt_c["E"]; T = len(E_gt)
    Wr, Hr = 640, 360  # 1280x720 output

    hw = np.load(f"{wdir}/color/all_hand_meshes.npz")
    eh = np.load(f"{wdir}/egohand/all_hand_meshes_egohand.npz")
    hl_f = torch.tensor(hw["left_faces"].astype(np.int64), device=device)
    hr_f = torch.tensor(hw["right_faces"].astype(np.int64), device=device)
    tex_hl = TexturesVertex(verts_features=torch.tensor(np.tile([0.92, 0.36, 0.36], (778, 1)), dtype=torch.float32, device=device)[None])
    tex_hr = TexturesVertex(verts_features=torch.tensor(np.tile([0.36, 0.48, 0.94], (778, 1)), dtype=torch.float32, device=device)[None])
    el_f = torch.tensor(eh["left_faces"].astype(np.int64), device=device)
    er_f = torch.tensor(eh["right_faces"].astype(np.int64), device=device)
    tex_el = TexturesVertex(verts_features=torch.tensor(np.tile([0.92, 0.36, 0.36], (eh["left_vertices"].shape[1], 1)), dtype=torch.float32, device=device)[None])
    tex_er = TexturesVertex(verts_features=torch.tensor(np.tile([0.36, 0.48, 0.94], (eh["right_vertices"].shape[1], 1)), dtype=torch.float32, device=device)[None])

    loaded_objs = []
    for obj, gt_k, col_gt in objs_spec:
        oid = gt_k.split("_")[-1]
        gv, gf, _ = load_trimesh_textured(f"{TACO_BASE}/object_models_released/{oid}_cm.obj")
        gv = gv * 0.01
        tg = TexturesVertex(verts_features=torch.tensor(np.tile(col_gt, (len(gv), 1)), dtype=torch.float32, device=device)[None])
        P_gt = np.load(f"{TACO_BASE}/Object_Poses/{seq_str}/{gt_k}.npy")
        pv, pf, ptex = load_trimesh_textured(f"{wdir}/ref_obj_masks_000000/{obj}/{obj}.obj")
        rt_hw = load_Rt(f"{wdir}/tracking_{obj}/layout_cam_opt_hawor.json")
        rt_eh = load_Rt(f"{wdir}/tracking_{obj}/layout_cam_opt_egohand.json")
        loaded_objs.append((gv, gf, tg, P_gt, pv, pf, ptex, rt_hw, rt_eh))

    raw = f"{OUT_DIR}/{out_name}_raw.mp4"
    mp4 = f"{OUT_DIR}/{out_name}.mp4"
    if os.path.exists(mp4) and os.path.exists(f"{OUT_DIR}/{out_name}_mid.png"):
        print("Already exists:", mp4)
        return
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (Wr*2, Hr*2))
    snap = None
    for t in range(T):
        rgb = np.array(Image.open(f"{wdir}/all_frames/{t:06d}.png").convert("RGB").resize((Wr, Hr), Image.Resampling.BILINEAR))
        Km = np.load(f"{wdir}/all_frames/{t:06d}_intrinsics.npy").astype(np.float32)

        p0 = put_label(rgb, f"Original RGB (f={t})", (235, 240, 250))

        gt_items = [
            (torch.tensor(gt_c["left_verts"][t], device=device, dtype=torch.float32), hl_f, tex_hl),
            (torch.tensor(gt_c["right_verts"][t], device=device, dtype=torch.float32), hr_f, tex_hr),
        ]
        hw_o, eh_o = [], []
        for gv, gf, tg, P_gt, pv, pf, ptex, (sh, Rh, th), (se, Re, te) in loaded_objs:
            Tgt = E_gt[t] @ P_gt[t]
            gt_items.append((gv @ torch.tensor(Tgt[:3,:3].T, device=device, dtype=torch.float32) + torch.tensor(Tgt[:3,3], device=device, dtype=torch.float32), gf, tg))
            hw_o.append(((pv * sh) @ torch.tensor(Rh[t].T, device=device) + torch.tensor(th[t], device=device), pf, ptex))
            eh_o.append(((pv * se) @ torch.tensor(Re[t].T, device=device) + torch.tensor(te[t], device=device), pf, ptex))

        p1 = put_label(render_overlay(rgb, gt_items, K_gt, Hr, Wr), "TACO Ground Truth", (120, 255, 165))

        hw_h = []
        if bool(hw["left_valid"][t]):  hw_h.append((torch.tensor(hw["left_vertices"][t], device=device, dtype=torch.float32), hl_f, tex_hl))
        if bool(hw["right_valid"][t]): hw_h.append((torch.tensor(hw["right_vertices"][t], device=device, dtype=torch.float32), hr_f, tex_hr))
        p2 = put_label(render_overlay(render_overlay(rgb, hw_h, K_gt, Hr, Wr), hw_o, Km, Hr, Wr), "Baseline: Do As I Do (HaWoR)", (255, 140, 120))

        eh_h = []
        if bool(eh["left_valid"][t]):  eh_h.append((torch.tensor(eh["left_vertices"][t], device=device, dtype=torch.float32), el_f, tex_el))
        if bool(eh["right_valid"][t]): eh_h.append((torch.tensor(eh["right_vertices"][t], device=device, dtype=torch.float32), er_f, tex_er))
        p3 = put_label(render_overlay(render_overlay(rgb, eh_h, K_gt, Hr, Wr), eh_o, Km, Hr, Wr), "Baseline: Do As I Do (EgoHand)", (255, 185, 95))

        comp = np.vstack([np.hstack([p0, p1]), np.hstack([p2, p3])])
        vw.write(cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
        if t == T // 2:
            snap = comp
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-i", raw, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21", mp4], check=True)
    os.remove(raw)
    if snap is not None:
        Image.fromarray(snap).save(f"{OUT_DIR}/{out_name}_mid.png")
    print("Saved", mp4)

render_case_2x2(
    "brush_bowl_20231005_188", "(brush, brush, bowl)/20231005_188",
    [("brush", "tool_178", (0.20, 0.86, 0.42)), ("bowl", "target_023", (0.16, 0.82, 0.68))],
    "/tmp/taco_vis/brush_bowl_gt.npz", "case1_brush_bowl_gt_vs_baseline", fps=20
)

render_case_2x2(
    "cut_knife_plate_20231020_219", "(cut, knife, plate)/20231020_219",
    [("knife", "tool_063", (0.20, 0.86, 0.42))],
    "/tmp/taco_vis/cut_knife_plate_gt.npz", "case2_cut_knife_gt_vs_baseline", fps=24
)

# Case 3: 3D Metric Top-Down (X-Z) & Side View (Z-Y) Trajectory Comparison Video on brush_bowl (shows 3D depth jitter & scale inflation directly!)
wdir = f"{WORK_BASE}/brush_bowl_20231005_188"
gt_c = np.load("/tmp/taco_vis/brush_bowl_gt.npz")
E_gt = gt_c["E"]; T = len(E_gt)
g_b = trimesh.load(f"{TACO_BASE}/object_models_released/178_cm.obj", force="mesh", process=False)
g_bv = np.asarray(g_b.vertices, np.float32) * 0.01
g_o = trimesh.load(f"{TACO_BASE}/object_models_released/023_cm.obj", force="mesh", process=False)
g_ov = np.asarray(g_o.vertices, np.float32) * 0.01
P_b = np.load(f"{TACO_BASE}/Object_Poses/(brush, brush, bowl)/20231005_188/tool_178.npy")
P_o = np.load(f"{TACO_BASE}/Object_Poses/(brush, brush, bowl)/20231005_188/target_023.npy")

p_b = trimesh.load(f"{wdir}/ref_obj_masks_000000/brush/brush.obj", process=False)
if isinstance(p_b, trimesh.Scene): p_b = p_b.dump(concatenate=True)
p_bv = np.asarray(p_b.vertices, np.float32)[::8]
p_o = trimesh.load(f"{wdir}/ref_obj_masks_000000/bowl/bowl.obj", process=False)
if isinstance(p_o, trimesh.Scene): p_o = p_o.dump(concatenate=True)
p_ov = np.asarray(p_o.vertices, np.float32)[::24]
sb, Rb, tb = load_Rt(f"{wdir}/tracking_brush/layout_cam_opt_hawor.json")
so, Ro, to = load_Rt(f"{wdir}/tracking_bowl/layout_cam_opt_hawor.json")

raw3d = f"{OUT_DIR}/case3_raw.mp4"
mp43d = f"{OUT_DIR}/case3_brush_bowl_3d_metric_gt_vs_baseline.mp4"
vw3 = None
for t in range(T):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 5.2), dpi=100)
    Tgt_b = E_gt[t] @ P_b[t]; Tgt_o = E_gt[t] @ P_o[t]
    vgb = (g_bv @ Tgt_b[:3,:3].T + Tgt_b[:3,3])[::12]
    vgo = (g_ov @ Tgt_o[:3,:3].T + Tgt_o[:3,3])[::12]
    vpb = (p_bv * sb) @ Rb[t].T + tb[t]
    vpo = (p_ov * so) @ Ro[t].T + to[t]

    for ax, (d0, d1, xl, yl, ttl) in [
        (ax1, (0, 2, "X (right, m)", "Z (depth, m)", f"Top-Down 3D (X-Z) | Frame {t}/{T-1}")),
        (ax2, (2, 1, "Z (depth, m)", "Y (down, m)",  f"Side View 3D (Z-Y) | Frame {t}/{T-1}")),
    ]:
        ax.scatter(vgb[:, d0], vgb[:, d1], s=5, c="#10b981", alpha=0.65, label="TACO GT")
        ax.scatter(vgo[:, d0], vgo[:, d1], s=5, c="#10b981", alpha=0.65)
        ax.scatter(vpb[:, d0], vpb[:, d1], s=5, c="#ef4444", alpha=0.60, label="Baseline (DAID+HaWoR)")
        ax.scatter(vpo[:, d0], vpo[:, d1], s=5, c="#ef4444", alpha=0.60)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ttl, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if d0 == 0:
            ax.set_xlim(-0.35, 0.35); ax.set_ylim(0.45, 1.05)
        else:
            ax.set_xlim(0.45, 1.05); ax.set_ylim(0.35, -0.35)
    ax1.legend(loc="upper left", markerscale=2.5, fontsize=9)
    plt.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    if vw3 is None:
        H3, W3 = buf.shape[:2]
        vw3 = cv2.VideoWriter(raw3d, cv2.VideoWriter_fourcc(*"mp4v"), 20, (W3, H3))
    vw3.write(cv2.cvtColor(buf, cv2.COLOR_RGB2BGR))
    if t == T // 2:
        Image.fromarray(buf).save(f"{OUT_DIR}/case3_mid.png")
vw3.release()
subprocess.run(["ffmpeg", "-y", "-i", raw3d, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21", mp43d], check=True)
os.remove(raw3d)
print("ALL GT VS BASELINE VIDEOS DONE")
