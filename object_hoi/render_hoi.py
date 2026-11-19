"""Render side-by-side Original RGB vs. 3D Hand + Object (Route 3 + K_vipe + EgoHand) overlay video."""
import argparse
import glob
import json
import os
import subprocess
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
from pytorch3d.renderer import (
    BlendParams,
    MeshRasterizer,
    MeshRenderer,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
)
from pytorch3d.structures import Meshes, join_meshes_as_scene


def load_K(work_dir: str, intrinsics_json: str = "", intrinsics_txt: str = "") -> np.ndarray:
    if intrinsics_json and os.path.exists(intrinsics_json):
        with open(intrinsics_json) as f:
            kd = json.load(f)
        if "K" in kd:
            return np.asarray(kd["K"], dtype=np.float32).reshape(3, 3)
        return np.array(
            [[float(kd["fx"]), 0.0, float(kd["cx"])], [0.0, float(kd["fy"]), float(kd["cy"])], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
    if intrinsics_txt and os.path.exists(intrinsics_txt):
        return np.loadtxt(intrinsics_txt).astype(np.float32)
    return np.load(os.path.join(work_dir, "all_frames", "000000_intrinsics.npy")).astype(np.float32)


def load_trimesh_textured(path: str, device: torch.device, color_rgb=(0.25, 0.78, 0.45)):
    tm = trimesh.load(path, process=False)
    if isinstance(tm, trimesh.Scene):
        tm = tm.dump(concatenate=True)
    v = torch.tensor(np.asarray(tm.vertices, np.float32), device=device)
    f = torch.tensor(np.asarray(tm.faces, np.int64), device=device)
    tex = None
    try:
        vis = tm.visual
        if hasattr(vis, "uv") and vis.uv is not None and hasattr(vis, "material"):
            img = getattr(vis.material, "image", None)
            if img is None and hasattr(vis.material, "baseColorTexture"):
                img = vis.material.baseColorTexture
            if img is not None:
                arr = np.asarray(img.convert("RGB"), np.float32) / 255.0
                uv = np.asarray(vis.uv, np.float32)
                ih, iw = arr.shape[:2]
                u = np.clip((uv[:, 0] % 1.0) * (iw - 1), 0, iw - 1).astype(np.int64)
                v_px = np.clip((1.0 - (uv[:, 1] % 1.0)) * (ih - 1), 0, ih - 1).astype(np.int64)
                vcols = torch.tensor(arr[v_px, u], dtype=torch.float32, device=device)[None]
                tex = TexturesVertex(verts_features=vcols)
    except Exception:
        tex = None
    if tex is None:
        c = torch.tensor(color_rgb, dtype=torch.float32, device=device).view(1, 1, 3).expand(1, len(v), 3)
        tex = TexturesVertex(verts_features=c)
    return v, f, tex


def load_layout_Rt(json_path: str, T: int):
    with open(json_path) as f:
        lay = json.load(f)
    s = float(lay.get("translation_scale_optimization", {}).get("mesh_scale", 1.0))
    Rs = np.repeat(np.eye(3, dtype=np.float32)[None], T, axis=0)
    ts = np.zeros((T, 3), dtype=np.float32)
    valid = np.zeros((T,), dtype=bool)
    for o in lay["objects"]:
        fi = int(o["frame_idx"])
        if 0 <= fi < T:
            lts = o["local_to_scene"]
            if "translation_scale_optimization" not in lay:
                s = float(lts["scale"][0])
            qw = lts["quat_wxyz_camera_frame"]
            Rs[fi] = Rot.from_quat([qw[1], qw[2], qw[3], qw[0]]).as_matrix().astype(np.float32)
            ts[fi] = np.asarray(lts["translation_camera_frame"], np.float32)
            valid[fi] = True
    return s, Rs, ts, valid


def render_overlay(bg_rgb: np.ndarray, meshes_list, K: np.ndarray, device: torch.device, alpha: float = 0.85):
    if not meshes_list:
        return bg_rgb.copy()
    H, W = bg_rgb.shape[:2]
    scene = join_meshes_as_scene(meshes_list)
    R_cam = torch.diag(torch.tensor([-1.0, -1.0, 1.0], device=device))[None]
    T_cam = torch.zeros((1, 3), device=device)
    cams = PerspectiveCameras(
        focal_length=torch.tensor([[K[0, 0], K[1, 1]]], dtype=torch.float32, device=device),
        principal_point=torch.tensor([[K[0, 2], K[1, 2]]], dtype=torch.float32, device=device),
        in_ndc=False,
        image_size=torch.tensor([[H, W]], dtype=torch.float32, device=device),
        R=R_cam,
        T=T_cam,
        device=device,
    )
    raster = MeshRasterizer(
        cameras=cams,
        raster_settings=RasterizationSettings(image_size=(H, W), blur_radius=0.0, faces_per_pixel=1),
    )
    lights = PointLights(
        device=device,
        location=[[0.0, -0.4, -0.5]],
        ambient_color=[[0.55, 0.55, 0.55]],
        diffuse_color=[[0.55, 0.55, 0.55]],
        specular_color=[[0.15, 0.15, 0.15]],
    )
    shader = SoftPhongShader(
        device=device,
        cameras=cams,
        lights=lights,
        blend_params=BlendParams(background_color=(0.0, 0.0, 0.0)),
    )
    img = MeshRenderer(rasterizer=raster, shader=shader)(scene)[0].detach().cpu().numpy()
    rgb = np.clip(img[..., :3] * 255.0, 0, 255).astype(np.uint8)
    mask = (img[..., 3:4] > 0.01).astype(np.float32) * alpha
    return np.clip(bg_rgb.astype(np.float32) * (1.0 - mask) + rgb.astype(np.float32) * mask, 0, 255).astype(np.uint8)


def put_label(img: np.ndarray, text: str, color=(255, 255, 255)):
    out = img.copy()
    cv2.rectangle(out, (8, 8), (8 + len(text) * 11 + 16, 38), (15, 23, 42), -1)
    cv2.putText(out, text, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description="Render 3D Hand + Object overlay video")
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--obj", action="append", required=True, help="Object name (repeatable)")
    ap.add_argument("--layout_json", action="append", required=True, help="Object pose layout JSON (matches --obj)")
    ap.add_argument("--hand_npz", default="", help="Optional EgoHand/HaWoR hand meshes npz")
    ap.add_argument("--intrinsics_json", default="", help="Optional ViPE intrinsics JSON")
    ap.add_argument("--intrinsics_txt", default="", help="Optional 3x3 intrinsics txt")
    ap.add_argument("--output_mp4", required=True)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--max_width", type=int, default=640, help="Per-panel width in output side-by-side video")
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()

    if len(args.obj) != len(args.layout_json):
        raise ValueError("Number of --obj arguments must match --layout_json arguments")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    frame_files = sorted(glob.glob(os.path.join(args.work_dir, "all_frames", "[0-9][0-9][0-9][0-9][0-9][0-9].png")))
    if not frame_files:
        raise FileNotFoundError(f"No frames found under {args.work_dir}/all_frames/")
    T = len(frame_files)
    K = load_K(args.work_dir, args.intrinsics_json, args.intrinsics_txt)

    palette = [(0.20, 0.86, 0.42), (0.16, 0.78, 0.88), (0.95, 0.65, 0.20)]
    objs = []
    for i, (name, ljson) in enumerate(zip(args.obj, args.layout_json)):
        mesh_path = os.path.join(args.work_dir, "ref_obj_masks_000000", name, f"{name}.obj")
        v, f, tex = load_trimesh_textured(mesh_path, device, color_rgb=palette[i % len(palette)])
        s, Rs, ts, valid = load_layout_Rt(ljson, T)
        objs.append((v, f, tex, s, Rs, ts, valid))

    hand = np.load(args.hand_npz) if (args.hand_npz and os.path.exists(args.hand_npz)) else None
    hand_faces = {}
    if hand is not None:
        for side in ("left", "right"):
            if f"{side}_faces" in hand:
                hand_faces[side] = torch.tensor(hand[f"{side}_faces"], dtype=torch.int64, device=device)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_mp4)), exist_ok=True)
    raw_mp4 = args.output_mp4 + ".raw.mp4"
    vw = None
    for t, fpath in enumerate(frame_files):
        rgb = np.array(Image.open(fpath).convert("RGB"))
        H, W = rgb.shape[:2]
        meshes = []
        for v, f, tex, s, Rs, ts, valid in objs:
            if not valid[t]:
                continue
            Rt = torch.tensor(Rs[t], dtype=torch.float32, device=device)
            tt = torch.tensor(ts[t], dtype=torch.float32, device=device)
            vt = (v * s) @ Rt.T + tt
            meshes.append(Meshes(verts=[vt], faces=[f], textures=tex))
        if hand is not None:
            for side, col in (("left", (0.90, 0.32, 0.32)), ("right", (0.32, 0.38, 0.92))):
                if f"{side}_valid" in hand and t < len(hand[f"{side}_valid"]) and bool(hand[f"{side}_valid"][t]):
                    vh = torch.tensor(hand[f"{side}_vertices"][t], dtype=torch.float32, device=device)
                    fh = hand_faces[side]
                    ch = torch.tensor(col, dtype=torch.float32, device=device).view(1, 1, 3).expand(1, len(vh), 3)
                    meshes.append(Meshes(verts=[vh], faces=[fh], textures=TexturesVertex(verts_features=ch)))
        ov = render_overlay(rgb, meshes, K, device)
        scale = min(1.0, float(args.max_width) / W)
        pw, ph = int(round(W * scale)) // 2 * 2, int(round(H * scale)) // 2 * 2
        left_p = put_label(cv2.resize(rgb, (pw, ph)), f"Original RGB (f={t})")
        right_p = put_label(cv2.resize(ov, (pw, ph)), "EgoHandWorld + 3D Object (Route 3 + K_vipe)", (120, 255, 160))
        panel = np.concatenate([left_p, right_p], axis=1)
        if vw is None:
            vw = cv2.VideoWriter(raw_mp4, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (panel.shape[1], panel.shape[0]))
        vw.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    if vw is not None:
        vw.release()
    subprocess.run(
        ["ffmpeg", "-y", "-v", "warning", "-i", raw_mp4, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21", args.output_mp4],
        check=True,
    )
    os.remove(raw_mp4)
    print(f"Saved HOI visualization video ({T} frames): {args.output_mp4}")


if __name__ == "__main__":
    main()
