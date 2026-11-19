# EgoHandWorld

第一视角（Egocentric）单目视频 → **世界坐标系下的米制双手 3D 关节与手部参数序列**。

核心组合：
- **相机内参 & 米制轨迹（默认）**：NVIDIA **ViPE**（`pose_only` 模式，无需已知相机内参、不依赖手部检测，无手视频也可稳定建图；亦支持通过配置切回 HaWoR masked DROID-SLAM + Metric3D）。
- **逐帧 3D 手部重建**：NVIDIA **EgoHand**（相机系 SOMA-Hand）。
- **假阳性过滤**：使用原始 RT-DETR 检测分数复核 EgoHand 输出的每帧手框，过滤无手帧的跟踪插值误检。
- **Web 平台**：内置基于 Flask + Gunicorn + SQLite 的单机任务队列与 2D/3D 可视化界面。

---

## 1. 整体流水线架构

```text
Input Video (.mp4) [+ 可选内参 K.json]
        │
        ▼
[1. preprocess.py]      统一重编码到 30 fps，解析可选外置内参
        │
        ├────────────────────────────────────────┐
        ▼                                        ▼
[2a. vipe_camera.py] (默认)              [3. egohand_infer.py]
  ViPE pose_only (独立 vipe 环境)          EgoHand 推理 (独立 egohand 环境)
  → K_vipe, R_c2w, t_c2w (米制)            + RT-DETR 原始框分数复核 (hand_scores.npz)
  (可选切换为 hawor_camera.py)
        │                                        │
        └──────────────────┬─────────────────────┘
                           ▼
[4. lift_world.py]      有效帧判定 (score_thr / max_gap / min_frames)
                        SOMA-25 → 21 关节映射，p_world = R_c2w · p_cam + t_c2w
                           │
                           ▼
             result.npz + summary.json (+ 可选 Web 渲染 overlay.mp4 / web.json)
```

各阶段通过独立子进程调用对应 Python 环境（`hawor` / `egohand` / `vipe`），阶段产物落盘并记录在 `manifest.json` 中，支持断点续跑（`--force` 可强制重跑）。

---

## 2. 仓库目录结构

```text
EgoHandWorld/
├── configs/
│   └── default.yaml          # 全局配置：环境路径、camera_backend、ViPE/EgoHand 参数、有效帧阈值
├── processors/               # 流水线 4 个核心处理阶段
│   ├── common.py             # 配置加载、manifest 读写、通用工具
│   ├── preprocess.py         # 视频统一 30 fps 与内参解析
│   ├── vipe_camera.py        # [默认] ViPE 相机内参 + 米制轨迹估计
│   ├── hawor_camera.py       # [可选] HaWoR masked DROID-SLAM + Metric3D（需提供内参）
│   ├── egohand_infer.py      # EgoHand 官方推理 + 原始 RT-DETR 分数复核
│   └── lift_world.py         # 有效帧过滤、SOMA25→21、变换到世界坐标系并写 result.npz
├── tools/                    # 命令行入口
│   ├── run_video.py          # 单视频端到端运行（支持断点续跑）
│   ├── launch_batch.py       # 多卡并发批处理（按显存阈值自动调度空闲卡）
│   └── eval_egodex.py        # EgoDex + 无手视频回归测试脚本
├── web/                      # 可视化 Web 服务与后台任务队列
│   ├── app.py                # Flask + Gunicorn Web API 与页面路由
│   ├── worker.py             # SQLite 串行后台任务执行器
│   ├── render.py             # 生成 2D overlay 视频与 3D 浏览器渲染数据 (web.json)
│   ├── manage.py             # 一键 start / stop / restart / status
│   ├── import_case.py        # 将已有运行结果导入为首页展示案例
│   ├── public_tunnel.sh      # 启动 Cloudflare Quick Tunnel 公网访问
│   ├── common.py             # Web 配置、端口、限制与 SQLite 表结构
│   ├── static/               # 前端样式与 Three.js 3D 查看器 (viewer.js)
│   └── templates/            # HTML 模板 (index.html / task.html)
├── scripts/                  # 环境搭建、依赖冻结与权重管理
│   ├── setup_envs.sh         # 搭建 hawor / egohand 环境（uv venv）
│   ├── setup_vipe.sh         # 搭建 vipe 环境
│   ├── fetch_weights.sh      # 下载所需权重
│   ├── link_weights.sh       # 在仓库内建立指向权重目录的软链接
│   └── freeze_{hawor,egohand,vipe,web}.txt  # 4 个环境的精确依赖版本快照
└── third_party/              # 内嵌第三方代码（egohand / hawor / chumpy）
```

---

## 3. 92 服务器现成部署路径（开箱即用）

当前仓库与大文件/环境均已在 **92 服务器**（`10.29.150.92`）配置完毕：

| 模块 | 路径 | 说明 |
| :--- | :--- | :--- |
| 本仓库 | `/home/wangwen/code/EgoHandWorld` | 主代码仓库（Git 管理） |
| ViPE 源码 | `/home/wangwen/code/third_party_src/vipe` | 上游 `nv-tlabs/vipe`（commit `ead2e00`，无本地修改） |
| 大文件根目录 | `/data/wenwang/` | 环境、权重、数据与运行产物根目录 |
| Python 环境 | `/data/wenwang/envs/{hawor,egohand,vipe,web}` | 基于 `uv` 创建（底层 Python 在 `/data/wenwang/cache/uv-python`，请勿删除） |
| 模型权重 | `/data/wenwang/checkpoints/` | 仓库内 `checkpoints` 及 `third_party/hawor` 下的权重均软链至此 |
| 回归测试集 | `/data/wenwang/data/regression_videos.txt` | 4 段 EgoDex（含 GT 内参）+ 1 段无手户外视频 |
| Web 运行数据 | `/data/wenwang/web/` | 任务 SQLite、日志、展示案例与上传视频 |

---

## 4. 快速开始

### 4.1 单视频推理

```bash
cd /home/wangwen/code/EgoHandWorld
PY=/data/wenwang/envs/hawor/bin/python
nvidia-smi  # 先查看空闲 GPU

# 无内参模式（默认用 ViPE 自动估计内参与米制轨迹）
$PY tools/run_video.py --video /abs/path/to/video.mp4 --gpu 4

# 指定内参模式（json 含 fx,fy,cx,cy 或 EgoHand PINHOLE 格式）
$PY tools/run_video.py --video /abs/path/to/video.mp4 --intrinsics /abs/path/to/K.json --gpu 4 --run_dir /data/wenwang/work/exp01
```

### 4.2 多卡批量推理

准备列表文件 `videos.txt`，每行格式为 `/abs/video.mp4 [可选 /abs/K.json]`（若视频同目录下存在同名 `.json` 或 `_intrinsics.json` 也会自动识别）：

```bash
$PY tools/launch_batch.py --list videos.txt --run_dir /data/wenwang/work/batch_001 --gpus 0,3,4,7
```

### 4.3 回归测试

```bash
$PY tools/launch_batch.py --list /data/wenwang/data/regression_videos.txt --run_dir /data/wenwang/work/regress --gpus 0,3,4,7
$PY tools/eval_egodex.py --run_dir /data/wenwang/work/regress
# 预期：无手户外视频的左右手 valid_frames 均为 0
```

### 4.4 Web 可视化平台

```bash
cd /home/wangwen/code/EgoHandWorld
/data/wenwang/envs/web/bin/python web/manage.py status   # 查看状态（支持 start | stop | restart | status）
```

- **本机通过 SSH 转发访问**（推荐，无上传体积限制）：
  ```bash
  ssh -N -L 15100:127.0.0.1:5100 remote92
  # 浏览器打开 http://127.0.0.1:15100
  ```
- **临时公网分享**（单次上传上限 100 MB）：
  ```bash
  bash web/public_tunnel.sh
  ```
- **添加首页示例 Case**：
  ```bash
  /data/wenwang/envs/hawor/bin/python web/import_case.py --vdir /data/wenwang/work/<run>/<video_id> --id case-demo --title Demo --desc 说明
  ```

---

## 5. 输出格式说明

单视频输出位于 `<run_dir>/<video_id>/`，核心产物为 `result.npz` 与 `summary.json`。

坐标系约定：世界坐标系 = **第 0 帧相机坐标系**（OpenCV 约定：x 右、y 下、z 前），单位为**米**（无重力对齐）。`T` 为重编码至 30 fps 后的总帧数。

| `result.npz` 字段 | 形状 | 说明 |
| :--- | :--- | :--- |
| `fps`, `K`, `image_size` | scalar / (3,3) / (2,) | 帧率、最终使用的内参（用户给定优先，否则 ViPE 估计）、图像宽高 |
| `R_c2w`, `t_c2w` | `(T, 3, 3)`, `(T, 3)` | 每帧相机到世界坐标系的米制位姿 |
| `{left,right}_valid` | `(T,)` bool | 该帧对应手部是否有效（经 RT-DETR 分数过滤） |
| `{left,right}_det_score` | `(T,)` | 该帧手框内匹配到的最高原始检测置信度 |
| `{left,right}_joints_world` | `(T, 21, 3)` | 世界系 21 关节（OpenPose / MANO 顺序）；无效帧为 `NaN` |
| `{left,right}_joints_world_soma25` | `(T, 25, 3)` | 世界系 SOMA-Hand 25 关节；无效帧为 `NaN` |
| `{left,right}_joints_cam` | `(T, 21, 3)` | 相机系 21 关节；无效帧为 `NaN` |
| `{left,right}_soma_*` | — | EgoHand 原始参数（`hand_poses`, `identity_coeffs`, `global_scale`, `scale_params`, `cam_t`, `kp2d_presence`, `bbox_cxcys`） |
| `{left,right}_hawor_mano_*` | — | 仅当 `camera_backend: hawor` 时包含 HaWoR 相机系 MANO 参数 |

---

## 6. 关键配置（`configs/default.yaml`）

- `camera_backend`: `vipe`（默认，免内参）或 `hawor`（需提供内参）。
- `fps`: 默认 `30`；`min_free_mb`: 多卡调度时的最小可用显存阈值（默认 `16000` MB）。
- `egohand.dtype`: 默认 `bfloat16`；`perspective_cropping: true`。
- `validity`:
  - `score_thr: 0.5`（原始检测分数阈值；EgoDex 真实手 p10 ≥ 0.76，无手帧 ≤ 0.44）
  - `max_gap: 5`（允许保留的最大插值间隔帧数）
  - `min_frames: 10`（整段视频有效检测不足该帧数则整只手丢弃）

---

## 7. 在新机器 / 新账号复现部署

1. 修改以下位置中的路径变量（指向新的代码目录与大文件根目录）：
   - `configs/default.yaml`（`data_root`、`python.*`、`vipe.*`）
   - `web/common.py`（默认 `EHW_WEB_DATA` 与 `hawor_python`，也可通过环境变量 `EHW_WEB_DATA` 和 `settings.json` 覆盖）
   - `scripts/{setup_envs,setup_vipe,fetch_weights,link_weights}.sh` 顶部的 `B` / `P` / `SRC` 变量
2. 依次运行环境与权重脚本：
   ```bash
   bash scripts/setup_envs.sh      # 创建 hawor / egohand 环境
   bash scripts/setup_vipe.sh      # 克隆并安装 ViPE 环境
   bash scripts/fetch_weights.sh   # 下载权重
   bash scripts/link_weights.sh    # 建立仓库内权重软链接
   ```
   Web 环境可直接使用 `scripts/freeze_web.txt` 安装：
   ```bash
   uv venv /path/to/envs/web --python 3.12
   uv pip install --python /path/to/envs/web/bin/python -r scripts/freeze_web.txt
   ```

---

## 8. 后续开发与扩展切入点

- **替换/升级相机轨迹模块**：参考 `processors/vipe_camera.py`，保持输出 `camera.npz`（`R_c2w`, `t_c2w`, `K`）接口不变，在 `tools/run_video.py` 中注册新的 `camera_backend`。
- **替换/升级手部重建或过滤策略**：修改 `processors/egohand_infer.py` 与 `processors/lift_world.py`。
- **定制 Web 页面或限制**：编辑 `web/common.py`（或运行期覆盖 `/data/wenwang/web/settings.json`）、`web/app.py` 与 `web/static/viewer.js`。
