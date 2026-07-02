# SPIDER 集成（XHAND 抓取）

[SPIDER](https://github.com/facebookresearch/spider)（Meta）提供 **physics-informed retargeting**，原生支持 `robot_type=xhand`，比从零 PPO 快几个数量级。

## 快速开始

```bash
bash scripts/setup_spider.sh
python scripts/run_spider_xhand_demo.py --copy-official-video
```

输出视频：`data/spider/xhand_p36-tea_mjwp_replay.mp4`

## 本环境实测结果（2026-07-01）

| 步骤 | 状态 | 说明 |
|------|------|------|
| `uv sync` 安装 SPIDER | ✅ | Python 3.12 + mujoco 3.7 + mujoco-warp |
| `run_mjwp_fast.py` 优化 | ❌ CPU | `RuntimeError: Must be a CUDA device` — MJWP 图捕获必须 GPU |
| `mjcpu_viewer.py` 回放预计算轨迹 | ✅ | 双手 XHAND 抓茶壶 `p36-tea`，~108s 录屏 |
| 平躺瓶自定义 task | ⏳ | 需 GPU 跑优化，或人工演示 + retarget |

## 与自研 PPO 对比

| | 自研 PPO | SPIDER |
|--|---------|--------|
| 训练步数 | 50万–90万 | 0（回放）或 8–32 轮采样优化 |
| 本机耗时 | 数小时 CPU | 回放 ~2 分钟 |
| 需要 GPU | 否（但学不动） | 优化需要，回放不需要 |
| 手部控制 | `qpos` 瞬移 | 6-DoF 滑轨/旋转关节 + `kp=300` 伺服 |
| 碰撞 | 60 geom 混 visual | 显式 capsule + contact pair |

## 路径 B：CPU 抓取搜索（无 GPU、无 RL）

不跑 MJWP，直接复用 SPIDER 的 XHAND 碰撞/执行器模型，在平躺瓶场景上做 **CPU 网格+随机搜索**，用 `support_z` + 微抬升探针评分。

```bash
bash scripts/setup_spider.sh
python3 scripts/build_spider_bottle_scene.py
PYTHONPATH=src python3 scripts/run_spider_bottle_grasp.py --refine 100 --video
```

| 输出 | 说明 |
|------|------|
| `models/xhand_spider/bottle_scene.xml` | 手 + 水平柱瓶 + 显式 contact pair |
| `data/spider_bottle/best_grasp_ctrl.npz` | 最优 18 维 ctrl（6 臂 + 12 指） |
| `data/spider_bottle/spider_bottle_grasp.mp4` |  settle → 抬升 20cm 录屏 |
| `data/spider_bottle/result.json` | 支撑力、抬升高度、是否通过 |

### 本环境实测（2026-07-02，纯 CPU）

- 粗搜 ~400 组 + 精搜 100 轮（含 5cm 抬升探针）：约 2–3 分钟
- 典型结果：`support_z≈1.05N`（mg=1.47N），`lift_dz≈10cm`，末端支撑归零（瓶被甩脱但未穿模）
- 对比自研 PPO：有真实接触和物理抬升，不是 `qpos` 瞬移或 visual mesh 假间隙

### 与路径 A（MJWP 回放）对比

| | 路径 A MJWP | 路径 B CPU 搜索 |
|--|------------|----------------|
| GPU | 优化必须 | 不需要 |
| 输入 | 预录 mocap / HF 轨迹 | 无演示，纯搜索 |
| 场景 | SPIDER 官方 task | 自定义平躺瓶 |
| 抬升 | 轨迹里已有 | 搜索 tz ramp，当前 ~10cm |

## 许可证

SPIDER: CC BY-NC 4.0（非商业研究可用）。`third_party/spider` 为 vendored 副本，不随本仓库分发 LFS 大文件。
