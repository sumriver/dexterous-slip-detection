# SPIDER 集成（XHAND 抓取）

[SPIDER](https://github.com/facebookresearch/spider)（Meta）提供 **physics-informed retargeting**，原生支持 `robot_type=xhand`。

## 推荐路线：开源 E2E（无 GPU）

**先跑通轨迹 → 仿真 → 能量流日志，再考虑自研算法。**

```bash
bash scripts/setup_spider.sh
python3 scripts/run_spider_e2e.py --copy-official-video
# 握勺后抬升 10cm（自动在抓取帧截断 + grasp-sync 抬升）
python3 scripts/run_spider_e2e.py --lift 0.10
```

| 输出 | 说明 |
|------|------|
| `data/spider_e2e/oakinkv2_xhand_right_pick_spoon_bowl_replay.mp4` | CPU 回放录屏 |
| `data/spider_e2e/..._replay_lift10cm.mp4` | 握勺 + 抬升 10cm |
| `data/spider_e2e/oakinkv2_xhand_right_pick_spoon_bowl_energy.json` | 逐步接触数 + 能量流 m̃ + 滑动事件 |

默认任务：`oakinkv2 / xhand / right / pick_spoon_bowl`（单手舀勺入碗，SPIDER 预优化轨迹）。

### 本环境实测（2026-07-02，纯 CPU）

| 指标 | 结果 |
|------|------|
| 仿真步数 | 300 |
| 有接触步数 | 273 |
| 物体位移 Δz | 1.2 cm |
| 能量流日志 | ✅ JSON 逐步输出 |
| `--lift 0.10` 握勺抬升 | ❌ **物理不成立** — 见下 |

`pick_spoon_bowl` 是**勺舀入碗**任务，不是垂直握持抬升：
- 勺碗端始终贴地（floor contact）
- 手指/拇指捏在勺柄上方，接触点比重心高 ~40–50mm、偏水平 ~30–60mm
- 这是**杠杆**，不是重心下方的支撑，纯物理无法垂直抬升 10cm

`--lift` 现在会先跑 `grasp_validate`（离地、拇指–指对握、支撑力、接触点相对 COM），**不通过则跳过抬升**，不再使用 `grasp_sync` 运动学作弊。

垂直抬升应换场景：**平躺瓶 + 侧向三指对握**（后续 GPU MJWP 或专用 task），不是本 task。

其他任务：

```bash
# 双手抓茶壶（gigahand）
python3 scripts/run_spider_xhand_demo.py --copy-official-video
```

## 快速开始（茶壶 demo）

```bash
bash scripts/setup_spider.sh
python3 scripts/run_spider_xhand_demo.py --copy-official-video
```

输出视频：`data/spider/xhand_p36-tea_mjwp_replay.mp4`

## 本环境实测结果（2026-07-01）

| 步骤 | 状态 | 说明 |
|------|------|------|
| `uv sync` 安装 SPIDER | ✅ | Python 3.12 + mujoco 3.7 + mujoco-warp |
| `run_mjwp_fast.py` 优化 | ❌ CPU | `RuntimeError: Must be a CUDA device` — MJWP 图捕获必须 GPU |
| `mjcpu_viewer.py` 回放预计算轨迹 | ✅ | 双手 XHAND 抓茶壶 `p36-tea`，~108s 录屏 |
| `run_spider_e2e.py` oakinkv2 pick_spoon_bowl | ✅ | 回放 + 能量流 JSON + MP4 |

## 与自研 PPO 对比

| | 自研 PPO | SPIDER E2E |
|--|---------|------------|
| 训练步数 | 50万–90万 | 0（回放） |
| 本机耗时 | 数小时 CPU | ~8 秒 |
| 需要 GPU | 否（但学不动） | 回放不需要 |
| 手部控制 | `qpos` 瞬移 | 6-DoF 滑轨 + kp=300 伺服 |
| 碰撞 | 60 geom 混 visual | 显式 capsule + contact pair |
| 下游能量流 | 未接通 | `spider_replay.py` 已输出 JSON |

## 暂停的路线

- **路径 B CPU 抓取搜索**：自造握/抬判定，已证明不可靠，不再推荐
- **平躺瓶自定义 task**：需 GPU 跑 MJWP，或后续提供 mocap

## 许可证

SPIDER: CC BY-NC 4.0（非商业研究可用）。`third_party/spider` 为 vendored 副本，不随本仓库分发 LFS 大文件。
