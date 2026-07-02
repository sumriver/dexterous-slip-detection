# SPIDER 集成（XHAND + 能量流）

[SPIDER](https://github.com/facebookresearch/spider)（Meta）提供 physics-informed retargeting，原生支持 `robot_type=xhand`。

本仓库在 **纯 CPU** 下回放 SPIDER 预优化轨迹，并输出能量流 / 滑动检测 JSON，**不修改** `third_party/spider` 官方资源。

## 推荐场景：番茄酱瓶单手抓取（三步流水线）

| 步骤 | 说明 | 命令 |
|------|------|------|
| 1 | 官方双手 bimanual 回放 | `--dataset arcticv2 --task s01-ketchup_use_01 --embodiment bimanual` |
| 2 | 派生右手 workspace（去左手） | `python3 scripts/build_spider_ketchup_right.py` |
| 3 | 原轨迹 + 2s 末秒模仿 + 腕部抬高 10cm | `--ketchup-right --extend 2 --lift 0.10` |

一键运行（setup → 建 workspace → 第三步 E2E + 视频）：

```bash
bash scripts/setup_spider.sh
bash scripts/run_ketchup_pipeline.sh
```

### 分步命令

```bash
# 0. 拉取 SPIDER + LFS（含 ketchup 官方轨迹）
bash scripts/setup_spider.sh

# 1. 官方双手回放（对照，不写 third_party）
python3 scripts/run_spider_e2e.py \
  --dataset arcticv2 --task s01-ketchup_use_01 --embodiment bimanual \
  --copy-official-video

# 2. 生成右手 workspace → data/spider_ketchup_right/
python3 scripts/build_spider_ketchup_right.py

# 3. 右手回放 + 扩展 2s（循环末 1s 控制，腕 z +10cm）
python3 scripts/run_spider_e2e.py \
  --ketchup-right --extend 2 --mimic-last 1 --lift 0.10
```

### 输出文件

| 路径 | 说明 |
|------|------|
| `data/spider_ketchup_right/scene.xml` | 右手单手机械臂场景（由 bimanual 剥离） |
| `data/spider_ketchup_right/trajectory_mjwp_fast.npz` | 裁剪后的右手轨迹（nq=25, nu=18） |
| `data/spider_e2e/..._replay_extend2s_lift10cm.mp4` | 第三步完整录屏 |
| `data/spider_e2e/..._energy.json` | 逐步接触数、m̃ 估计、滑动事件（phase 含 `extend_mimic_lift`） |

### 第三步物理语义

- **原轨迹 3s**：与第二步完全相同，不截断、不 reset。
- **扩展 2s**：循环播放末 1s 的 `ctrl`（手指/手臂节奏不变）；`R_forearm_tz` 从轨迹结束值线性抬高 10cm。
- 物体随 grasp 接触**物理抬升**（非运动学 `grasp_sync`）。

## 通用 E2E 入口

```bash
python3 scripts/run_spider_e2e.py [选项]
```

| 选项 | 说明 |
|------|------|
| `--ketchup-right` | 使用 `data/spider_ketchup_right/` workspace |
| `--workspace PATH` | 自定义 workspace（含 `scene.xml` + `trajectory_*.npz`） |
| `--extend S` | 轨迹结束后追加 S 秒（末秒模仿 + 腕部抬高） |
| `--mimic-last S` | 扩展段循环的尾段时长（默认 1s） |
| `--lift M` | 扩展段腕部 tz 抬高米数；仅 `--extend` 时默认 0.10 |
| `--copy-official-video` | 复制 SPIDER 官方 HF 可视化 MP4 |

其他任务示例：

```bash
# oakinkv2 舀勺（不适合垂直抬升，见 grasp_validate）
python3 scripts/run_spider_e2e.py --copy-official-video

# gigahand 双手茶壶
python3 scripts/run_spider_xhand_demo.py --copy-official-video
```

## 代码结构

```
scripts/
  setup_spider.sh              # 克隆 SPIDER + LFS
  build_spider_ketchup_right.py # Step 2：bimanual → right-only workspace
  run_spider_e2e.py            # 统一回放 + 视频 + energy JSON
  run_ketchup_pipeline.sh      # 三步一键脚本
src/sim/
  spider_replay.py             # MuJoCo 回放、扩展段、能量流日志
  spider_ketchup.py            # 场景剥离 + 轨迹切片
  grasp_validate.py            # 物理抓取门控（legacy --lift 用）
```

## 本环境实测（番茄酱右手，CPU）

| 阶段 | 物体 Δz | 接触 |
|------|---------|------|
| 第二步（原轨迹） | +7.0 cm | 151 步 |
| 第三步（+2s 扩展） | 扩展段 +9.0 cm，全程 +16.0 cm | 扩展 200/200 步 |

`pick_spoon_bowl` 等舀取类任务：`--lift`（无 `--extend`）会走 `grasp_validate`，不通过则跳过抬升。

## 与自研 PPO 对比

| | 自研 PPO | SPIDER E2E |
|--|---------|------------|
| 训练 | 50万+ 步 CPU | 0（回放预优化轨迹） |
| 需要 GPU | 否（但难收敛） | 回放不需要 |
| 能量流 | 未接通 | `spider_replay.py` 输出 JSON |

## 许可证

SPIDER: CC BY-NC 4.0。`third_party/spider` 为 vendored 副本；大文件通过 `example_datasets` 子仓库 LFS 按需 checkout。
