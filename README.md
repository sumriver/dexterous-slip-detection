# dexterous-slip-detection

灵巧手抓取滑动检测研究项目：基于触觉能量流方法的 MuJoCo 仿真与 XHAND1 部署路径。

**目标硬件**：XHAND1 / XHAND1 PRO  
**核心算法**：NAIST + Honda 能量流方法（[arXiv:2512.21043](https://arxiv.org/abs/2512.21043)）  
**仿真平台**：MuJoCo + **XHAND1**（主）/ Shadow Hand E3M5（对照）

## 研究问题

如何让灵巧手在抓取物体时，通过触觉反馈实时检测并抑制滑动？

## 项目结构

```
dexterous-slip-detection/
├── docs/                    # 研究文档
├── models/                  # 场景与物体模型（Shadow Hand 通过 setup 脚本拉取）
├── src/energy_flow/         # 能量流算法核心模块
├── scripts/                 # 仿真与实验脚本
├── tests/                   # 单元测试
└── data/                    # 实验输出（gitignore）
```

## 快速开始

### 1. 环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 下载 Shadow Hand 模型

```bash
bash scripts/setup_models.sh
```

### 3. 运行 XHAND1 瓶子抓取仿真（推荐）

```bash
python scripts/run_xhand_grasp_sim.py
# 关键帧: data/xhand_grasp/keyframes/
```

URDF 来自 [worldstring](https://github.com/MaureenZOU/worldstring)，由 `setup_models.sh` 自动拉取并转换为 MJCF。详见 [docs/XHAND-SIM.md](docs/XHAND-SIM.md)。

### 4. 运行 Shadow Hand 瓶子抓取仿真（对照）

```bash
python scripts/run_bottle_grasp_sim.py
```

录制可视化视频（MP4，1280×720，30 fps）：

```bash
python scripts/run_bottle_grasp_sim.py --video
# 输出: data/bottle_grasp/phase1_bottle_grasp.mp4
```

**注意**：当前为物理仿真（无运动学绑瓶）。Shadow Hand + 竖立圆柱瓶极难抓稳，
预期结果为 FAIL。详见 [docs/PHASE1-PHYSICS.md](docs/PHASE1-PHYSICS.md)。
关键帧 PNG 保存在 `data/bottle_grasp/keyframes/`。

场景：桌面竖立细长瓶 → 中部抓握 → 抬升 20 cm → 空中翻转 90° 至水平。

### 5. 运行最小方块仿真（能量流日志）

```bash
python scripts/run_minimal_sim.py
```

### 6. 运行测试

```bash
pytest tests/ -v
```

## 推进路线

| Phase | 内容 | 状态 |
|-------|------|------|
| 1 | 方块抓取 + 能量状态计算 + 滑动检测 | ✅ 基础完成 |
| 1b | **XHAND1 瓶子抓取** + 触觉 taxel 映射 | ✅ 框架完成（抓取待调参） |
| 1c | Shadow Hand 瓶子抓取（对照） | ✅ 可用 |
| 2 | LGM-FF 在线学习（NumPy） | 待开始 |
| 3 | pMPC 力优化 | 待开始 |
| 4 | 适配 XHAND1 传感器格式 | 待开始 |

## 关键约束

- Shadow Hand 拇指对立角约 35.7°，**无法真正对握圆柱体** → 仿真使用方块物体 + 三指抓取
- XHAND1 **无振动感知通道** → 不适用振动检测路线，优先能量流方法
- 当前开发环境 **无 GPU** → LGM-FF / 能量流用 CPU（NumPy）即可

## 参考

- [NAIST+Honda 论文](https://arxiv.org/abs/2512.21043) — Online Model-Based RL with Physical Energy Abstraction
- [MuJoCo Menagerie — Shadow Hand](https://github.com/google-deepmind/mujoco_menagerie/tree/main/shadow_hand)

## License

Apache-2.0（Shadow Hand 模型遵循其原始许可证）
