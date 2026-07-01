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

## 平躺瓶下一步

1. **有 GPU 的机器**上跑：`uv run examples/run_mjwp_fast.py +override=gigahand_fast task=<custom> robot_type=xhand device=cuda:0`
2. 或提供 **Hot3D / 自建 mocap** → SPIDER `process_datasets` → IK → MJWP
3. 把 SPIDER 的 `right.xml` 碰撞/执行器模型 **替换** 当前 `xhand_right_sim.xml` 的 weak position 执行器

## 许可证

SPIDER: CC BY-NC 4.0（非商业研究可用）。`third_party/spider` 为 vendored 副本，不随本仓库分发 LFS 大文件。
