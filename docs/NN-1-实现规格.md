# NN-1 实现规格（一页）

**文档编号**：DS-SLIP-NN-1-IMP  
**版本**：v0.1  
**日期**：2026-07-15  
**前置**：NN-0 已合入（`data/slip_nn/`，T=40，D=26，L2–L4 校验通过）  
**关联**：[方案一二神经网络防滑检测设计](./方案一二神经网络防滑检测设计.md) §5 / §7 / §10  

---

## 1. 目标

用 NN-0 特征窗训练轻量时序二分类器，蒸馏 **融合教师**，在 ketchup 仿真闭环中以 `--antislip-nn` 驱动现有 `GripBoostController`，不低于规则方案二（friction÷2）闭环表现。

**本阶段不做**：多任务 Δgrip（NN-2）、原始 taxel 端到端（NN-3）、ONNX 真机（NN-4）。

---

## 2. 数据接口（冻结）

| 项 | 规格 |
|----|------|
| 输入张量 `X` | `(N, 40, 26)`，特征名 = `FEATURE_NAMES` / `manifest.json` |
| 归一化 | 训练集 per-feature z-score：`manifest.norm.mean/std`；推理同样变换 |
| **默认标签** | **`y_event`**（未来 0.5 s 内物体掉落 ≥1 cm）；`y_fused` / `y_scheme2` 仅作对照 |
| 对照标签（必跑 ablation） | `y_fused`、`y_scheme2`（规则教师在 baseline extend 上约 190/200 步触发，蒸馏后 τ 无法压误报） |
| 不用作默认教师 | `y_gt`（仿真运动学，当前 pos rate ≈0.97–1.0，易学成常报警） |
| 划分 | train / val / test 按 NN-0 manifest（test 含 `friction_div2`） |

---

## 3. 网络架构

**Baseline：1D-TCN + MLP**（与设计书 §5.1 一致，D 改为 26）

```
Input (B, T=40, D=26)
  → permute → (B, D, T)
  → Conv1d(26→64, k=3, padding=1) + ReLU
  → Conv1d(64→64, k=3, padding=2, dilation=2) + ReLU
  → AdaptiveAvgPool1d(1)          # 全局时间池化
  → Flatten → FC(64→32) + ReLU
  → FC(32→1) + Sigmoid
Output: p_slip ∈ [0, 1]
```

| 约束 | 值 |
|------|-----|
| 参数量 | < 50K（目标） |
| 推理 | CPU，单窗 < 2 ms（含特征环缓，不含 MuJoCo） |
| 决策 | `p_slip > τ`，默认 `τ=0.5`，val 上按 F1 可调 |

**备选（不阻塞）**：`GRU(hidden=64)×1 → 末步 → MLP`；同接口，脚本 `--arch {tcn,gru}`。

**损失**：`BCEWithLogits` 或 `BCE(p, y_fused)`；类不平衡用 `pos_weight = N_neg/N_pos`（按 train）。首版不加 focal。

**训练增强（可选，默认开轻量）**：力相关维高斯噪声 `σ=0.05·|x|`；5% 时间维随机丢帧（掩码后用邻帧填）。

---

## 4. 文件与接口清单

| 路径 | 职责 |
|------|------|
| `src/sim/slip_nn_model.py` | `SlipTCN` / 可选 `SlipGRU`；`count_params()` |
| `src/sim/slip_nn_detector.py` | `SlipNeuralDetector`：环形缓冲 T=40，归一化，前向，`update(x_t)→reading` |
| `scripts/train_slip_tcn.py` | 读 NPZ+manifest，训/验，写 checkpoint + metrics JSON |
| `scripts/eval_slip_nn_offline.py` | val/test 上 Precision/Recall/F1、提前量粗估 |
| `models/slip_nn/README.md` | 版本、教师标签、τ、指标、git SHA |
| `models/slip_nn/slip_tcn_v1.pt` | 权重（可 git-lfs 或本地；规格要求可复现训练） |
| `spider_replay.py` | 接入 `antislip_nn=True` → detector + GripBoost |
| `run_ketchup_robustness_sweep.py` | 增加 `--antislip-nn` |

**推理接口（对齐规则检测器语义）**：

```python
class SlipNeuralDetector:
    def __init__(self, model_path, norm, *, threshold=0.5, device="cpu"): ...
    def reset_extend(self) -> None: ...
    def update(self, features: np.ndarray) -> SlipNnReading:
        """features: (26,) current step; maintains T-window internally."""
# SlipNnReading: p_slip, slip_active (p>τ), n_valid_steps
```

触发后行为与规则一致：`GripBoostController.on_slip()` + `apply()`。

---

## 5. 训练协议（初版）

| 超参 | 值 |
|------|-----|
| Optimizer | Adam，lr=`1e-3` |
| Batch | 64 |
| Epochs | ≤50，early stop patience=8（监控 val F1） |
| Seed | 固定 `42`（写入 README） |
| 设备 | CPU 或 GPU；默认 CPU 可训完 |

命令草稿：

```bash
python3 scripts/train_slip_tcn.py \
  --data data/slip_nn --label y_fused --arch tcn --out models/slip_nn
python3 scripts/train_slip_tcn.py --label y_scheme2 --out models/slip_nn/ablate_s2
```

---

## 6. 验收（完成定义）

| # | 检查 | 门槛 |
|---|------|------|
| 1 | Offline val F1（对 **`y_fused`**） | ≥ 0.90 |
| 2 | Ablation：同配置 `y_scheme2` 报告 F1 / 闭环 | 有表即可，不挡合并 |
| 3 | friction÷2 + `--antislip-nn` | extend Δz ≥ **6 cm**，接触 ≥ **200/200**（对齐规则÷2 口径；目标贴近 +8.7 cm） |
| 4 | baseline + `--antislip-nn` | 仿真 PASS；extend 误触发步数 **< 100/200** |
| 5 | 单步推理延迟 | **< 2 ms** CPU（T=40,D=26） |
| 6 | 回归 | `pytest`；fixed seed 复现训练指标写入 `models/slip_nn/README.md` |

**风险（与融合教师相关）**：`y_fused` 继承方案一高敏 → baseline 误报可能升高。若 #4 失败：先升 `τ` 搜 F1/误报 Pareto；仍失败则回退默认教师到 `y_scheme2`，`y_fused` 降为对照（写进 README 决策记录）。

---

## 7. 实施顺序

1. `slip_nn_model.py` + 单测（形状、参数量）  
2. `train_slip_tcn.py`（`y_fused` → checkpoint）  
3. `slip_nn_detector.py` + offline eval  
4. 接入 `spider_replay` / sweep `--antislip-nn`  
5. friction÷2 + baseline 闭环表 → 达标后开 PR  

---

## 8. 非目标（显式排除）

- 改 NN-0 特征维数或重做全量场景  
- 部署期输入 `y_gt` / MuJoCo 内滑移速度  
- ONNX / 真机（NN-4）  
- Δgrip 回归头（NN-2）  

---

*一页规格 · NN-1 · dexterous-slip-detection · 2026-07-15*
