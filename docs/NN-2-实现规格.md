# NN-2 实现规格（一页）

**文档编号**：DS-SLIP-NN-2-IMP  
**版本**：v0.1  
**日期**：2026-07-20  
**前置**：NN-1 已合入（默认 `y_event` + leak drop + latch + confirm；闭环门闩 PASS）  
**关联**：[NN-1-实验报告](./NN-1-实验报告.md) · [设计书](./方案一二神经网络防滑检测设计.md) §5.3 / §10  

---

## 1. 目标

在保持 NN-1 friction÷2 闭环表现（Δz ≥ 6 cm，接触 200/200，目标贴近 +8.7 cm）的前提下：

1. **进一步压 baseline 误触发**：目标 **&lt; 50/200**（设计书：相对规则 ~195/200 的 &lt;50%，即 &lt;~98；本阶段加严到 **&lt;50**，相对 NN-1 的 93 再砍约一半）。  
2. 引入 **多任务 Δgrip 头**（可选启用），用日志中的 `y_grip`（规则闭环时累积握力）作回归目标，减少「检测对了但握力策略粗糙」的问题。  
3. 探索 **难工况**：friction÷4 是否可部分挽救（不挡合并；有表即可）。

**本阶段不做**：原始 taxel CNN（NN-3）、ONNX 真机（NN-4）。

---

## 2. 数据接口（继承 NN-0/1）

| 项 | 规格 |
|----|------|
| 输入 | `(N, 40, 26)`，部署仍 `drop_leak_features` |
| 分类标签 | 默认 **`y_event`**（与 NN-1 一致） |
| 回归标签 | **`y_grip`** ∈ [0, 0.25]，窗末步或窗内均值（实现时固定一种并写进 meta） |
| 划分 | 同 NN-0 manifest |

**`y_grip` 注意（2026-07-20）**：开环导出时 `grip_extra≡0`，NPZ 中 `y_grip` 全零。打通流水线可用合成 `y_grip_syn = max_grip * y_event`；正式教师需导出时开 `--antislip` 写入真实握力。

---

## 3. 网络

**Baseline：共享 TCN 骨干 + 双头**

```
Input (B, T, D)
  → 同 NN-1 Conv 骨干 → h ∈ R^{64}
  → head_slip: FC → logit  (BCEWithLogits)
  → head_grip: FC → Δgrip̂ ∈ R  (MSE；训练时可用 softplus/clamp 到 [0, 0.25])
```

| 约束 | 值 |
|------|-----|
| 参数量 | 仍 &lt; 80K |
| 损失 | `L = BCE + λ · MSE`，默认 `λ=0.1`；可用 `--lambda-grip 0` 退化为 NN-1 |
| 推理 | 分类头驱动 `slip_active`；Δgrip 可选覆盖 `GripBoostController` 步长/目标 |

**备选（不阻塞）**：更深 TCN / 小 GRU；阈值校准（temperature / Platt）仅分类头。

---

## 4. 部署策略候选（择一或组合，需闭环扫表）

| ID | 策略 | 预期作用 |
|----|------|----------|
| D0 | 保持 NN-1：confirm=15, τ=0.7, latch | 基线对照 |
| D1 | 提高 confirm 或 τ（分类 only） | 压误报，可能伤 ÷2 |
| D2 | 多任务头：`slip_active` 时用 `Δgrip̂` 设握力（替代固定 +0.015/步） | 少触发但每次加力更准 |
| D3 | 两段阈值：baseline 用高 τ，低 μ 用低 τ（需 friction_scale 仅训练可见、部署禁用） | **禁止**用泄漏 μ；改用力特征自适应 |

默认实现顺序：**先 D1 扫表 → 再 D2 多任务**。

---

## 5. 文件清单

| 路径 | 职责 |
|------|------|
| `src/sim/slip_nn_model.py` | `SlipTCNMulti`（或 `SlipTCN` 加 optional grip head） |
| `scripts/train_slip_tcn.py` | `--lambda-grip`、`--grip-label y_grip` |
| `src/sim/slip_nn_detector.py` | 可选读 `delta_grip`；接口向后兼容 |
| `scripts/eval_slip_nn_closedloop.py` | 增加 baseline 误触发 &lt;50 报告；可选 ÷4 |
| `models/slip_nn_v2/` | NN-2 权重与 meta（勿覆盖 v1） |
| `docs/NN-2-实验报告.md` | 完成后撰写 |

---

## 6. 验收

| # | 检查 | 门槛 |
|---|------|------|
| 1 | friction÷2 + antislip-nn | Δz ≥ 6 cm，接触 ≥ 200/200（不低于 NN-1） |
| 2 | baseline 误触发 | **&lt; 50/200** |
| 3 | NN-1 回归 | v1 checkpoint 闭环门闩仍 PASS（CI / 手工） |
| 4 | 若启用 Δgrip | 报告 grip MAE；闭环表对比 D0 |
| 5 | 延迟 | CPU mean &lt; 2 ms |
| 6 | friction÷4 | 有开环/闭环对比表（可不 PASS） |

---

## 7. 实施顺序

1. D1：在 v1 上扫 confirm ∈ {15,20,25} × τ ∈ {0.7,0.8,0.85} → 看能否不训新模型就过 &lt;50  
2. 多任务模型 + `λ` 扫 + 闭环对比  
3. ÷4 摸底表  
4. 实验报告 + 默认切到 v2（若门闩全过）  

### D1 摸底结论（2026-07-20，`deploy_sweep_d1.json`）

仅调 τ / confirm **无法**同时满足 baseline &lt;50 与 friction÷2 PASS：

| τ | confirm | baseline nn_slip | friction÷2 |
|---|---------|------------------|------------|
| 0.7 | 15 | 93 | +8.7 PASS |
| 0.7 | 25 | **73** | +8.7 PASS（NN-1 内最优） |
| 0.8 | 20 | 80 | +8.7 PASS |
| 0.8 | 25 | 70 | **FAIL**（−15.8 cm） |
| 0.85 | 20/25 | 79/70 | **FAIL** |

→ **必须进入 D2 多任务 / 新训练**；D1 可将默认 confirm 试提到 20–25 作为过渡（仍 &gt;50）。

### D2 脚手架结论（同日，`models/slip_nn_v2/`）

- `SlipTCNMulti` + `train_slip_multitask.py` 已通；开环 `y_grip≡0`，现用合成 grip。  
- confirm=25 时 v2 丢失 ÷2；confirm=15 时 ÷2 PASS，baseline nn_slip ≈ **97**（未优于 v1）。  
- 下一步：`--antislip` 重导出真实 `y_grip`，再训并把 Δgrip 接到握力控制器。

---

*一页规格 · NN-2 · dexterous-slip-detection · 2026-07-20*