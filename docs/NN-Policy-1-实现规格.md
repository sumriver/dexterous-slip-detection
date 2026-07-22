# NN-Policy-1 实现规格（一页）

**文档编号**：DS-SLIP-NN-POL-1-IMP  
**版本**：v0.1  
**日期**：2026-07-21  
**前置**：NN-1 / NN-2 闭环门闩已过（v2：baseline nn_slip **43/200**，friction÷2 **+8.7 cm**）  
**关联**：[设计书](./方案一二神经网络防滑检测设计.md) §5.3 / §10 · [NN-2-实现规格](./NN-2-实现规格.md) · [NN-1-实验报告](./NN-1-实验报告.md)  

---

## 1. 目标

在 **冻住或轻量微调检测器** 的前提下，新增 **握力策略头**，用仿真可算的 **最小充分握力** 作教师，替代「规则 ramp / soft-preempt 启发式」，做到：

1. friction÷2 闭环 **不低于** NN-2（Δz ≥ 6 cm，接触 200/200，目标贴近 +8.7 cm）。  
2. baseline：**峰值 grip 与加力步数** 优于 NN-2 默认部署（soft=0.7 / τ=0.99 / confirm=30）；nn_slip 步数 **不劣于** 43/200。  
3. 策略输出为真机可执行的 **标量 Δgrip / 目标 grip**，CPU 单步（含检测）仍 **&lt; 2 ms**。

**本阶段不做**：分指动作、整臂 RL、原始 taxel 策略（NN-3）、ONNX 真机（NN-4）。

---

## 2. 问题形式

| 项 | 规格 |
|----|------|
| 角色 | **检测 + 策略两段式**；检测提供风险，策略决定剂量 |
| 观测 `s_t` | 近窗特征（同 NN-0，`T=40,D=26`）+ `p_slip`（检测头）+ 当前 `grip_extra` |
| 部署禁入 | `friction_scale`、`slip_rule_s2`、`phase_extend`（及任何仿真专有泄漏维） |
| 动作 `a_t` | **目标握力** `g* ∈ [0, 0.25]` 或增量 `Δg`（实现二选一，写入 meta；默认 **目标 grip**） |
| 执行 | `grip_extra ← max(grip_extra, g*)`（ratchet）+ 既有 `apply_grip_boost`；速率上限可选 |
| 启用段 | 默认仅 **extend**（与现 antislip 一致）；轨迹段提前介入为可选 ablation |

---

## 3. 教师：最小充分握力（优先于再抄 scheme-2）

规则 antislip 的 `y_grip` 在 baseline 上常顶满，不适合作为「省力」策略教师。

**默认教师生成（仿真 only）：**

1. 对每个 export case 跑开环，记录是否 FAIL。  
2. 对可救工况（至少 friction÷2），在成功的 NN-2 / 规则闭环轨迹上，对 extend 段做 **事后压缩**：从末时刻向前，将 `grip_extra(t)` 降为仍满足  
   - 接触步数 ≥ 200/200（或约定阈值），且  
   - extend Δz ≥ 6 cm，且物体下落 ≤ 3 cm  
   的 **逐时最小序列** `g_min(t)`。  
3. 窗标签：`y_policy = g_min[t_end]`（窗末）或窗内均值（固定一种写进 meta）。  
4. 对照教师（ablation）：规则 `--antislip` 的 `y_grip`、NN-2 Δgrip 头输出。

| 标签 | 用途 |
|------|------|
| `y_policy` | **默认**策略回归目标 |
| `y_grip` | 对照（规则闭环） |
| `y_event` / `p_slip` | 检测侧；策略训练时检测头默认 **冻结** |

**剂量平衡（NN-0 补数）**：旧网格中 `friction_div2` 整族进 test，训练几乎只有低 `G*`（baseline/mass）窗，策略头会系统性欠剂量。补 μ∈{0.70,0.60,0.55,0.45,0.40} 与 `mass×{2,4}×μ{0.5,0.6}`（**新 case 名**进 train）；policy 导出跳过 `÷4/÷8` 与 `mass×{8,16,32}`。验收：train 中 `y_policy≥0.13` 显著高于 ~5%。

---

## 4. 网络

**推荐：残差策略头（P1）挂在共享 TCN 上**

```
X (B,T,D) → [冻结] SlipTCN 骨干 → h
  → [冻结] head_slip → p_slip
  → head_policy (tier A): concat(h, p_slip, grip_extra)  # 34-D
       → Linear(W)→LN→ReLU → Linear(W)→LN→ReLU → Linear(1)→sigmoid·max_grip
可选：ĝ = clip( g_ref + residual )   # g_ref 为 NN-2 Δgrip；`--residual`
```

默认 `W=64`（≈6.7k 可训参数）。消融：`--policy-width 32`。

| 约束 | 值 |
|------|-----|
| 新增参数 | &lt; 20K（整模仍 &lt; 100K）；默认 tier A ≈ 6721 |
| 损失 | `L = MSE(ĝ, y_policy) + λ_sparse · mean(ĝ)`，默认 `λ_sparse=0.05`（抑常顶满） |
| 训练 | 检测头默认 `requires_grad=False`；`--unfreeze-detect` 为可选 |
| 推理 | 确定性；不采样 |

**备选（不阻塞）**：独立小 MLP 只吃 `(p_slip, S_ratio, sep, grip_extra)` 手工摘要——作延迟/消融对照。

---

## 5. 与现控制器的关系

| 模式 | 行为 |
|------|------|
| `policy=off` | 现 NN-2：soft preempt + confirm + `set_grip(Δgrip_nn2)` |
| `policy=replace`（默认目标） | 检测仍出 `p_slip`；**握力只听** `ĝ`（可仍要求 `p_slip>τ_gate` 才允许加力） |
| `policy=residual` | `g = clip(g_nn2 + π_θ)` |

CLI 草稿：`--antislip-nn --policy models/slip_nn_policy1`。

安全层（所有模式保留）：`g ∈ [0,0.25]`、可选每步 `|Δg|≤δ`、仅约定 phase 使能。

---

## 6. 文件清单

| 路径 | 职责 |
|------|------|
| `scripts/export_min_grip_teacher.py` | 事后最小握力 → `y_policy` 写入 NPZ / 旁路 shard |
| `src/sim/slip_nn_policy.py` | `SlipPolicyHead` / `SlipDetectAndPolicy` |
| `scripts/train_slip_policy.py` | 冻检测 + 训策略；写 `models/slip_nn_policy1/` |
| `src/sim/slip_nn_detector.py` | 扩展：`policy_grip` 读取与 `τ_gate` |
| `src/sim/spider_replay.py` | `--policy` / residual\|replace |
| `scripts/eval_slip_policy_closedloop.py` | baseline + ÷2（+ 可选 ÷4 表） |
| `models/slip_nn_policy1/` | ckpt + `train_meta.json` + README |
| `docs/NN-Policy-1-实验报告.md` | 完成后撰写 |

---

## 7. 验收

| # | 检查 | 门槛 |
|---|------|------|
| 1 | friction÷2 | Δz ≥ 6 cm，接触 ≥ 200/200（不低于 NN-2） |
| 2 | baseline 加力 | 峰值 `grip_extra` **&lt; NN-2 默认** 或加力步数更少（二者报一主一辅） |
| 3 | baseline 检测 | `nn_slip_events` **≤ 43**（不劣于 NN-2） |
| 4 | 延迟 | 检测+策略 CPU mean **&lt; 2 ms** |
| 5 | 消融 | `y_policy` vs `y_grip` 教师各一表 |
| 6 | ÷4 | 有开环/闭环对比即可，不挡合并 |

---

## 8. 实施顺序

1. `export_min_grip_teacher.py`：在 ÷2 成功轨迹上产出 `y_policy`，抽查曲线非恒 0.25。  
2. `SlipPolicyHead` + 单测（形状、clamp、冻骨干）。  
3. `train_slip_policy.py`（replace 模式）。  
4. 接入 `spider_replay` → closed-loop 表 vs NN-2。  
5. residual 消融；÷4 摸底；实验报告。  

**风险**：最小握力 DP/压缩若过猛 → ÷2 边缘失败；对策：压缩时留 **margin**（Δz≥7 cm 或 grip 地板 0.05）。教师若仍接近规则顶满 → 提高 `λ_sparse` 或只在 `y_event=1` 邻域回归。

---

*一页规格 · NN-Policy-1 · dexterous-slip-detection · 2026-07-21*
