# NN-1 实验报告：时序网络防滑检测（TCN + 事件标签）

**文档编号**：DS-SLIP-NN-1-RPT  
**日期**：2026-07-17  
**项目**：dexterous-slip-detection  
**分支**：`cursor/nn1-tcn-impl-f222`  
**PR**：https://github.com/sumriver/dexterous-slip-detection/pull/8  
**规格**：[`NN-1-实现规格.md`](./NN-1-实现规格.md)  
**前置**：NN-0 特征窗与 L2–L4 校验（`validate_report.json` overall **PASS**）  
**对照报告**：[方案一与方案二实验报告](./防滑算法方案一与方案二实验报告.md)

---

## 摘要

本报告整理 **NN-1**：在 NN-0 的 26 维特征窗上训练轻量 **1D-TCN** 二分类器，以 `--antislip-nn` 驱动既有 `GripBoostController`，在 ketchup / XHAND 仿真中验收闭环防滑。

| 工况 | 开环 | 规则方案二 | **NN-1（默认）** |
|------|------|-----------|------------------|
| baseline（μ×1） | PASS | PASS（误触发 ~191/200） | **PASS**（误触发 **93/200**） |
| **friction÷2（μ×0.5）** | **FAIL**（Δz ≈ −20.7 cm） | **PASS**（+8.7 cm, 200/200） | **PASS**（**+8.7 cm**, 200/200） |

**核心结论**：

1. **闭环达标**：默认配置在 friction÷2 上与规则方案二同量级恢复（+8.7 cm / 满接触），并在 baseline 上将误触发压到 **&lt;100/200**。  
2. **教师选择是关键**：蒸馏 `y_fused` / `y_scheme2` 时，即便 τ 提到 0.99，baseline 仍约 **190–200/200** 步触发——规则教师本身在 baseline 上近乎常开，τ 无法救。  
3. **默认教师改为 `y_event`**（未来 0.5 s 内掉落 ≥1 cm），配合 **泄漏特征清零 + latch + confirm=15 + τ=0.7** 后闭环门闩全部 PASS。  
4. 离线 val F1（对 `y_event`，τ=0.7）约 **0.76**，低于规格初稿对 `y_fused` 的 0.90 软门槛；本阶段以**闭环门闩优先**。

---

## 1. 实验目标与验收门闩

### 1.1 目标

- 用可复现的仿真数据训练轻量网络，替代/蒸馏规则检测器。  
- 在 **friction÷2** 上不低于规则方案二闭环表现。  
- 在 **baseline** 上显著降低误触发（相对规则 ~190+/200）。

### 1.2 验收（相对规格 §6，闭环优先）

| # | 检查 | 门槛 | 结果 |
|---|------|------|------|
| A | friction÷2 + `--antislip-nn` | Δz ≥ 6 cm，接触 ≥ 200/200；目标贴近 +8.7 cm | **PASS**：+8.71 cm，200/200 |
| B | baseline + `--antislip-nn` | 仿真 PASS；误触发步数 **&lt; 100/200** | **PASS**：Δz +9.38 cm；**93/200** |
| C | 参数量 / 延迟 | &lt;50K；CPU 单步 &lt;2 ms | **PASS**：19521 params；mean **0.18 ms** |
| D | Ablation 教师 | 报告 `y_fused` / `y_scheme2` | **有表**（闭环均无法过 B） |
| E | Offline F1（规格初稿对 `y_fused` ≥0.90） | 软门槛 | `y_event`@τ=0.7 val F1 **0.76**（闭环优先，未挡合并） |

数据来源：`data/slip_nn/closedloop_smoke.json`、`models/slip_nn/{train_meta,eval_val}.json`。

---

## 2. 数据与标签

### 2.1 特征与划分（NN-0）

| 项 | 值 |
|----|-----|
| 窗形 | `(N, T=40, D=26)` |
| 总量 | 16596 窗（train 11064 / val 3688 / test 1844） |
| 场景 | 单任务 `s01-ketchup_use_01`；质量/摩擦/延伸变体 |
| 归一化 | train per-feature z-score（`manifest.json`） |
| 校验 | L2–L4 `validate_nn0_dataset.py` → **PASS** |

导出摘要：`data/slip_nn/export_summary.json`。

### 2.2 标签定义

| 标签 | 含义 | 用途 |
|------|------|------|
| `y_event` | 未来 **0.5 s**（50 步）内物体竖直掉落 ≥ **1 cm** | **默认教师** |
| `y_scheme2` | 规则方案二检测器输出 | 消融对照 |
| `y_fused` | 融合教师（方案一∪方案二语义） | 消融对照 |
| `y_gt` | 仿真运动学滑移 | 不用作默认（正样本率过高） |

### 2.3 为何放弃融合教师作默认

规则方案二在 **baseline extend** 上约 **191/200** 步触发（latch 后）。蒸馏该行为后：

| 教师 | τ | baseline `nn_slip_events` | friction÷2 |
|------|---|---------------------------|------------|
| `y_fused` | 0.5 → **0.99** | **200/200** | PASS（常触发） |
| `y_scheme2` | 0.5 → 0.9 | **190–193/200** | PASS |
| 规则 scheme-2（参考） | — | ~191/200 | +8.7 cm |

→ **提高 τ 无效**；必须换监督信号。完整扫表见 `data/slip_nn/tau_sweep_y_fused.json`、`tau_sweep_y_scheme2.json`。

---

## 3. 模型与部署配置

### 3.1 网络

```
Input (B, 40, 26) → Conv1d 26→64 (k=3) → Conv1d 64→64 (k=3, dil=2)
  → AdaptiveAvgPool1d(1) → FC 64→32 → FC 32→1 → sigmoid p_slip
```

| 项 | 值 |
|----|-----|
| 架构 | TCN（备选 GRU，同脚本 `--arch`） |
| 参数量 | **19521** |
| 损失 | BCE + `pos_weight`（train 负/正比） |
| 优化 | Adam lr=1e-3，batch 64，seed **42** |
| 训练 | early-stop 监控 val F1；本 ckpt best_epoch=1，best val F1@τ=0.5 = **0.880** |
| Checkpoint | `models/slip_nn/slip_tcn_v1.pt` |

### 3.2 默认部署栈（写入 `train_meta.json`）

| 项 | 值 | 说明 |
|----|-----|------|
| 教师 | `y_event` | 事件前瞻，非规则蒸馏 |
| `drop_leak_features` | true | 推理时清零泄漏维：**17** `slip_rule_s2`、**24** `phase_extend`、**25** `friction_scale` |
| `deploy_latch` | true | 触发后保持 `slip_active`（握力侧） |
| `confirm_steps` | 15 | 连续确认后才计 raw fire / 触发 |
| `default_threshold` τ | **0.7** | 闭环默认 |

握力响应与规则一致：`GripBoostController`，每步 +0.015 rad，上限 0.25 rad，仅 extend 段。

### 3.3 推理延迟（CPU，本机点测）

零特征输入、环形缓冲已预热，500 次 `update`：

| 统计 | ms |
|------|-----|
| mean | **0.176** |
| p50 | 0.174 |
| p95 | 0.188 |
| max | 0.213 |

远低于 **&lt;2 ms** 门闩。

---

## 4. 离线评估

### 4.1 默认模型（`y_event`）

训练期 best（τ=0.5 扫 F1）：val F1 **0.880**（P 0.95 / R 0.82）。

部署阈值 τ=**0.7** 下 offline val（`eval_val.json`）：

| Split | τ | Precision | Recall | F1 | pos_rate |
|-------|---|-----------|--------|-----|----------|
| val | 0.7 | 0.970 | 0.618 | **0.755** | 0.313 |

说明：提高 τ 换更低误报、牺牲 recall；与闭环「少误触发」目标一致。  
（`eval_test.json` 仍为早期 `y_fused`@0.5 残留，**不以该文件作为当前默认模型指标**。）

### 4.2 消融：`y_scheme2`（`models/slip_nn/ablate_s2/`）

| 项 | 值 |
|----|-----|
| best val F1 @0.5 | **0.997** |
| 闭环 baseline 误触发 | ~190/200（τ 0.5–0.9） |

离线极高 F1 **不能**转化为 baseline 低误报——标签本身在无真实滑落时仍大量为正。

---

## 5. 闭环实验

协议：`scripts/eval_slip_nn_closedloop.py` / `run_ketchup_robustness_sweep.py --antislip-nn`  
扩展段：2 s，腕 tz 目标 +10 cm；判定与方案报告一致（Δz≥6 cm 等）。

### 5.1 默认配置最终结果

来源：`data/slip_nn/closedloop_smoke.json`（τ=0.7，完整部署栈）

| Case | status | extend Δz | 接触 | `nn_slip_events` | max grip |
|------|--------|-----------|------|------------------|----------|
| baseline | **pass** | **+9.38 cm** | 200/200 | **93** | 0.25 |
| friction_div2 | **pass** | **+8.71 cm** | 200/200 | 150 | 0.25 |

门闩：`baseline_false_trigger_ok=true`，`friction_div2_gate_ok=true`。

与规则方案二对照（摘自方案报告）：

| 工况 | 开环 | 方案二 | NN-1 |
|------|------|--------|------|
| friction÷2 Δz | −20.7 cm | +8.7 cm | **+8.7 cm** |
| friction÷2 接触 | 99/200 | 200/200 | **200/200** |
| baseline 误触发 | — | ~195/200 | **93/200** |

### 5.2 教师消融（闭环，τ 扫描摘要）

**`y_fused`**（无有效 τ）：baseline 一律 **200/200** 触发。  
**`y_scheme2`**：baseline **190–193/200**，均未过 &lt;100 门闩。  
**`y_event`** 早期 τ 扫（`tau_sweep_y_event.json`，部署栈未完全对齐最终 confirm）：baseline `nn_slip` 约 117–123，仍偏高；加入 **confirm_steps=15 + leak drop + latch** 后降至 **93**。

| 配置 | baseline nn_slip | friction÷2 Δz | 过门闩 B？ |
|------|------------------|---------------|-----------|
| 规则 / 蒸馏规则教师 | ~190–200 | ~+8.7 cm | 否 |
| `y_event` + 完整部署 | **93** | **+8.7 cm** | **是** |

### 5.3 可视化

| 类型 | 路径 / 链接 |
|------|-------------|
| 误触发对比图 | [`figs/nn1_baseline_false_triggers.png`](../data/slip_nn/figs/nn1_baseline_false_triggers.png) |
| 闭环指标图 | [`figs/nn1_closedloop_metrics.png`](../data/slip_nn/figs/nn1_closedloop_metrics.png) |
| τ sweep 图 | [`figs/nn1_tau_sweep_y_event.png`](../data/slip_nn/figs/nn1_tau_sweep_y_event.png) |
| 离线指标图 | [`figs/nn1_offline_val_metrics.png`](../data/slip_nn/figs/nn1_offline_val_metrics.png) |
| **开环 FAIL vs NN PASS** | [friction_div2_openloop_vs_nn.mp4](https://github.com/sumriver/dexterous-slip-detection/blob/cursor/nn1-tcn-impl-f222/data/slip_nn/videos/friction_div2_openloop_vs_nn.mp4) |
| NN baseline / ÷2 | [`videos/nn_baseline.mp4`](../data/slip_nn/videos/nn_baseline.mp4)、[`nn_friction_div2.mp4`](../data/slip_nn/videos/nn_friction_div2.mp4) |

直链下载对比片：  
https://raw.githubusercontent.com/sumriver/dexterous-slip-detection/cursor/nn1-tcn-impl-f222/data/slip_nn/videos/friction_div2_openloop_vs_nn.mp4

---

## 6. 复现命令

```bash
# 数据（含 y_event）
python3 scripts/export_slip_dataset.py
python3 scripts/validate_nn0_dataset.py --align

# 训练默认模型
python3 scripts/train_slip_tcn.py \
  --label y_event --drop-leak-features --deploy-latch --confirm-steps 15

# 离线 / 闭环
python3 scripts/eval_slip_nn_offline.py --split val
python3 scripts/eval_slip_nn_closedloop.py

# 图表与演示视频
python3 scripts/plot_nn1_results.py
python3 scripts/render_nn1_demo_videos.py

# 教师消融（对照）
python3 scripts/train_slip_tcn.py --label y_scheme2 --out models/slip_nn/ablate_s2
```

单用例视频：

```bash
python3 scripts/run_ketchup_robustness_sweep.py \
  --case friction_div2 --antislip-nn --video
```

---

## 7. 讨论与局限

1. **标签与门闩对齐**：规则蒸馏的高离线 F1 与高误触发可并存；事件标签更贴近「要不要加握力」的物理后果。  
2. **泄漏特征**：训练窗含规则输出 / phase / μ；部署必须清零，否则网络可抄规则。  
3. **confirm 滞后**：15 步确认降低误报，但可能推迟真实滑落初期的响应；本任务 friction÷2 仍达标。  
4. **离线 F1 软门槛**：对 `y_event` 未达 0.90；若产品侧要求离线指标，需另扫 τ / 损失 / 窗长，或单独报告 event 口径下的目标。  
5. **工况范围**：与规则相同，本阶段主验 baseline + friction÷2；÷4/÷8 规则亦失败，未作为 NN-1 必过项。  
6. **泛化**：单任务 ketchup；跨物体 / 真机属 NN-3/NN-4。

---

## 8. 结论与下一步

| 结论 | 状态 |
|------|------|
| NN-1 默认配置闭环门闩 A/B | **通过** |
| 相对规则：÷2 持平，baseline 误触发约减半（93 vs ~191） | **达成** |
| 默认教师由 `y_fused` 修订为 `y_event` | **已写入规格与 train_meta** |

**建议下一步（非本报告范围）**：

- NN-2：多任务 / Δgrip，或更强时序（仍用事件标签）。  
- 刷新 `eval_test.json`（`y_event` 口径）并固化 latency 进 CI。  
- 合并 PR #8 后把演示视频链到 `main`。

---

## 附录 A：关键文件

| 路径 | 说明 |
|------|------|
| `models/slip_nn/slip_tcn_v1.pt` | 默认权重 |
| `models/slip_nn/train_meta.json` | 教师、τ、部署开关、训练曲线 |
| `models/slip_nn/eval_val.json` | 离线 val @ τ=0.7 |
| `data/slip_nn/closedloop_smoke.json` | 闭环门闩结果 |
| `data/slip_nn/tau_sweep_y_*.json` | 教师 × τ 扫表 |
| `src/sim/slip_nn_{model,detector,data}.py` | 模型 / 推理 / 数据 |
| `scripts/train_slip_tcn.py` | 训练 |
| `scripts/eval_slip_nn_{offline,closedloop}.py` | 评估 |

## 附录 B：验收清单勾选

相对设计书 §12 / 规格 §6：

- [x] friction÷2 闭环：Δz ≥ 6 cm，接触 200/200（实测 +8.7 cm）  
- [x] baseline 闭环：PASS，误触发 &lt; 100/200（93）  
- [x] 参数量 &lt; 50K；推理 &lt; 2 ms  
- [x] `y_scheme2` / `y_fused` 消融有表  
- [ ] Offline F1≥0.90（`y_fused` 口径）— **改事件标签后作软门槛，当前 0.76 @τ=0.7**  
- [x] 可视化图表与对比视频  

---

*本报告由 2026-07-17 训练与闭环结果整理；数值以仓库内 JSON 为准。*
