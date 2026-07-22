# NN-Policy-1 实验报告：最小充分握力策略头

**文档编号**：DS-SLIP-NN-POL-1-RPT  
**日期**：2026-07-22  
**项目**：dexterous-slip-detection  
**分支**：`main`（PR #11 / #12 合入后）  
**PR**：https://github.com/sumriver/dexterous-slip-detection/pull/11 · https://github.com/sumriver/dexterous-slip-detection/pull/12（已合并）  
**规格**：[`NN-Policy-1-实现规格.md`](./NN-Policy-1-实现规格.md)  
**前置**：NN-1 / NN-2 闭环门闩已过（v2：baseline `nn_slip` **43/200**，friction÷2 **+8.7 cm**）  
**对照**：[`NN-1-实验报告.md`](./NN-1-实验报告.md) · [`NN-2-实现规格.md`](./NN-2-实现规格.md)

---

## 摘要

本报告整理 **NN-Policy-1**：在冻结的 NN-2 检测骨干上增加 **握力策略头**，用仿真可算的 **最小充分握力** `y_policy` 作教师，以 `policy=replace` 替代 NN-2 的 Δgrip 启发式，在 ketchup / XHAND 上验收闭环。

| 工况 | NN-2（`policy=off`） | **Policy-1（`replace`，终版）** |
|------|----------------------|----------------------------------|
| baseline（μ×1） | PASS · events **43** · max_grip **0.241** | **PASS** · events **12** · max_grip **0.130** |
| friction÷2（μ×0.5） | PASS · **+8.7 cm** · 200/200 | **PASS** · **+8.8 cm** · 200/200 |
| friction_s045（μ×0.45） | PASS · +11.2 cm · grip 0.245 | **PASS** · **+10.8 cm** · grip **0.177** |
| friction_s040（μ×0.40） | FAIL（掉落） | FAIL（掉落；与 NN-2 同类） |

**核心结论**：

1. **门闩达标**：÷2 不低于 NN-2；baseline 峰值握力与触发步数均优于 NN-2。  
2. **主因是数据构成，不是网络宽度**：旧网格把 `friction_div2` 整族放 test，train 几乎只有低 `G*`；仅加宽策略头（tier A）无法救 ÷2。  
3. **剂量网格 + 划分修正有效**：μ 邻域 / mass×μ 交叉进 train 后 ÷2 过门；再把 `friction_s045` 从 val 挪进 train 后，s045 由 FAIL→PASS。  
4. **s040 属包络外**：NN-2 与 Policy 均无法在 μ×0.40、`grip_max=0.25` 下救回，不计入策略回归。

---

## 1. 实验目标与验收门闩

### 1.1 目标

- 冻住检测器，学习 **标量目标握力** `ĝ ∈ [0, 0.25]`（ratchet 执行）。  
- friction÷2 **不低于** NN-2。  
- baseline **更省力**（峰值 grip 或加力步数优于 NN-2）。  
- 不引入 taxel / 真机（NN-3 / NN-4）。

### 1.2 验收（规格 §7）

| # | 检查 | 门槛 | 结果 |
|---|------|------|------|
| 1 | friction÷2 | Δz ≥ 6 cm，接触 ≥ 200/200 | **PASS**：+8.78 cm，200/200 |
| 2 | baseline 加力 | 峰值 grip **&lt; NN-2** 或步数更少 | **PASS**：0.130 vs 0.241；events 12 vs 43 |
| 3 | baseline 检测 | `nn_slip_events` **≤ 43** | **PASS**：12 |
| 4 | 延迟 | 检测+策略 CPU mean **&lt; 2 ms** | **PASS**：约 **0.22 ms**/step（本机 smoke） |
| 5 | 教师消融 | `y_policy` vs `y_grip` 各一表 | **未做**（见 §7 后续） |
| 6 | ÷4 | 有开环/闭环对比即可 | **有表**：÷4/÷8 两边均 FAIL（§5） |

主表：`models/slip_nn_policy1/closedloop_s045_fix.json`。

---

## 2. 数据与教师

### 2.1 特征

与 NN-0/NN-2 相同：窗 `(T=40, D=26)`；推理时清零泄漏维（含 multitask 的 `grip_extra` 通道进骨干前归零；策略头另取 **原始** `grip_extra` 作条件）。

### 2.2 教师 `y_policy`（最小充分握力）

对每个导出 case：

1. NN-2 闭环记 `y_grip(t)`。  
2. 若 PASS（margin：Δz≥7 cm、接触满、掉落≤3 cm），二分搜索最小 `grip_max = G*` 仍 PASS。  
3. `y_policy(t) = min(y_grip(t), G*)`。  
4. 失败轨：`fail_keep_grip`（保留原握力序列；÷4/÷8 等无成功 `min_cap`）。

脚本：`scripts/export_min_grip_teacher.py`。

### 2.3 旧网格为何训出「欠剂量」

| 现象 | 数值（补数前） |
|------|----------------|
| 窗质量 `g* < 0.08` | ~**74%** |
| train `y_policy ≥ 0.13` | ~**5%** |
| `friction_div2` 划分 | **整族进 test** → 成功 ÷2 `min_cap` 几乎不进 train |
| 条件偏差（val `y≥0.13`） | pred≈0.056 vs y≈0.22（bias ≈ −0.16） |
| 闭环峰值（replace） | ≈**0.07** ≪ ÷2 所需 ~0.13 → **÷2 FAIL** |

整体 val MAE≈0.0065 **好看但误导**：被大量近零标签淹没。

### 2.4 剂量网格（PR #12）

新增 **新 case 名**（可进 train，不与 test 的 `friction_div2` 撞名）：

| 类型 | case |
|------|------|
| μ 邻域 | `friction_s070/060/055/045/040`（μ×0.70…0.40） |
| 交叉 | `mass_x{2,4}_friction_{div2,s060}` |
| Policy 导出跳过 | `÷4/÷8`、`mass×{8,16,32}`（fail-heavy / 低 `G*`） |

Policy 划分（终版）：

| split | bases |
|-------|-------|
| train | baseline、mass×2/×4、s070/060/045/040、交叉（除 val）等 |
| val | **`friction_s055`**、`mass_x4_friction_div2` |
| test | **`friction_div2`**（OOD 门闩） |

**s045 修正**：初版曾把 `friction_s045` 整族放 val → 成功 `g*≈0.17–0.20` 无训练监督；挪入 train 后 s045 闭环恢复。

### 2.5 终版数据规模

| 项 | 值 |
|----|-----|
| 总量 | 23972 窗（train **18440** / val 3688 / test 1844） |
| train `y≥0.13` | **~13.2%**（旧 ~5%） |
| train mean `y_policy` | ~0.042 |
| 摘要 | `data/slip_nn_policy/export_summary.json` |

重划分工具：`scripts/resplit_policy_shards.py`（改 val/test 时不必重跑教师）。

---

## 3. 模型与训练

### 3.1 结构

```
X → [冻] SlipTCN-multi → h, p_slip, g_ref(NN-2)
  → policy: concat(h, p_slip, grip_extra)  # 34-D
       → Linear(64)→LN→ReLU → Linear(64)→LN→ReLU → sigmoid·max_grip
```

| 项 | 值 |
|----|-----|
| 骨干 | `models/slip_nn_v2/slip_tcn_v1.pt`（冻结） |
| 策略可训参数 | **6721**（tier A；&lt;20K） |
| 损失 | `MSE(ĝ, y_policy) + λ_sparse·mean(ĝ)`，`λ_sparse=0.05` |
| 优化 | Adam 1e-3，batch 64，seed 42，early-stop on val MAE |
| 终版 ckpt | `models/slip_nn_policy1/slip_policy_v1.pt`（best val MAE ≈ **0.026**） |

代码：`src/sim/slip_nn_policy.py`、`scripts/train_slip_policy.py`。

### 3.2 部署

| 项 | 值 |
|----|-----|
| 模式 | **`replace`**（握力听 `policy_grip`） |
| 检测 | τ=**0.99**，confirm=**30**，latch，soft=**0.7**（继承 NN-2） |
| 执行 | `set_grip` ratchet，`g∈[0,0.25]`，仅 extend |
| 加载 | `load_detector_from_dir(models/slip_nn_policy1)` → `arch=detect_and_policy` |

评测：`scripts/eval_slip_policy_closedloop.py`。

### 3.3 弯路：只改架构无效

| 版本 | 可训参数 | val MAE | ÷2 闭环 |
|------|----------|---------|---------|
| tiny `34→32→1` | 1153 | ≈0.0069 | （未作为主部署） |
| tier A `34→64→64→1`+LN（旧数据） | 6721 | ≈0.0065 | **FAIL**（峰值 grip≈0.07） |
| tier A + 剂量网格 + s045∈train | 6721 | ≈0.026 | **PASS** |

→ 闭环改善来自 **监督分布**，不是加宽 MLP。val MAE 升高因 val 含更多高剂量窗，属预期。

---

## 4. 闭环主结果（终版）

来源：`closedloop_s045_fix.json`（Policy vs NN-2 同脚本、同门闩）。

| case | μ× | NN-2 status | Policy status | NN-2 Δz | Policy Δz | NN-2 grip | Policy grip | events NN-2→P1 |
|------|-----|-------------|---------------|---------|-----------|-----------|-------------|----------------|
| baseline | 1.00 | pass | **pass** | +7.1 | +7.9 | 0.241 | **0.130** | 43→**12** |
| friction_div2 | 0.50 | pass | **pass** | +8.7 | +8.8 | 0.249 | 0.175 | 70→5 |
| friction_s045 | 0.45 | pass | **pass** | +11.2 | +10.8 | 0.245 | **0.177** | 59→41 |
| friction_s055 | 0.55 | pass | **pass** | +7.0 | +7.1 | 0.249 | 0.157 | 58→19 |
| friction_s040 | 0.40 | **fail** | **fail** | −20.0 | −20.0 | 0.246 | 0.195 | 0→0 |

---

## 5. 扩展扫表

### 5.1 标准 robustness（mass / ÷4 / ÷8）

来源：`closedloop_full_sweep.json`（剂量网格后、s045 划分修正前；mass/÷4/÷8 结论与终版一致）。

| case | NN-2 | Policy-1 | 备注 |
|------|------|----------|------|
| mass×2…32 | pass | pass | Policy 握力/触发更省 |
| friction÷2 | pass | pass | 与门闩一致 |
| friction÷4 / ÷8 | fail | fail | 两边均不可救 |

### 5.2 μ 邻域（修正前后）

| case | 修正前 Policy | 修正后 Policy | 根因摘要 |
|------|---------------|---------------|----------|
| s070 / s060 / s055 | pass | pass | — |
| mass×2/×4 × ÷2 | pass | pass | — |
| **s045** | **fail**（grip≈0.125） | **pass**（grip≈0.177） | val 扣留成功 `min_cap` → 欠剂量 |
| **s040** | fail | fail | NN-2 亦 fail；无成功教师；包络外 |

---

## 6. 失败分析（简）

### 6.1 s045（已修复）

- NN-2 可过；Policy 曾以 ~0.125 峰值掉落，而教师 `G*≈0.17–0.20`。  
- `friction_s045` 曾在 `POLICY_VAL_CASES` → train 无该 μ 的成功监督。  
- 挪入 train 后闭环握力升至 ~0.177，PASS。

### 6.2 s040（不修策略）

- 教师 4 variant 全为 `fail_keep_grip`，无成功 `min_cap`。  
- NN-2 满握力仍掉落。  
- 结论：超出 `grip_max=0.25` + 现控制栈可救范围；扩展包络或承认 OOD，而非再堆 fail_keep 高标签。

### 6.3 总体 MAE 陷阱

应用 **条件 MAE / `y≥0.13` 占比 / 闭环峰值 grip** 验收策略头；禁止只看 overall MAE。

---

## 7. 未完成与后续

| 项 | 状态 | 建议 |
|----|------|------|
| `docs/NN-Policy-1-实验报告.md` | **本文件** | — |
| 延迟正式写入 `latency.json` | smoke 0.22 ms | 可并入 CI 脚本 |
| `y_policy` vs `y_grip` 教师消融 | 未跑 | 规格 §7.5；应用同网格对比闭环 |
| `policy=residual` | 未系统评 | 可选；replace 已过门闩 |
| s040 | 包络外 | 不作为 Policy-1 主指标 |
| NN-3 / NN-4 | 未开始 | taxel / ONNX·真机（设计书下一阶段） |

---

## 8. 复现命令

```bash
# 教师（耗时）
python3 scripts/export_min_grip_teacher.py

# 改划分后快速重切
python3 scripts/resplit_policy_shards.py

# 训练
python3 scripts/train_slip_policy.py

# 门闩对比
python3 scripts/eval_slip_policy_closedloop.py

# NN-0 剂量网格（检测数据，可选）
python3 scripts/export_slip_dataset.py --antislip
```

产物：`models/slip_nn_policy1/`、`data/slip_nn_policy/closedloop_*.json`。

---

## 9. 结论

1. **NN-Policy-1 在 ketchup 仿真上达到规格主门闩**：÷2 不低于 NN-2，baseline 更省握力与触发。  
2. **成功关键是教师工况分布与划分**，不是策略 MLP 容量。  
3. **s045 证明「val 误扣高剂量 μ」会直接导致闭环回归**；s040 证明不可把包络外 fail 当成策略失败。  
4. 收尾项为教师消融表与 residual 摸底；下一产品阶段按设计书转向 **NN-3 / NN-4**。

---

*NN-Policy-1 实验报告 · dexterous-slip-detection · 2026-07-22*
