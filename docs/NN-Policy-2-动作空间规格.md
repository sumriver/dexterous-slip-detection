# NN-Policy-2 动作空间规格（一页）

**文档编号**：DS-SLIP-NN-POL-2-ACT  
**版本**：v0.1  
**日期**：2026-07-22  
**前置**：[NN-Policy-1 实验报告](./NN-Policy-1-实验报告.md)（1D `grip_extra` 门闩已过；μ×0.40 包络外）  
**关联**：[Policy-1 实现规格](./NN-Policy-1-实现规格.md) · [设计书](./方案一二神经网络防滑检测设计.md) · ketchup `scene.xml` actuators  

---

## 1. 动机

Policy-1 动作是 **1 维**：全体手指 `+grip_extra ∈ [0, 0.25]`，臂/腕跟演示轨迹。  
μ×0.45 可救、μ×0.40 上 NN-2/Policy 顶满仍掉 → **可选项不足**，不是策略头容量问题。

本规格只定 **动作空间与教师协议**；网络/RL 训练另开实现页。

---

## 2. 目标（仿真 only）

1. 把策略从「剂量调节」升级为 **握力 + 腕姿残差**（仍单手）。  
2. 在 **不降低** Policy-1 门闩（baseline 省力、÷2 ≥ +6 cm）前提下，证明 **μ×0.40 出现可救轨迹**（至少开环/搜索教师 PASS）。  
3. 动作可解释、有界、可复现；默认 **先脚本搜索教师 → BC**，不先上端到端 RL。

**本阶段不做**：双手协调、分指独立策略、整臂自由运动（取消演示）、taxel 策略（NN-3）、真机。

---

## 3. 执行器地图（现状）

`data/spider_ketchup_right/scene.xml`：`nu=18`

| idx | 名称 | 角色 | Policy-1 | **Policy-2** |
|-----|------|------|----------|--------------|
| 0–2 | forearm tx/ty/tz | 臂平移 | 轨迹 + tz lift | **默认冻结残差**（可选 tz 微调 ablation） |
| 3–5 | forearm roll/pitch/yaw | **腕旋转** | 纯轨迹 | **残差使能（核心）** |
| 6–17 | fingers | 握持 | 统一 `+grip_extra` | **保留统一 grip**（分指 = 后续） |

注释：轨迹仍提供名义 `ctrl_ref`；策略只加 **有界残差**，避免推翻 SPIDER 演示抓取。

---

## 4. 动作定义

### 4.1 默认动作 `a_t`（P2-A）

\[
a_t = \big(g^\star,\; \Delta r,\; \Delta p,\; \Delta y\big)
\]

| 分量 | 含义 | 范围（初值，写入 meta 可调） | 执行 |
|------|------|------------------------------|------|
| `g*` | 目标握力偏置 | `[0, g_max]`，默认 `g_max=0.25`（扫表可至 0.35） | `grip_extra ← max(grip_extra, g*)`（ratchet） |
| `Δr,Δp,Δy` | 相对 **当步轨迹腕角** 的残差 | 各 `[-Δ_max, Δ_max]`，默认 **`Δ_max=0.25 rad`** | `ctrl[3:6] ← clip(ctrl_ref[3:6] + Δ, ctrlrange)` |

速率限制（安全）：

- `|Δgrip|/step ≤ δ_g`（默认 0.02）  
- `|Δwrist|/step ≤ δ_w`（默认 0.02 rad）  
- 仅 **extend**（与 P1 同）；轨迹段使能为 ablation  

语义目标（教师侧优先，不硬编码进网络）：**增大支撑有利性**（例如瓶轴更水平 / 接触法向更抗重力），不是任意挥腕。

### 4.2 明确降级 / 升级

| 代号 | 动作 | 用途 |
|------|------|------|
| P2-0 | `(g*)` only | 回归 Policy-1 |
| **P2-A** | `(g*, Δr, Δp, Δy)` | **默认** |
| P2-B | P2-A + 有界 `Δtz` | 微调提升高度 |
| P2-C | 分指 `g_i` | 后置；需新教师 |
| P2-D | 双手 | **另立项目**，不在本页 |

---

## 5. 观测（策略输入）

与 P1 对齐，禁止泄漏：

| 入 | 出 |
|----|----|
| NN-0 窗 `X`（漏维清零）+ `p_slip` + `grip_extra` | 可 |
| 当前腕残差 `Δwrist`、可选物体四元数/轴角摘要（**仿真可算、部署需有对应估计**） | 仿真教师与训练可用；部署门闩外做 ablation |
| `friction_scale` / `slip_rule_s2` / case-id | **禁** |

检测骨干默认 **冻结**（NN-2 / Policy-1 检测头）。

---

## 6. 教师协议（先于网络）

**阶段 0 — 可解性证明（必须先做）**

在 μ∈{0.50, 0.45, 0.40}（+ 既有 mass 抽检）上，对 extend 段做有界搜索：

1. 固定演示轨迹；决策变量：分段常数或少量 knot 的 `(g*, Δr, Δp, Δy)`。  
2. 方法：网格 / CEM / 随机射击（实现任选，写入 meta）。  
3. PASS 定义同现门闩：Δz≥6 cm（教师可要求 ≥7 cm margin）、接触 200/200、掉落≤3 cm。  
4. **成功判据**：μ×0.40 至少 **1 条** 搜索轨迹 PASS → 包络被动作扩维打开。  
5. **失败判据**：`g_max=0.35` 且 `Δ_max=0.5 rad` 仍无解 → 需改任务物理/轨迹，而不是训网络。

**阶段 1 — 行为克隆**：成功轨迹 → 窗标签 `y_g, y_Δr, y_Δp, y_Δy` → 训策略头。  
**阶段 2（可选）**：BC warm-start 后短程 RL（奖励：PASS − λ·握力 − μ·腕动作幅度）。

---

## 7. 验收

| # | 检查 | 门槛 |
|---|------|------|
| 1 | 可解性 | μ×0.40 搜索教师 **≥1 PASS**（P2-A 动作内） |
| 2 | 回归 | baseline / ÷2 不低于 Policy-1 终版门闩 |
| 3 | 省力 | baseline 峰值 `g*` **≤ Policy-1**（或同级且腕动作幅度有报告） |
| 4 | 消融 | P2-0 vs P2-A 同工况表（证明腕残差贡献） |
| 5 | 安全 | 残差限幅 + 速率限制始终生效；无爆炸 ctrl |

---

## 8. 文件清单（草案）

| 路径 | 职责 |
|------|------|
| `docs/NN-Policy-2-动作空间规格.md` | 本页（SSOT） |
| `src/sim/antislip_control.py` | `WristResidualController` / 合并 apply |
| `src/sim/spider_replay.py` | extend 段应用 `Δwrist` |
| `scripts/search_policy2_teacher.py` | 阶段 0 可解性搜索 |
| `scripts/export_policy2_teacher.py` | 成功轨 → 窗标签 |
| `src/sim/slip_nn_policy2.py` | 多维头（实现阶段） |
| `docs/NN-Policy-2-实验报告.md` | 搜索 + BC 结果 |

---

## 9. 实施顺序

1. 实现腕残差执行器 + 限幅（P2-0 行为不变）。  
2. `search_policy2_teacher.py`：先打 μ×0.40 可解性。  
3. 不可解 → 调 `Δ_max`/`g_max` 或任务物理；**不进入训练**。  
4. 可解 → 导出教师 → BC（P2-A）→ 闭环表 vs P1。  
5. 可选 RL 微调；双手单独立项。

**风险**：腕残差过大导致失抓或自碰 → 硬限幅 + 接触跌落即失败；搜索目标加「接触连续性」约束。

---

## 10. 阶段 0 实测（2026-07-22）

脚本：`scripts/search_policy2_teacher.py`（seed + random + CEM），`--expand` 对 s040 试到 `g_max=0.35`、`Δ_max=0.5`。

| case | solvable | 摘要 |
|------|----------|------|
| ÷2 / s045 | **yes** | 开环已大量 PASS；已导出 `data/slip_nn_policy2/` 窗数据 |
| **s040** | **no** | 218 trials 无 PASS → **包络未打开**；按 §6 失败判据暂停对 s040 的 BC/RL |

产物：`data/slip_nn_policy2/search/search_summary.json`。

### 训练 / 闭环（P2-A）

```bash
python3 scripts/train_slip_policy2.py
python3 scripts/eval_slip_policy2_closedloop.py
```

- 检测 norm：**沿用 NN-2 backbone**（勿在 PASS-hit 上重算）。
- 部署默认 `wrist_scale=0.5`（全量手腕在 s045 闭环有害）。
- 闭环门：baseline / ÷2 / s045 PASS；s040 仍 FAIL（包络外）。
- 产物：`models/slip_nn_policy2/`，`data/slip_nn_policy2/closedloop_policy2.json`。

---

*一页规格 · NN-Policy-2 动作空间 · dexterous-slip-detection · 2026-07-22*
