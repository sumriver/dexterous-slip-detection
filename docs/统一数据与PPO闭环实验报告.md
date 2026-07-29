# 实验报告：统一数据同训评测 + Policy-2 闭环 PPO 小实验

**文档编号**：DS-SLIP-UNIFIED-RL-RPT  
**日期**：2026-07-28  
**项目**：dexterous-slip-detection  
**任务**：ketchup / XHAND SPIDER 抬升防滑（`s01-ketchup_use_01`）  
**相关 PR**：
- #19 统一数据集 + 同训重训  
- #20 评测口径清理 / 硬 case / 数据质量  
- #21 extend-only PPO smoke  
- #22 门控 PPO 接入 `replay_spider_task`（Step1–3）  

**门闩（PASS）**：extend 抬升 Δz ≥ **6 cm** 且接触步数 ≥ **200**（2 s @ 0.01 s）。

---

## 摘要

本阶段做了三件事：

1. **统一训练分布**：把 `slip_nn` / `policy` / `policy2` / `heavy` 按 schema GCD 合成一份数据，在同数据上重训并只排 **NN-2 / grip-only / grip+wrist**。  
2. **硬 case 诊断**：`friction_div2` 有开环 teacher 但闭环常挂；`friction_s040` 开环常值搜索 **0 hits**（不可解）。  
3. **RL 小实验**：extend 段 PPO → 门控接入完整 `replay_spider_task`，与 BC 同场对比。

**核心结论**：

| 命题 | 结论 |
|------|------|
| 同训下显式 policy vs NN-2 | policy（grip / wrist）frontier **5/7**，NN-2 **1/7** |
| grip-only vs grip+wrist | 同训下 PASS 翻转 **0**；腕部不必然更好 |
| 跨数据混排 | **禁止**作为算法排名（只能当 transfer） |
| BC 在 μ×0.45 / 0.40 | 仍大量 FAIL；s040 无开环 teacher |
| PPO（80k，extend）闭环 | `always` **3/4**（含 s045）；`on_detect` **2/4**；s040 仍 FAIL |

---

## 1. 数据

### 1.1 仿真任务

- **一条** SPIDER 轨迹：`arcticv2` / XHAND right / `s01-ketchup_use_01`  
- 评测「案例」= 同一轨迹上改 **质量 × 摩擦**（及 g-cap）  
- 开环摸底格 **9** 条；统一评测表 **11** 条；本次 PPO 对比 **4** 条硬/参考格  

### 1.2 源数据集（合并前）

| 源 | 角色 | 约略规模 | 关键标签 |
|----|------|----------|----------|
| `slip_nn` | 检测 + 附带握力 | ~8.3k 窗 | `y_event`, `y_grip` |
| `slip_nn_policy` | Policy-1 min-grip 教师 | ~6.0k | `y_policy` |
| `slip_nn_policy2` | P2 开环 PASS hits | 2.0k（1 窗/hit） | `y_grip_p2`, `y_wr/wp/wy` |
| `slip_nn_policy2_heavy` | 重物 + g-cap | 1.5k | 同上 |

特征公约数：**26 维 × 40 步**。缺失 wrist / slip → 填 0。

### 1.3 统一集 `data/slip_nn_unified`（质量过滤后）

构建脚本：`scripts/build_unified_slip_dataset.py`

| 步骤 | 效果 |
|------|------|
| schema 并集 | 共享标签键 |
| 去弱教师 | 同 `base_case` 保留 policy2/heavy > policy > slip_nn（丢 8759） |
| 配额 | 每 `(source, case)` ≤ 500 |
| 难例 boost | `friction_div2` ×3 |
| 分层划分 | `source\|base_case`（去掉 `_hitNNNN`） |

**规模**：train **7229** / val **901** / test **902**（合计 9032；wrist 教师约占 **39%**）

| 源（过滤后） | 窗数 |
|--------------|------|
| slip_nn | 2305 |
| slip_nn_policy | 3227 |
| slip_nn_policy2 | 2000 |
| slip_nn_policy2_heavy | 1500 |

难例计数（boost 后）：div2 **1500** · s045 **500** · s040 **461**（无 wrist teacher，来自较弱源）

### 1.4 硬 case 教师现状

| Case | 开环 teacher | 说明 |
|------|--------------|------|
| `friction_div2`（μ×0.5） | 有（≥1000 hits） | 开环可解；BC 闭环不稳定 |
| `friction_s045`（μ×0.45） | 有 | 搜索可解 |
| `friction_s040`（μ×0.40） | **无** | 激进搜索 306 次（g≤0.45,d≤0.6）仍 **0 hits**，best Δz≈−0.37 cm |

产物：`data/slip_eval/hard_cases_status.json`，`s040_aggressive_search.json`

---

## 2. 算法

### 2.1 同训三类（算法对比口径）

| 名称 | 结构 | 闭环用法 |
|------|------|----------|
| **NN-2** | multitask TCN：`y_event` + grip | `policy_mode=off`（检测驱动握力） |
| **grip-only** | 冻结 NN-2 + grip 头（`train_slip_policy`） | `policy_mode=replace` |
| **grip+wrist** | 冻结 NN-2 + grip+wrist（`train_slip_policy2`，`--mask-zero-wrist`） | `policy_mode=p2a` |

**不再混排**：旧「P1 vs P2-grip」在同数据上只是同方法不同种子 → 仅作方差，不进排名。

动作空间（Policy-2）：`(grip, Δroll, Δpitch, Δyaw)`，经 `Policy2OpenLoopController` 限速执行。

### 2.2 RL（PPO smoke）

| 项 | 设定 |
|----|------|
| 环境 | `KetchupExtendRLEnv`：仅 extend 200 步 |
| 算法 | SB3 PPO，80k steps，2 envs |
| Obs | 26 维特征 + grip/wrist/t/μ/m（33 维） |
| Action | Box[-1,1]^4 → grip∈[0,0.25]，wrist∈[-0.25,0.25] |
| 未训 | 轨迹回放、NN detect backbone |

闭环接入（门控，默认 off）：

| mode | 何时问 PPO | 用途 |
|------|------------|------|
| `off` | 从不 | 默认 |
| `always` | 每步 | 接近 Gym 训练 |
| `on_detect` | soft/hard 滑移 | 接近 BC 部署触发 |
| `inspect` | 每步记录 | 人工检查 |

`apply_actions=False` 时只提议不下手。`on_detect` 用 NN-2 **detect-only**（无 GripBoost 兜底，避免污染对比）。

---

## 3. 结果

### 3.1 同训统一评测（11 格）

产物：`data/slip_eval/unified_suite_latest.json`  
排名规则：只排 NN-2 / grip / wrist；envelope（s040）不进 frontier 排名。

| Rank | 模型 | Frontier | Econ mean grip | Envelope |
|------|------|----------|----------------|----------|
| 1 | grip+wrist | **5/7** | 0.248 | 0/1 |
| 2 | grip-only | **5/7** | 0.227 | 0/1 |
| 3 | NN-2 | **1/7** | 0.173 | 0/1 |

腕部消融（同 seed）：**helps_pass=0 / hurts_pass=0**（无 PASS 翻转）。

种子方差：grip seed42 vs seed43 frontier 仅约 **3/7** 一致 → 单次 PASS 表噪声大。

**逐格（P/F + Δz_cm / max_grip）**：

```
cell                     nn2        grip       wrist
mass_x2_s045         F-16/0.14  F-16/0.24   F-1/0.25
mass_x4_s045          P+6/0.19   P+6/0.23   P+6/0.25
mass_x2_s042         F-20/0.13  F-16/0.24  F-17/0.22
friction_s045        F-19/0.13  P+11/0.24  P+13/0.25
mass_x4_div2_g012     F-5/0.12   P+7/0.12   P+7/0.12
mass_x2_div2_g010     F-7/0.08   P+7/0.10   P+6/0.10
mass_x8_s045_g015    F-14/0.15   P+7/0.15   P+7/0.15
baseline              P+9/0.15   P+7/0.22   P+9/0.25
friction_div2        F-16/0.18  F-21/0.23  F-15/0.25
mass_x4               P+8/0.19   P+8/0.24   P+9/0.25
friction_s040        F-20/0.17  F-20/0.23  F-20/0.24
```

### 3.2 闭环 BC vs PPO（同场 4 格）

产物：`data/slip_eval/ppo_closedloop_vs_bc.json`  
全部经 `replay_spider_task`（完整轨迹 + extend）。

| case | BC grip | BC wrist | PPO always | PPO on_detect |
|------|---------|----------|------------|---------------|
| friction_div2 | **PASS** +8.7 | FAIL −15.8 | **PASS** +9.1 | **PASS** +8.9（q=146） |
| baseline | **PASS** +7.7 | **PASS** +8.4 | **PASS** +8.6 | **PASS** +8.6 |
| friction_s045 | FAIL −18.2 | FAIL −22.5 | **PASS** +11.9 | FAIL −18.1 |
| friction_s040 | FAIL −20.0 | FAIL −19.8 | FAIL −19.9 | FAIL −0.6 |

**通关数**：bc_grip **2** · bc_wrist **1** · **ppo_always 3** · ppo_on_detect **2**

解读：

1. **PPO always** 最强：每步可动手，与训练分布一致；独特地过了 s045。  
2. **PPO on_detect** 与 BC 触发更可比：能救 div2，但 s045 仍挂（检测时机/动作分布错位）。  
3. **s040** 全员失败；开环亦无 teacher → 不是「再堆 BC」能解。  
4. PPO 平均 max_grip（~0.17–0.18）低于 BC（~0.23–0.25）→ 有握力经济性迹象（样本少，待多种子确认）。

### 3.3 公平性备忘

| 对比 | 是否可归因算法 |
|------|----------------|
| 同 `slip_nn_unified` 的 NN-2 / grip / wrist | **可以** |
| 旧跨数据域混排（P1 旧数据 vs P2-heavy） | **不可以**（仅 transfer） |
| Gym extend-only PPO vs BC 闭环 | **弱可比**（规则不同） |
| `replay` 内 PPO always / on_detect vs BC | **可以**（同考试） |

---

## 4. 复现命令

```bash
# 统一数据
python3 scripts/build_unified_slip_dataset.py --boost-cases friction_div2:3
python3 scripts/retrain_unified_models.py          # 或 --skip-eval 后再单独评
python3 scripts/eval_slip_unified_suite.py

# PPO
python3 scripts/train_policy2_extend_rl.py --timesteps 80000 --n-envs 2
python3 scripts/smoke_extend_hook_step1.py --friction 0.5
python3 scripts/smoke_extend_hook_step2.py --friction 0.5
python3 scripts/eval_ppo_closedloop_vs_bc.py
```

关键路径：

- 数据：`data/slip_nn_unified/`  
- 同训模型：`models/slip_nn_unified_{nn2,grip,wrist}/`  
- PPO：`models/rl/policy2_extend/p2_extend_ppo_final.zip`  
- 检查点说明：`docs/RL-闭环接入检查点.md`

---

## 5. 下一步建议（按优先级）

### P0 — 把结论钉牢
1. **多种子**（≥3）重跑 unified suite 与 PPO 闭环表，报均值±方差（已见 grip 种子 3/7 不一致）。  
2. 固定评测随机性 / 记录 `nn_threshold`、soft/hard 配置进 JSON，避免「同模型不同天不同 PASS」。

### P1 — 扩大 PPO 的可靠增益
3. **加长训练**（≥5e5）+ 提高 s045/s040 采样权重；可选 BC teacher **actor 热启动**。  
4. 专攻 **`on_detect` 分布偏移**：用检测触发时段做 DAgger / 在线微调，使 always 的 s045 增益迁移到部署触发模式。  
5. 保持门控：默认 `off`，对比实验显式开 `always` / `on_detect`。

### P2 — s040 / 动作空间
6. s040 常值 P2-A **已证伪** → 试 **时变 / 分段** 动作（前半握、后半腕）或放宽执行器/运动，再搜教师；仍 0 hits 则标为 **envelope 界外**，不进算法排名。  
7. 评估是否需要超出当前 4D 动作（臂基座微调等）——超出则单独立项，勿与当前 P2-A 混报。

### P3 — 工程收尾
8. 合并 #19–#22 中已稳定部分；CI 加 `smoke_extend_hook_step1/2` + 短步数 PPO 不回归。  
9. 统一评测入口文档：算法对比 **必须** `slip_nn_unified` + 三类头；RL 对比 **必须** `eval_ppo_closedloop_vs_bc.py`。

---

## 6. 一句话总结

**同训下，显式 grip/wrist policy 明显强于 NN-2；腕部相对 grip-only 增益不稳定。**  
**短训 PPO 在「每步可动手」闭环上已超过 BC（含 s045），但「检测触发才动手」仍未全面超越；s040 仍是开环不可解边界。**  
**下一主线：多种子稳结论 + 把 always 的增益迁到 on_detect（DAgger/加长 RL），而不是继续混数据硬比旧模型。**
