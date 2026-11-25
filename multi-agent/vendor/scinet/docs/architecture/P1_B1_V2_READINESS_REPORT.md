# P1-B1 — V2 Pilot 就绪度报告（P1-C / P2-03A 门槛）

**结论先行：P1-A' 五项修复全部完成、V2 pilot 独立验收 10/10 PASS；
但【仍不批准进入 P1-B2 全量数据生成或 P2-03A 训练】，因为任务书明确要求 STOP 等待人工批准，
且 Q4 / Q6 等模型阶段决策尚未关闭。**

---

## 1. Gate 状态

| Gate | 要求 | 状态 |
|------|------|------|
| Gate 0 Git 与历史核对 | 分支关系、历史证据、提交一致性 | ✅ `P1_A_BRANCH_INTEGRITY_REPORT.md` §1–§7 |
| Gate 1 P1-A 科学验收 | 5 个审查问题修复 + 独立复核 | ✅ `P1_A_REVIEW_FIX_REPORT.md` |
| **P1-B1 V2 pilot** | 恢复规格的 20 条 pilot + 独立验收 | ✅ **10/10 PASS**，本报告 |
| **P1-C Data Acceptance** | 全量数据上的独立验收 + G0 STOP GATE | ⛔ **未开始**（全量数据尚未生成） |
| P2-03A Single-step JEPA | E0–E4、5 seeds、G1–G3 | ⛔ 未开始 |
| P2-03B Multi-step JEPA | E5、horizons 1/2/5/10/20 | ⛔ 未开始（须 G1–G3 通过） |
| P2-04 MPC | Learned vs Ground-truth 对照 | ⛔ 未开始 |

---

## 2. 本轮完成度对照任务书

| 任务书要求 | 完成 | 证据 |
|------------|------|------|
| Phase 0 核验 GitHub 状态 | ✅ | `git_integrity.json`、分支报告 §7 |
| P1 修 D-A 物理验收 + 解析单测 | ✅ | Readiness Plan §5.6 AMENDED + `test_action_intervention_analytic.py` |
| P2 加固验证器 + 负向测试 | ✅ | `verify_p1a_reports.py` v2 48/48 + 6 个负向用例 |
| P3 修 config 注释 + 显式键 + 一致性检查 | ✅ | `controlled_pendulum_jepa.yaml`、`check_config_impl_consistency.py` 10/10 |
| P4 加固 simulator 校验 | ✅ | `controlled_pendulum_v2.py` + 17 tests / 25 subtests（含 v1 逐位零回归） |
| P5 分支报告标注历史快照 | ✅ | 分支报告头部声明 + §7 |
| Phase 2 恢复原规格配置 | ✅ | `controlled_pendulum_v2_pilot.yaml` |
| Phase 3 regime 分层 / stress 隔离 | ✅ | 20/0/0 bounded；v1 保留为 eval-only |
| Phase 4 新 20 条 V2 pilot + 完整 manifest | ✅ | `datasets/controlled_pendulum_v2_pilot/` |
| Phase 5 十项独立验收 | ✅ | `audit_report.json` 10/10 |
| 4 份报告 | ✅ | 本文件 + 另 3 份 |
| 不覆盖原 pilot | ✅ | `npz_sha256` / `config_hash` 复核逐字一致 |
| 不改 E0–E5 / G0–G6 | ✅ | `P2_03_ACTION_JEPA.md` 正文零改动 |

---

## 3. 关键实测汇总（全部可追溯）

| 指标 | 值 | 来源 |
|------|----|------|
| 当前 HEAD（核验时） | `3143772cf8ede786254a01ee6d0013a4d4c243d1` | `git_integrity.json` |
| ahead / behind | **11 / 0** | 同上 |
| 相对 main 文件 | **36**（34 A + 2 M + 0 D） | 同上 |
| P2-02 证据文件 | 78 | `git ls-tree` 实读 |
| 数字可追溯性 v2 | **48/48**，全部数值比较 + anchor 行定位 | `number_traceability_v2.json` |
| 配置-实现一致性 | **10/10** | `config_impl_consistency.json` |
| v1 config_hash | `41bb7d66...`（未变） | manifest 复核 |
| V2 `dataset_version` | `cpv2-ff53b15f8996` | v2 manifest |
| V2 `npz_sha256` | `0f48914d...` | v2 manifest / audit |
| V2 DOP853 最大误差 | **1.6515e-05** | `audit_report.json` |
| V2 能量平衡残差 | **7.4512e-04** | 同上 |
| V2 max \|Δθ\|/step | **0.1390 rad** | 同上 |
| V2 初态多样性 | θ0/ω0 各 **20** 个互不相同 | 同上 |
| V2 划分 | **14/3/3**，零泄漏 | 同上 |
| Q-C stress 误差降低 | **×696.8**（7.628e-03 → 1.095e-05） | 同上 |
| 全量回归 | **297 passed + 292 subtests**（零回归） | 本轮本地执行记录 |

---

## 4. 测试执行记录

| 命令 | 结果 |
|------|------|
| `pytest tests/dynamics/test_action_intervention_analytic.py -q` | 5 passed + 8 subtests |
| `pytest tests/dynamics/test_traceability_verifier.py -q` | 11 passed + 96 subtests |
| `pytest tests/dynamics/test_controlled_pendulum_v2.py -q` | 17 passed + 25 subtests |
| `python scripts/verify_p1a_reports.py` | PASS 48/48 |
| `python scripts/check_config_impl_consistency.py` | PASS 10/10 |
| `python scripts/audit_controlled_pendulum.py --config ...`（v1） | 8/8 PASS |
| `python scripts/audit_controlled_pendulum_v2.py` | **10/10 PASS** |
| `pytest tests/ --ignore=tests/test_pysr.py -q` | **297 passed + 292 subtests** |

基线对照：上一轮 `255 passed + 101 subtests` → 本轮 `297 passed + 292 subtests`，
净增 **+42 tests / +191 subtests**，恰好等于本轮新增 4 个测试文件之和
（5+11+17+9 / 8+96+25+62），**零回归**。

> **执行环境声明**：以上均在 Windows 本地 `D:/python313/python.exe`
> （numpy 2.2.6 / scipy 1.16.3 / matplotlib 3.10.8 / torch 2.14.1 / pytest 9.1.1）执行。
> GitHub 本轮**未返回可独立确认的 CI 状态**，因此这些是**本地执行记录**，
> 不是 CI 结果，也不是 ChatGPT 独立复跑的结果。

---

## 5. 后续实现计划（保留既有设计，不擅自开工）

按 ChatGPT 的「分成三个独立开发任务」建议，P1-C 通过后依次进行：

### Task 1 — `dynamics/trajectory_windows.py`
严格复用 `dynamics_batch_from_windows()`，只做薄适配：
`(context_block, action_t, target_block)` 三元组、`[B,K*D]` 展平、
多步 `target_k_block = obs[s+k : s+k+K]`（正好 K 帧）、
边界 `0 <= s <= L-K-k_max`、h=1 与单步 target 逐元素相同、episode 隔离、mask。
**先只做数据窗口层，不训练模型。**

### Task 2 — `models/action_conditioned_world_model.py`
| 组件 | 实现 |
|------|------|
| Online Encoder | 复用 `models/encoder.py:76`（吃 `[B,K*D]`） |
| Conditional Predictor | 复用 `models/conditional_predictor.py:57`，**子模块命名 `action_predictor`**（降低与 legacy `predictor` 的 state_dict 键碰撞风险） |
| Target Encoder | `deepcopy` + `requires_grad_(False)` + EMA |
| VICReg | 复用 `utils/losses.py:83` |
| dispatch | 显式 `predictor_mode ∈ {conditional, unconditioned}` |
| `model_class` | `"ActionConditionedWorldModel"`，新增双向 checkpoint 加载测试以关闭 Q6 |

Q4 建议默认 `latent_dim=8`、保留 16 的配置支持，先做低成本 smoke test。

### Task 3 — E0–E5（**严格保留原规格定义**）
第一轮只跑 E0–E4、5 seeds：latent collapse、Raw-state Linear Probe、
RMSE/MAE、Persistence Skill（显式报告分母）、episode-level paired bootstrap CI。
**只有 G1–G3 达标才进入 E5**（horizons 1/2/5/10/20）。

---

## 6. P1-B2 全量数据 manifest 要求（尚未生成）

按 ChatGPT 建议，全量数据 manifest 必须包含完整实验身份：
`dataset_version` / `config_hash` / `git_commit` / `integrator` / `dt` /
`episode_steps` / `theta_representation` / `torque_bounds` /
`initial_state_distribution` / `parameter_ranges` / `collapse_thresholds` /
`ood_groups`（**四组，不新增第五组**）/ `episodes[]`。
每条 episode 记录 `initial_state`、`physical_parameters`、`action_profile`、
`action_hold_steps`、`seed`、`regime`、`split` 与可验证的数据身份。

V2 pilot 的 manifest 已按此结构实现，可作为全量数据的模板。

---

## 7. 待人工决策 / 未关闭项

| 项 | 状态 |
|----|------|
| **是否批准进入 P1-B2 全量数据生成** | ⏸ **等待人工批准（本轮 STOP）** |
| Q4 `latent_dim` 8 vs 16、`block_size` 10 | OPEN（模型阶段） |
| Q6 `action_predictor` 命名 + `model_class` + 双向加载 | OPEN（模型阶段，Task 2 关闭） |
| 全量数据 dt 是否维持 0.02 | 建议维持（V2 实测 DOP853 误差 1.65e-05），最终由 P1-C 数值证据决定 |
| 四组 Parameter OOD 的范围/计数 | 已在 v2 config 声明，尚未生成 |

---

## 8. STOP

**本轮到此停止。** 未生成全量数据集、未实现 `trajectory_windows.py` 或
`ActionConditionedWorldModel`、未训练 JEPA / E0–E5、未实现 MPC、未合并 main、未开 PR。
保留 E0–E5 与 G0–G6 全部定义，保留 v1 与 v2 两份 pilot 及全部历史证据。
