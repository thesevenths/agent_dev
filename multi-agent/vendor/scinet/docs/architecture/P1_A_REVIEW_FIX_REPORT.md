# P1-A' — 审查问题修复报告 (Review Fix Report)

**结论先行：ChatGPT 在 GitHub 远端源码审查中指出的 5 个问题，经本地独立实证复核，全部成立，已全部修复。**
修复过程未改动任何历史 pilot 数据、未覆盖历史证据、未合并 main、未开 PR、未开始 JEPA 训练。

- 基准分支：`feature/p2-03a-action-jepa`
- 核验时 HEAD：`3143772cf8ede786254a01ee6d0013a4d4c243d1`
- `origin/main`：`e4eccc0b1da2c8a365706f97ded95f3723e0f5e9`（仍是直接祖先）

---

## 1. Phase 0 — GitHub 状态核验（先于一切改动）

| 检查 | 实测 |
|------|------|
| HEAD full SHA | `3143772cf8ede786254a01ee6d0013a4d4c243d1` |
| `git merge-base origin/main HEAD` | `e4eccc0b1da...`（= `origin/main`，main 是 HEAD 的祖先） |
| `git rev-list --left-right --count origin/main...HEAD` | **0** behind / **11** ahead |
| 相对 main 文件变更 | **36** files = 34 A + 2 M + 0 D |
| P2-02 证据文件（`git ls-tree` 实读） | 78 |
| `a5b8396..HEAD` 两个审查提交 | `eb1dc51`（analysis）、`3143772`（docs） |
| 未提交改动 | 无（核验时工作树 clean） |

与 ChatGPT 的 GitHub 实测（11 ahead / 0 behind / 34 added + 2 modified + 0 deleted）**逐项一致**。
机器可读快照：`datasets/controlled_pendulum_pilot/analysis/git_integrity.json`
（由 `scripts/capture_git_integrity.py` 用真实 git 命令采集，记录全部命令与 full SHA）。

---

## 2. 五个问题的独立复核判定

我没有照单全收，而是逐条用命令/代码实证：

| # | 问题 | 独立复核结果 | 判定 |
|---|------|--------------|------|
| 1 | D-A「正/负力矩导致相反方向角加速度」 | 实算 `α=-(g/L)sinθ-bω+τ/(mL²)`：θ0=2 时基项 `-8.920208`；+2.5→`-6.420208`、-2.5→`-11.420208`，**同为负** | ✅ 成立（验收条件错误，非模拟器错误） |
| 2 | 数字可追溯性 28/28 有验证器漏洞 | AST 统计：28 case 中 **9 个 `decimals=None`** 直接 `numeric_ok=True`；`78` 硬编码；四份报告被拼成 `all_text` 单一字符串 | ✅ 成立 |
| 3 | config 头部 seed 注释过期 | config 第 13 行仍写 `master_seed*1000 + ...`；实现为 `sha256(...)[:4]` | ✅ 成立 |
| 4 | 规格符合性审计自身不严格 | config 中 `grep -nE "^(integrator\|theta_representation\|...)"` 返回 **NONE**，但 Readiness Plan §2 把 `integrator` / `theta unwrapped` 标为 ✅；`simulator` 内 `isfinite` 逐步断言缺失 | ✅ 成立 |
| 5 | 分支报告需标注历史快照 | 报告头部仍写 HEAD `a5b8396` / 9 commits / 18 files，而当前为 11 / 36 | ✅ 成立 |

**结论：必须改。** 下面逐项给出修复与证据。

---

## 3. P1 — 修正 D-A Action Intervention 物理验收条件

**修改**：`P2_03A_READINESS_PLAN.md` §5.6 加 AMENDED 段落，写明原条件在 θ0=2 下不成立，
改用 **θ0=0, ω0=0**（重力矩为零，`α=τ/(mL²)`），并要求**固定同一 context block**（同一 episode、
同一起点 `s`），只改变动作序列。一般初态下只要求「差异方向与 `Δα=Δτ/I` 一致」，不要求符号相反。
D-A 保持**附加诊断**，不替代 E0–E5 / G0–G6。

**新增解析级单测**：`tests/dynamics/test_action_intervention_analytic.py`（5 tests + 8 subtests，全过）

| 测试 | 内容 |
|------|------|
| `test_theta0_zero_positive_negative_zero_torque_have_expected_signs` | θ0=0：+τ→α>0、-τ→α<0、τ=0→α≈0；解析值 == `derivatives()` 数值 |
| `test_theta0_zero_alpha_equals_tau_over_inertia` | `α == τ/(mL²)` 逐元素 |
| `test_original_theta0_two_condition_is_indeed_invalid` | **回归护栏**：θ0=2 时两者同为负，且 `Δα = Δτ/I` |
| `test_general_theta_direction_rule_is_delta_alpha_not_absolute_sign` | 10 组随机状态：只有 `Δα=Δτ/I` 恒成立 |
| `test_numeric_derivatives_match_analytic_for_random_states` | 20 组随机状态解析/数值一致 |

实测：`alpha0(θ=2)=-8.920208`，`+2.5 → -6.420208`、`-2.5 → -11.420208`；
`θ=0` 时 `+2.5 → +2.500000`、`-2.5 → -2.500000`、`0 → 0.000000`（nominal I=1）。
ChatGPT 给出的数字**完全正确**。

---

## 4. P2 — 加固数字可追溯性验证器

**重写** `scripts/verify_p1a_reports.py`（v2）。v1 的三个漏洞与修复：

| v1 漏洞 | v2 修复 |
|---------|---------|
| `decimals=None` ⇒ `numeric_ok=True`（9/28 未比较） | **每个 case 必填显式 `tolerance`**，全部执行 `abs(parsed - scale*source) <= tol`；无默认 True |
| `78` 硬编码 | 从 `analysis/git_integrity.json`（真实 git 命令采集）读取 `p2_02_evidence_file_count` |
| 四份报告拼成 `all_text` | **每个 case 绑定唯一 `target_report`**；且新增 **`anchor` 行定位**：字面量必须出现在含该锚点的行上 |

额外加固（由负向测试反逼出来）：只绑定报告还不够 —— `7.628e-03` 在同一报告出现 **3 次**，
若数字被挪到错误位置仍会通过。因此 v2 增加 **per-line anchor**（如 `全局最大`、`| 0.02000 |`、
`| train | **14** |`），把数字钉在它应该出现的具体表格行/句子上。

- case 数：**28 → 48**，每个绑定 (source JSON, JSONPath, target report, anchor, literal, tolerance)
- JSONPath 支持 `.` 与 `[i]`（如 `per_episode[5].theta_final_rad`）
- 解析支持科学计数法、`**粗体**`、`%` 百分数、带符号小数
- 结果：`number traceability v2: PASS (48/48)` → `analysis/number_traceability_v2.json`

**负向测试**：`tests/dynamics/test_traceability_verifier.py`（11 tests + 96 subtests，全过），
其中 6 个负向用例：

| 负向用例 | 篡改方式 | 要求 |
|----------|----------|------|
| `test_negative_scientific_notation` | 把锚点行 `全局最大 7.628e-03` 改掉 | verifier FAIL |
| `test_negative_integer` | `**14**` → `**15**` | FAIL |
| `test_negative_git_count_is_not_hardcoded` | 把证据快照的 78 改成 79 | FAIL |
| `test_negative_wrong_report_binding` | 从目标报告移除全部 `7.628e-03`，塞进另一份报告 | FAIL |
| `test_anchor_binding_rejects_wrong_location` | 目标报告里数字只出现在**非锚点行** | FAIL |
| `test_negative_missing_jsonpath` | 删掉证据 JSON 的一个键 | FAIL |

**历史 28/28 的处理**：`analysis/number_traceability.json` **原样保留**为历史产物，
并在 v2 输出的 `strictness.historical_v1_28_28` 中明确标注其验证强度边界
（28 个字面量存在性 + 19 个数值匹配；**不能**作为"全部数值级追溯"的证明）。
`P1_A_SCIENTIFIC_REVIEW.md` §14 的表述也已相应收敛。

---

## 5. P3 — 修正配置文档并补齐显式配置项

**`configs/dynamics/controlled_pendulum_jepa.yaml`（v1 历史配置）**

- 修正过期 seed 注释 → 明确写出 SHA-256 派生公式与 `derive_episode_seed()` 位置，
  并注明旧乘法公式因溢出 `RandomState` 32 位范围已被替换。
- 新增**如实声明**的显式键：`integrator: rk4`、`theta_representation: unwrapped`、
  `torque_bounds: [-5.0, 5.0]`、`episode_steps: 200`、`action_hold_steps: 1`。
- 新增 `resolved_pilot_config` 块，**诚实标注 v1 未实现的能力**：
  `initial_state_distribution: fixed_point`、`parameter_sampling: fixed_scenarios`、
  `parameter_ood_groups: not_generated_in_v1`、`collapse_thresholds: not_applicable_v1_pilot`。
  不虚构 v1 没有的训练语义。

> ⚠️ **历史可复现性验证（关键）**：v1 pilot 的 `config_hash` / `dataset_version` 必须保持不变。
> `dataset_spec()` 用显式白名单读键，新增键不进入哈希。实测：
> `recomputed config_hash = 41bb7d66887037e64da18626084f704e001f49ed6bbd92cebe8356a8cb936b82`
> == manifest 记录值；`dataset_version = cpilot-41bb7d668870` 也逐字一致。**历史证据零破坏。**

**新增 v2 配置** `configs/dynamics/controlled_pendulum_v2_pilot.yaml`：
包含规格 §4.4 要求的**全部**显式键（见 `P1_B1_V2_PILOT_REPORT.md` §2），
collector 把 resolved 值写入 v2 manifest。

**新增一致性检查** `scripts/check_config_impl_consistency.py` → **PASS 10/10**：
校验 seed 声明 vs 实现、integrator/theta_representation/torque_bounds/episode_steps 一致性、
`resolved_pilot_config` 一致性，以及模拟器确实不做 `mod 2π` 归约。

---

## 6. P4 — 加固 simulator 校验

**设计取舍**：ChatGPT 要求「Keep the current simulator and dataset unchanged until the new
implementation passes its tests」。我采用**继承复用 + 校验层**：

- **原 `data/controlled_pendulum.py` 零改动**（历史证据）；
- 新建 `data/controlled_pendulum_v2.py`，`ValidatedPendulum` 继承 v1 数值核心，
  **RK4 算术逐位一致**（有单测证明）。

新增校验：

| 类别 | 内容 |
|------|------|
| 物理参数 | `m>0, L>0, dt>0, g>0, b>=0`、全部 finite、拒绝 bool |
| 逐步 finite guard | `rk4_step` 输入/输出、`derivatives` 输入/输出、`simulate` 初态/终态/全轨迹 |
| 力矩校验 | 每个 torque finite 且在 `torque_bounds` 内 |
| profile 校验 | 类型白名单（含新增 `zero`）、fraction/magnitude/amplitude 值域、frequency≥0、bounds 必须跨 0 |
| v2 能力 | `action_hold_steps`、真正的零力矩 episode、初态/物理参数分布采样、预注册 regime 分类 |

**新增单测** `tests/dynamics/test_controlled_pendulum_v2.py`（17 tests + 25 subtests，全过），
含**零回归证明** `TestZeroRegressionAgainstV1`：6 组随机合法输入下
`np.testing.assert_array_equal(v1.simulate(...), v2.simulate(...))` 逐位相同。

**Readiness Plan §2 的判定也已修正**：`integrator` / `theta_representation` 现在区分
「代码实现行为 ✅」与「YAML 显式声明 ❌ / Manifest 记录 ❌」三层，不再笼统标 ✅；
并新增 `readiness_gap` 说明 P1-B 前必须补齐 config + manifest 追溯。

---

## 7. P5 — 分支完整性报告标注历史快照

`P1_A_BRANCH_INTEGRITY_REPORT.md`：

- 头部新增 **「⚠️ 历史审计快照声明」**，明确 §1–§6 是 `a5b8396` 时点快照，
  数字（9 commits / 18 files / 4853 insertions）**保留不覆盖**；
- **追加 §7 当前 HEAD 审计**：实测 11 ahead / 0 behind、36 files（34A+2M+0D）、
  78 个 P2-02 证据文件、两个新审查提交、与 ChatGPT GitHub 实测的逐项一致性表；
- §7.4 明确数字口径：旧值仅适用于 `a5b8396` 基准。

---

## 8. 修复后的验证结果

| 检查 | 命令 | 结果 |
|------|------|------|
| D-A 解析单测 | `pytest tests/dynamics/test_action_intervention_analytic.py -q` | **5 passed + 8 subtests** |
| 验证器 + 负向测试 | `pytest tests/dynamics/test_traceability_verifier.py -q` | **11 passed + 96 subtests** |
| v2 simulator 校验 | `pytest tests/dynamics/test_controlled_pendulum_v2.py -q` | **17 passed + 25 subtests** |
| v2 报告核验 + 负向测试 | `pytest tests/dynamics/test_v2_report_verifier.py -q` | **9 passed + 62 subtests** |
| v2 报告数字可追溯性 | `python scripts/verify_v2_reports.py` | **PASS 32/32** |
| 数字可追溯性 v2 | `python scripts/verify_p1a_reports.py` | **PASS 48/48** |
| 配置-实现一致性 | `python scripts/check_config_impl_consistency.py` | **PASS 10/10** |
| v1 config_hash | 重算 vs manifest | **逐字一致（历史完好）** |
| v1 既有 audit | `python scripts/audit_controlled_pendulum.py --config ...` | **8/8 PASS（未改动）** |
| V2 pilot 独立验收 | `python scripts/audit_controlled_pendulum_v2.py` | **10/10 PASS** |
| **全量回归** | `pytest tests/ --ignore=tests/test_pysr.py -q` | **297 passed + 292 subtests**（本轮本地实跑） |

回归基线对照：此前 `255 passed + 101 subtests` → 本轮 `297 passed + 292 subtests`，
净增 **+42 tests / +191 subtests**，与本轮新增的 4 个测试文件
（`test_action_intervention_analytic.py` 5+8、`test_traceability_verifier.py` 11+96、
`test_controlled_pendulum_v2.py` 17+25、`test_v2_report_verifier.py` 9+62）
精确吻合：5+11+17+9=42、8+96+25+62=191。**零回归**。

> **关于 ChatGPT 的诚实提醒**：此前 `255 passed + 101 subtests` 是本任务在
> Windows 本地环境（`D:/python313/python.exe`）的执行记录，**不是** ChatGPT 独立复跑的结果；
> GitHub 也未返回可供独立确认的 CI 状态。本轮我在同一环境**重新实跑**全量回归得到
> `297 passed + 292 subtests`，明确标注为**本地执行记录**，不冒充 CI、不冒充远端独立验证。

---

## 9. 限制与合规

- 未修改 `data/controlled_pendulum.py`、`scripts/collect_controlled_pendulum.py`、
  `scripts/audit_controlled_pendulum.py`、`scripts/analyze_p1a_pilot.py`、
  `datasets/controlled_pendulum_pilot/` 下的 `npz` / `manifest.json` / `audit_report.json`。
- 未改动任何受限 Core / Adapter 文件、Pusher-v1、历史 checkpoint。
- 未生成全量数据集、未实现 `trajectory_windows.py` / `ActionConditionedWorldModel`、
  未训练 JEPA、未实现 MPC、未合并 main、未开 PR。
- 保留 E0–E5 与 G0–G6 全部定义；`P2_03_ACTION_JEPA.md` 正文未被修改。
