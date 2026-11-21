# M4.3-02 Integrated End-to-End Value Validation — 验收报告

**分支**：`feature/arr-0.1-domain-registry`（未 merge main，未建 PR）
**阶段**：M4.3-02（Pilot + Confirmatory 框架 + 分析 + 报告 + CLI + 真实运行）
**结论先行**：执行系统已完整交付并**真实跑通**（3 个 Block × A/B/C = 9 条真实
SciNet++ 链、真实 12-step Gate、真实外部审批、真实 Report/Experience、可恢复
Registry、108 个真实引擎日志归档）。
正式实验（Confirmatory）**未启动**：Pilot 的人工盲评尚未由两名真实评审提交，
样本量估计因此为 `evidence_insufficient`，协议冻结按设计被拒绝。这是规格要求的
行为，不是缺陷。

真实数据一句话总结：**9 条完成的真实运行全部是 `negative_finding`（假设被拒/
不确定），三组有效发现率均为 0，差异方差为 0；人工盲评缺失 → 证据不足。**

---

## 1. 交付清单

| # | 交付物 | 文件 | 状态 |
|---|---|---|---|
| 1 | Pilot Runner | `research_runtime/m4_3/pilot_runner.py` | ✅ |
| 2 | Confirmatory Runner | `research_runtime/m4_3/confirmatory_runner.py` | ✅（框架完成，人工门未开） |
| 3 | Pilot Analysis | `research_runtime/m4_3/pilot_analysis.py` | ✅ |
| 4 | Sample Size Estimator | `research_runtime/m4_3/sample_size.py` | ✅ |
| 5 | Protocol Builder / Freeze | `research_runtime/m4_3/confirmatory_protocol_builder.py` | ✅ |
| 6 | Statistical Analysis | `research_runtime/m4_3/statistical_analysis.py` | ✅（Holm + 配对 t） |
| 7 | Value Report Generator | `research_runtime/m4_3/value_report.py` | ✅ |
| 8 | Unified CLI | `research_runtime/m4_3/cli.py` | ✅ |
| 9 | Recovery | `experiment_recovery.py` + `experiment_state.py` | ✅（真实中断后续跑） |
| 10 | 人工盲评入口 | `review_queue.py` + `cli review` | ✅（评审仍 pending） |
| 11 | README | `M4_3_RUNTIME_README.md` | ✅ |
| 12 | CI | `.github/workflows/arr-research-runtime.yml` | ✅（引擎无关 + opt-in 真引擎） |
| 13 | 真实 Pilot 数据与证据包 | `M4_3_runs/pilot/**` | 见第 6 节 |

## 2. P0：Gate 语义修复（terminal ≠ successful）

`research_service_gate.py` 原先把 `CAMPAIGN_TERMINAL_STATUSES`（含 `failed` /
`cancelled`）当作 Gate 成功。现在：

* `campaign_terminal` 与 `campaign_successful` 分离；只有
  `completed` / `completed_with_failures` 且（有 trial 记录时）至少一个 trial
  成功，才算成功；
* `failed` / `cancelled` **永不**产生 Gate 成功；
* `valid_finding` 额外要求 `valid_finding_evidence`：成功 Campaign + 报告携带真实
  科学结论（`not_evaluated` 不算）+ 六个 artefact 精确互相关联 + Experience refs
  全部解析到真实行；
* `negative_finding` 只要求 `gate_passed`：真实完成但没有预期正向发现也算结果；
* 关联校验（9 项）：plan↔session objective、approval↔plan、campaign↔plan、
  campaign↔session objective、session 拥有 campaign、report↔session、
  report↔objective、session.report_id、report 列出该 campaign；
* 跨 run 复用同一 campaign 的成功证据 → `RegistryConflict`（内存 + SQL 双查）；
* 只读：只用既有 query 方法，不重写 12-step gate。

反例测试（`test_m4_3_02_p0_gate_success.py`，16 项）：failed/cancelled、仅失败
trial、跨 session、跨 plan、错误 plan SHA、伪造 experience ref、`not_evaluated`
报告、只读性、重放拒绝。

## 3. P0：统一 Experiment Controller

23 个状态、显式转换表，非法转换直接拒绝（`IllegalTransition`）；每次状态变化
追加持久化事件（`m4_3_experiment_events`，append-only）；`BudgetGuard` 在预算
不足时以 `BudgetExhausted` 停止并记录 `stopped_reason`；`recover` 只允许进入
active 状态。

真实恢复证据：本次 Pilot 在 block 2 选题阶段因盲评泄漏失败（见 5.2），
`status` 显示 `SELECTING_OBJECTIVES` + block 1 `BLOCK_COMPLETED` + block 2 A 已
注册未 finalize；`recover` 对账后继续执行，**未重复创建 block 1 的任何 Campaign**。

## 4. P0：Pilot Runner 的 A/B/C 等价性控制

| 控制 | 实现 |
|---|---|
| 同一研究机会 | 每个 block 一个真实 Evidence 子集；A/B/C 同一 snapshot_id、同一 seed |
| A 不看历史 | `ARM_CONTEXT_RULES` + `build_arm_context` fail-closed；A 的 snapshot 无历史 objectives/sessions |
| B/C 历史一致 | 同一个 `DiscoverySnapshot` 对象（同一 evidence_view + 同一历史 objectives/sessions/experience/failures） |
| C 真实 Discovery | `ObjectiveDiscovery(generator, critic, ranker).discover()`，记录完整 `DiscoveryResult`（候选/critique/ranked/版本） |
| 同一服务/门/引擎 | 一个 `ResearchService`、一个 `ResearchServiceGateAdapter`、一个 SQLite |
| 相同预算 | `candidate_budget`、`human_time_budget_s`、`campaign_budget` 每 block 固定 |
| 相同 paired seeds | 注入 domain policy，plan 实际使用 42/43 |
| 真实外部审批 | 每个 arm 由独立进程 `approval_cli grant --approved-by <operator>` |
| 全部结果保留 | valid/negative/inconclusive/invalid/failed/interrupted 全部入库 |

**诚实标注**：A/B 的“人工”选题由**预注册的 delegated-operator 规则**执行
（selector_id=`delegated-operator`，规则 `delegated_operator_heuristic_v1`：可解析
证据数 → 可证伪性完备度 → 生成顺序），并非独立人类研究者。该限制写入
`pilot_analysis.limitations` 与价值报告。

## 5. 真实执行

### 5.1 环境

* SciNet++ checkout：`E:\LLM\SciNet++`（含 `research_api/cli.py`）
* 引擎解释器：`D:\conda_envs\scinet_gpu\python.exe`（真实训练，CPU smoke config）
* 真实历史证据：`arr_m34_001` → 118 个 evidence entity
* 源实验：由真实 EvidenceStore 选出带结构化诊断的 evaluate 结果（非硬编码）

### 5.2 过程中发现并修复的真实缺陷

1. **盲评泄漏**：历史 objective id（如 `obj_sel_pilot_block_1_A`）出现在候选
   evidence refs 中，触发 blinding fail-closed。修复：`is_group_leaking()` +
   评审可见区过滤（withheld 列表写入 run notes，不静默丢弃）。
2. **续跑不可行**：`ExperimentRegistry` / harness 未从同一 SQLite 恢复，重复 block
   冲突。修复：`_build_harness` 用 `ExperimentRegistry.recover`，`ensure_block`
   重连既有 block，按 arm 跳过已注册选题。
3. **候选池被自身消耗**：A 落库后其候选对 B 变成 duplicate。修复：候选池在任何
   落库之前构建。
4. **B/C 证据范围**：C 的 Discovery 若读全量语料会把整库候选一次性落库，后续 block
   无候选。修复：每个 block 的 Discovery 只用该 block 的真实证据子集，历史只提供
   context。

### 5.3 真实运行结果（Final）

命令：

```bash
python -m research_runtime.m4_3.cli run --stage pilot --blocks 4 \
    --seeds 42 43 --candidate-budget 3 --approver operator --max-campaigns 12
```

| 项目 | 真实值 |
|---|---|
| 注册 Block | 4（block 1-3 完整，block 4 因候选耗尽 + 预算停止而排除） |
| 可配对 Block | **3** |
| Runs | 12（9 完成 + 2 interrupted + 1 invalid） |
| 结果分布 | `negative_finding` 9 / `invalid` 1 / `interrupted` 2 / `valid_finding` 0 |
| Gate 完整率 | 0.75（9/12，= 完成 run 全部 12/12 步） |
| 外部审批率 | 0.75，`approval_bypass_count` = 0 |
| Report / Experience 完整率 | 0.75 / 0.75（与完成 run 一致） |
| 真实 Campaign | 9 个互不相同（无跨 run 复用，`duplicate_campaign_count` = 0） |
| 真实训练时间 | 2019.87 s（9 个 campaign，2 个 paired seed） |
| 引擎日志 | 108 个作业日志已归档 |

各 arm（4 个 run，其中 3 个完成）：

| 指标 | A | B | C |
|---|---|---|---|
| valid_research_findings_per_compute_budget | 0.0 | 0.0 | 0.0 |
| independently_reviewed_objective_quality | n/a（无人工评审） | n/a | n/a |
| plan_validity_rate | 0.75 | 0.75 | 0.75 |
| objective_acceptance_rate | 0.75 | 0.75 | 0.75 |
| evidence_grounding_rate | 1.0 | 1.0 | 0.75 |
| invalid_experiment_rate | 0.0 | 0.0 | 0.25 |

Block 级配对（n=3，描述性，**不做显著性判断**）：

| 对比 | 指标 | n | 均值差 | 标准差 | bootstrap 95% CI |
|---|---|---|---|---|---|
| B-A | valid_findings/compute | 3 | 0.0 | 0.0 | [0.0, 0.0] |
| C-B | valid_findings/compute | 3 | 0.0 | 0.0 | [0.0, 0.0] |
| C-A | valid_findings/compute | 3 | 0.0 | 0.0 | [0.0, 0.0] |
| B-A / C-B / C-A | objective_quality | 0 | 缺失 | 缺失 | n/a（无人工盲评，不插补） |

**解读（严格按数据）**：3 个 Block 上三组的有效发现率完全相同（都是 0），
差异方差为 0；objective quality 因人工盲评 pending 完全缺失。因此

* `pilot_analysis.conclusion.verdict = evidence_insufficient`
* `sample_size.status = evidence_insufficient`（三个对比都缺方差）
* `M4_3_PILOT_VALUE_REPORT.md` 结论 = `evidence insufficient`，
  `significance_claimed = false`

这就是 Pilot 的真实结论：**在单域 smoke 规模上，A/B/C 没有可观测差异，且人工
评审缺失，无法给出任何价值判断**。报告不会把它包装成"C 胜出"或"无价值"。

证据包：`M4_3_runs/pilot/evidence/`
（`registry_export.json` + `controller_export.json` + `campaigns.json` +
`engine_logs/` 108 个真实作业日志 + `evidence_bundle.json`；
`registry_integrity = true`，`missing_artefacts = []`）。

## 6. 分析与样本量

`analyze-pilot` 输出：

* `pilot_analysis.json`：每组 16 指标、block 级配对、Snapshot/Block/Seed 三层、
  失败/中断/无效率、Gate/Approval/Report/Experience 完整率、Campaign 时间与
  compute、Objective Quality 均值/标准差/分布、A-B/B-C/A-C 配对差异 + 方差 +
  bootstrap CI、数据完整性与排除原因、预算估算；
* `pilot_report.md`：人读报告；
* `sample_size_recommendation.json`：SEOI 驱动、方差膨胀、敏感性网格、多重比较
  校正后的 alpha；证据不足时明确 `evidence_insufficient` 且不给数字。

**Pilot 不宣布胜出**：`significance_conclusion = null`，结论固定为
`evidence_insufficient`（人工盲评未完成 + block 数不足以估计方差）。

## 7. Confirmatory 门（现状：按设计拒绝）

`prepare-confirmatory` 在 Pilot 样本量估计为 `evidence_insufficient` 时写出
`confirmatory_draft.json`（`status: blocked` + 明确原因），`freeze-confirmatory`
拒绝冻结，`approve-confirmatory` 拒绝（无冻结协议），`run --stage confirmatory`
拒绝（无人工批准 + 非 `CONFIRMATORY_AWAITING_APPROVAL`）。

要真正进入 Confirmatory，需要（按顺序、人工执行）：

1. 两名独立评审通过 `cli review submit` 提交全部 queued objectives；
2. `cli review unblind`（全部完成后才可解盲）；
3. `cli analyze-pilot` 重算（此时 objective-quality 有数据，方差可估）；
4. `cli freeze-confirmatory --blocks N --seoi X`；
5. `cli approve-confirmatory --approved-by <人>`；
6. `cli run --stage confirmatory`。

## 8. 测试与回归

| 范围 | 结果 |
|---|---|
| `pytest research_runtime/tests/ -k m4_3` | **321 passed, 2 skipped** |
| `pytest research_runtime/tests/`（全量回归） | **575 passed, 3 skipped, 37 subtests** |
| 真实引擎 E2E | opt-in（`M4_3_RUN_REAL=1`），CI 中显式 skip，本次由真实运行替代 |

引擎无关与真引擎结果严格分开报告：CI 通过**不**代表真实引擎通过；本报告第 5 节
的真实数据来自本机真实执行（真实 SciNet++ 训练 + 真实 SQLite 记录）。

新增测试文件：`test_m4_3_02_p0_gate_success.py`、
`test_m4_3_02_state_machine.py`、`test_m4_3_02_review_queue.py`、
`test_m4_3_02_analysis_report.py`、`test_m4_3_02_confirmatory.py`、
`test_m4_3_02_pilot_units.py`、`test_m4_3_02_cli.py`、
`test_m4_3_02_evidence_bundle.py`、`test_m4_3_02_real_engine.py`（opt-in）。

## 9. 未启动项与限制

* M5 未启动。
* Confirmatory 未启动（人工门未开）。
* A/B 选题为 delegated-operator，不是独立人类。
* 人工盲评为 pending（队列已入 3×N 条，评审需真人提交）。
* 单域（SciNet++ pendulum smoke），非多域泛化实验。

## 10. 复现命令

```bash
python -m research_runtime.m4_3.cli preflight
python -m research_runtime.m4_3.cli run --stage pilot --blocks 4 --seeds 42 43 \
    --candidate-budget 3 --approver operator --max-campaigns 12
python -m research_runtime.m4_3.cli analyze-pilot
python -m research_runtime.m4_3.cli report --stage pilot
python -m research_runtime.m4_3.cli status
python -m research_runtime.m4_3.cli verify-integrity
```
