# M5 Integrated Autonomous Research Runtime — 集成验收报告

分支 `feature/arr-0.1-domain-registry`，起点基线 `609934da`，最终 HEAD `b1d51bf9`。
未 merge main、未建 PR。未重写 ResearchService / Director / 12-step Gate / Discovery /
Scheduler / Evidence Reader / Decision Engine。未用 FakeRunner 充当真实引擎，未自动授予
正式 Human Approval，未把 Gate Pass 当 Scientific Finding。

## 结论先行

三层能力均有真实证据，其中"研究能力"和"RSI 能力"是本次重点：

| 层次 | 要证明的事 | 结果 | 关键真实证据 |
|---|---|---|---|
| Runtime | 持续运行、调度、恢复、控预算 | ✅ | hash-chain journal 10 事件、3 域预算消费=3、recovery fail-closed、10/10 validation |
| Research | 发现问题、设计实验、分析失败、积累经验 | ✅ | 三域真实 Campaign+Gate+Experience；Round2 三个新 objective 全部 `generation_source=historical_experience` |
| RSI | 通过实验改进研究模型和 Harness | ✅（Domain 层真晋级；Harness 层诚实 blocked/INCONCLUSIVE） | embodied `state_mse@1` 独立 holdout 0.001495→0.001233（+0.000278，3/3 方向一致），PROMOTED |

**"下一轮是否因经验和改进而变得更好"的直接证据**：Round2 的三个新 objective 由 Round1 的
真实 Experience 驱动（非 bootstrap）；Domain Model 候选在**独立 holdout seeds (60/61/62)**
上把 `state_mse@1` 从基线 0.001495 降到 0.001233，且 direction fraction=1.0。这是可复核的
真实指标改善，不是布尔声明。

---

## 1. 完整架构与数据流

```
run_m5_integrated.py --phase all
  ├─ RuntimeConfig + RuntimeState(SQLite)         [全局状态/预算/journal/checkpoint]
  ├─ phase_research
  │    └─ AutonomousRuntime -> RuntimeCoordinator
  │         ├─ CrossDomainScheduler (scinet/embodied/quant adapter)
  │         │    └─ ResearchDirector -> ResearchService -> 12-step Gate
  │         │         -> 外部 approval_cli -> 真实 Campaign -> Report -> Experience
  │         ├─ build_cross_domain_snapshot (全字段合并+来源/SHA)
  │         └─ ObjectiveDiscovery (Critic + deterministic Ranker)  [Experience-driven]
  ├─ phase_rrsi (run_m5_rrsi.py)
  │    ├─ Layer1 DOMAIN_MODEL: 真实 embodied runner -> 候选评估 -> 独立 holdout
  │    ├─ Layer2 RESEARCH_HARNESS: SciNet++ ingest (子进程超时) / 真实历史产物回退
  │    └─ RRSICoordinator(require_structured_holdout=True) -> EvolutionRegistry
  └─ phase_validate (run_m5_validation.py)
       └─ 3 rounds / 崩溃注入 / 重启恢复 / 预算 / stop / 隔离 / 幂等 / rollback
```

所有状态写入共享 `ResearchDatabase`（WAL + `BEGIN IMMEDIATE` + RLock），append-only journal
为 SHA 哈希链，checkpoint 内容不可变。

## 2. 顶层 Runtime 入口与用法

```python
from research_runtime.autonomous_runtime import AutonomousRuntime, RuntimeConfig
rt = AutonomousRuntime.from_services(
    config=RuntimeConfig(run_id="m5_integrated_v1", max_rounds=2, global_budget=12),
    state_db=shared_db, scheduler=cross_domain_scheduler, discovery=objective_discovery,
    rrsi=None, dataset_shas={"scinet": sha, ...},
    approver=external_human_approver,   # 必须外部注入；None 则 fail-closed
    out_dir="...", objective_specs=specs)
rt.start(); rt.recover_or_raise()
rt.run_round(1, objective_specs=specs)
rt.cross_domain_discovery(round_no=2, objective_ids=[...])
rt.finalize(round_no=2)
```

模块：`autonomous_runtime.py`（入口）、`runtime_coordinator.py`（闭环）、`runtime_state.py`
（SQLite 状态/journal/预算/checkpoint/recovery）、`runtime_report.py`（证据包）。

## 3. 真实三域 E2E（Round1）+ 数据流

| 域 | Campaign | Director | 审批 | Gate |
|---|---|---|---|---|
| scinet（SciNet++ 真引擎） | `campaign_a84704db42775e9f` | COMPLETED | 外部 CLI | 原 12-step |
| embodied（真实仿真） | `campaign_ba9bb6554bc95bf5` | COMPLETED | 外部 CLI | 原 12-step |
| quant（真实回测，synthetic 标注） | `campaign_583d7d8d6f3458a0` | COMPLETED | 外部 CLI | 原 12-step |

证据包 `M5_INTEGRATED_runs/evidence/phase_research_evidence.json`，
SHA `b7dcafb3...`，journal 10 事件全部 integrity_ok，全局预算 consumed=3 / limit=12。

## 4. Round2 跨域 Experience 回流证据

- 跨域 Snapshot：`source_domains=[embodied, quant, scinet]`，3 objectives / 3 sessions，
  `experience_available=true`，三域各保留 artifact SHA（`016c1c56...`/`bf93bd03...`/`0ad21034...`），
  `unresolved_gaps=[]`。
- 3 条 Experience 全部保留 `source_domain` + `source_run_id`。
- Discovery 产出 3 个新 objective：`obj_disc_cand_experience_*`，
  `generation_source` 全部为 `historical_experience`——真实经验驱动，非 bootstrap。

## 5. RRSI 两层真实评估证据

证据包 `M5_INTEGRATED_runs/rrsi/rrsi_result.json`，SHA `adedbbc1...`。

**Layer 1 — DOMAIN_MODEL（本会话真实引擎新跑，晋级）**

两个真实候选经过相同真实 embodied runner + paired(decision seeds 42/43) + 独立 holdout(60/61/62)：

| 候选 | 决策 | 结果 |
|---|---|---|
| `train.lr=0.003`（search-space） | REJECTED | 失败候选保留，candidate_sha `d5404f24...` |
| `model.drift_correction=0.75`（diagnosis-targeted） | ACCEPTED → **PROMOTED** | 独立 holdout 通过，candidate_sha `572b7caf...` |

指标（`state_mse@1`，minimize）：holdout 基线 `[0.001495, 0.001543, 0.001476]` →
候选 `[0.001233, 0.001264, 0.001185]`，`holdout_improvement=+0.000278`，
`holdout_direction_fraction=1.0`（3/3 改善）。结构化 Holdout 证据绑定真实 manifest SHA、
split id、candidate/parent SHA、task ids、result SHAs、指标与统计结论，不再相信布尔值。

**Layer 2 — RESEARCH_HARNESS（诚实标注）**

实时的 SciNet++ `research_ingest(real=True)` 在子进程 180s 超时内未通过环境钩子
（sandbox 批量清理被 safe-delete 保护拦截），驱动**诚实回退**到磁盘上真实持久化的
harness-evolution 产物（`experiments/arr_1_0_evolve/harness_evolution_HYPOTHESIS_QUALITY_FAILURE_*.json`，
含真实 holdout Q、result path、gate trace），经同一结构化 Holdout 门判定为
**INCONCLUSIVE**（未晋级）。`evaluation_source=replayed_real_persisted_artifact`，
不伪装为实时晋级。

**双层隔离与回滚**：两层独立 Registry 记录、独立 parent SHA；`parent_for()` 返回最新
PROMOTED 版本作为 rollback 目标，失败候选保留但不作为回滚目标。

## 6. Checkpoint / Recovery / Budget 证据

- `RuntimeState`：全局 SQLite 状态、Reservation 在 Campaign 前、同 campaign 幂等不重复扣费、
  Run Config SHA + Dataset Manifest SHA 绑定、启动对账、不可判定 fail-closed。
- 预算耗尽、人工 stop、wall-clock、单域失败隔离均在 validation 中验证。
- at-most-once + resume-in-place（明确不宣称 exactly-once）。

## 7. Crash Injection 结果

`M5_INTEGRATED_runs/validation/validation_result.json`，SHA `f9048f68...`，**10/10 PASS**：
3 轮连续运行、每域预算上限、12 个崩溃注入点（≥10）、3 次进程重启恢复、预算耗尽安全停止、
人工停止、单域隔离、Campaign 幂等+证据不可覆盖、corrupt 状态 fail-closed、24h 诚实记账。

## 8. 24 小时连续运行

**BLOCKED（未执行，不宣称通过）**：本会话资源/时间预算不足，无法完成真实 24h wall-clock
连续运行。validation 将其如实记为 `status=blocked`，并明确"NOT claimed as passed"。

## 9. CI 对应最新 HEAD

- HEAD `b1d51bf9`，ARR Research Runtime workflow：
  <https://github.com/thesevenths/agent_dev/actions/runs/43841581029>（查询时 `in_progress`）。
- Pre-commit checks：<https://github.com/thesevenths/agent_dev/actions/runs/43841580781>（`success`）。
- 本地完整 engine-free 回归：**683 passed / 8 skipped / 37 subtests passed**（248.79s）。
- opt-in 真实三层证据测试：`M5_INTEGRATED_RUN_REAL=1` → 3 passed。
- engine-free 与 real-engine 测试分离（skipif 门控），新增模块均已加入根 CI。

## 10. 独立 Commit 清单（609934da..HEAD）

| Commit | 内容 |
|---|---|
| `656baf25` | P0：跨域 snapshot 全字段合并、真实状态/退出码、诚实 smoke 审批 |
| `8b8c8231` | P0：结构化 Holdout 晋级门（禁止布尔-only 晋级） |
| `9c0241fe` | Phase1/2：顶层 Runtime（持久全局状态/预算/journal/checkpoint/恢复/证据包） |
| `663f0477` | Phase1/2：Runtime engine-free 测试 |
| `9b8f3a34` | P4：RRSI facade dataclass 序列化修复、holdout 构造器、候选身份、证据不可变 |
| `87ce4c8a` | Phase4/5/6：三域 E2E / 双层 RRSI / production validation 三个 driver |
| `bc181d06` | Phase4/5/6：Domain 真实晋级 + 诚实 harness 回退 + all-phase driver |
| `b1d51bf9` | opt-in 真实三层证据测试 + CI 门禁 |

## 11. 未完成事项与真实风险

1. **Harness 层未实时晋级**：实时 SciNet++ ingest 被环境 safe-delete 钩子拦截，回退到真实历史
   产物判 INCONCLUSIVE。要拿到本会话实时 Harness 晋级证据，需在允许 sandbox 批量清理的
   环境重跑 `python run_m5_rrsi.py`（去掉超时/钩子限制）。
2. **24h 连续运行 BLOCKED**：资源不足，未执行。
3. **每域每轮 ≤2 Campaign 上限**：以配置绑定 + 上游调度断言验证；本轮实际每域各 1 个 Campaign。
4. **Confirmatory / 真人盲评**：仍为历史状态（0 提交、双 Primary Metric evidence_insufficient），
   本阶段未触碰。
5. **quant 数据为 synthetic**：仅用于流程/引擎验证，不作为真实市场收益或 Alpha 证据。
6. **CI ARR workflow 尚在运行**：查询时 in_progress，最终结论以 run 43841581029 为准。

现在暂停，等待代码审查。

---

# M5 Acceptance Hardening（源码审查后加固，最终章节）

用户基于真实源码审查提出 P0/P1 缺陷后，逐条修复并重新真实运行。起点 `49438f25`，
终点以本节提交为准。**这是本报告的最新结论**；上方章节保留为加固前历史记录。

## H-P0 五项缺陷修复（源码级，附真实证据）

| 优先级 | 审查发现 | 修复 | 源码/测试证据 |
|---|---|---|---|
| P0 | 预算执行后才 reserve | 新增 `rt_reservation` 表，确定性 reservation_id，Campaign 启动前原子预留；bind→consume 幂等；崩溃后 orphan reservation 不重获预算 | `runtime_state.reserve_execution/bind_reservation/consume_reservation`；`test_budget_one_three_domains_only_one_campaign_starts`（budget=1 三域只启动 1 个 Campaign） |
| P0 | start() 无条件设 INIT；并发启动无保护 | 终态（COMPLETED/STOPPED/…）拒绝启动；RUNNING 重启先分类恢复；`claim_executor` 原子单例执行者 | `start()` / `claim_executor`；`test_start_refuses_terminal_run_and_running_restart`、`test_concurrent_start_only_one_executor` |
| P0 | Round 状态字段非空即判 COMPLETED；report/experience 可空 | 细化 CAMPAIGN_COMPLETED/REPORT_PENDING/EXPERIENCE_PENDING/ROUND_COMPLETED；`put_round` 拒绝无证据的 ROUND_COMPLETED；`classify` 用 `reality_verifier` 重读真实 Store（campaign binding→session→report→experience）；finalize 只在 C_COMPLETE 时标记完成 | `runtime_state.classify`、`runtime_coordinator._verify_round_reality`；`test_classify_uses_real_store_not_round_flag` |
| P0 | Round2 只生成 Objective 未执行；RRSI/Validation 脱离 Runtime | `phase_research` 真实执行 Round2 新 Objective 的 hypothesis/plan/approval/real campaign；RRSI 在 `--phase all` 中 in-process 写同一 RuntimeState journal；finalize 统一收口 | `run_m5_integrated.phase_research`；真实运行 `e2e_final`（见下） |
| P0 | Holdout result SHA 自造；Harness candidate SHA 身份不一致 | builder 不再自造 SHA（从真实 result_files 读）；verifier 支持 `artifact_reader` 重读磁盘 + manifest SHA 复核；candidate_sha 统一由 coordinator 对同一 dict 计算；parent_sha 来自 Registry | `holdout_evidence.build/verify`；`test_holdout_rejects_self_derived_result_sha`、`test_holdout_candidate_identity_uses_coordinator_sha` |

## H-P1 两项修复

- **P1-A 跨域 Objective 归属**：ObjectiveCandidate 的 domain 现从证据来源推断（experience/failure 的 source_domain、OOD 的 dname），Discover 按 candidate.domain 分别写入对应域 Store 并标 `cross_domain`；Scheduler 按真实归属路由。真实 Round2 证据：3 个 objective 分别路由到 quant/scinet/embodied（2 个 cross_domain=True）。
- **P1-B 代码版本与 Scientific Judgment**：Evidence Package 记录 agent_dev/SciNet++ 的 Git HEAD + dirty 状态 + Python 版本（真实证据包 HEAD `fc19ded2`，dirty=true）；新增确定性 `scientific_judgment`（novelty 明确 `literature_check=not_performed`，不假称新颖；research_value 为规则评分，不发 ACCEPT/REJECT 裁决）。

## H-真实完整闭环（M5 Acceptance Hardening 运行）

输出目录 `M5_INTEGRATED_runs/e2e_final/`，evidence SHA `bfd9e146…`，journal 20 事件 integrity_ok：

- **Round1**：scinet/embodied/quant 三域真实 Campaign 全部成功（外部 CLI approval + 原 12-step Gate）。
- **Round2**：3 个 Experience-driven objective（generation_source 全部 `historical_experience`）分别路由到 quant/scinet/embodied，其中 2 个标 `cross_domain=True`；三域**真实执行**新 Campaign：`campaign_22e74d797b185534` / `campaign_8da0f3bb9b83581e` / `campaign_d57f6d171ca45a72`，全部 `ROUND_COMPLETED` 且带真实 `experience_id`。
- **预算**：全局 consumed=6（两轮各 3），limit=12；域预算按轮隔离（P0-D）。
- **终态**：`integrated_result.json` state=COMPLETED，finalize checkpoint SHA `ae982f29…`（P0-C：所有 round 真实 Store 核验通过才允许 COMPLETED）。
- **opt-in 真实证据测试**：`M5_INTEGRATED_RUN_DIR=.../e2e_final M5_INTEGRATED_RUN_REAL=1` → 3 passed（含 Round2 真执行、experience_id、跨域路由、代码版本绑定断言）。

## H-验证与边界

- engine-free 加固测试 17 项（`test_m5_hardening_p0.py`）全绿；相关 discovery/runtime 回归通过；语法门禁通过。
- 24h 连续运行仍 BLOCKED；真人 Confirmatory Blind Review 仍未启动；Harness 层实时晋级仍受环境 safe-delete 钩子限制（回退真实历史产物判 INCONCLUSIVE）。这些均未宣称通过。
- SciNet++ 训练/验证/Test split 隔离、按 split 保存 result/meta 的设计已核实正确；建议后续补 Dataset Manifest SHA 绑定与 Resume 身份复核（P2，非本次 M5 阻塞项）。
