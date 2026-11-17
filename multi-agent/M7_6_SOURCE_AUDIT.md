# M7.6 Source Audit — Production Runtime Closure

> 本审计基于当前 checkout 的真实源码逐行复核，不依赖 README、验收报告或任何 Agent 总结。
> 每条缺陷均附 `file:line` 与亲自复核的当前行为。审计时间：2026-11-17。

## 0. 基线（Baseline）

| 项 | 值 |
|---|---|
| Git root | `E:/agent_dev` |
| 工作目录 | `E:/agent_dev/multi-agent` |
| 最新连贯实现 | `feature/arr-m7-capability-validation` @ `6888ac950810a1527e57955f4cb0726ff6cee58c` |
| 与 discovery 分支关系 | `git merge-base --is-ancestor 729184c4 HEAD` = YES（discovery 是其祖先） |
| 与 main 关系 | main 是其祖先；`main..HEAD` = 210 commits |
| M7.6 分支 | `feature/arr-m7-6-production-runtime-closure`（从 `6888ac95` 建立） |
| 工作区 | 无 tracked 修改；untracked 仅为 `.pytest_bt_*` 临时目录与 `m7_capability_clean/` 旧快照（非本轮产物，保留不动） |
| 远端 feature HEAD | validation=`6888ac95`，discovery=`729184c4`，main=`96d85296` |

结论：`feature/arr-m7-capability-validation` 是最新连贯实现，作为 M7.6 唯一基线；两个 feature 分支祖先关系明确，无需 reconcile；不 merge main、不建 PR、不 force push。

## 1. 当前生产 Runtime 入口（已核实）

| 能力 | 入口 | file:line |
|---|---|---|
| 顶层编排 | `RuntimeCoordinator.start / run_round` | `runtime_coordinator.py:52 / :92 / :215` |
| 正式 Dispatcher | `ResearchService.run_research_campaign` → `_run_campaign` | `research_service.py:773 / :777` |
| Dispatcher 实际执行 | `runner_factory().run(plan, run_spec)` | `research_service.py:912` |
| Campaign 生命周期 | `CampaignRunner.run / resume / get_status / cancel` | `campaign/runner.py:281 / :336 / :373 / :392` |
| CampaignRecord 状态机 | created→preflighting→running→terminal | `campaign/schemas.py:22-23,127-139` |
| 确定性 campaign id | `make_campaign_id` | `research_service.py:208` |
| 生产装配 | `build_default_research_service(..., harness_loader=...)` | `research_environment.py:161 / :210` |
| 预算预留状态机 | reserved→bound→consumed\|released | `runtime_state.py:60-63`；`reserve/bind/consume/release` = `:377/:425/:452/:474` |
| Journal 校验 | `RuntimeState.verify_journal` | `runtime_state.py:281` |
| 预留/结算正确范式 | `RuntimeCoordinator.run_round` 成功与异常释放 | `runtime_coordinator.py:236-270 / :318-356` |

## 2. M6 HarnessLoader / Promotion / Rollback 接口（已核实）

| 接口 | file:line |
|---|---|
| `LoadedHarness`（harness_sha/component_shas/parent_sha/activation_id） | `harness_loader.py:41-47` |
| `HarnessLoader.verify_approved`（PROMOTED+审批+activation+磁盘复核） | `harness_loader.py:164-202` |
| `HarnessLoader.resolve_for_campaign`（内含 bind + bind_campaign） | `harness_loader.py:204-223` |
| `CampaignVersionBinding.bind / resolve` | `harness_loader.py:77 / :107` |
| `verify_formal_promotion / formal_promote` | `promotion.py:82 / :157` |
| `AdoptionManager.adopt / verify_loaded / active_sha / bind_campaign` | `adoption.py:59 / :146 / :53 / :161` |
| `real_rollback` | `rollback_v2.py:58` |
| 真实已批准 harness 的生产装配范例 | `runtime_seam.py:31-153`（真实 activation + registry + artifact_index） |
| 真实 M6.1 采纳证据 | `M6_1_EVIDENCE/run_final4/adoption/{activation.json,evolution.sqlite,bindings.sqlite,runtime.sqlite}` |

## 3. M7 DiscoveryCoordinator / DomainPack 接口（已核实）

| 项 | file:line | 当前状态 |
|---|---|---|
| 17-stage 状态机 | `discovery_coordinator.py:46-53` | 完整，可恢复、幂等 |
| executor 注入 | `:86, :111` `executor or self.pack.run_experiment` | **BYPASS（P0-B）**：默认直连 domain pack |
| harness_sha 注入 | `:93, :118` 来自构造参数 | **BYPASS（P0-B）**：未经 Loader 校验 |
| formal_loader seam | `:94, :244-262` | **PARTIAL**：仅存证 SHA，不路由执行 |
| 执行调用 | `:584, :669` `self.executor(...)` | **BYPASS**：从不调用 Dispatcher |
| campaign_id | `:580, :666` 合成 `m7sel_/m7rep_` | **BYPASS**：无真实 CampaignRecord |
| reservation | `:227-242` 创建+绑定；`:264-270` 结算 | PARTIAL：绑伪 campaign，异常不结算 |
| 自述集成级别 | `:291-293` | 自认默认 `PARTIAL_LOCAL_DIRECT_EXECUTOR` |
| DomainPack 真实引擎 | `domain_packs/embodied_world_model/`（run_experiment + evaluator） | 真实，非 mock |

## 4. 审批边界（Approval Boundary）

M4.3 已有完整的带外审批机制，设计正确，可复用：

| 机制 | file:line |
|---|---|
| `ApprovalStore.request/grant/reject/revoke` | `approval_store.py:63-142`（request 只能建 pending；grant 带外、不可变、SHA 重校验） |
| 带外 CLI（独立进程） | `approval_cli.py:1-18, :34-124`（list/show/grant/reject/revoke；grant 重读 plan 当前 SHA） |
| 12 步 fail-closed gate | `research_service.py:651-770` 核心 `:663-692`（status/source/approver/scope/SHA/幂等） |
| ApprovalPolicy | `research_policy.py:23-30`（allowed_sources=(human,), require_approved_by） |
| 只读 gate 适配器 + 防伪造 | `m4_3/research_service_gate.py:88-90,172-182` |
| 信任模型声明 | `APPROVAL_BOUNDARY_M4_3.md:37-43`：`approved_by` 是未经认证的 operator 字符串；无签名/证书/ACL；`source=human` 是 caller-supplied label |

**必须保留的诚实措辞**：out-of-band operator approval、SHA-bound、independent CLI process、fail-closed gate；不得声称密码学/企业身份认证。现有 store 状态仅 `pending/approved/rejected/revoked`，**不支持 EXPIRED**。

## 5. 现有 Benchmark 与证据（已核实）

| Benchmark | 关键实现 | 状态 |
|---|---|---|
| M7.2 Scientific Judgment | `capability_bench/scientific_judgment.py:138-202` | verdict **仅由 novelty_accuracy 决定**（:198-201），其余指标不参与门控；expert_blind_review=NOT_PERFORMED → 实际恒 INCONCLUSIVE |
| M7.3 Memory | `capability_bench/memory_bench.py:95-131` | `repeated_failure_rate` 用模型**预测**（:102-103），未使用 GT 的 `expected_repeat_failure`；4 臂齐全含 EQUAL_TOKEN |
| M7.5 Integrated | `capability_bench/integrated_bench.py:31-39,111-138` | `select_tasks` 每桶 `[:max(1,n//buckets)]` 可返回少于 n（实测 7<8）；`min_families` 用结果行数冒充 unique families（:134）；无 cluster-aware CI；无 holdout |
| M7.4 RSI | `rsi_bench.py:15-32` + `run_m7_4_pilot.py:49` | Candidate 为写死 prompt，无瓶颈诊断证据链；OOD split 记录未执行 |
| 真实 E2E | `run_m7_e2e.py` | 自审批 + 直连 executor + git HEAD 冒充 harness SHA |
| 测试规模 | `research_runtime/tests/` | M7.1–M7.5 累计 797 passed / 12 skipped（见 M7_1_TEST_REPORT.md） |

## 6. 缺陷登记表（Defect Register，均经源码亲自复核）

### P0（阻断生产 E2E，必须先修）

| ID | 严重 | 源码位置 | 当前行为 | 根因 |
|---|---|---|---|---|
| P0-A-1 | P0 | `run_m7_e2e.py:56-62` | 定义 `approve()` 闭包，`approver="human-operator"`、`decision="APPROVED"` | E2E 进程内自审批 |
| P0-A-2 | P0 | `run_m7_e2e.py:61` | 将伪造审批 `freeze_json` 成 `approval.json` 证据 | 伪造审批证据链 |
| P0-A-3 | P0 | `run_m7_e2e.py:67` | `approval=approve` 注入自审批回调 | 生产入口注入自批 |
| P0-A-4 | P0 | `discovery_coordinator.py:201-225` | `_check_approval` 信任 callable/dict，仅黑名单 `approver`，`:203-205` 字符串缓存重放；不查 ApprovalStore、无 source/scope | 自建弱 gate 取代 M4.3 |
| P0-A-5 | P0 | `test_m7_discovery_coordinator.py:121-123` | 正向 E2E 用假审批 lambda 跑通 | 测试掩盖 P0 |
| P0-A-6 | P1 | `approval_store.py` / `schemas.py:32` | 无 EXPIRED 状态；`approved_by` 无认证（M4.3 已声明的信任模型） | 状态机与信任模型限制 |
| P0-B-1 | P0 | `discovery_coordinator.py:111,584,669` | 生产直连 `pack.run_experiment`，生产无 Dispatcher | 执行旁路 |
| P0-B-2 | P0 | `discovery_coordinator.py:580,666` | 合成 campaign_id，无真实 CampaignRecord，reservation 绑伪 ID | campaign 生命周期旁路 |
| P0-B-3 | P0 | `run_m7_e2e.py:64-71` | 未注入 formal_loader/executor；`harness_sha=env or _git_head()` | 生产 driver 无正式链路；git HEAD 冒充 harness SHA |
| P0-B-4 | P1 | `discovery_coordinator.py:873-880` | 通用异常仅标 FAILED，不结算 reservation → bound 孤儿 | 异常路径预算泄漏 |
| P0-B-5 | P1 | `research_service.py:825-826` | `existing is not None` 分支在 `campaign_id` 赋值（:847/:858）前引用它 → NameError | 恢复路径变量顺序缺陷（接线后暴露） |
| P0-B-6 | P1 | `discovery_coordinator.py:257-260` | loader 内已 bind 后再 bind，语义冗余 | 重复绑定（幂等但语义错） |

### P1（科研质量与评测，P0 后修）

| ID | 源码位置 | 缺陷 |
|---|---|---|
| P1-A | `questions.py:63-122` | 逐 abstract 独立生成问题，无多论文综合；缺 RQ 扩展字段 |
| P1-B | `scientific_judgment.py:198-201` | verdict 仅由 novelty 决定；无 preregistered primary/co-primary |
| P1-C | `memory_bench.py:102-103,116,128-131` | repeat_failure 用模型预测率，未用 GT `expected_repeat_failure` |
| P1-D | `integrated_bench.py:31-39,134` | 采样不足额；用行数冒充 families；无 cluster CI；无 holdout |
| P1-E | `rsi_bench.py:15-32`,`run_m7_4_pilot.py:49` | Candidate 写死，无 evidence→bottleneck→candidate；OOD 未执行 |
| P1-F | `novelty/assessor.py:39,90-113,134-139` | Jaccard 直接决定 novelty 状态，与 "retrieval-only" 声明矛盾 |

### 已核实为 OK（不得误改）

- `questions.py:162-194` 真实 likelihood + KL 的 EIG，无默认常数。
- `questions.py:200-232,253-279` evidence 门控与 fail-closed。
- `discovery_coordinator.py:431-433` `NO_VALUE_EVIDENCE_FALLBACK` 是诚实标签，非伪造分数。
- `capability_bench` GT 隔离 + SHA + expert/blind review 标记（NOT_PERFORMED 正确 fail-closed）。
- `hypotheses/engine.py:92-157` IndependentCritic 规则审查，与生成器分离并标注 model identity。

## 7. 计划（最小复用式，不新建 Runtime/Campaign/审批系统）

1. **P0-A**：把 `_check_approval` 改为只读查 `ApprovalStore`（复用 `research_service.py:663-692` 断言），删除 callable/dict 注入；E2E 拆 `--prepare`（冻结协议+建 pending+BLOCKED 停止）与 `--resume`（带外 CLI 审批后重读 SHA 绑定）；所有 engine-free 测试夹具显式标 `TEST_FIXTURE`。
2. **P0-B**：生产路径注入按 `runtime_seam.py:91-95` 装配的真实 `HarnessLoader`；`harness_sha` 从 `resolve_for_campaign` 读取（不再用 git HEAD）；执行改走正式 Dispatcher；修 P0-B-4/B-5/B-6；生产缺 Dispatcher/Campaign/Loader/审批一律 BLOCKED。
3. **P1-A~F**：逐项复用已有组件（`candidate_generation.detect_bottleneck/generate_candidate`、`evaluation._clustered_bootstrap` 等）修复。
4. **P0 全绿后**才跑真实 E2E 与多轮科研验收。

---

**Phase 0 结论**：基线清晰、P0/P1 缺陷全部有源码证据。P0 未修前，M7 的"真实 E2E"不构成生产链路证据；本审计不支持任何 L1/L2/L3 PASS 声明。
