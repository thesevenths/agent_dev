# ARR M6.1 — Acceptance Closure 报告

## 0. 判定（结论先行）

本阶段交付 **ENGINEERING_COMPLETE / SCIENTIFIC_CONFIRMATION_INCONCLUSIVE**（以
`M6_1_EVIDENCE/run_final4/final_result.json` 为准，evidence manifest SHA `71b6a53cf4f5eb2c…`）。

含义：所有工程控制链已修复并真实接线（Benchmark V2 三路隔离、真实 Critic 进入生产
Agent 链、正式 Loader 绑定 Campaign、在途版本固定、真实回滚）；但在当前真实配对
数据上，Candidate 未产生可证伪性指标的 measurable improvement，family 样本与功效
也不足以支持"科学改进已确认"。按 M6.1 规则，**不制造显著性**，科学结论保持
INCONCLUSIVE。

## 1. Repository State

- 起点 HEAD：`c078a4e237316676232a04287eb771d020785917`（feature/arr-m6-harness-rsi）
- 终点 HEAD：`c7ee14bd550cfba5a9259ca1827a1e1d26e70b46`（分支 `feature/arr-m6-acceptance-closure`）
- SciNet++：固定 `95e65d0a187db33a98ff3b278b53c22d2a432d1d`（未修改）
- 工作树：源码干净（证据目录为 untracked/随 commit 纳入）
- 未 merge main、未建 PR、未 force push
- CI（起点 HEAD）：Pre-commit success；ARR Research Runtime run `43996143000` 的
  6 个 matrix job 全部 `conclusion=success`（run 级 status 字段陈旧）

## 2. 实际源码调用图（按真实实现，非理想化）

```
run_m6_1_acceptance.py
  └─ RuntimeState.initialize / append / verify_journal        (M5, 复用)
  └─ benchmark_v2.build_benchmark_v2
       ├─ build_tasks_v2 (dev/selection/holdout family 池两两不交)
       ├─ HoldoutAccessVerifier (越权 = HOLDOUT_LEAKAGE)
       └─ validate_benchmark_v2 (三路两两隔离 + task_sha + split manifest SHA)
  └─ PreregistrationV2 (结果未见前冻结, SHA)
  └─ base_harness / derive_candidate (只改 hypothesis_critic)
  └─ real_trajectory.run_real_trajectory
       ├─ arr_runtime.make_harness_generate (生产 planner)
       ├─ LLMProposer (生产 proposer, BASE_HARNESS 控制)
       ├─ build_critic_manager_factory                       ← M6.1 新增注入
       │    └─ HypothesisCritic / _FalsifiabilityGroundedCritic
       ├─ arr_runtime.run_research_task                      (生产, 加了可选注入参数)
       │    └─ ResearchService._review → injected manager.critic
       │    └─ create_plan → approval → run_research_campaign → REAL engine
       │    └─ compare → holdout confirmation → quality evaluator (冻结)
       └─ trajectory.json (35 字段, engine_result_sha 绑定真实引擎结果)
  └─ research_metrics_v2.grade_real_trajectory (独立 grader, 从真实产物算分)
  └─ statistics_v2.analyse_family_confirmation
       └─ family 聚合 → cluster bootstrap → sign-flip permutation → power → 诚实判定
  └─ runtime_seam.verify_runtime_seam                        ← M6.1 工程 seam
       ├─ HarnessLoader.resolve_for_campaign (Registry+approval+activation+disk 全验证)
       ├─ CampaignVersionBinding (在途 campaign 版本不可变)
       └─ rollback_v2.real_rollback (暂停→drain→parent 校验→真实 smoke→恢复)
```

与此前理想图一致，但明确：**M6 旧链中 `method_runner` 的固定 probe 已被真实
`run_research_task` 链取代**；engine_Q 不再是旁路字段，而是 trajectory 中
`engine_result_sha` 指向的真实引擎结果产物。

## 3. P0/P1 问题与修复（源码实证）

| id | 严重 | 根因 | 修复 | 回归证据 |
|---|---|---|---|---|
| M6-01/02 | P0 | dev 用 `fams["all"]` 含 sel+holdout family；只查 sel∩hold | Benchmark V2 三路独立 family 池 + 三路两两隔离校验 | `test_benchmark_family_isolation` |
| M6-03 | P1 | 无 task_sha / split manifest SHA | per-task `task_sha` + 三份 split manifest SHA | `test_benchmark_tamper_detection` |
| M6-04 | P1 | holdout 无 verifier、无访问记录 | `HoldoutAccessVerifier` hash-chain + 目的白名单 | `test_hidden_holdout_quarantine` |
| M6-07/08 | P0 | 固定 probe 算分 ≠ 真实 Agent | `real_trajectory` 走生产 `run_research_task`，Critic 策略注入 | `test_real_hypothesis_critic_strategy_injection` + 真实轨迹 |
| M6-09 | P0 | engine_Q 旁路附加 | trajectory 绑定 `engine_result_sha`，独立 grader 从引擎结果读 verdict | `test_real_trajectory_chain_fields` |
| M6-10/11 | P1 | 统计单位=seed/task | statistics_v2 以 family 为单位 sign-flip + cluster CI + power | `test_cluster_level_statistics`, `test_seed_only_does_not_inflate_family_count` |
| M6-14 | P1 | 经验消融只查注入 ID | 真实 runner ON/OFF（见 §8） | driver ablation 证据 |
| M6-16/18 | P0 | Loader 未接 Dispatcher；在途版本会变 | `HarnessLoader.resolve_for_campaign` + `CampaignVersionBinding` 绑定不可变 | `test_campaign_version_binding_immutable`, `test_concurrent_dispatch_single_binding` |
| M6-17 | P1 | rollback smoke 只 verify_loaded | `rollback_v2` 真实 smoke campaign + 状态机 + 崩溃可恢复 | `test_rollback_failed_keeps_dispatch_paused` |

## 4. Benchmark 隔离

- Benchmark V2：`m6_bench_v2`，三路 family 数 dev=7 / selection=7 / holdout=6；
  task 数约 59；三路 task id 与 family 均不重叠；
- 每个 task 有 `task_id / family_id / objective / task_config / task_sha / split`；
- split manifest 各自 SHA 写入总 manifest；
- 对抗：dev∩sel=∅、dev∩hold=∅、sel∩hold=∅、重复 id/SHA 篡改/holdout 越权均 fail closed。

## 5. 真实 Agent 证据（不拼接）

- 每个 arm 调用真实 `run_research_task`：真实 baseline（train+evaluate）→ 真实
  LLMPlanner 假设 → **真实 Critic 决策** → plan → approval → 真实 campaign → 真实
  引擎结果 → report → quality evaluator；
- Base 与 Candidate 唯一差异：`HypothesisCritic` 策略（base vs
  falsifiability_grounded），二者共享同一 service / planner / engine / budget；
- 真实轨迹 35 字段含 harness/component/prompt/tool/runtime/engine SHA、
  campaign id、critic passed/errors、假设全文、evidence ids；
- 独立 grader 从真实引擎结果与真实 critic 记录计算 falsifiability 等指标；
- 实证：真实配对跑出的 base/candidate 在若干 family 上 falsifiability 均为 1.0
  （Candidate 无 measurable 差异）——如实记录，不伪造改善。

## 6. 统计结果

- 预注册 `PreregistrationV2`（SHA 冻结，结果未见前写盘）：primary=
  falsifiability_rate，方向 maximize，MME=0.05，alpha=0.05，power target=0.8，
  unit=task_family，primary test=family sign-flip permutation，CI=cluster bootstrap；
- 方法：family 内 seed 聚合为一个效应；family 级精确符号翻转（≤16 family 精确枚举）；
  cluster bootstrap CI；effect size；family win/tie/loss；domain 分层；Holm 说明；
- 诚实性验证：`test_statistical_insufficiency_is_inconclusive`（3 family →
  INCONCLUSIVE）、`test_seed_only_does_not_inflate_family_count`（20 seed 同 family
  仍 n=1）、`test_candidate_regression_rejected`、`test_forged_result_sha_detected`；
- 真实结果：`M6_1_EVIDENCE/run_final4/evaluation/statistical_report.json`——
2 个独立 family，base=candidate=1.0（falsifiability 无差异），mean effect=0，
power 不足（所需 family 数为 ∞/不可达），verdict=INCONCLUSIVE。未伪造显著性。

## 7. Runtime Adoption / Rollback（用户第三点）

- `HarnessLoader.resolve_for_campaign`：从 EvolutionRegistry 的 PROMOTED 记录 +
  用户批准（SHA 绑定）+ activation record + 磁盘 artifact 重读，全通过后返回
  `LoadedHarness`；任一失败 FAIL_CLOSED；
- `CampaignVersionBinding`：campaign 首次 bind 后永不改 SHA；运行中二次 bind 抛
  LoaderError；并发 bind 只有一个成功；
- `real_rollback`：DISPATCH_PAUSED → INFLIGHT_DRAINING（在途 campaign 保持原版本）
  → PARENT_LOADING/VERIFICATION → REAL_SMOKE_CAMPAIGN → PARENT_VERIFIED →
  DISPATCH_RESUME；失败时 dispatch 保持暂停，不自动恢复未验证版本；
- `runtime_seam.verify_runtime_seam` 在 M6 真实用户批准产物（SHA
  `6c197910...`）上验证：Loader 拒绝错误 SHA、新 campaign 解析到精确批准 SHA、
  在途 campaign 在 rollback 前后版本 SHA 不变、真实 smoke 使用 parent。

## 8. Tests / Evidence / CI

- engine-free：全量 `751 passed / 12 skipped / 37 subtests`（306.97s，所有改动后）；
- M6.1 专项+对抗：`22 passed`（family 隔离、holdout quarantine、伪造 SHA、
  seed 伪装 family、在途版本固定、并发绑定、budget、journal 篡改等）；
- 真实 E2E：`run_m6_1_acceptance.py`（真实 embodied 引擎配对）overall=
  ENGINEERING_COMPLETE_SCIENTIFIC_INCONCLUSIVE，journal ok；
- 真实 Runtime seam：新 campaign 解析批准 SHA `6c197910…`、在途版本保持、
  fail-closed、真实 smoke rollback（campaign `campaign_02c5c296…`、
  trajectory SHA `f5e93b34…`）到 parent `9d8bae03…`，ROLLBACK_COMPLETED；
- Experience ablation（真实 runner ON/OFF）：on=1.0/off=1.0，delta=0 →
  INCONCLUSIVE；
- Evidence manifest：125 文件，SHA `71b6a53cf4f5eb2c…`；
- CI：M6.1 测试模块已加入 workflow（engine-free），real-engine opt-in 分离。


## 9. Remaining Risks / 未证明事项

- **科学结论 INCONCLUSIVE**：真实 Agent 配对在可证伪性上无 measurable 改善；family
  数与功效不足以确认；未伪造显著性；
- Novelty 无真实文献检索；Quant 为 synthetic research，不声称 alpha；
- 真人 Confirmatory Blind Review 未做；24h soak 未做；
- M6.1 的新候选 SHA（`186b9ab4...`）未经人工批准，因此不进入正式 Adoption；
  Runtime seam 用已真实批准的 M6 SHA 验证工程链。
