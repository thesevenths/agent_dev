# M5-00.1 + M5-01/02 Acceptance Report

**结论先行**：Part A/B/C/D/E 的全部代码改动已按逻辑 commit 推送到 `feature/arr-0.1-domain-registry`。三域真实 Smoke 全部通过：scinet / embodied / quant 各自完成真实 Campaign、外部审批、原 12-step Gate 和 Experience；跨域下一轮 Discovery 产出 3 个新 Proposed Objectives。未 merge main，未创建 PR，未使用 FakeRunner，未自动授予 Human Approval，未把 Gate Pass 当作 Scientific Finding。

## 1. 提交清单

| 范围 | Commit |
|---|---|
| P0-01 持久化预算 | `39f0d008` |
| P0-02/03/04 真实 Recovery / 自动接线 / 不可覆盖 Round Evidence | `e2a24332` |
| P0-05 Crash Injection | `91e7095f` |
| M5-01 跨域薄模块 | `9060c1f3` |
| M5-01 engine-free 测试 | `da9a0cf5` |
| M5-02 RRSI 薄模块 | `6181aa73` |
| M5-02 engine-free 测试 | `ea1c2367` |
| M5-01/02 真实 Smoke driver + opt-in 测试 + CI | `3d7d4bed` |

## 2. Part D 十项证明

| # | 要求 | 证据 |
|---|---|---|
| 1 | 三个 Domain Adapter 可注册/查询 | `test_three_domain_adapters_register_and_preflight`：embodied / quant / scinet |
| 2 | Objective 正确路由 | `test_objective_routes_to_owning_domain_and_capability_rejects` |
| 3 | capability 不符拒绝执行 | 同上；`DomainCapabilityError` |
| 4 | 每域 Budget 独立 | `test_domain_budgets_are_independent`；Smoke 三域各 consumed=1 |
| 5 | 域失败隔离 | `test_one_domain_failure_does_not_abort_others`；`run_objectives` 逐域 try/except |
| 6 | 跨域 Experience 可读且保留来源 | `test_cross_domain_experience_keeps_source_and_run`；Smoke 3 行 Experience 带 `source_domain` |
| 7 | Domain Model 与 Harness 不混用实验 | `test_experiment_id_cannot_be_reused_across_layers`；`assert_single_layer` |
| 8 | 两层独立版本和 Evidence | registry 按 layer 记录；`parent_for(layer)` 按层继承 |
| 9 | held-out 不参与训练/候选选择 | `test_holdout_never_reaches_generation...`；generator 无 holdout 参数 |
| 10 | 只有 Gate 通过才晋级 | `test_gate_never_accepts_training_only_or_frozen_test`；PROMOTED 需 ACCEPTED + holdout |

## 3. 三域真实 Smoke 证据

| 域 | Campaign | Report | Experience | Grounding |
|---|---|---|---|---|
| scinet | `campaign_07cf3e22efff5f17` | `report_40d85e60e496` | `exp_0d005cd6834f` | 真实 `cand_H_HORIZON_LATENT_MSE_seed_43` |
| embodied | `campaign_fcb6cc156d6859d2` | `report_6c8178069472` | `exp_0ed5557e940c` | 真实本地 train+evaluate `m5s2_src_embodied` |
| quant | `campaign_8b247041db585098` | `report_c05cd8405281` | `exp_6c00a94b9aa4` | 真实本地 train+evaluate `m5s2_src_quant`；synthetic data 明示 |

- 结果文件：`M5_01_02_runs/e2e/m5_01_02_result.json`，SHA256 `c2d36993a21ef5d9612a1e4e305ba9c4321c0ef0424628b248540e0504285153`。
- 三个 Director 状态均为 `COMPLETED`；Report 保留原 12-step Gate critic reviews；outcome 均为 `inconclusive`，`promotion=not_requested`，`frozen_test_used=false`。
- 跨域 Snapshot：3 objectives / 3 sessions / `experience_available=true` / `unresolved_gaps=[]`。
- 下一轮 Discovery 候选：`obj_disc_cand_experience_04f886607910`、`obj_disc_cand_experience_33837b07bd12`、`obj_disc_cand_experience_575be2715ad3`。
- 全局 append-only `evidence.jsonl` 因真实 Smoke 新事件 SHA 前进到 `1efebb02543c9ce708b15a4e35d81578c240084787d76bcb9b4`；历史记录未改写，旧产物 SHA 保持不变。

## 4. 测试与 CI

| 检查 | 结果 |
|---|---|
| engine-free 全量 `pytest research_runtime/tests/` | **668 passed / 5 skipped / 37 subtests passed** |
| opt-in `M5_01_02_RUN_REAL=1 test_m5_01_02_real_engine.py` | 1 passed |
| opt-in `M5_00_RUN_REAL=1 test_m5_00_real_engine.py` | 1 passed（M5-00 历史结果保留） |
| 语法门禁 `precommit_checks.py ast` | 通过 |
| CI YAML 解析 | 通过；新模块与 opt-in 测试已加入根 workflow |

CI workflow：<https://github.com/thesevenths/agent_dev/actions/workflows/arr-research-runtime.yml>。当前交付 HEAD push 后的具体 run 链接在最终交付消息中给出。

## 5. 未完成项与风险

- **真实 RRSI 晋级 E2E 未执行**：本阶段 RRSI 只有 engine-free 契约证明 + 三域真实 Smoke；Domain Model / Harness 的真实 held-out 晋级需下一步预算。
- **Confirmatory 仍未启动**：Pilot 真人盲评 0 提交、双 Primary Metric `evidence_insufficient`，前置条件不满足。
- quant 数据为 synthetic；只能用于流程与引擎验证，不能作为真实市场科学结论。
- 本地域 Smoke 的源证据由各自真实 engine 的 train+evaluate 生成，避免跨领域科学事实借用；以后接入真实外部数据需重跑 grounding。
- M3.x `UnifiedResearchScheduler` 仍保留用于既有 dual-layer 实验；M5-01 新 scheduler 是独立薄模块，未删除历史产物。
