# M6 Acceptance Gates（G1–G10）

所有门以**实际运行产物和源码证据**判定，不以文档或 import 成功判定。任何一项未通过
必须在最终报告中标注 BLOCKED / INCONCLUSIVE，不得宣称 M6 COMPLETE。

| Gate | 名称 | 必须提供的证据 | 代码/产物位置 |
|---|---|---|---|
| G1 | Benchmark | 60 任务冻结 manifest SHA；family 不重叠；sealed gold 与 generator_view 分离 | `benchmark/benchmark_manifest.json`、`benchmark/sealed/gold_rubric.json`、`test_benchmark_family_disjoint_and_sealed` |
| G2 | Harness Evolution | 4 个独立版本化方法组件；bundle 完整 SHA；只改一个组件 | `harness.py`、`test_bundle_sha_covers_complete_artifact`、`test_failure_driven_generation_single_component` |
| G3 | Failure-driven Improvement | 真实 dev 执行 → 失败分类 → 瓶颈 → 经验来源 → 候选 | `run_final2/dev_base/*`、`contamination.json`、`test_failure_driven_generation_single_component` |
| G4 | Controlled Evaluation | 同任务/seed/预算；随机顺序；全结果留存；infra 分离 | `paired_selection/paired_trajectory.json`、`test_paired_execution_retains_all_outcomes`、`test_adv_crash_during_paired_evaluation` |
| G5 | Independent Evaluation | 预注册 SHA；8 指标；配对/聚类/校正统计；evaluator SHA 绑定 | `preregistration.json`、`independent_evaluation.json`、`test_independent_evaluation_detects_real_improvement` |
| G6 | Verified Promotion | 结构化 holdout；磁盘重读；task/family 不重叠；禁 legacy bool；statistical gate | `promotion_verdict.json`、`test_adv_*`（20 项）、`run_formal_harness_promotion` |
| G7 | Actual Adoption | 精确 SHA 加载；journal loaded SHA；campaign 绑定；人工批准绑定 SHA | `adoption.json`、`human_approval.json`、journal `harness_loaded`/`harness_version_verified` |
| G8 | Generalization | 独立 holdout 通过；post-adoption 新任务再次测量 | `holdout_*`、`post_adoption/*`、`final_result.json` |
| G9 | Real Rollback | 停调度 → 校验 parent → 切换 → smoke → 恢复；实际 loaded SHA 改变 | journal `harness_dispatch_stopped`/`harness_version_switched`、`final_result.rollback` |
| G10 | Full Runtime Integration | 单一 driver 全流程；同一 RuntimeState journal；checkpoint；恢复 | `run_m6_harness_rsi.py`、`runtime.sqlite`、`final_result.json` |

## 人工批准规则

- 唯一人工关卡在 G6 之后。driver 自带 approval template（`approved:false`），**never**
  自行写批准文件；
- 批准必须包含 `approved:true` 且 `reviewed_candidate_sha` 精确等于被晋级候选 SHA；
- 缺批准 → promotion 保持 INCONCLUSIVE，G7/G8/G9 标为 BLOCKED_PENDING_HUMAN_APPROVAL。

## 对抗用例清单（强制）

holdout leakage / empty selection / empty holdout / candidate modified after freeze /
result modified after evaluation / wrong parent SHA / candidate SHA mismatch / reused
holdout / candidate self-grading / budget mismatch / missing evaluator / missing artifact
reader / partial task execution / crash during paired evaluation / crash during promotion /
crash during adoption / concurrent promotion / stale loader / failed rollback / incorrect
loaded SHA / missing human approval / legacy boolean path / task family overlap。

均在 `research_runtime/tests/test_m6_harness_rsi.py` 中实现为可执行测试。

## 通过标准

G1–G10 全绿才允许 `M6_ACCEPTANCE_REPORT.md` 写 COMPLETE。任何一项失败需写明：
现象、根因、源码证据、是否已修、重跑结果。
