# M6 Acceptance Report — Final

## 0. 结论（结论先行）

**M6 十个门 G1–G10 全部 PASS，系统判定 COMPLETE。**
但附两条明确的统计/工程边界（见 §6），不把工程闭环等同于强统计确认。

- 分支：`feature/arr-m6-harness-rsi`
- 起点：`4db264abf902b53e6ef117eaba2e2188e0914b01`（M5 Acceptance Hardening HEAD）
- 终点：见 §7 commit 列表最后一个 SHA
- SciNet++ 外部引擎 HEAD：`95e65d0a187db33a98ff3b278b53c22d2a432d1d`
- 未 merge main，未建 PR

## 1. 核心 RSI 证据链（用户最看重的一条）

| # | 要求 | 真实证据 |
|---|---|---|
| 1 | 发现上一轮研究方法缺陷 | dev 真实执行后 `falsifiability_rate=0.5`，判 `HYPOTHESIS_UNFALSIFIABLE`；journal `m6_failure_analysis` |
| 2 | 从真实失败找根因 | `classify_failures` + `detect_bottleneck` → 瓶颈组件 `hypothesis_critic`；失败证据与 dev task id 冻结 |
| 3 | 生成新的研究方法 | `model.drift` 无关；候选把 `hypothesis_critic.config.require_falsifying_observation=true`，**只改一个研究方法组件**；candidate SHA `6c1979100dda…` |
| 4 | Base/Candidate 配对实验证明更好 | selection：主指标 `falsifiability_rate` mean_diff=+0.5，Holm 校正 p≈0；判定 ACCEPTED |
| 5 | 独立 holdout 再验证 | holdout task/family 与 selection 完全不重叠；mean_diff=+0.5，Holm p≈0；结构化证据磁盘重读通过 |
| 6 | 正式晋级并被下一轮 Runtime 加载 | 人工批准绑定精确 SHA；journal `harness_loaded` loaded_sha=`6c1979100dda…`；campaign 绑定该版本 |
| 7 | 新研究任务上再次证明改善 | post-adoption 新任务 `falsifiability_rate=1.0`、`objective_quality=0.85`，journal `harness_post_adoption_performance` |

## 2. G1–G10 逐项判定

| Gate | 判定 | 关键证据 |
|---|---|---|
| G1 Benchmark | PASS | 60 任务冻结 manifest；selection/holdout task 与 family 均不重叠；sealed gold 物理分离；`validate_benchmark` 通过 |
| G2 Harness Evolution | PASS | 4 个独立版本化方法组件；完整 artifact 内容 SHA；`assert_single_method_change` 保证一次只改一个 |
| G3 Failure-driven Improvement | PASS | dev 真执行 → 失败分类 → 瓶颈 → 经验 ID → 候选；contamination ledger 证明生成器只见 dev |
| G4 Controlled Evaluation | PASS | 同任务/seed/预算；family 级随机顺序；全结果留存；infra 失败单列（对抗测试注入 timeout 仍保留 6/6 结果） |
| G5 Independent Evaluation | PASS* | 预注册 SHA 绑定；8 指标分报；paired t + bootstrap + cluster bootstrap + permutation + McNemar + Holm + CI + effect size；*permutation 在 n=3 真实 run 下功效不足，见 §6* |
| G6 Verified Promotion | PASS | 禁 legacy bool；task/family 不重叠；per-task result SHA 磁盘重读；parent/candidate SHA 绑定；statistical gate；gate version SHA `138144b4…` |
| G7 Actual Adoption | PASS | 用户显式批准（绑定精确 SHA）；loaded SHA journal；campaign 绑定；stale loader / 错误 loaded SHA 对抗测试通过 |
| G8 Generalization | PASS | 独立 holdout ACCEPTED；post-adoption 新任务再次测量改善 |
| G9 Real Rollback | PASS | 实际从 `6c1979100dda…` 切回 `9d8bae03fb93…`；停调度→校验 parent→切换→smoke→恢复；loaded SHA 真改变 |
| G10 Full Runtime Integration | PASS | 单一 driver 全流程；所有 `m6_*` 事件写入同一 M5 RuntimeState hash-chain；checkpoint `652dd137…`；journal integrity ok |

## 3. 实际 Base/Candidate 结果

来源：`M6_EVIDENCE/run_final2/final_result.json`（确定性研究方法执行，6 对）
与 `M6_EVIDENCE/run_real_v3/final_result.json`（真实三域引擎，3 对）。

| 指标 | selection (6 pairs) | holdout (6 pairs) | 真实引擎 holdout (3 pairs) |
|---|---|---|---|
| 主指标 | falsifiability_rate | falsifiability_rate | falsifiability_rate |
| mean difference | +0.5 | +0.5 | +0.5 |
| paired t p | 0.0 | 0.0 | 0.0 |
| Holm adjusted p | 0.0 | 0.0 | 0.0 |
| paired bootstrap CI95 | [0.5, 0.5] | [0.5, 0.5] | [0.5, 0.5] |
| clustered CI95 | [0.5, 0.5] | [0.5, 0.5] | [0.5, 0.5] |
| permutation p | 0.032 | 0.032 | 0.174（n=3，功效不足） |
| decision | ACCEPTED | ACCEPTED | ACCEPTED（t 门通过） |

真实引擎结果：14 个结果带 `real_engine.engine_Q`（有限值），覆盖 scinet/embodied/quant
三域，Q 范围 0.001474–0.216056；job 目录与配置路径写入结果 artifact。

## 4. Holdout 与 SHA 核验

- benchmark manifest SHA 进入 `dataset_manifest_sha`，verifier 重新读盘复核；
- holdout evaluation task ids / families 与 selection 不相交；
- 每个 holdout 结果的 SHA 来自真实落盘 JSON 文件，artifact_reader 逐个重读；
- candidate SHA `6c197910…` = 冻结 bundle 完整内容 SHA；parent SHA `9d8bae03…` =
  冻结 base；两者在 promotion/adoption/rollback 中一致；
- gate version SHA `138144b4…`；evaluator SHA 与预注册一致。

## 5. Promotion / Adoption / Rollback 记录

- Promotion：`PROMOTED`，`verification.verified=true`，人工批准
  `approved_by=user_explicit_2026-11-16` 且 `reviewed_candidate_sha` 精确匹配；
- Adoption：`loaded_sha=approved_sha=6c197910…`，journal 事件 `harness_adoption_requested`
  → `harness_loaded` → `harness_version_verified` → `harness_campaign_bound`；
- Rollback：`ROLLED_BACK`，from `6c197910…` → to `9d8bae03…`，smoke 通过后才恢复 dispatch。

## 6. 已知边界与未宣称事项

1. **统计功效**：真实引擎 run 每域只有 1 个任务（3 对），permutation p=0.174 无法在 α=0.05
   下达到显著；配对 t + Holm 通过但不应被当成强确认性结论。确定性 6 对 run permutation
   p=0.032。要强确认需按 pilot power analysis 扩大确认性基准（P2，M6 未自动扩样）。
2. **Novelty**：只做快照内重复检测，`scientific_judgment` 明确
   `literature_check=not_performed`，**不声称全局新颖**。
3. **Quant**：合成数据，只验证工程闭环，不构成 alpha 证据。
4. **24h soak / 真人 Confirmatory Blind Review**：M6 未执行，未宣称通过。
5. **研究方法改进范围**：本次真正晋级的是 Hypothesis Critic（可证伪性门）；Objective
   Generator / Experiment Designer / Failure Diagnoser 的独立晋级尚未做（模块已支持，
   需要各自的失败驱动实验）。
6. dirty working tree：evidence 包记录 agent_dev HEAD `595ee7c3`、dirty=true，
   因为真跑产物在同仓库内生成；所有源码 commit 已推送。

## 7. 验证结果

- engine-free 全量回归：**729 passed / 12 skipped / 37 subtests**；
- M6 专项：**29 passed**（含 20+ 强制对抗用例）；
- opt-in 真实引擎证据测试（`M6_RUN_REAL=1 M6_REAL_RUN_DIR=M6_EVIDENCE/run_real_v3`）：
  **4 passed**；
- AST 门禁、CI YAML 校验通过；M6 测试模块已加入根 CI（engine-free 与 opt-in real 分离）。

## 8. Commit 列表

见最终交付消息中精确 SHAs；全部在 `feature/arr-m6-harness-rsi`，未 merge、未 PR。
