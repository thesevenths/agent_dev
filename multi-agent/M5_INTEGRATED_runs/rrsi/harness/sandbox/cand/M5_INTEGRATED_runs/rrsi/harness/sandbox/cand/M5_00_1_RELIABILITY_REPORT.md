# M5-00.1 Reliability Hardening Report

**结论先行**：M5-00.1 的 5 项可靠性加固已全部落地并推送。预算从进程内存改为 SQLite 持久化账本并在 Campaign 执行前 reservation；Recovery 从“round JSON 有 ID”升级为逐 ID 真实 Store 对账；`run()` 启动即 fail-closed 对账；Round Evidence 改为 O_EXCL 不可覆盖；10 个崩溃注入场景证明 **at-most-once + resume-in-place**。没有宣称 exactly-once。

分支：`feature/arr-0.1-domain-registry`。未 merge main，未创建 PR。

| 项 | Commit | 状态 |
|---|---|---|
| P0-01 持久化预算 | `39f0d008` | 完成 |
| P0-02/03/04 真实对账、自动恢复、不可覆盖证据 | `e2a24332` | 完成 |
| P0-05 Crash Injection | `91e7095f` | 完成 |

## 1. P0-01 持久化 Budget Ledger

新增 `research_runtime/loop_budget_ledger.py`，两张表：
- `loop_budget_state`：每个 loop 的 `config_sha`、limits、持久化的 `started_epoch`、`reserved`、`consumed`、`status`；
- `loop_budget_entries`：`PRIMARY KEY(loop_id, campaign_id)`，保证同一 Campaign 无法重复扣费。

语义：`reserved` 为已执行前预占但未终结，`consumed` 为完成/失败/中断后的实际消耗，`remaining = limit - reserved - consumed`。Reservation 使用 `ResearchService.make_campaign_id(plan_id, approval_id, plan_sha)` 在 `director.run_to_completion()` 之前写入；执行后若真实 campaign_id 漂移则 fail-closed。

- 墙钟预算读取持久化的 `started_epoch`，进程重启不重置。
- `LoopConfig` 与 `LoopRecord.config_sha` 双向绑定；恢复时修改预算或实验配置直接拒绝。
- 预算耗尽 / 人工 stop 写入 ledger status 并安全停止。
- 单测：`test_m5_00_1_budget.py`（4 项）覆盖重启后消耗不归还、同 campaign 幂等、SHA 绑定、墙钟不重置。

## 2. P0-02/P0-03 真实 Recovery 对账与自动接线

新增 `research_runtime/loop_reality_reconciler.py`。round JSON 只是 claim；每个 ID 必须真实解析：

| 层 | 验证 |
|---|---|
| Objective | `ObjectiveStore.get`，SHA/内容存在 |
| Hypothesis | `HypothesisStore.get` + `latest_review.passed` |
| Plan | `ExperimentPlanStore.get` + `plan_sha256` |
| Approval | `ApprovalStore.get`，status=approved，resource_id/resource_sha256 与 plan 一致 |
| Campaign | `CampaignStore.load_campaign`，status=completed，binding 三元组一致，无 failed trials |
| Report | `ResearchReport.report_sha256` 重算一致，frozen_test_used=false，promotion=not_requested，有 critic review |
| Experience | `report_has_real_experience` + Experience Store 真实行 |
| Director | `<state_dir>/round_NN/<director_id>.json` 状态 COMPLETED，journal 末事件 FINALIZE |
| Budget | reservation 已 consume，未重复扣费 |

分类：`ROUND_COMPLETED` / `AWAITING_APPROVAL` / `CAMPAIGN_IN_FLIGHT` / `EXPERIENCE_PENDING` / `RECOVERY_REQUIRED` / `CORRUPT`。`run()` 启动先 `recover(apply=False)`；`CORRUPT` 或不可判定状态 fail-closed，等待人工。只有全部真实产物解析成功才恢复为 completed 并继续下一轮。Recovery 全程只读真实 Store，不创建 Campaign、不 grant approval、不伪造 Experience。

Experience 已写入但 Round JSON 丢失时，`_rebuild_completed_round()` 仅从真实 Director ctx + Store 重建候选记录，并再次通过同一 reconciler 才能写回；否则 fail-closed。

## 3. P0-04 Round Evidence 不可覆盖

`_write_round()` 改为 `os.open(O_CREAT|O_EXCL|O_WRONLY)`：
- 内容 SHA 排除 `written_at`，只覆盖证据 payload；
- 同 round 同 SHA：幂等复用，不写第二份；
- 同 round 不同 SHA：`ResearchStoreConflict`，绝不用 `tmp.replace()` 覆盖；
- 历史 M5-00 round JSON 仍可读（缺失 evidence_sha 时按内容重算）。

## 4. P0-05 Crash Injection 证据

`test_m5_00_1_crash_injection.py` 8 个测试覆盖 10 个要求：

| 场景 | 验证 |
|---|---|
| Objective 后崩溃 | 分类非 completed，Campaign 列表为空 |
| Plan 后崩溃 | plan SHA 存在但无 Campaign |
| Approval 后、Campaign 前 | `AWAITING_APPROVAL`，无 Campaign |
| Campaign 执行中崩溃 | `CAMPAIGN_IN_FLIGHT`，不判 completed |
| Report 写入、Experience 未确认 | `EXPERIENCE_PENDING` |
| Experience 写入、Round JSON 丢失 | `_rebuild_completed_round` 从真实 Store 恢复，无重跑 |
| Round JSON 写入、LoopRecord 未更新 | 从不可变文件同步，无重跑 |
| 恢复后再次启动 | 新实例只从 SQLite + 文件恢复，campaign ids 不变 |
| 预算不重置 | `consumed=1, reserved=0, remaining=1` |
| 已完成 Campaign 不重复 | 重启后 `_run_round` 被调用次数为 0 |

**可证明结论**：at-most-once（一个 campaign_id 至多执行一次、预算至多扣一次）+ resume-in-place。不可证明且不宣称 exactly-once。

## 5. 回归

- engine-free 全量：668 passed / 5 skipped / 37 subtests（含本阶段新增测试）。
- AST 语法门禁：通过。
- CI 新模块已接入根 workflow（见 `M5_01_02_ACCEPTANCE_REPORT.md`）。
