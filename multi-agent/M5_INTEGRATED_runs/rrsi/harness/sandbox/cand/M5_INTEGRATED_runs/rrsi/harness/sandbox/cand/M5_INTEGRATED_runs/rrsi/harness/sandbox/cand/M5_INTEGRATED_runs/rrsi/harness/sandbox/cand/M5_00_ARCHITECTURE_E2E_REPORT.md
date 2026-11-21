# M5-00 Architecture / E2E Report

**结论先行**：M5-00 单域 Autonomous Research Loop 已完成，并在 SciNet++ 真实引擎上通过两轮闭环验收。两轮都由真实 Discovery → ResearchDirector → 外部 CLI 审批 → 真实 Campaign → 原 12-step Gate → Experience 串成；Round-2 的 objective 明确由 Round-1 Experience 生成（`generation_source=historical_experience`，`prior_experience_ref=exp_edd967b5c112`）。事件链完整（`integrity_ok=true`），历史产物未被覆盖。

分支：`feature/arr-0.1-domain-registry`。未 merge main，未创建 PR。Part B Confirmatory 仍未启动；M5-01/M5-02 未启动。

---

## 1. 架构：薄编排，不重写引擎

| 层 | 组件 | 本次状态 |
|---|---|---|
| 真实证据输入 | `DiscoveryContextProvider`（真实 `run_id=arr_m34_001`） | 只读复用，未改 |
| 选题 | `ObjectiveDiscovery` / `ObjectiveCandidateGenerator` / `ObjectiveCritic` / `ObjectiveRanker` | 直接调用，未重写 |
| 假设/计划/审批/执行/收尾 | `ResearchDirector` | 直接调用，未重写 |
| 审批 | `research_runtime.research_store.approval_cli` 子进程 | 外部授权；Loop 自身从不 grant |
| 执行引擎 | SciNet++ 真实 paired seeds `(42, 43)` | `resolve_engine()` 预检通过 |
| 判定 | 原 12-step Gate + 决策引擎 | 每轮 `COMPLETED`，未改 |
| 循环记账 | `autonomous_loop_state.py`（hash-chained append-only）、`autonomous_loop_recovery.py` | M5-00 新增，仅记账/恢复 |
| 编排 | `autonomous_research_loop.py` | M5-00 新增薄编排层 |

执行流：

```
DiscoverySnapshot(真实 run_id，含 prior objective_ids / Experience)
  -> ObjectiveDiscovery -> Proposed Objective
  -> ResearchDirector -> Hypothesis -> Plan -> Approval request
  -> 外部 approval_cli grant（SHA-bound）
  -> ResearchDirector -> Campaign -> Evaluate -> 12-step Gate -> Report -> Experience
  -> 下一轮 Snapshot 读取该 Experience/Objective（本次完成 2 轮）
```

## 2. 过程中发现并修复的两个真实缺陷

| 缺陷 | 根因 | 修复 | Commit |
|---|---|---|---|
| Round-2 Discovery 报 `ResearchStoreConflict` | 同一确定性 `candidate_id` 在新快照中带了不同的上下文 `source_objective_ids`；`ObjectiveStore` 正确地拒绝覆盖历史 | 仅在薄编排层增加 loop-scoped adapter：同一 `candidate_id` + 同一 `generation_source` 且该 objective 已由此 loop/run 提出时复用历史记录；其它冲突仍 fail-closed。未改 Discovery/Store 通用语义 | `1fca9ab7` |
| 事件日志轮次错位 | 事件写入发生在 `LoopRecord.round_no` 更新之前，Round-1 事件被标为 0 | 用 `self._current_round` 标记正在执行的轮次；create/preflight 仍为 0，后续事件与 payload 中的 round_no 对齐 | `69a038a7` |

框架初版提交：`3db5ee05`；E2E driver / opt-in 测试 / CI 门禁：`6a5f7cc5`。

## 3. 最终真实 E2E（干净目录重跑）

| 项 | 值 |
|---|---|
| `loop_id` / `run_id` | `m5_e2e_loop` / `arr_m34_001` |
| 源实验 / paired seeds | `cand_H_HORIZON_LATENT_MSE_seed_43` / `(42, 43)` |
| 状态 / 轮数 | `LOOP_COMPLETED` / 2 |
| 耗时 | 472.54 s |
| 事件链完整性 | `true`（21 条事件） |
| 结果文件 SHA256 | `561f00b9edb1d77c3baac4eb21cd81a33a42d2ee70c8145c12c30b6c0042ea49` |

| Round | Objective（generation_source） | Plan / Approval | Campaign | Report | Experience | prior_experience_ref | Gate |
|---|---|---|---|---|---|---|---|
| 1 | `obj_disc_cand_ood_exploration_21403ca260ca`（`id_ood_exploration`） | `plan_m5_round_1_m5_e2e_loop` / `approval_60c6ebc1c4f3` | `campaign_11e65ad60417552d` | `report_3658da99b36e` | `exp_edd967b5c112` | — | `COMPLETED` / `gate_passed=true` |
| 2 | `obj_disc_cand_experience_ada4a2caac17`（`historical_experience`） | `plan_m5_round_2_m5_e2e_loop` / `approval_6dbbb55528e3` | `campaign_c4453645c5285bcd` | `report_af1a85551d4a` | `exp_c059c853924f` | `exp_edd967b5c112` | `COMPLETED` / `gate_passed=true` |

两份 Report 的真实 outcome 都是 `inconclusive`，`frozen_test_used=false`，`promotion=not_requested`，各有 2 条 `deterministic_critic` review；没有伪造成功结论，也没有触碰 frozen final split。

Round-2 另有 13 个此前已提出的 objective 被判定为 `reused_prior`，其历史内容 SHA 未被改写；新增的 Experience-based objective 正常落库。

## 4. 证据链与完整性

- 落盘文件：`M5_00_runs/e2e/m5_e2e_result.json`、`M5_00_runs/e2e/rounds/round_01.json`（SHA `981a82c6…`）、`round_02.json`（SHA `a9ef385f…`）、`M5_00_runs/e2e/research.sqlite`。
- DB 计数：21 条 hash-chained loop events、15 个 objectives、2 个 research sessions、2 条 Experience、2 份 Report、2 个 approval。
- 事件轮次：`create/preflight=0`；Round-1 全部 9 条事件为 1；Round-2 全部 9 条事件为 2；`complete_loop=2`，与 payload 对齐。
- 审批边界：两个 approval 都由独立 `approval_cli` 子进程写入，`source=human`、`approved_by=m5_e2e_operator`；Loop 只观察已发生的授权。
- 幂等性：同一 loop 已完成轮次不会重跑；失败/中断记录保留；恢复只做 reconcile，不新建 Campaign、不 grant approval。

## 5. 验收与回归

| 检查 | 结果 |
|---|---|
| `M5_00_RUN_REAL=1 pytest research_runtime/tests/test_m5_00_real_engine.py` | 1 passed |
| `pytest research_runtime/tests/ --basetemp=.pytest_bt_m5_final` | **643 passed, 4 skipped, 37 subtests passed** |
| `python tools/precommit_checks.py ast` | 通过（无输出） |
| CI workflow YAML 解析 | 通过；新增 `test_m5_00_loop_state.py` 与 opt-in `test_m5_00_real_engine.py` |
| opt-in 语义 | 4 个 skip 均为真实引擎/CI 显式 opt-in，未计为通过 |

说明：一次完整回归曾因环境 safe-delete 钩子清理旧 pytest 临时目录而中断（非断言失败）；改用独立 `--basetemp` 后得到上表干净结果。

## 6. 边界与未完成项

- **Part B Confirmatory 未启动**：Closure 条件仍不满足——11 条真人盲评目标 0 条提交，两个 Primary Metric 样本量均为 `evidence_insufficient`，协议冻结/真人审批前置不成立。详见 `M4_3_02_1_CLOSURE_REPORT.md` §5。
- **M5-01 跨域调度、M5-02 双层 RRSI 演化未启动**。
- 历史 Campaign / Report / Experience / Verdict 未修改；新 E2E 以 append-only 方式向全局 `evidence.jsonl` 追加真实事件，文件级 SHA 前进到 `1f9ca4f4…`，已在 Closure Report §7 注明。
- 运行产物保留在本地 `M5_00_runs/`，不删除失败/中断记录。
