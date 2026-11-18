# ARR v1.0 — 01 Source Audit

> 基于当前 checkout 逐行复核，不依赖 README/验收报告/Agent 总结。审计时间 2026-11-18。
> 基线分支 `feature/arr-m7-6-production-runtime-closure` @ `f0de986fb84fa752383c17c4019dad8f28a7113b`。

## 1. 基线核查

| 项 | 实测 |
|---|---|
| Git root | `E:/agent_dev`；工作目录 `E:/agent_dev/multi-agent` |
| 任务书给定基线 SHA | `f0de986f...` |
| 实际本地 HEAD | `f0de986f...`（一致） |
| 远端 `feature/arr-m7-6-production-runtime-closure` | `f0de986f...`（一致） |
| v1.0 分支 | `feature/arr-v1-0-complete-system`（从基线建立） |
| 工作区未提交改动 | `config.py`(+18) / `llm.py`(+6)：用户的 `LLM_EXTRA_BODY` 透传（Qwen3 关 reasoning），与 ARR 无关；已备份到 `E:/agent_dev/_user_uncommitted_backup/config_llm_20261118.patch`，全程保留、不覆盖、不提交 |
| stale HEAD.lock | 中断的 checkout 残留，已清理 |

结论：基线可信；两个 feature 分支祖先关系清晰；用户改动受保护。

## 2. 架构资产（复用，不重写）

| 组件 | 现状 | file:line |
|---|---|---|
| M5 RuntimeState | journal(hash-chain)/reservation/claim_executor/checkpoint/classify 完整 | `runtime_state.py:157-199,252-292,369-503,594-728` |
| M5 ResearchService | 正式 ML ExperimentPlan Dispatcher | `research_service.py:773-923` |
| M5 CampaignStore | campaign 状态+事件单事务（无 atomic claim） | `campaign/campaign_store.py:26-96,104-215` |
| M6 HarnessLoader | resolve_for_campaign + verify_approved + 版本绑定 | `harness_rsi/harness_loader.py:164-223` |
| M6 Promotion/Adoption/Rollback | 完整（PROMOTED/human approval/8 态 rollback） | `promotion.py:82-203`、`adoption.py:59-181`、`rollback_v2.py:58-182` |
| M7 DiscoveryCoordinator | 16 stage 单轮、磁盘状态、resume | `discovery_coordinator.py:47-55,153-201,962-1030` |
| FormalRuntimeAdapter | 生产执行 seam（M5 RuntimeState + M6 Loader + DomainPack） | `discovery/formal_runtime.py:82-156` |
| ClaimEvidenceGraph | append-only、SHA 校验、边/生命周期 | `discovery/knowledge/graph.py` |
| VerifiedResearchMemory | candidate/verified 两级 + promote 验证链 | `discovery/memory/verified.py:28-29,130-231,248-282` |
| candidate_generation | 失败驱动瓶颈诊断 + EvidenceBoundary | `harness_rsi/candidate_generation.py:89-114,129-229` |
| capability_bench | 四臂 OFF/ON/SHUFFLED/EQUAL_TOKEN + 四组 A/B/C/D + family clustered CI | `capability_bench/memory_bench.py:10,91-193`；`integrated_bench.py:10-15,138-258` |
| RQ/假设/Novelty | 多维 fail-closed 评分、竞争假设 Critic、novelty screening-only | `questions.py:16-17,233-361`；`hypotheses/engine.py:18-35,163-182`；`novelty/assessor.py:120-167` |
| CLI | run/benchmark/evolve/report/verify/loop | `cli.py:405-529,694` |
| SciNet++ | 外部科学引擎（未改） | `thesevenths/LLM/SciNet++` |

## 3. 有源码证据的真实缺口

### P0（阻断 L1，必须先修）

| ID | 位置 | 当前行为 | 后果 |
|---|---|---|---|
| P0-1 | `formal_runtime.py:125-128` | 重复 dispatch 无条件重跑 `backend()`；注释 146-149 声称幂等但无 guard | 二次执行非幂等实验、破坏 at-most-once；现有测试 `test_m7_6_production_runtime.py:170-183` 未断言 backend 调用次数（盲点） |
| P0-2 | `discovery_coordinator.py:867-872` | verdict `INCONCLUSIVE` 落入 `else` 写成 `REFUTES`/`REFUTES` 边 | 不确定性被伪造成证伪证据；与 886-888 的 `INCONCLUSIVE` claim 状态矛盾；污染 memory `_claim_gate`（verified.py:182-203） |
| P0-3 | `formal_runtime.py:82-156` | dispatch 内无 per-campaign 原子 claim；无并发互斥 | 两进程同 run_id 可同时执行 backend（claim_executor 仅在 Coordinator 层硬编码） |

### P1（L2/L3/L4 相关）

| ID | 位置 | 缺口 |
|---|---|---|
| P1-1 | `discovery/` 全局 | 无 ResearchProgram 层：DiscoveryCoordinator 单轮即终态；`loop/controller.py:35-174` 多轮但不调用 discovery、不跨轮继承 graph/memory |
| P1-2 | `memory/verified.py:28-29,142-145` | 六类记忆缺 `methodology`/`contradiction`（failed 仅状态、limitation 仅字段） |
| P1-3 | `harness_rsi/` 全局 | 无 method-candidate registry（两个现存 registry 均不匹配 CandidateRecord） |
| P1-4 | `capability_bench/` 全局 | 无外部 benchmark adapter（ScienceAgentBench/FIRE-Bench 等 0 命中） |
| P1-5 | `experiment/designer.py:60-114` | 无实验×假设区分度量化矩阵（仅文本层 review_competition） |
| P1-6 | `approval_gate.py:100-102` | 审批仅绑 protocol SHA，未绑 dataset/method/harness 复合 SHA |
| P1-7 | `formal_runtime.py:140-144` | result 仅 journal 记 summary_sha，无结构化结果对象持久化 |
| P1-8 | `cli.py` | 无 discovery/program 稳定子命令 |

## 4. 已满足、不得返工

- reservation 生命周期、hash-chain journal、确定性 campaign_id、生产 fail-closed、M6 harness SHA pin、at-most-once 诚实声明、审批外部化+protocol SHA 绑定、crash recovery 分类、四臂/四组对照与 family clustered CI、RQ/假设/novelty fail-closed。

## 5. 外部阻塞（记入 Blocked Register）

- 独立人类专家盲评：无外部专家资源 → L2/L4 科学认证阻塞。
- 外部 benchmark（ScienceAgentBench 等）访问/许可未验证 → 不假设可访问、不伪造结果。
- 8 独立 Research Family + ≥3 paired seeds 的算力/数据：现有 embodied pack family 有限 → L4 样本量阻塞。

## 6. 审计结论

L1 工程闭环在 M7.6 已建立，但 P0-1/P0-2/P0-3 是可复现的真实缺陷，必须先修并通过负向测试，才允许真实多轮执行。P1 缺口按依赖顺序补。外部阻塞如实记录，不做假。
