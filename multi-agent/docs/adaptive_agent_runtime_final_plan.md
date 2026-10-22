# Adaptive Agent Runtime — 最终实施方案（V3 / Final）

> **项目**：`E:\agent_dev\multi-agent`
> **文档性质**：可开工、可验收的实施规格（Implementation Spec），合并三份前序文档并修正其结构性缺陷
> **前序文档**
> - `docs/adaptive_agent_runtime_rebuild_plan.md`（ChatGPT V1，研究议程）
> - `docs/adaptive_agent_runtime_v2_implementation_spec.md`（ChatGPT V2，已可开工）
> - `docs/reviews/review-01-runtime-rebuild.md` / `docs/reviews/review-02-implementation-spec.md`（评审与替换规格）
> **本文档地位**：与代码冲突时，以本文档为准；与真实代码冲突时，以代码为准并回改本文档
> **状态**：设计定稿，未改动任何生产代码
> **日期**：2026-10-22

---

## 0. 这份文档解决什么问题

V2 已经解决了「要不要做、大致怎么做」，但有三类东西仍然缺一：

| 缺口 | 表现 | 本文档的处理 |
|---|---|---|
| **判据缺数字** | `C_max / R_max / δ / β0 / β1 / n` 全是占位符 | 全部给出取值、来源与反推公式（§8、§9.3、附录 A） |
| **尺子未验证** | 11 条验收全是「能跑通」 | 增加**两个阴性对照**作为唯一强制通过项（§16） |
| **与仓库脱节** | 目录树里的文件不存在；冻结按目录做，一定漏 | 改为真实文件/符号映射（§2），冻结改为**符号级 + 双 hash**（§11） |

一句话概括本文档的立场：

> **这套系统的技术含量不在「让 Agent 改自己」，而在「证明改动真的有效」。**
> 前者的工程量约占 30%，后者占 70%，且必须**先做后者**。

### 0.1 相对 V2 的 12 处实质变更

| # | V2 的做法 | 本文档改为 | 理由 |
|---|---|---|---|
| 1 | 主目标是 `Verifier / Task Quality`（隐含 LLM 打分） | 主目标改为 **L1 确定性合成分 Q**，LLM rubric 降为次要证据与 tie-break | LLM 打分 sd≈12 时需 71 配对才能检出 +4 分，成本不可行（§8.2） |
| 2 | `ΔQ 或 ΔB` 过门即可 | 去掉「或」。ΔB **只作归因证据，不进 Gate** | 加法型行为指标极易刷分（多列三个来源即「改进」） |
| 3 | 目录级 `FROZEN_PATHS` | **符号级冻结 + 文件 hash + 解析后 env hash** | 真正的评分逻辑在 `agents.py` 与 `.env`，不在任何独立目录 |
| 4 | 未定义 `Current Harness` 与 `Base Harness` 关系 | 定义为 **Base / Task / Incumbent** 三层 + 人工 promote 门（§4） | 二者混用必然退回「无限累积」或「改进永不生效」 |
| 5 | `n=5` 重复 | `n` 由 **power 反推**，并先用 L1 把 sd 压下来（§8.3） | 固定 5 次在 sd=12 时 δ≈15，等于测不出任何东西 |
| 6 | 单轮多候选（2 个起） | **单轮单候选**（P0/P1），预算集中换统计功效（§14） | 一次完整评估已是百万 token 级，多候选等于每个都测不准 |
| 7 | `agents/ runtime/` 目录树 | 只新增 `adaptive_runtime/` 一个包，**现有文件一行不改**（§18） | 6 个 node 共享 gate/handoff 逻辑，拆文件会引入 import 环 |
| 8 | 12 类 taxonomy | 先启用 **5 类可判定**，其余记 `UNKNOWN` 并监控占比（§12） | 现有日志只有 3 个 `kind=`，9 类无字段可判 |
| 9 | Trace 从零建 | **旁路解析 `runlog`** 已有事件，不改现有埋点（§6） | `[step-failure] kind= agent= WHY=` 已经在写 |
| 10 | 无 token 计量落点、无预算 | 明确挂 `llm.py:173 wrap_model_call`，并设 **单轮 ≤ 8M token** 硬上限（§14） | 目前全仓库**零 token 统计**，成本规则无法计算 |
| 11 | Day2 冻结 Evaluator | Day4 通过阴性对照后才冻结（§17） | 冻结一把没验证过的尺子 = 把错误固化 |
| 12 | `§35 成功/失败轨迹对比` 直接做 | P1 做**跨任务分布对比**（弱化版），P2 再用同任务多跑做严格版（§13.5） | 95 条 run 是 95 个不同问题，不存在同任务成功/失败对 |

---

## 1. 结论先行（五条）

1. **先造尺，再造流水线。** 在 `evaluator` + `calibration` + 两个阴性对照通过之前，不写 `proposer`。这是本文档唯一的硬顺序约束。
2. **主目标必须是确定性的。** 研报类产物没有 ground truth，但有**可验证的结构事实**（gate 是否通过、产物是否落盘、数字是否带来源、是否被截断）。把 Q 定义在这些事实上，`sd(Δ)` 可以降到 2–4 分，配对样本量随之从 71 降到 8。
3. **ΔB 永远不进 Gate。** 行为指标的职责是解释「为什么可能有效」，不是决定「是否准入」。一旦它有了准入权，闭环第一件事就是学会堆来源。
4. **单轮单候选。** 一次完整评估 = `n` 对 × 2 臂 × 单次 run 成本，已经是百万 token。把预算摊给 2 个候选，等于两个都测不准。RRSI 单轮 40 候选的前提是存在可判定的仿真器，本仓库不具备该前提。
5. **接受 evolve 分数可能下降。** 如果验收标准写成「这版比上版好」，系统会必然优化成「在这个任务上更好」。P0 的验收标准是「尺子准」，不是「分数高」。

---

## 2. 系统现状（据真实代码核对，2026-10-22）

> 本节替代 V2 §5/§6 的错误映射。**任何按 V2 目录树开工的行为都应被阻止。**

### 2.1 真实文件与职责

| 文件 | 真实职责 | 在本文档中的角色 |
|---|---|---|
| `plan.py` | replan 判定；`_REPLAN_NOOP_MAX=2` 等 8 个阈值 | 结构层演化对象（P2） |
| `planutil.py` | `members` / `options` 成员枚举、`_normalize_plan`、结构化重试 | **冻结**（拓扑即评分口径） |
| `supervisor.py` | 规划、路由、早停、失败仲裁、replan | 被测 Runtime（不改） |
| `agents.py` | `create_nodes()`（L918）+ 6 个 node；`_rule_gate`(L209) / `_critic_gate`(L281) / `_action_contract_gate`(L182) / `_is_deliverable_step`(L113) | 被测 Runtime + **评分实现（冻结）** |
| `graph.py` | `build_graph_with_memory()`(L108) 拓扑装配 | 被测 Runtime（不改） |
| `runlog.py` | `log_event()`(L130) / `timing_summary()`(L182) | **Trace 数据源（只读）** |
| `llm.py` | `CustomContextMiddleware.wrap_model_call`(L173) | **Token 计量唯一落点** |
| `prompt.py` | 系统提示与角色提示 | L2 演化对象 |
| `config.py` | 仅 LLM/路径配置；**大部分参数在 `.env` 与模块常量** | 不是参数中心，不要当参数中心用 |
| `state.py` / `context.py` / `summary.py` / `compress.py` / `memory/` / `longterm.py` | 状态、上下文压缩、长期记忆 | P2 之后才允许触及 |

**结论：不存在** `agents/planner.py`、`agents/executor.py`、`agents/reviewer.py`、`runtime/orchestrator.py`。
「Reviewer」不是一个节点，而是**每一步内部的三道 gate**。这一点决定了「改 workflow」在本仓库意味着改 `agents.py`（高风险），所以在 P0/P1 必须排除。

### 2.2 评分逻辑的真实位置（冻结清单的由来）

| 冻结对象 | 位置 | 为什么属于「评分」而非「harness」 |
|---|---|---|
| `_rule_gate` / `_critic_gate` / `_action_contract_gate` | `agents.py` | 决定「这一步算不算完成」 |
| `_is_deliverable_step` + `_DELIVERABLE_VERBS` + `_PATH_EXTS` | `agents.py` | 决定「什么算交付物」 |
| `_CRITIC_MIN_LEN` | `agents.py:96`（env `AGENT_CRITIC_MIN_LEN=4`） | 判定阈值 |
| `_ACTION_PROBES_ENV` | `agents.py:159` | 动作契约词表 |
| `_STEP_FAILURES_MAX` | `agents.py:323`（=20） | 何时判整体失败 |
| `members` / `options` | `planutil.py:26/30` | 拓扑可选项 = 搜索空间本身 |
| `log_event` / `timing_summary` | `runlog.py` | trace 完整性 |

### 2.3 历史证据（可用于冷启动的资产）

| 事实 | 数值 | 用途 |
|---|---|---|
| 历史 run 日志 | `log/*.log` 共 95 条 | 冷启动诊断语料 |
| `[step-failure]` 事件 | 36 次 | 失败模式分布 |
| 其中 `kind=recursion` | 28 次（78%） | 主失败模式 |
| `kind=truncated` / `exception` | 7 / 1 | 次失败模式 |
| 失败涉及 `agent=CodeAgent` | 35 次（97%） | 失败高度集中 |
| `AGENT_MAX_ITERATIONS_CODE` 现值 | **60**（全局 `AGENT_MAX_ITERATIONS=15` 的 4 倍） | 即使已是 4 倍仍 28 次 recursion → **瓶颈不在此参数** |
| token 统计 | **不存在**（`runlog`/`llm.py` 均无 `usage_metadata`） | Day 1 必须补 |
| 原始提问可还原 | 日志含 `[longterm] recalled ... for query '<原文>'` 与 `[supervisor] ... goal='<目标>'` | FRS 冻结回放集可直接从历史构建 |

**两个直接可用的测试用例**

- `AGENT_CRITIC_MIN_LEN=4` —— 看着最像「质量旋钮」，实际是 4 个字符，**等于没有门槛**。`agents.py` 的注释写明它「只是空泛非答复的下限，不是质量门槛」，即设计上就是惰性阈值。因此任何「调高它来提质」的候选收益恒为 0，应被 Critic 直接拒。这是 Inert Critic 的第一个真实用例。
- `AGENT_MAX_ITERATIONS_CODE: 60 → 15` —— 这是**第二个阴性对照**（故意变差）。它若不判负，说明比较符号或 base/candidate 接反了。

---

## 3. 不变量（任何阶段都不得违反）

1. **模型冻结**：不微调、不做 RL/LoRA，只改模型外围 harness。
2. **现有 Runtime 不重写**：`supervisor.py` / `agents.py` / `plan.py` / `graph.py` 在 P0–P1 **一行不改**；Adaptive Runtime 是**控制平面**，通过 worktree + env 注入干预，不通过 import 侵入。
3. **评分器不可自改**：Frozen Core 的修改请求一律 `FROZEN_CORE_VIOLATION`，包括「只改一点点」。
4. **证据不完整即失败**：任一臂出现超时/缺失 run，整轮判 `INCOMPLETE_EVIDENCE`，**不丢样本、不做插补**。
5. **改动一次只碰一层**：P0/P1 每个候选只允许改 L1 参数或 L2 提示中的**一处**。
6. **可复现三元组**：`base_commit + patch_hash + judge_version` 缺一不可。

---

## 4. 术语与版本模型（解决 V2 §2 vs §29 的冲突）

V2 同时出现 `Current Harness` 与 `Base Harness`，未定义关系。本文档定义**三层**：

| 名称 | 定义 | 生命周期 | 谁可修改 |
|---|---|---|---|
| **Base Harness** | git tag `base/vN`（`base_commit` + 冻结的 `.env` 快照） | 长期，人工推进 | 仅人工（promote 门） |
| **Task Harness** | `Base + 本轮检索到的 ≤3 条经验`（组装出的临时覆盖层，落 `runs/<task>/<round>/task_harness.json`） | 单轮 | 系统（只读组装，不改 base） |
| **Incumbent** | 本轮内已被接受的候选（可多个候选串行比较） | 单轮 | 系统 |
| **Candidate** | worktree 上的一次具体改动 | 单轮 | Proposer |

规则：

```
执行用的是 Task Harness；比较的基线是 Base（不是上一轮的 Candidate）。
本轮结束后 Task Harness 与 Incumbent 全部丢弃，只留 Experience。
每 N 轮（建议 N=10）或某条经验在 ≥2 个任务族验证通过后，人工把 ≤3 条合并进 Base，打新 tag base/vN+1。
```

这样同时满足：**不漂移**（base 只在人工门下变化）、**改进能落地**（不是只生成建议）、**可归因**（每个候选都对着同一个 base 比较）。

---

## 5. 最终架构

相对 V2 架构图有 5 处修改，全部标注了下标。

```
                          USER TASK
                              │
                              ▼
                     Task Analyzer ──► Experience Retrieval
                              │                │
                              └───────┬────────┘
                                      ▼
                            Task Harness(= Base + lessons)
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │  FROZEN CORE (只读横条)    │  ← [新增] 评分/协议/预算/trace
                        │  evaluator · protocol ·   │
                        │  split · budget · runlog  │
                        └───────────────────────────┘
                                      │
                                      ▼
                        Existing Multi-Agent Runtime
                     supervisor → 6 nodes（每步内三道 gate）
                                      │
                                      ▼
                          Execution + Trace + Token
                                      │
                                      ▼
                            Failure Diagnosis
                                      │
                                      ▼
                       Hypothesis ──► Hypothesis Ledger  ← [新增] 失败假设不可重提
                                      │
                                      ▼
                            Component Selector
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
                 L1 参数           L2 提示           L3 结构
                    │                 │                 │
                    └─────────────────┼─────────────────┘
                                      ▼
                              Candidate (worktree)
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │  Gate A: Frozen / Leakage /   │  ← [前置] 评分前拦截
                      │  Inert / Cost-cap / Incomplete│
                      └───────────────────────────────┘
                                      │ pass
                                      ▼
                       Budget Governor ──► Probe (Racing)  ← [新增] 两级筛选
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
              L1 确定性 Q        L2 rubric          Cost
              (主目标)          (次要/tie-break)     (约束)
                    └─────────────────┼─────────────────┘
                                      ▼
                            Comparator (paired t)
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │  Gate B: 非补偿准入（Reason）  │  ← [补位] V2 缺拒绝权
                      └───────────────────────────────┘
                          │                       │
                    ACCEPT                    REJECT
                          │                       │
                          ▼                       ▼
                    Incumbent              Failure Ledger
                          │
                          ▼
                    Cross-task Test（held-out / OOD）
                          │
                          ▼
                   Experience Library
                          │
                          ▼
             Future Retrieval / 人工 promote 进 Base
```

**五处修改及其必要性**

| 修改 | 为什么必须 |
|---|---|
| Leakage / Frozen 检查**前置到 Probe 之前** | 候选一旦被评分就获得「曝光」，事后拦截已晚；且省掉无效 probe 的钱 |
| Comparator 与 Composer 之间插入 **Gate B** | V2 只有评分维度，没有判定权；没有拒绝权就没有「低接受率」这个健康信号 |
| 新增 **Frozen Core 横条** | 评分器与被优化对象在同一仓库，这是本方案与论文最大的差异 |
| 新增 **Budget Governor** | probe 是双份执行，无硬预算必然失控（§14） |
| Probe 改 **Racing 两级** | 粗筛（4 任务 × 2 臂）→ 精测（过筛者跑满 n），成本约降 60% |

---

## 6. 观测层：Trace 与 Token

### 6.1 原则

**不改现有埋点，旁路解析。** `runlog.py` 已经在写结构化程度足够的事件，第一步是**导出**而不是**重写**。

### 6.2 Token 计量（Day 1 唯一前置依赖）

落点：`llm.py:173 CustomContextMiddleware.wrap_model_call`

```python
# adaptive_runtime/token_probe.py（新增，不改 llm.py 的既有逻辑）
# 采用可插拔观测：wrap_model_call 返回的 response 上读 usage，
# 追加一行 jsonl：{ts, run_id, agent, model, in, out, cached, latency_ms}
# 若 response 无 usage_metadata（部分网关不返回），退化为字符估算并置 estimated=true
```

要求：
- 每条记录必须带 `run_id` 与 `agent`，否则无法归因到 step。
- `estimated=true` 的占比必须上报；>30% 说明网关不返回 usage，成本规则只能用估算口径，需在报告中标注。
- 这一步**零回归**：只是旁路观测，不改变任何调用参数。

### 6.3 Trace Schema

```jsonc
{
  "run_id": "20261022T101500_ab12cd34",
  "task_id": "FRS-007",
  "round_id": "r03",
  "arm": "base",                       // base | cand
  "harness": {"base_tag": "base/v3", "patch_hash": null, "judge_version": "ev-0.2"},
  "plan": [{"step_no": 1, "agent": "crawler_agent", "title": "..."}],
  "events": [
    {"t": 12.4, "type": "gate", "agent": "code_agent", "step_no": 1,
     "gate": "rule|critic|action_contract", "pass": false, "reason": "len<4", "retry": 1},
    {"t": 41.0, "type": "step_failure", "kind": "recursion", "agent": "code_agent",
     "step_no": 2, "why": "...", "iterations": 60},
    {"t": 88.2, "type": "tool", "name": "web_search", "ok": true, "ms": 1830},
    {"t": 95.0, "type": "replan", "trigger": "noop", "noop_count": 2, "grow": 1},
    {"t": 300.1, "type": "llm", "agent": "supervisor", "model": "...", "in": 12034, "out": 812}
  ],
  "artifacts": [{"path": "documents/xxx.md", "bytes": 18422, "deliverable": true}],
  "result": {"finalized": true, "truncated": false, "salvaged": false},
  "usage": {"in": 812340, "out": 96120, "estimated": false}
}
```

**关键点**：`events` 里必须保留 **gate 的每一次拒绝及其 reason 子串**。这是 A（契约达成率）唯一的计算依据，也是事后解释「为什么这轮没过」的唯一证据。

### 6.4 回放能力

- **L0 回放（P0 必须有）**：给定 `trace.jsonl`，能重新计算 Q、重新跑 Gate、重放 Comparator。**不重跑 LLM。**
- **L1 回放（P2 再做）**：同 seed + 同 worktree 重跑执行。当前系统对 web 检索结果无快照，严格重放不可能，必须诚实标注为「近似重放」。

---

## 7. 评价层：三级打分

### 7.1 为什么主目标必须是 L1

研报类任务没有 ground truth，这是事实；但**「过程是否合规、产物是否完整、结论是否有来源」是确定性的**。

| 级别 | 内容 | 噪声 | 在 Gate 中的位置 |
|---|---|---|---|
| **L1 确定性** | gate 通过率、step-failure、replan 空转、产物落盘与完整度、数字/结论的来源覆盖率、截断/兜底率 | 0（同输入同输出） | **主目标**（唯一准入依据） |
| **L2 rubric** | LLM 对 5 个维度的 1–5 分（见 7.3） | 高（sd 10+） | 次要证据 / tie-break / 报告 |
| **L3 行为** | sources、primary_sources、verified_claims、unsupported_claims | 中 | **仅归因，永不进 Gate** |

> 这是本文档相对 V2 最重要的一处改动：把「没有 ground truth 就只能靠 judge」变成「先用确定性事实压缩噪声，再让 judge 只做归因」。

### 7.2 L1 合成分 Q（0–100）

```python
Q = 100 * (0.50*A + 0.30*E + 0.20*C)

A = 契约达成率 = 通过三道 gate 的 step / 计划 step 数
    （gate 拒绝后重试成功计 1；最终未通过计 0；被封顶重试仍失败计 0）
E = 证据支持率 = 带可点击来源的关键数字与结论数 / 关键数字与结论总数
    （确定性抽取：正则抓数字+单位，检查同段是否有 URL/文件名/表名；无来源即 unsupported）
C = 完整性 = 1 - (截断段数 + 兜底段数 + 缺失计划产物数) / (计划产物数 + 1)
    任一计划产物缺失，C 直接 ≤ 0.5
```

三个设计约束：

1. **A 必须是硬的**：它是「契约层」的直接读数，不允许被 E/C 补偿（否则又会回到「答案不全但来源很多」）。
2. **E 的抽取必须确定性**：用规则，不用 LLM。LLM 只能**抽样复核** E 的准确率（每轮 20 条），复核一致率 <85% 时回改规则，而不是让 LLM 接管 E。
3. **Q 的量纲是「百分点」**：这样 β0/β1 与 δ 的口径统一，避免百分比与分数混用。

### 7.3 L2 rubric（仅 5 维，1–5 分）

| 维度 | 锚点（1 分 / 3 分 / 5 分） |
|---|---|
| 事实边界 | 把推断当事实 / 部分标注 / 明确区分事实与推断 |
| 结构完整 | 缺章节 / 结构齐但浅 / 结论—证据—限定条件齐全 |
| 可执行性 | 只有结论 / 有条件建议 / 给出可操作步骤与失效条件 |
| 一致性 | 前后矛盾 / 基本一致 / 数字与结论互相印证 |
| 表达 | 堆砌 / 可读 / 信息密度高且无冗余 |

**反偏置措施（缺一不可）**

- **双向呈现**：同一对产物按 `A→B` 与 `B→A` 各评一次，取均值；两次差异 >1.5 分则该对作废并记 `JUDGE_UNSTABLE`。
- **异构 judge**：`AGENT_CRITIC_LLM` 已存在。要求 **judge 与 proposer 不同配置**（不同模型或不同厂商）。同源 = self-preference，无低成本替代方案。
- **盲评**：呈现给 judge 的文本剥离 `run_id`、时间戳、文件名中的 `cand/base` 标记。

### 7.4 为何 ΔB 不得进入 Gate（以本仓库工作流为例）

```
sources 4 → 7          # 多列三个二手来源
verified_claims 8 → 13 # 多写五条「已核实」
```

这两项涨了，报告质量可以完全不变，甚至更差（堆砌来源稀释结论）。而 `ΔB` 是**加法型指标**，刷它的成本远低于提升 Q。所以：

> **ΔB 的输出位置只有一处：报告里的「为什么可能有效」。它不出现在 `admit()` 函数里。**

---

## 8. 噪声与统计

### 8.1 判定式（配对 t）

```
Δ_i = Q_cand(i) − Q_base(i)          # 同一任务、同一 judge 版本、两臂配对
δ   = t_{0.975, n−1} · sd(Δ) / √n     # 单侧判定门限（等价双侧 95%）
ACCEPT(统计意义上) iff mean(Δ) > δ
```

`n` 是**配对数**（不是总 run 数；总 run = 2n）。`sd(Δ)` 必须实测，不能假设。

### 8.2 固定 n 时的门限（为什么 n=5 不可用）

| sd(Δ) \ n | 5 | 10 | 25 | 50 |
|---|---|---|---|---|
| 2（L1 主导，判分几乎无噪声） | 2.5 | 1.4 | 0.8 | 0.6 |
| 4（好 rubric + 结构化锚点） | 5.0 | 2.9 | 1.7 | 1.1 |
| 7（rubric 化 + 双向呈现） | 8.7 | 5.0 | 2.9 | 2.0 |
| 12（裸 LLM 打分） | **14.9** | 8.6 | 5.0 | 3.4 |

**读法**：`sd=12, n=5` 时 δ≈14.9 —— 意味着「任何小于 15 分的改进都无法与噪声区分」。而 V2 §47 的示例写的是 `+7.2`，**在裸 LLM 打分 + 5 次重复下恰好落在噪声带内，会被正确拒绝**。

### 8.3 反推 n（本文档采用的正确口径）

检出效应 Δ 且达到 80% 把握所需配对数：

```
n ≈ (2.8 · sd(Δ) / Δ)²        # α=0.05 双侧，power=0.8
```

| 要检出的 Δ | sd=2 | sd=4 | sd=7 | sd=12 |
|---|---|---|---|---|
| **+4 分** | 2 | 8 | 25 | **71** |
| +8 分 | 1 | 3 | 7 | 18 |

> **一处必须说明的口径修正**：前序评审里用的是「δ ≤ Δ」这一判据（sd=7→13 对，sd=12→36 对）。那只相当于 **50% 把握**，即「抛硬币」。本文档改用 80% power 口径，`sd=12` 需要 **71 对**，成本不可接受——这恰好证明 **§7.1 把主目标换成 L1 不是优化，而是前提**：只有把 sd 压到 2–4，8 对以内才可能得出结论。

### 8.4 标定协议（P0 的核心交付）

```
输入：Base Harness
步骤：
 1. 取 calibration 集（6 个任务，覆盖：检索型 / 计算型 / 绘图型 / 多步长链 / 短问答 / 需外部数据的）
 2. 每个任务跑 k=6 次（同一 base，无任何改动）
 3. 对每一对 (i,j) 计算 |Q_i − Q_j|，得到 Q 的重复性噪声 sd_run
 4. 构造「零改动候选」pairs（第二次跑即视为候选），算 sd(Δ_null)
 5. 写入 experiments/calibration.json
输出字段：
  sd_run（单 run 的 Q 波动）
  sd_diff_null（零改动下的配对差波动）→ 这就是 δ 的 σ
  per_task（每个任务的 sd，识别「哪些任务本身不稳定」）
  judge_version、model、采样参数、时间
```

**退出门：`sd_diff_null < 6`。** 若 >6，先改判分（rubric 锚点、E 的抽取规则、双向呈现），**不要**用加样本量掩盖——那是用钱买一个本该由设计解决的问题。

### 8.5 Racing 两级筛选（控制成本）

```
Stage 1（粗筛）：4 个任务 × n=2/任务 = 8 对
   ├─ 若 mean(Δ) < 0 且 |mean| > 1.5·sd/√n  → REJECT_COARSE（止损，不进 Stage 2）
   └─ 其余（含「看起来有效」与「不确定」）→ Stage 2
Stage 2（精测）：补齐到 n=8（或按 8.3 表反推的 n）
   总 run 数 ≈ 8×2 + 4×2 = 24 run
```

粗筛只做**止损**，不做录取——因为 n=2 没有任何判定力，它的唯一职责是「明显变差的别再花钱」。

---

## 9. 判定层：非补偿 Gate

### 9.1 Gate A（评分前）

| 检查 | 失败 Reason | 实现 |
|---|---|---|
| 触及冻结符号或冻结 env | `FROZEN_CORE_VIOLATION` | §11 双 hash |
| diff 中出现任务 ID / 数据集路径 / 基准文本 | `BENCHMARK_LEAKAGE` | 正则 + 关键字表 |
| 改动是惰性机制（如 `AGENT_CRITIC_MIN_LEN`） | `INERT_MECHANISM` | 冻结表内容直接拒 |
| 预估成本 > 硬上限 | `COST_CAP` | 按改动类型估 token |
| 改动层 > 本轮 edit budget | `EDIT_BUDGET` | 候选 schema 校验 |

### 9.2 Gate B（评分后，非补偿）

```python
class Reason(Enum):          # 拒绝原因必须是枚举，不能是自由文本
    FROZEN_CORE_VIOLATION = "frozen_core_violation"
    BENCHMARK_LEAKAGE     = "benchmark_leakage"
    INERT_MECHANISM       = "inert_mechanism"
    COST_CAP              = "cost_cap"
    EDIT_BUDGET           = "edit_budget"
    INCOMPLETE_EVIDENCE   = "incomplete_evidence"
    CONTRACT_FLOOR        = "contract_floor"     # A < A_FLOOR
    WITHIN_NOISE          = "within_noise"
    COST_RULE             = "cost_rule"
    REGRESSION            = "regression"
    JUDGE_UNSTABLE        = "judge_unstable"

def admit(cand, ev, cfg) -> Decision:
    # ---- Hard constraints（任一失败即拒，不可补偿）----
    if touches_frozen(cand):              return Reject(FROZEN_CORE_VIOLATION)
    if leakage(cand.diff):                return Reject(BENCHMARK_LEAKAGE)
    if ev.missing_or_timeout_runs > 0:    return Reject(INCOMPLETE_EVIDENCE)
    if ev.A_cand < cfg.A_FLOOR:           return Reject(CONTRACT_FLOOR)   # 0.85
    if ev.judge_unstable_ratio > 0.2:     return Reject(JUDGE_UNSTABLE)

    # ---- Primary objective（唯一通过依据，L1 合成分）----
    dQ, se = paired_delta(ev.Q)            # 配对均值与标准误
    thr    = cfg.t_crit(ev.n) * se
    if dQ <= thr:
        # 未过噪声门的三种例外（ΔB 不在其中）
        if ev.dC < -cfg.C_SAVE_MIN:        return Accept("CHEAPER_EQUAL")   # 3%
        if cand.adds_novel_component:      return Accept("NOVELTY")
        if ev.hard_failure_removed:        return Accept("HARD_FAILURE_REMOVED")
        return Reject(WITHIN_NOISE)

    if ev.regression_rate > cfg.R_MAX:     return Reject(REGRESSION)        # 0.15
    if ev.dC > cfg.BETA0 + cfg.BETA1 * dQ: return Reject(COST_RULE)         # 0.05 + 0.015*dQ
    return Accept("VALIDATED")

# 注意：ev.dB 不出现在这个函数里。行为指标只进报告，不进 gate。
```

### 9.3 门槛取值表

| 常量 | 取值 | 来源 / 依据 | 何时复核 |
|---|---|---|---|
| `A_FLOOR` | **0.85** | 契约达成率是过程健康的地板；低于它谈质量无意义 | 标定后按 base 实测分布定 10 分位 |
| `R_MAX` | **0.15** | 允许 15% 任务出现负 Δ，避免「平均涨、局部崩」 | P1 用 held-out 复核 |
| `C_SAVE_MIN` | **3%** | 小于 3% 的成本下降在噪声内，不构成「更省」 | 成本埋点稳定后复核 |
| `BETA0` | **0.05** | 允许 5% 的固定成本上浮（多一次校验调用） | 用前 5 轮实测拟合 |
| `BETA1` | **0.015 / 分** | 每 +1 分 Q 允许 +1.5% token（+4 分 → +11%） | 同上 |
| `COST_CAP_ROUND` | **8,000,000 token** | 见 §14 | 每轮实测更新 |
| `sd_diff_null` 目标 | **< 6** | §8.4 退出门 | 每换 judge 版本重测 |

**β0/β1 的意义**：`ΔC ≤ 5% + 1.5%·ΔQ` 把「改动必须自己证明其成本合理」变成一条可计算的规则。它同时挡住本仓库最诱人的伪改进：**「把 `AGENT_MAX_ITERATIONS_CODE` 调大，让 recursion 失败归零」**——失败率会瞬间降，token 会暴涨，而 β1 会要求它同时带来足够 Q 增量才准过。

---

## 10. 演化层

### 10.1 L1 真实参数搜索空间（替换 V2 §12 的编造名字）

| 参数（真实名） | 现值 | 建议搜索空间 | 分级 | 备注 |
|---|---|---|---|---|
| `AGENT_MAX_ITERATIONS_CODE` | 60 | {60, 80, 100} | L1 | **只作证伪候选**：已是全局 15 的 4 倍仍 28 次失败 |
| `AGENT_MAX_ITERATIONS`（全局兜底） | 15 | {15, 25, 40} | L1 | 按 `AGENT_MAX_ITERATIONS_<TYPE>` 派生；爬虫未单独配置，实际落到 15 |
| `AGENT_PARSE_TRIES` | 3 | {2, 3, 5} | L1 | 结构化解析重试 |
| `AGENT_REPLAN_ARTIFACT_MAX` | 3（默认） | {2, 3, 5} | L1（弱） | 影响 replan 触发 |
| `AGENT_REPLAN_NOOP_MAX` | 2 | — | **L3，P2 才开** | 改它等于改控制流 |
| `AGENT_REPLAN_OBS_WINDOW` | 20 | — | L3 | 同上 |
| `AGENT_PLAN_GROW_MAX` | 1 | — | L3 | 同上 |
| `AGENT_CTX_*` / `AGENT_SUMMARIZE_*` / `AGENT_MEMORY_*` | 见 `.env` | — | L3 | 影响长上下文行为 |
| `AGENT_CRITIC_MIN_LEN` | **4** | **冻结，不搜** | — | 惰性机制 |
| `AGENT_ACTION_PROBES` / `AGENT_CRITIC_RETRIES` / `AGENT_STEP_FAILURES_MAX` / `AGENT_TRUNCATED_SALVAGE` / `AGENT_SALVAGE_EXTS` | 见 `.env` | **冻结** | — | 评分口径 |

**判据：一个参数能不能搜，取决于「改它会不会改变判分口径」**，不取决于它名字像不像旋钮。

### 10.2 三级演化与 edit budget

| 级别 | 内容 | 允许阶段 | budget |
|---|---|---|---|
| L1 参数 | 上表「L1」行 | P0 | 1 处 |
| L2 提示 | `prompt.py` 的角色/系统提示、工具指引 | P1 | 1 处 + ≤15 行 diff |
| L3 结构 | `plan.py` 阈值、`agents.py` 控制流、拓扑、子 agent、脚本 | P2 | 1 处 + 需人工复核 |

**annealed budget（P2）**：`b(k) = 1 + floor(k / 20)`，`k` 为已完成的被拒轮数；连续 5 轮 ACCEPT 则 `b` 归 1。含义：**长期无进展才允许放大改动幅度**，而不是一上来就改工作流。

### 10.3 Hypothesis Ledger（失败假设不可重提）

```jsonc
// experiments/ledger/hypotheses.jsonl
{
  "hyp_id": "hyp_004",
  "fingerprint": "replan_noop_max|2->5|COMPLETION_FAILURE",  // 去重 key
  "claim": "提高 AGENT_REPLAN_NOOP_MAX 可减少过早结束",
  "target": "plan.py:_REPLAN_NOOP_MAX",
  "status": "rejected",                  // proposed|running|accepted|rejected|inconclusive
  "evidence": {"n": 8, "dQ": 0.6, "dC": 0.09, "reason": "within_noise"},
  "tasks": ["FRS-002", "FRS-005"],
  "created_at": "..."
}
```

规则：
- `fingerprint` 相同 → Proposer **必须**引用原 `hyp_id` 并说明「新假设与旧的差异」，否则 `DUPLICATE_HYPOTHESIS` 直接拒。
- `inconclusive`（样本不足/判分不稳）与 `rejected` 分开记，**不能互相覆盖**。
- 连续 3 轮同一 `inconclusive` → 提升为 `rejected` 并附「该假设在本 harness 下不可判定」的结论。

### 10.4 Exploration（防搜索坍缩）

维护 `component_usage = {L1: a, L2: b, L3: c}`（按**被接受**次数计，不按提出次数）。选择规则：

```
若 max(usage) > 4 × min(usage) 且 max ≥ 5：
    下一轮强制在被冷落层中提出候选（不强制接受）
```

**注意**：本仓库的「层」是不均衡的——L1 只有 4 个可搜参数，很容易被穷举完。所以 exploration 的冷落对象应该是**失败模式 × 层**的组合（如「`truncated` 类失败从未用 L2 处理过」），而不只是层本身。

### 10.5 Candidate Critic（前置，便宜）

只查五件事，任一命中即拒（都在 Gate A）：

1. 是否改评分器/协议/预算（Frozen 双 hash）
2. 是否 benchmark-specific（diff 中出现任务 ID、固定日期、固定个股、固定文件名）
3. 是否答案泄漏（把期望结论写进提示）
4. 是否惰性机制（改了也不影响任何行为路径 —— 静态检查「被改符号是否在决策路径上」）
5. 是否复杂度不合理（+80 行只为 +1 分）

### 10.6 Pruning（淘汰无效组件）

只有一条判据：**某组件下提出的候选，连续 10 次 `WITHIN_NOISE`** → 该组件进入 `dormant`，30 天内不再由该组件生成候选。同时输出报告：「该组件在此 harness 上是惰性的」，**这才是 pruning 真正有价值的产品——它证明某类机制没用**。

---

## 11. Frozen Core：符号级冻结 + 双 hash

### 11.1 为什么目录冻结必然失效

V2 的 `FROZEN_PATHS = {"adaptive_runtime/evaluation/", ...}` 保护的是一个**新建的空目录**。真正的评分逻辑在 `agents.py` 的模块函数和 `.env` 的键里。按目录冻结 → 上面 §2.2 的 7 项全部落在「可编辑区」→ 闭环只要把 `AGENT_CRITIC_MIN_LEN` 改成 1，通过率立刻上升，而这一「改进」会被记入经验库。

### 11.2 两层冻结

```python
# adaptive_runtime/frozen.py（新增）
FROZEN_SYMBOL_FILES = {                 # 文件级 hash：防改符号
    "agents.py":    ["_rule_gate", "_critic_gate", "_action_contract_gate",
                     "_is_deliverable_step", "_DELIVERABLE_VERBS", "_PATH_EXTS",
                     "_CRITIC_MIN_LEN", "_ACTION_PROBES_ENV", "_STEP_FAILURES_MAX"],
    "planutil.py":  ["members", "options"],
    "runlog.py":    ["log_event", "timing_summary"],
    "graph.py":     ["build_graph_with_memory"],
}

FROZEN_ENV = [                          # 解析后 env hash：防改 .env / 防被环境变量覆盖
    "AGENT_CRITIC_MIN_LEN", "AGENT_ACTION_PROBES", "AGENT_CRITIC_RETRIES",
    "AGENT_STEP_FAILURES_MAX", "AGENT_TRUNCATED_SALVAGE", "AGENT_SALVAGE_EXTS",
    "AGENT_FINAL_ADJUDICATE", "AGENT_CRITIC_LLM",
]

# 两道断言，缺一不可
assert sha256_files(FROZEN_SYMBOL_FILES) == frozen_file_hash     # 防改文件
assert sha256_resolved_env(FROZEN_ENV)   == frozen_env_hash      # 防改 .env / 防 env 覆盖
```

**第二道断言是关键**：本仓库 37 个 `AGENT_*` 参数全部可由 `.env` 或进程环境覆盖。只 hash 文件 = 没冻结。

### 11.3 运行时校验 + 副作用

- `probe` 前后各算一次双 hash，不一致 → 整轮 `FROZEN_CORE_VIOLATION`，**并且该候选的 patch 进 `experiments/quarantine/`**（不是删除，用于事后审计「闭环试图改什么」）。
- 冻结值本身写进 `experiments/calibration.json` 与每轮 `meta.json`，保证「用哪把尺子量」可追溯。

---

## 12. 诊断层：taxonomy 必须被「可判定字段」约束

现有日志只有 3 个值：`recursion 28 / truncated 7 / exception 1`。V2 的 12 类里 **9 类当前无字段可判**。

### 12.1 首版映射（只启用 5 类）

| 现有信号 | taxonomy | 判定规则（全部用现有字段） |
|---|---|---|
| `kind=recursion` + 有产物且过 gate | `COMPLETION_FAILURE` | 预算耗尽但交付了 |
| `kind=recursion` + 无产物 | `EXECUTION_FAILURE` | 预算耗尽且没交付 |
| `kind=truncated` + `salvaged=true` | `COMPLETION_FAILURE` | 截断但有兜底产物 |
| `kind=truncated` + `salvaged=false` | `EXECUTION_FAILURE` | 截断且无兜底 |
| `kind=exception` | `EXECUTION_FAILURE` | 再按异常类细分（工具/基础设施） |
| gate 拒绝且重试耗尽（`[critic+summary]` 有记录） | `REVIEW_FAILURE` | 按 `_critic_gate` reason 子串分类 |
| `noop_count ≥ AGENT_REPLAN_NOOP_MAX` | `PLANNING_FAILURE` | supervisor 日志计数 |
| 其余 | `UNKNOWN` | **先统计占比** |

### 12.2 运营规则

```
UNKNOWN 占比 > 40%  → 停止增加 taxonomy 类别，转去补埋点
UNKNOWN 占比 20–40% → 允许 LLM 对 UNKNOWN 做 root cause 归因，但结果标记 llm_inferred=true
UNKNOWN 占比 < 20%  → 才允许把 UNKNOWN 聚类成新类别
```

> **taxonomy 的规模应被「可判定字段」约束，而不是被想象力约束。**

---

## 13. 经验层

### 13.1 Experience schema

```jsonc
{
  "experience_id": "exp_012",
  "hyp_id": "hyp_009",
  "problem": "COMPLETION_FAILURE / code_agent / step_no>=2",   // 失败模式指纹
  "claim": "在 code_agent 提示中要求先落盘中间产物，可降低 recursion 耗尽率",
  "component": "L2/prompt/code_agent",
  "edit": {"patch_hash": "a1b2c3", "lines": 9},
  "positive_tasks": ["FRS-002", "FRS-005"],
  "negative_tasks": ["FRS-007"],
  "dQ": 5.2, "dB": 0.31, "dC": 0.08,          // dB 仅供参考，不进判定
  "regression_rate": 0.09,
  "transfer_score": null,                      // P2 才填
  "confidence": 0.62,                          // 见 13.2
  "status": "validated",                       // validated|weakened|refuted
  "judge_version": "ev-0.2",
  "created_at": "..."
}
```

### 13.2 confidence 的定义（不能凭感觉）

```
confidence = w1 * (n_pairs / n_target)          # 样本充分度
           + w2 * (1 - regression_rate)          # 稳健性
           + w3 * (1 - p_value)                  # 统计显著度（截断到 [0,1]）
w = (0.4, 0.3, 0.3)
```

`confidence < 0.5` 的经验**允许被检索，但只能作为「待验证提示」注入**（Task Harness 中标注 `unverified`），不得作为既定规则。

### 13.3 检索（P1）

检索 key = 当前任务的 `(failure_mode, agent, step_no, task_family)`，不是语义相似度。原因：语义相似会把「金融研报」和「产业分析」检到一起，但它们的失败模式未必相同；失败模式匹配才是可操作的。

返回 **≤3 条**，并强制附加：

```
[经验 exp_012 · confidence 0.62 · 在 2 个任务上有效，1 个无效]
```

**带负面证据的经验必须一起呈现**。只给正面证据等于教模型 overfit。

### 13.4 Regression 集（P1）

```python
# experiments/regression/tasks.json
# 规则：任一 accepted candidate 必须在 regression 集上复跑；
#       regression 集永不参与 evolution（不用于生成假设，只用于否决）
# 规模：≥12 个任务，覆盖 6 个任务族，每族 ≥2
# 判据：regression_rate = 显著负 Δ 的任务数 / 总任务数  ≤ R_MAX(0.15)
```

### 13.5 成功/失败轨迹对比（对 ModularRSI 的诚实降级）

| 版本 | 前提 | 做法 | 阶段 |
|---|---|---|---|
| **弱化版** | 只需历史日志 | 把 36 次失败按 `(kind, agent, step_no)` 聚类，与成功 run 的同步骤做**分布对比**（如「失败 run 的 CodeAgent 平均 41 次工具往返，成功 run 18 次」） | P1 |
| **严格版** | 同任务多次执行且有成有败 | 用 FRS 把同一任务跑 5–8 次，取成败轨迹做逐步对照 | P2 |

弱化版必须标注：「与 ModularRSI 的严格定义不同，这是跨任务分布对比，不能归因单条轨迹」。**不要在没有数据的地方假装有。**

---

## 14. 预算治理

### 14.1 成本模型

```
单轮成本 ≈ n_pairs × 2 × T_run + proposer + judge_overhead
           （T_run = 单次 run 的 token，Day 1 实测；先假设 15–20 万）
```

以 `n=8, T_run=180k` 计：`8×2×180k = 2.88M`，加 judge/proposer 约 **3.2M token/轮**。
以 `n=25`（sd=7 且不降噪）计：**9M token/轮**，且要 2–4 小时。

### 14.2 硬规则

| 规则 | 值 |
|---|---|
| 单轮上限 | **8M token**，超限中止并记 `COST_CAP` |
| 单候选上限 | **4M token**（超限不补齐 Stage 2） |
| 每轮候选数 | **1**（P0/P1）；P2 起允许 2，但每轮总量不变 |
| 中断策略 | Stage 1 明显变差立即止损，不进 Stage 2 |
| 预算不足时 | **降 n 并标注结论为 `inconclusive`**，不允许「少跑几次但结论照旧」 |

### 14.3 为什么把「单轮单候选」写成硬规则

RRSI 一轮 40 候选的前提是它有针对固定任务的**可判定仿真器**，单次评估近乎免费。本仓库的单次评估是「跑一个真实多 agent 任务」，成本高 3 个数量级。**在这个成本结构下，宽搜（多候选少样本）必然得到一堆噪声结论，窄搜（单候选足样本）才有信息量。** 这不是保守，是成本结构决定的。

---

## 15. 路线图与阶段退出条件

### P0（2 周）：能看见、能量、能拒绝

范围（**顺序不可调换**）：

```
Day0  token 埋点 + trace 旁路导出
Day1  evaluator L1（Q 公式）+ 历史 run 回放
Day2  calibration（sd_run / sd_diff_null）
Day3  comparator + 两个阴性对照      ← 唯一硬门
Day4  冻结（双 hash）+ Gate A/B + Reason 枚举
Day5-6 worktree probe（只做 L1 参数，2 个候选）
Day7  experience.jsonl + 拒绝原因分布报告
```

**不做**：L2/L3 演化、experience 检索、transfer、pruning、自动改代码、多候选。

**退出条件（全部满足）**：

- [ ] `sd_diff_null < 6`，写入 `calibration.json`
- [ ] 零改动候选 → `WITHIN_NOISE`
- [ ] `AGENT_MAX_ITERATIONS_CODE: 60→15` → `REGRESSION` 或 `WITHIN_NOISE`
- [ ] 双 hash 在 probe 前后一致，且人为改 `.env` 能被抓出
- [ ] 一轮完整流程 token ≤ 8M 且有账
- [ ] 每个拒绝都有 Reason 枚举，能输出分布

### P1（3–4 周）：泛化与经验

- 多任务配对、held-out、regression 集（12 任务）
- L2 提示演化（15 行 diff 上限）
- Failed Hypothesis Ledger + Exploration
- Experience 检索（失败模式 key）+ 负面证据共同呈现
- 成功/失败**分布对比**弱化版

退出：held-out 上的 ΔQ 与 evolution 集同号；regression_rate ≤ 0.15；能回答「10 个候选为什么只过 1 个」。

### P2（4 周+）：正则化 RSI

- annealed edit budget、pruning、stalled search 检测
- L3 结构演化（`plan.py` 阈值起步，`agents.py` 控制流需人工复核）
- Generalization Gap（evolution vs held-out vs OOD）报表
- 严格版轨迹对比（同任务多跑）

退出：OOD 损失 ≤ evolve 收益的 50%；连续 10 轮无 ACCEPT 时能自动降级搜索空间而不是硬凑。

### P3：Modular / Plugin RSI

把已验证机制原子化为可组合 plugin（`guardrail` / `verifier` / `tool_guidance` / `workflow_fragment`），支持 Task Harness 现场组装。**前置条件**：P2 至少积累 8 条 `validated` 经验，否则没有可组合的原子。

---

## 16. P0 验收标准 v2

V2 的 11 条全部保留，但**不足以证明尺子是对的**。分四组：

### A. 能力（V2 原 11 条，保留）

```
[✓] 每次运行都有完整 Trace
[✓] Trace 可回放（L0，不重跑 LLM）
[✓] Diagnosis 输出结构化 root cause（含 UNKNOWN 占比）
[✓] 每个 Candidate 有 hypothesis 与 fingerprint
[✓] Candidate 使用独立 worktree
[✓] Frozen Core 双 hash 无法被绕过
[✓] Candidate 可自动运行并采集
[✓] Base/Candidate 可重复配对比较
[✓] 可 ACCEPT/REJECT 并输出 Reason
[✓] Experience 可保存
[✓] 失败 Candidate 可回滚（worktree 删除 + quarantine）
```

### B. 数值

```
[✓] 同一任务重复 6 次，sd_run 与 sd_diff_null 已实测（calibration.json）
[✓] comparator 输出 mean(Δ) / sd(Δ) / SE / t / δ / 是否过门，六项齐全
[✓] 一轮成本已记账，≤ 8M token
[✓] 每个 accepted candidate 有 base_commit + patch_hash + judge_version
```

### C. 阴性对照（**最重要，不过则 P0 失败**）

```
[✓] 阴性对照 1（零改动）：只改一行注释的候选 → 必须 WITHIN_NOISE
[✓] 阴性对照 2（故意变差）：AGENT_MAX_ITERATIONS_CODE 60→15 → 必须 REGRESSION 或 WITHIN_NOISE
[✓] 敏感性对照（可选但强烈建议）：base 复制一份跑两次，结论也必须 WITHIN_NOISE
```

判 ACCEPT 说明三种可能之一：δ 错了、judge 有偏、base/candidate 路径不对称。**这三种都无法靠后续实验补救，只能回 Day2。**

### D. 原因可统计

```
[✓] 所有拒绝输出 Reason 枚举
[✓] 能输出「10 个候选：6 WITHIN_NOISE / 2 COST_RULE / 1 REGRESSION / 1 VALIDATED」这类分布
[✓] 能回答「为什么这轮被拒」而不需要读日志（gate_trace 逐项布尔）
```

**接受率 10–25% 是正常状态，不是故障。** 若首轮接受率 >50%，先怀疑 δ 太小或尺子有偏。

---

## 17. 第一周实施计划（修订版）

原 V2 把「冻结 Evaluator」放在 Day 2 —— 但那时 Evaluator 还不存在。顺序必须是**先造尺、再标定、最后才冻**。

| Day | 任务 | 完成标志 | 依赖 |
|---|---|---|---|
| **0** | token 埋点（`llm.py:173`）+ trace 旁路导出 | 任意一次 run 导出结构化 jsonl，token 非 0，`estimated` 占比已上报 | — |
| **1** | evaluator L1：Q = 0.5A+0.3E+0.2C | 任一历史 run 能打出 0–100 的 Q，且 A/E/C 三项可分别读 | Day0 |
| **2** | calibration：6 任务 × 6 次 | `calibration.json` 有真实 `sd_run` / `sd_diff_null` | Day1 |
| **3** | comparator + 两个阴性对照 | 两个对照都判对；否则回 Day1 | Day2 |
| **4** | 双 hash 冻结 + Gate A/B + Reason | 手改 `.env` 能被抓；拒绝有枚举 | Day3 |
| **5** | worktree probe（L1 参数，1 个候选） | 一条 `run_adaptive.py` 端到端跑通 | Day4 |
| **6** | Racing Stage 1/2 + 成本记账 | 预算超限能中止 | Day5 |
| **7** | experience.jsonl + 拒绝分布报告 | 能回答「本轮为什么没过」 | Day6 |

**若 Day3 的 sd_diff_null ≥ 6**：停止后续所有工作，回到 Day1 修判分（锚点更具体、E 抽取规则化、双向呈现）。**不要**用加样本量掩盖——那是用钱买一个设计问题。

---

## 18. 目录与产物：只新增，不搬家

### 18.1 代码

```
multi-agent/
├── adaptive_runtime/                 # 全新独立包，不 import 现有模块的内部实现
│   ├── __init__.py
│   ├── token_probe.py                # Day0：token 观测（旁路）
│   ├── trace_export.py               # Day0：解析 runlog → trace.jsonl（只读）
│   ├── evaluator.py                  # Day1：L1 Q（A/E/C）+ L2 rubric
│   ├── calibrate.py                  # Day2
│   ├── comparator.py                 # Day3：配对 t / δ / regression_rate
│   ├── frozen.py                     # Day4：双 hash
│   ├── gate.py                       # Day4：Gate A/B + Reason
│   ├── probe.py                      # Day5：worktree 执行与采集
│   ├── budget.py                     # Day6
│   ├── diagnosis.py                  # P1
│   ├── proposer.py                   # P1
│   ├── ledger.py                     # P1
│   ├── experience.py                 # P1
│   └── run_adaptive.py               # 唯一入口
├── experiments/
│   ├── calibration.json
│   ├── tasks/{calibration,regression,heldout,ood}.json
│   ├── runs/<task_id>/<round_id>/...
│   ├── ledger/hypotheses.jsonl
│   ├── quarantine/
│   └── reports/
└── （supervisor.py / agents.py / plan.py / graph.py / runlog.py 一行不改）
```

**明确禁止**：`mkdir agents/`、`git mv` 现有文件、把 `prompt.py` 抽成 `prompt.yaml`（双源漂移）、给 `agents.py` 加 import 反向依赖。

### 18.2 单轮实验产物（含复现指纹）

```
experiments/runs/<task_id>/<round_id>/
├── meta.json          # base_tag, base_commit, file_hash, env_hash, judge_version,
│                      # cfg_snapshot, started_at, cost_tokens, budget_cap
├── task_harness.json  # base + 本轮检索到的经验（≤3）
├── base/{trace.jsonl, score.json, artifacts/}
├── candidates/<cid>/{patch.diff, worktree_head, trace.jsonl, score.json, artifacts/}
├── comparison.json    # per-task Δ 明细 + mean/sd/SE/t/δ + regression_rate + 分层(n)
├── decision.json      # {decision, reason, gate_trace:[{"check":..,"pass":bool,"value":..}]}
└── experience.jsonl
```

`gate_trace` 是关键：记录**每条判据的逐项布尔值**，而不只是最终结论。没有它，事后无法回答「为什么这轮被拒」。

---

## 19. 风险登记与反模式

| 风险 | 触发信号 | 处置 |
|---|---|---|
| **闭环学会改尺子** | `AGENT_CRITIC_MIN_LEN` 出现在候选 diff | 双 hash + quarantine + 该候选永久标记 |
| **伪改进：调大预算** | `AGENT_MAX_ITERATIONS_CODE` 上升且 recursion→0、token↑ | β1 拦截；并把结论写入负证据 |
| **样本量通胀** | 有人提议「跑 30 次让它显著」 | 先查 sd_diff_null；>6 一律驳回 |
| **judge 同源自嗨** | judge 与 proposer 同模型 | 强制异构，写进配置校验 |
| **经验污染** | confidence<0.5 的经验被当既定规则 | Task Harness 标 `unverified`，且必须附负面证据 |
| **指数漂移** | base 上出现「上一轮的改动」 | Base 只经人工 promote；tag 只增不改 |
| **长任务不可比** | 一个 run 20 分钟以上 | 该任务移出 calibration 集，单列「长任务」分层报告 |
| **判分器与产物强耦合** | 改产物格式导致 Q 跳变 | judge_version 必须随格式变更递增，跨版本禁止比较 |

**三个反模式（看到了就要停）**

1. **先跑一个月再总结**：没有标定就没有结论，跑得越多噪声越大。
2. **用「感觉变好了」替代 gate**：这正是 V1 的病。
3. **为了接受率好看而放松 δ**：接受率是设计结果，不是 KPI。

---

## 20. 需人工决策的三件事

| # | 决策 | 选项 | 建议 |
|---|---|---|---|
| 1 | judge 模型选型 | 与 proposer 同源 / 异构（不同厂商） | **异构**，且 judge 侧固定温度 0；用 `AGENT_CRITIC_LLM` 旁挂一套独立配置 |
| 2 | FRS（冻结回放集）从哪批 run 取 | 全部 95 条 / 抽样 24 条 / 新建 12 条 | 从历史 95 条里挑 **24 条**（6 族 × 4），已有 `query` 与 `goal` 可还原；先只读不重跑 |
| 3 | 是否接受「evolve 分数可能下降」 | 接受 / 不接受 | **必须接受**。否则这套系统会退化成「在这个任务上更好」的优化器 |

---

## 附录 A：常量速查表

```python
# adaptive_runtime/config.py（建议初值，随标定更新）
A_FLOOR          = 0.85
R_MAX            = 0.15
C_SAVE_MIN       = 0.03
BETA0            = 0.05
BETA1            = 0.015          # per Q point
COST_CAP_ROUND   = 8_000_000      # tokens
COST_CAP_CAND    = 4_000_000
N_DEFAULT        = 8              # 由 (2.8*sd/Δ)^2 反推，sd 目标 2-4
K_CALIB          = 6              # 每任务重复次数
SD_DIFF_TARGET   = 6.0
UNKNOWN_MAX      = 0.40
EDIT_BUDGET_P0   = 1
JUDGE_TEMP       = 0.0
ALPHA            = 0.05           # 双侧
POWER            = 0.80
```

## 附录 B：与 V2 原文的逐节处置

| V2 节 | 处置 | 说明 |
|---|---|---|
| §2 架构、§7 四个 Loop | **保留**，补 Gate B / Budget Governor / Frozen 横条 | §5 |
| §5 现有代码映射 | **替换** | 按真实文件重写（§2） |
| §6 目录树 | **删除** | 文件不存在，改为「只新增」（§18） |
| §8 Trace、§9 Taxonomy | 保留骨架，**补映射与 UNKNOWN 规则** | §6、§12 |
| §12 三级演化 | 保留框架，**替换参数清单** | §10.1 |
| §14/§15 Frozen Core | 保留思想，**替换实现（符号+双 hash）** | §11 |
| §16 Noise Calibration | **替换** | 给公式与标定协议（§8.4） |
| §17 Paired Evaluation | 保留，**补 power 反推 n** | §8.3 |
| §18 Verifier 优先级 | 保留，**主目标改为 L1** | §7.1 |
| §19/§20 行为与多维分 | 保留，降级为次要证据 | §7.4 |
| §21 Accept/Reject | **替换伪代码**（非补偿 + Reason） | §9.2 |
| §22 四层优先级 | **保留**（V2 最好的设计），在 §9.2 落地 | — |
| §23 Cost Rule | 保留，**填 β0/β1** | §9.3 |
| §24 Edit Budget | 保留，**补 annealed 公式与阶段** | §10.2 |
| §25 Failed Hypothesis | 保留，**补 fingerprint 与 inconclusive** | §10.3 |
| §26 Exploration / §27 Critic / §28 Worktree | 保留，补真实判据 | §10.4–10.6 |
| §29 Base 不累积 | 保留，**补三层版本模型** | §4 |
| §30/§31 Experience | 保留，**补 confidence 公式与检索 key** | §13 |
| §32–§34 Transfer/Generalization | P2 落地，P0 不实现 | §15 |
| §35 成功/失败轨迹对比 | **降级为弱化版** | §13.5 |
| §38–§40 P0 与验收 | **重排顺序 + 替换验收** | §15、§16 |
| §44 第一周 | **替换** | §17 |
| §45 不改 Python | 保留（P1 只碰 `prompt.py` 的常量） | §10.2 |
| §46–§49 实验与结果解读 | 保留为 P1 参考，判据以 §9 为准 | — |

## 附录 C：文献定位（已核验）

| 简称 | 定位 | 本文档实际借鉴 |
|---|---|---|
| Harness² | arXiv:2610.14224 | 任务级 propose–probe–compose；每任务从 base 出发；lessons 单独存 |
| RRSI | 2609.24972 / regularized-rsi.com | edit budget、failed hypothesis、noise-adjusted selection、cost rule、pruning |
| ModularRSI | 2609.14857 | benchmark-disjoint、成败轨迹对比（本文档诚实降级） |
| PluginRSI | 2609.32423 | 机制原子化为 plugin（P3） |

**注意**：RRSI 的关键取舍是**放弃 evolve 分数换 OOD 与成本**（evolve 90.5 vs 92.8；OOD 43.6 vs 40.3；token 2.42M vs 3.80M）。如果本方案只抄了机制而没抄这个取舍，等于没抄。

---

## 附：一句话验收

> **这套系统做对了的标志，不是「它改进了自己」，而是「它能拒绝一个看起来像改进的改动，并说出理由」。**
