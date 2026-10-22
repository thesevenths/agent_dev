# Adaptive Agent Runtime V2
## `thesevenths/agent_dev/multi-agent` → Evidence-Driven Adaptive Agent Runtime
### Implementation Specification / V2

> **目标**
>
> 不更换模型、不做模型权重训练，保留现有 `Multi-Agent → Planner / Executor / Reviewer → Memory` 体系，在外面增加一个 **Adaptive Runtime**：
>
> `Execution → Trace → Diagnosis → Hypothesis → Candidate → Probe → Evaluation → Accept/Reject → Experience → Transfer`
>
> V2 的核心不是“自动改 Prompt”，而是：
>
> **让 Agent 能发现自身瓶颈，提出可验证的 Harness 修改，隔离实验，依据证据选择修改，并把经过跨任务验证的机制沉淀为可复用经验。**

---

# 1. V2 相比 V1 的核心升级

V1 更像“研究架构”。

V2 必须变成可以真正开工的 Implementation Specification。

主要吸收 Harness²、RRSI、ModularRSI、PluginRSI 的思想：

| 问题 | V2 方案 |
|---|---|
| 没有 Ground Truth | Verifier 优先，Behavioral Evidence fallback |
| 单次执行噪声 | Noise Calibration + Paired Repeated Runs |
| Task-specific overfit | Multi-task + Held-out + OOD |
| Agent 修改评分器 | Frozen Core |
| 无限修改 Harness | Edit Budget |
| 重复失败假设 | Failed Hypothesis Memory |
| 只改 Prompt | Component-level Evolution |
| 修改太复杂无法归因 | Parameter → Prompt → Structure 分级演化 |
| `main` 无限累积修改 | Git Worktree / Candidate Branch |
| 成功机制无法复用 | Experience Library |
| 经验直接污染未来 Harness | Lesson Retrieval，而非自动永久 merge |
| 单体 Harness 难以演化 | 后期 Modular / Plugin Evolution |

Harness² 当前公开实现明确采用 Task-level recursion 的 Propose–Probe–Compose，并在更高一层保留 adopted harness 与 evidence-supported lessons；其 Harness 暴露 system prompt、guardrails、memory、tool guidance、subagents、skills、scripts、plugins 等组件。V2 借鉴这个分层思想，但不直接复制 Harness² 的代码结构。 

RRSI 则强调 evolution budget、失败假设历史、探索、candidate critic、noise-adjusted selection、cost rule 和 pruning；这些会成为 V2 的 regularization 层。 

ModularRSI 强调 benchmark-disjoint evolution、成功/失败 trajectory 对比、跨任务证据以及模块化演化；V2 会把这些作为 P2/P3 的泛化能力。 

PluginRSI 则把成功机制原子化为可复用 plugin，再组合到未来 Harness 中；V2 后期采用类似方向。 

---

# 2. 最终架构

```text
                              USER TASK
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Task Analyzer    │
                         └────────┬─────────┘
                                  │
                                  ▼
                         Experience Retrieval
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │     Current Harness     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                         Existing Multi-Agent
                                 │
                 ┌───────────────┼───────────────┐
                 ▼               ▼               ▼
              Planner         Executor        Reviewer
                 │               │               │
                 └───────────────┼───────────────┘
                                 ▼
                              Execution
                                 │
                                 ▼
                                Trace
                                 │
                                 ▼
                         Failure Diagnosis
                                 │
                                 ▼
                            Hypothesis
                                 │
                                 ▼
                       Component Selector
                                 │
             ┌───────────────────┼──────────────────┐
             ▼                   ▼                  ▼
        Parameter             Prompt            Structure
        Evolution            Evolution           Evolution
             │                   │                  │
             └───────────────────┼──────────────────┘
                                 ▼
                           Candidate Harness
                                 │
                                 ▼
                         Frozen-Core Gate
                                 │
                                 ▼
                         Leakage / Cost Gate
                                 │
                                 ▼
                               Probe
                                 │
                                 ▼
                       Evidence Collection
                                 │
               ┌─────────────────┼─────────────────┐
               ▼                 ▼                 ▼
            Verifier          Behavior           Cost
               │                 │                 │
               └─────────────────┼─────────────────┘
                                 ▼
                            Comparator
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
          Accept / Reject                 Cross-task Test
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
                         Experience Library
                                 │
                                 ▼
                         Future Task Retrieval
```

---

# 3. 最重要设计原则

## 3.1 模型保持 Frozen

第一阶段：

```text
LLM weights = frozen
```

变化的是：

```text
Prompt
Workflow
Tool Guidance
Parameters
Memory Policy
Subagents
Skills
Tools
Context Management
```

不是：

```text
Fine-tuning
RL
LoRA
Model Weight Update
```

这与 Harness² / RRSI 的基本问题设定一致：优化的是模型外围的 Harness，而不是模型参数。 

---

# 4. Existing Multi-Agent 不重写

当前系统：

```text
Planner
   ↓
Executor
   ↓
Reviewer
   ↓
Memory
```

保留。

新增一层：

```text
                 Adaptive Runtime
                       │
       ┌───────────────┴───────────────┐
       │                               │
Improvement Runtime              Existing Runtime
       │                               │
Diagnosis                         Planner
Candidate                         Executor
Probe                             Reviewer
Comparator                        Memory
Experience
```

核心原则：

> **Adaptive Runtime 是控制平面，不替代现有 Agent Runtime。**

---

# 5. 现有代码 → V2 映射

由于当前 `agent_dev` 的公开仓库页面无法稳定提供完整源码树，本节按此前确认的 `multi-agent` 架构进行映射；实际改代码时，应以本地 checkout 的真实文件名为准，不应凭空假设文件名。

建议映射：

```text
现有组件                    V2 角色
-------------------------------------------------
Planner                     Existing Agent Runtime
Executor                    Existing Agent Runtime
Reviewer                    Existing Agent Runtime
Memory                      Existing Runtime + Experience Retrieval
Orchestrator                Runtime Executor
Tool layer                  Runtime / Tool Registry

新增：
runtime/trace.py            Execution Trace
runtime/run_context.py      Run metadata
diagnosis/                  Failure Diagnosis
evolution/                  Candidate generation + selection
evaluation/                 Verifier / Behavior / Cost
experience/                 Lessons / Transfer
harness/                    Versioned harness specification
experiments/                Experiment protocol
```

---

# 6. 推荐 V2 目录

第一阶段不要把所有东西都拆得很细。

推荐：

```text
agent_dev/
└── multi-agent/
    ├── agents/
    │   ├── planner.py
    │   ├── executor.py
    │   └── reviewer.py
    │
    ├── runtime/
    │   ├── orchestrator.py
    │   ├── trace.py
    │   └── run_context.py
    │
    ├── adaptive_runtime/
    │   ├── diagnosis/
    │   │   ├── analyzer.py
    │   │   └── taxonomy.py
    │   │
    │   ├── evolution/
    │   │   ├── proposer.py
    │   │   ├── candidate.py
    │   │   ├── probe.py
    │   │   ├── comparator.py
    │   │   └── composer.py
    │   │
    │   ├── evaluation/
    │   │   ├── verifier.py
    │   │   ├── behavior.py
    │   │   ├── cost.py
    │   │   └── transfer.py
    │   │
    │   ├── experience/
    │   │   ├── store.py
    │   │   ├── retrieval.py
    │   │   └── confidence.py
    │   │
    │   └── policy/
    │       ├── budget.py
    │       ├── critic.py
    │       ├── selector.py
    │       └── pruner.py
    │
    ├── harness/
    │   ├── base/
    │   └── candidates/
    │
    ├── experiments/
    │   ├── tasks/
    │   ├── runs/
    │   └── reports/
    │
    └── tests/
```

注意：

> 第一阶段不要为了“架构漂亮”而把现有代码全部搬家。

---

# 7. V2 的四个 Loop

## Loop 1：Execution

```text
Task
 ↓
Planner
 ↓
Executor
 ↓
Reviewer
 ↓
Final Result
 ↓
Trace
```

目标：

> 先让 Agent 行为可观察。

---

## Loop 2：Improvement

```text
Trace
 ↓
Diagnosis
 ↓
Hypothesis
 ↓
Candidate
 ↓
Probe
```

目标：

> 找到 Harness 改动。

---

## Loop 3：Validation

```text
Candidate
 ↓
Frozen-Core Gate
 ↓
Noise Calibration
 ↓
Repeated Probe
 ↓
Verifier / Behavior / Cost
 ↓
Accept / Reject
```

目标：

> 证明 Candidate 不是随机波动。

---

## Loop 4：Experience

```text
Accepted Improvement
 ↓
Experience
 ↓
Cross-task Validation
 ↓
Generalizable Lesson
 ↓
Future Task Retrieval
```

目标：

> 把 Task-specific 改进变成可验证的通用机制。

---

# 8. Trace 是 P0

没有 Trace，后面所有 Self-Improvement 都只能靠猜。

建议每次运行生成：

```json
{
  "run_id": "run_20261022_001",
  "task_id": "finance_001",
  "harness_version": "base",
  "model": "qwen...",
  "seed": 42,

  "steps": [
    {
      "step_id": 1,
      "agent": "planner",
      "action": "plan",
      "input": "...",
      "output": "...",
      "latency_ms": 1200,
      "tokens": 1800
    }
  ],

  "tools": [],
  "review": {},
  "final_result": {},
  "cost": {}
}
```

必须至少记录：

```text
task
harness version
model
seed
agent
step
tool
tool input
tool output
error
review
final output
tokens
latency
cost
```

---

# 9. Failure Taxonomy

不要让 Diagnosis 每次自由发挥。

先建立固定 taxonomy：

```text
PLANNING_FAILURE
EXECUTION_FAILURE
TOOL_SELECTION_FAILURE
TOOL_USAGE_FAILURE
EVIDENCE_FAILURE
CONTEXT_FAILURE
MEMORY_FAILURE
DELEGATION_FAILURE
REVIEW_FAILURE
COMPLETION_FAILURE
COST_FAILURE
LOOP_FAILURE
```

例如：

```json
{
  "failure_type": "EVIDENCE_FAILURE",
  "evidence": [
    "claim_count=12",
    "supported_claims=5"
  ],
  "confidence": 0.87
}
```

---

# 10. Diagnosis 的职责

Diagnosis 只回答：

> **为什么失败？**

不要让它直接改代码。

输入：

```text
Task
+
Trace
+
Final Result
+
Review
```

输出：

```json
{
  "root_cause": "missing_evidence_verification",
  "component": "workflow",
  "confidence": 0.86
}
```

然后交给 Proposer。

这样：

```text
Diagnosis
≠
Proposer
```

避免一个 Agent 自己说：

> 我失败了，所以我决定修改自己。

---

# 11. Hypothesis

每个 Candidate 必须先有 Hypothesis。

格式：

```json
{
  "hypothesis":
    "Adding evidence verification after research
     will reduce unsupported claims.",

  "target_component": "workflow",

  "expected_behavior": [
    "increase_verified_claims",
    "decrease_unsupported_claims"
  ]
}
```

必须包含：

```text
Problem
Hypothesis
Target Component
Expected Behavior
Expected Cost
```

---

# 12. Evolution 分三级

这是 V2 的重要升级。

## Level 1：Parameter Evolution

第一优先。

例如：

```text
MAX_REPLAN
MAX_RETRIES
REVIEW_THRESHOLD
TIMEOUT
TOOL_BUDGET
MAX_STEPS
```

例如：

```text
MAX_REPLAN = 3
```

候选：

```text
1
2
4
5
```

优点：

- 安全
- 易归因
- 易回滚
- 实验成本低

---

## Level 2：Prompt / Guidance Evolution

例如：

```text
System Prompt
Tool Guidance
Reviewer Instruction
Memory Policy
Skill Description
```

---

## Level 3：Structural Evolution

最后才允许：

```text
Workflow
Agent Topology
Subagent
Context Pipeline
Tool
Script
```

原因：

> 改得越大，越难知道为什么有效。

---

# 13. Candidate Schema

建议：

```json
{
  "candidate_id": "cand_001",

  "base_commit": "abc123",

  "hypothesis_id": "hyp_004",

  "target_component": "workflow",

  "edit_scope": [
    "workflow/research.yaml"
  ],

  "expected_behavior": [
    "more_source_validation"
  ],

  "expected_cost_delta": "<20%",

  "patch": "...",

  "created_by": "proposer"
}
```

---

# 14. Frozen Core

这是 V2 必须有的。

Agent 可以修改：

```text
Prompt
Workflow
Tools
Memory Policy
Skills
Subagents
```

Agent 不可以修改：

```text
Evaluator
Verifier
Benchmark Split
Scoring
Experiment Protocol
Safety
Budget Enforcement
Trace Integrity
```

结构：

```text
                 Candidate
                     │
                     ▼
              Frozen-Core Gate
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
      Eval files   Split     Runtime policy
          │          │          │
          └──────────┼──────────┘
                     │
              reject if touched
```

---

# 15. Frozen Core 具体实现

候选提交前：

```bash
git diff --name-only base..candidate
```

然后检查：

```python
FROZEN_PATHS = {
    "adaptive_runtime/evaluation/",
    "adaptive_runtime/experiments/protocol/",
    "adaptive_runtime/policy/frozen/",
}
```

如果 candidate 修改这些路径：

```text
REJECT
reason = FROZEN_CORE_VIOLATION
```

进一步加入：

```text
hash evaluator files
```

运行前后比较：

```text
hash_before == hash_after
```

---

# 16. Noise Calibration

先不要规定：

```text
σ < 8
```

这种固定数字。

正确方式：

```text
Base Harness
 ↓
same task
 ↓
N repeated runs
 ↓
estimate noise
```

例如：

```text
Base scores:

72
69
71
70
73

mean = 71
std = 1.58
```

Candidate：

```text
75
76
74
77
75

mean = 75.4
```

那么：

```text
Δ = 4.4
```

再判断是否超过 evaluation noise。

---

# 17. Paired Evaluation

尽量：

```text
Task T
 ├── Base run
 └── Candidate run
```

而不是：

```text
一批 Task 跑 Base
另一批 Task 跑 Candidate
```

因为任务难度不同会污染结果。

理想：

```text
T1: Base / Candidate
T2: Base / Candidate
T3: Base / Candidate
...
```

然后：

```text
Δ_i = Candidate_i - Base_i
```

分析：

```text
mean(Δ)
median(Δ)
variance(Δ)
win_rate
```

---

# 18. Verifier 优先级

评价采用：

```text
Level 1:
Reliable Ground Truth / Verifier

Level 2:
Benchmark Judge

Level 3:
Behavioral Evidence

Level 4:
LLM Reflection
```

如果存在可靠 Ground Truth：

> 必须优先使用。

如果开放世界任务没有 verifier：

> 才使用 Behavioral Evidence。

因此：

```text
Behavior Improvement
≠
Task Correctness
```

必须明确区分。

---

# 19. Behavioral Evaluation

例如 Research Agent：

```text
Base:

sources = 4
primary_sources = 1
verified_claims = 8
unsupported_claims = 7

Candidate:

sources = 7
primary_sources = 5
verified_claims = 13
unsupported_claims = 2
```

可以说：

> Candidate 在“证据验证行为”上明显改善。

不能直接说：

> Candidate 最终答案一定更正确。

---

# 20. Multi-dimensional Score

不要只使用：

```text
score = 0.83
```

建议：

```json
{
  "task_quality": 0.83,
  "behavior_gain": 0.31,
  "transfer_gain": 0.12,
  "cost_delta": 0.08,
  "latency_delta": 0.05,
  "robustness": 0.91
}
```

---

# 21. Candidate Accept / Reject

第一版使用一个简单但明确的 Gate。

定义：

```text
ΔQ = quality(candidate) - quality(base)
ΔB = behavior(candidate) - behavior(base)
ΔC = cost(candidate) - cost(base)
ΔR = robustness(candidate) - robustness(base)
```

Candidate 至少满足：

```text
1. Frozen Core 未被修改
2. 无 benchmark leakage
3. ΔQ 或 ΔB 超过 noise floor
4. ΔC 不超过 budget
5. 没有明显 regression
```

可以定义：

```text
ACCEPT IF

EvidenceGain >= δ_noise
AND
CostDelta <= C_max
AND
Regression <= R_max
AND
FrozenCoreViolation == false
AND
Leakage == false
```

---

# 22. Quality / Behavior 的优先级

不要把所有指标粗暴加权。

推荐：

```text
第一层：Hard Constraints

Frozen Core
Leakage
Safety
Cost ceiling

        ↓

第二层：Primary Objective

Verifier / Task Quality

        ↓

第三层：Secondary Evidence

Behavior
Robustness
Transfer

        ↓

第四层：Optimization

Cost
Latency
Token
```

这样不会出现：

```text
答案错了
但行为很漂亮
所以总分很高
```

---

# 23. Cost Rule

例如：

```text
Base:
quality = 70
tokens = 10k

Candidate:
quality = 71
tokens = 30k
```

不应该自动接受。

定义：

```text
Quality Gain / Cost Increase
```

例如：

```text
ΔQ = +1
ΔTokens = +200%
```

Reject。

而：

```text
ΔQ = +8
ΔTokens = +12%
```

可能 Accept。

---

# 24. Edit Budget

第一版：

```text
Candidate 最多修改 1 个主要组件
```

例如：

```text
Prompt only
```

或者：

```text
Workflow only
```

不要：

```text
Prompt
+
Workflow
+
Tool
+
Memory
```

一起修改。

否则无法 attribution。

后续：

```text
P0: 1 edit
P1: 2 edits
P2: annealed budget
```

---

# 25. Failed Hypothesis Memory

如果：

```text
Hypothesis H1
```

失败：

```text
H1:
“增加 reviewer prompt 可以解决 tool failure”

Result:
No improvement
```

记录：

```json
{
  "hypothesis": "reviewer_prompt_improves_tool_failure",
  "result": "rejected",
  "evidence": "...",
  "tasks": ["T1", "T2"],
  "reason": "no measurable gain"
}
```

下次 Proposer 看到：

```text
H1
```

不能原样重新提出。

---

# 26. Exploration

如果系统一直修改：

```text
Prompt
Prompt
Prompt
Prompt
```

说明搜索空间坍缩。

维护：

```text
component_usage
```

例如：

```text
Prompt       12
Workflow      1
Tool          0
Memory        0
Subagent      0
```

此时提高：

```text
Workflow
Tool
Memory
```

的 exploration probability。

---

# 27. Candidate Critic

在运行昂贵实验之前：

```text
Candidate
 ↓
Critic
```

检查：

```text
Benchmark-specific hack?
Data leakage?
Frozen Core violation?
Unnecessary complexity?
Cost explosion?
Task-specific hard coding?
```

如果发现：

```text
Candidate:
if task_id == "T123":
    ...
```

直接 Reject。

---

# 28. Git Worktree

Candidate 不直接修改 main。

流程：

```text
main
 │
 ├── candidate/001
 ├── candidate/002
 └── candidate/003
```

更推荐：

```bash
git worktree add \
  ../worktrees/candidate-001 \
  -b candidate/001
```

每个 Candidate 独立：

```text
code
trace
logs
tests
```

完成后：

```text
ACCEPT
 ↓
merge
```

失败：

```text
delete worktree
```

RRSI 当前实现也采用 candidate git worktrees，每个 candidate 在独立 worktree/branch 上 draft、screen、evaluate，接受后再推进 incumbent。 

---

# 29. Base Harness 不应该无限累积

不要：

```text
Base
 ↓
v1
 ↓
v2
 ↓
v3
 ↓
v50
```

因为后面不知道哪个改动真正有效。

V2 采用：

```text
                 Base Harness
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
      Task A        Task B        Task C
        │             │             │
        ▼             ▼             ▼
    Candidate       Candidate     Candidate
        │             │             │
        └─────────────┼─────────────┘
                      ▼
                Evidence
                      │
                      ▼
             Experience Library
```

未来任务：

```text
Task D
 ↓
retrieve Experience
 ↓
Proposer
 ↓
new Candidate
```

而不是永久继承 Task A 的所有修改。

---

# 30. Experience 数据结构

建议 JSONL：

```json
{
  "experience_id": "exp_001",

  "problem": "unsupported_claims",

  "hypothesis":
    "evidence verification improves research reliability",

  "component": "workflow",

  "edit": "...",

  "positive_tasks": [
    "research_001",
    "research_005"
  ],

  "negative_tasks": [
    "coding_002"
  ],

  "quality_delta": 0.11,
  "behavior_delta": 0.31,
  "cost_delta": 0.08,

  "transfer_score": 0.14,

  "confidence": 0.87,

  "status": "validated"
}
```

---

# 31. Experience 不是普通 Memory

普通 Memory：

```text
“上次 Agent 做过什么。”
```

Experience：

```text
“什么问题”
+
“什么修改”
+
“为什么认为有效”
+
“在哪些任务有效”
+
“在哪些任务无效”
+
“成本是多少”
+
“泛化到哪里”
```

因此：

```text
Experience
=
Hypothesis
+
Evidence
+
Scope
+
Negative Evidence
+
Cost
+
Confidence
```

---

# 32. Transfer Evaluation

把任务分成：

```text
EVOLVE
HELD_OUT
OOD
```

例如：

```text
EVOLVE
├── research_001
├── research_002
└── finance_001

HELD_OUT
├── research_003
├── finance_002

OOD
├── coding_001
├── data_analysis_001
```

Evolution 阶段：

```text
只允许看到 EVOLVE
```

验证：

```text
HELD_OUT
OOD
```

---

# 33. Generalization Gap

定义：

```text
Evolution Gain
=
G_e

Held-out Gain
=
G_h

OOD Gain
=
G_o
```

重要指标：

```text
Generalization Gap
=
G_e - G_h
```

以及：

```text
OOD Transfer
=
G_o
```

理想：

```text
G_e > 0
G_h > 0
G_o > 0
```

如果：

```text
G_e = +20
G_h = +2
G_o = -3
```

说明：

> Harness 很可能 overfit。

---

# 34. Cross-task Evidence

不能：

```text
Task A +10%
```

就宣布：

> 这个 Lesson 泛化。

至少需要：

```text
Task A +10%
Task B +8%
Task C +7%
```

更理想：

```text
Task Family A
Task Family B
Task Family C
```

都能看到正向效果。

---

# 35. Successful + Failed Trajectory 对比

借鉴 ModularRSI：

```text
Task T

Success trajectory
        │
        ├── actions
        ├── tools
        └── observations

Failure trajectory
        │
        ├── actions
        ├── tools
        └── observations
```

比较：

```text
什么行为只出现在成功 trajectory？
什么行为经常出现在失败 trajectory？
```

然后：

```text
cross-task aggregation
```

寻找：

> recurring behavioral deficiency

ModularRSI 的核心就是利用成功/失败 trajectory 的 contrast，并跨任务聚合 recurring deficiencies，再在受限模块内演化。 

---

# 36. Modular Evolution：P2

第一版不做。

第二阶段把 Harness 拆成：

```text
Agent Loop
Tool Use
Observation Management
Context Management
Task Completion
```

这与 ModularRSI 的五模块划分接近。 

每个模块：

```text
独立 diagnosis
独立 candidate
独立 evaluation
```

最后：

```text
Integration
```

解决模块之间冲突。

---

# 37. Plugin Evolution：P3

后期将成功 Experience 抽象成：

```text
Plugin
```

例如：

```text
plugins/
├── evidence_verifier/
├── source_cross_checker/
├── retry_failed_tool/
├── completion_detector/
└── adversarial_reviewer/
```

PluginRSI 的核心思想就是：

> 把可复用机制原子化成 plugin，独立改进、放入共享库，再重新组合 Harness。 

V2 后期可以采用同样思想。

---

# 38. P0：第一版 MVP

第一版只做：

```text
1. Trace
2. Diagnosis
3. Hypothesis
4. Parameter / Prompt Candidate
5. Git Worktree
6. Probe
7. Behavioral Evaluation
8. Accept / Reject
9. Experience JSONL
```

暂时不做：

```text
Plugin
复杂 Memory Evolution
复杂 Agent Topology Evolution
自动修改 Evaluator
跨模型 Transfer
大规模 Benchmark
```

---

# 39. P0 的具体闭环

一个任务：

```text
Task
 ↓
Base Agent
 ↓
Trace
 ↓
Diagnosis
 ↓
Hypothesis
 ↓
Candidate A
Candidate B
 ↓
Critic
 ↓
Probe
 ↓
Compare
 ↓
Accept / Reject
 ↓
Experience
```

例如：

```text
Base:
MAX_REPLAN = 3

Diagnosis:
任务经常在第 4 步前过早结束

Hypothesis:
MAX_REPLAN=5 能减少 premature termination

Candidate:
MAX_REPLAN=5

Probe:
5 repeated runs

Result:
Base    68%
Candidate 76%

Cost:
+7%

Decision:
ACCEPT
```

---

# 40. P0 验收标准

P0 不以：

> “Agent 更聪明了”

作为验收。

而以：

```text
[✓] 每次运行都有 Trace
[✓] Trace 可以 replay
[✓] Diagnosis 输出结构化 root cause
[✓] 每个 Candidate 有 hypothesis
[✓] Candidate 使用独立 Git worktree
[✓] Frozen Core 无法被修改
[✓] Candidate 可以自动运行
[✓] Base/Candidate 可以重复比较
[✓] Candidate 可以 Accept/Reject
[✓] Experience 可以保存
[✓] 失败 Candidate 可以回滚
```

全部通过，才算 P0 完成。

---

# 41. P1：真正开始研究泛化

加入：

```text
Multi-task
Held-out
Noise Calibration
Cost-aware Selection
Failed Hypothesis
Experience Retrieval
Transfer Evaluation
Regression
```

P1 验收：

```text
[✓] 同一 Task 支持 repeated runs
[✓] 多 Task paired evaluation
[✓] Held-out 不参与 evolution
[✓] OOD 不参与 evolution
[✓] 能计算 Generalization Gap
[✓] 能记录 negative evidence
[✓] Experience 可以被新任务检索
```

---

# 42. P2：Regularized RSI

加入：

```text
Annealed Edit Budget
Candidate Critic
Exploration
Pruner
Cost Rule
Stalled Search Detection
```

P2 目标：

> 防止 Agent 为了 benchmark 分数不断堆复杂度。

RRSI 的实验结果表明，regularized evolution 的目的不仅是提高 evolve split，还包括缩小 OOD 损失并控制 policy-token 成本。V2 应借鉴这种“收益 + 泛化 + 成本”三者一起考虑的思路。 

---

# 43. P3：Modular / Plugin RSI

最终：

```text
Diagnosis
 ↓
Module
 ↓
Plugin
 ↓
Experience Library
 ↓
Composition
```

形成：

```text
Adaptive Agent Runtime
+
Reusable Improvement Library
```

---

# 44. 第一周实施计划

## Day 1：Trace

完成：

```text
TraceEvent
RunContext
RunSummary
```

目标：

```text
任何一次 Agent Run 都能完整记录。
```

---

## Day 2：Evaluation Contract

定义：

```text
Task
Run
Score
Evidence
Cost
```

同时冻结：

```text
Evaluator
Protocol
Benchmark split
```

---

## Day 3：Diagnosis

实现：

```text
FailureAnalyzer
FailureTaxonomy
```

先用 LLM + rule hybrid。

例如：

```text
Rule:
tool error > 2
→ TOOL_FAILURE

Rule:
unsupported claims > threshold
→ EVIDENCE_FAILURE
```

LLM 再负责复杂 root cause。

---

## Day 4：Candidate

实现：

```text
Candidate
Hypothesis
Patch
```

先支持：

```text
Parameter
Prompt
```

---

## Day 5：Probe

实现：

```text
Base
Candidate
```

自动：

```text
git worktree
run
collect trace
```

---

## Day 6：Comparator

输出：

```text
Base
Candidate
Δ
Cost
Behavior
Decision
```

---

## Day 7：Experience

保存：

```text
hypothesis
patch
evidence
tasks
cost
decision
```

最终跑通：

```text
Failure
 ↓
Diagnosis
 ↓
Hypothesis
 ↓
Candidate
 ↓
Probe
 ↓
Comparison
 ↓
Experience
```

---

# 45. 第一版不要追求“自动改任何 Python”

这是一个重要限制。

第一版 Candidate 修改范围：

```text
A. Parameters
B. Prompt
C. Tool Guidance
```

例如：

```text
config.py
prompt.py
tool_guidance.py
```

而不是：

```text
orchestrator.py
agent.py
evaluation.py
```

原因：

> 先验证 Adaptive Loop，而不是验证 Agent 能不能乱改代码。

---

# 46. 第一版实验任务

建议不要一开始就拿“开放世界所有任务”。

先建立 3 个 Task Family：

```text
Research
Finance
Coding
```

每类：

```text
10 tasks
```

即：

```text
30 tasks
```

划分：

```text
18 evolve
6 held-out
6 OOD
```

第一版不要求高质量 benchmark。

重点是：

> 能不能验证“一个 Harness 修改在新任务上仍然有效”。

---

# 47. 一个完整实验

例如：

```text
Hypothesis H1

Problem:
Research Agent evidence verification weak.

Candidate:
Add verification instruction.

Evolution:
6 tasks

Base:
68.2

Candidate:
75.4

Gain:
+7.2
```

然后：

```text
Held-out:
Base = 70.1
Candidate = 74.0

Gain:
+3.9
```

OOD：

```text
Base = 62.0
Candidate = 64.1

Gain:
+2.1
```

结论：

```text
Evolution Gain = +7.2
Held-out Gain = +3.9
OOD Gain = +2.1
```

这是比较有意义的泛化证据。

---

# 48. 如果出现这种结果

```text
Evolution = +20
Held-out = +1
OOD = -4
```

不要 merge。

Experience 状态：

```text
status = overfit
```

同时记录：

```text
negative evidence
```

未来 Proposer：

> 不应该再次提出完全相同的修改。

---

# 49. 如果出现这种结果

```text
Evolution = +7
Held-out = +6
OOD = +5
Cost = +3%
```

则：

```text
status = validated
```

Experience：

```text
confidence = high
scope = research + analysis
```

未来任务可以检索它。

---

# 50. Experience Retrieval

新任务：

```text
“分析 ETH 当前市场状态”
```

系统先提取：

```text
task type = financial research
failure risk = evidence
```

检索：

```text
Experience #001
Experience #014
Experience #031
```

然后交给 Proposer：

```text
Relevant lessons:
...
```

注意：

> Retrieval 只提供候选经验，不自动强制使用。

最终仍然：

```text
Hypothesis
→ Probe
→ Validate
```

这样不会因为旧经验错误而污染所有任务。

---

# 51. 一个关键原则：经验也是 Hypothesis

不要：

```text
Experience = Truth
```

应该：

```text
Experience = Evidence-supported hypothesis
```

例如：

```text
“Evidence verification is usually helpful
for research tasks”
```

而不是：

```text
“所有任务必须 evidence verification”
```

---

# 52. Regression Test

Candidate 即使提升当前 Task：

```text
Task A +10%
```

也必须检查：

```text
Task B
Task C
Task D
```

是否退化。

定义：

```text
Regression Rate
=
# tasks with significant negative Δ
/
# evaluated tasks
```

如果：

```text
Regression Rate > threshold
```

Reject。

---

# 53. 未来可以加入 Bandit / Bayesian Search

当 Candidate 数量增加后：

```text
Prompt A
Prompt B
Prompt C
Tool A
Workflow A
...
```

可以使用：

```text
Multi-armed Bandit
Bayesian Optimization
Evolutionary Search
```

选择下一批 Candidate。

但：

> 第一版不需要。

第一版：

```text
LLM proposer
+
controlled experiment
```

足够。

---

# 54. 未来可以加入自动 Experiment Planner

后期：

```text
Diagnosis
 ↓
Experiment Planner
```

自动决定：

```text
应该测试 Prompt？
还是 Workflow？
还是 Tool？
需要几次重复？
需要哪些 Task？
```

最终：

```text
Hypothesis
 ↓
Experiment Design
 ↓
Candidate
 ↓
Probe
```

这是比“自动改 Prompt”更有价值的能力。

---

# 55. 研究价值

最终这个项目不只是：

> Agent Framework。

而是可以研究：

### RQ1

```text
Behavioral contrast
是否可以作为 open-world Agent improvement signal？
```

### RQ2

```text
Diagnosis 是否比直接 Prompt Evolution 更有效？
```

### RQ3

```text
Parameter Evolution
vs
Prompt Evolution
vs
Structural Evolution
```

### RQ4

```text
Single-task optimization
vs
Cross-task experience
```

### RQ5

```text
Unregularized RSI
vs
Regularized RSI
```

### RQ6

```text
Monolithic Harness
vs
Modular Harness
```

### RQ7

```text
Experience Library
是否可以跨模型复用？
```

---

# 56. 最终实验矩阵

| Experiment | Baseline | Treatment | 主要问题 |
|---|---|---|---|
| E1 | Base | Adaptive | Adaptive 是否有效 |
| E2 | Prompt-only | Diagnosis-guided | Diagnosis 是否有价值 |
| E3 | Single-task | Multi-task | 是否泛化 |
| E4 | No regularization | RRSI-style | 是否减少 overfit |
| E5 | Monolithic | Modular | 是否更容易归因 |
| E6 | No experience | Experience retrieval | 是否可以复用 |
| E7 | Full harness merge | Lesson-only | 哪种长期更稳定 |
| E8 | Behavior-only | Verifier+Behavior | Ground Truth 存在时是否更可靠 |

---

# 57. 最终系统的核心数据流

```text
Task
 │
 ▼
Experience Retrieval
 │
 ▼
Existing Multi-Agent
 │
 ▼
Trace
 │
 ▼
Diagnosis
 │
 ▼
Hypothesis
 │
 ▼
Candidate Generator
 │
 ▼
Candidate Critic
 │
 ▼
Frozen Core Gate
 │
 ▼
Git Worktree
 │
 ▼
Repeated Probe
 │
 ▼
Verifier / Behavior / Cost
 │
 ▼
Comparator
 │
 ├── Reject
 │
 └── Accept
        │
        ▼
   Cross-task Validation
        │
        ▼
   Experience Library
```

---

# 58. 最重要的设计边界

### Agent 可以修改：

```text
Prompt
Tool Guidance
Parameter
Workflow
Memory Policy
Skill
Subagent
Tool
Plugin
```

### Agent 不能修改：

```text
Evaluator
Verifier
Benchmark Split
Experiment Protocol
Safety
Budget Enforcement
Trace Integrity
```

### Agent 可以提出：

```text
Hypothesis
```

### Agent 不能自己宣布：

```text
“我已经证明这个修改有效。”
```

证明必须来自：

```text
Evaluation Runtime
```

---

# 59. V2 与 Harness² / RRSI / ModularRSI / PluginRSI 的关系

```text
Harness²
│
├── Task-level recursion
│      └── Propose → Probe → Compose
│
└── Harness-level recursion
       └── Experience / Lessons


RRSI
│
├── Edit Budget
├── Failed Hypothesis
├── Exploration
├── Critic
├── Noise Floor
├── Cost Rule
└── Pruning


ModularRSI
│
├── Success/Failure Contrast
├── Cross-task Evidence
├── Benchmark-disjoint Evolution
└── Modular Evolution


PluginRSI
│
├── Atomized Plugins
├── Shared Plugin Library
└── Plugin Recombination


Adaptive Agent Runtime V2
│
├── Existing Multi-Agent Runtime
├── Diagnosis
├── Controlled Experiment
├── Frozen Core
├── Noise-aware Evaluation
├── Cross-task Transfer
├── Experience Library
├── Regularized Evolution
└── Future Modular / Plugin Evolution
```

因此项目定位不是：

> Harness² Clone。

而是：

> **A diagnosis-guided, evidence-driven adaptive agent runtime for reusable and generalizable harness evolution.**

---

# 60. 最终 README 可以这样描述

```text
Adaptive Agent Runtime

An evidence-driven runtime that enables an agent
to diagnose its own execution failures, propose
controlled harness modifications, evaluate them
under frozen evaluation protocols, and accumulate
validated improvements across tasks.
```

中文：

> **一个能够观察 Agent 执行过程、定位 Harness 瓶颈、提出可验证修改、通过受控实验选择改进，并将跨任务验证后的机制沉淀为可复用经验的 Adaptive Agent Runtime。**

---

# 61. 最终路线图

```text
                     Adaptive Agent Runtime V2
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
       P0                    P1                    P2/P3
   可观察 + 可实验          可泛化                可进化
        │                     │                     │
        ▼                     ▼                     ▼
      Trace              Multi-task             RRSI
      Diagnosis          Held-out               ModularRSI
      Hypothesis         OOD                    PluginRSI
      Candidate          Transfer
      Probe              Experience
      Compare            Regression
      Git
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                    Adaptive Agent Runtime
                              │
                              ▼
                 Reusable Improvement Library
```

---

# 62. 第一阶段真正应该写的代码

不要一次写几十个模块。

第一批只需要：

```text
adaptive_runtime/
├── trace.py
├── diagnosis.py
├── candidate.py
├── probe.py
├── evaluator.py
├── comparator.py
└── experience.py
```

然后把现有：

```text
Planner
Executor
Reviewer
Memory
```

接进去。

第一版只允许：

```text
Parameter
Prompt
Tool Guidance
```

三种修改。

---

# 63. P0 的 Definition of Done

最终必须能够执行：

```bash
python run_adaptive.py \
    --task tasks/research_001.json \
    --mode evolve
```

系统自动完成：

```text
1. Run Base Harness
2. Save Trace
3. Diagnose failure
4. Generate hypothesis
5. Generate Candidate A/B
6. Create Git worktrees
7. Run candidates
8. Collect traces
9. Compare
10. Accept / Reject
11. Save Experience
12. Produce report
```

最终输出：

```text
runs/
└── research_001/
    ├── base/
    │   ├── trace.json
    │   └── result.json
    │
    ├── candidates/
    │   ├── cand_001/
    │   └── cand_002/
    │
    ├── comparison.json
    ├── decision.json
    └── experience.json
```

---

# 64. 最终报告示例

```text
Adaptive Run Report
===================

Task:
research_001

Base:
score = 68
tokens = 12000

Diagnosis:
EVIDENCE_FAILURE

Hypothesis:
Add evidence verification after research.

Candidate:
workflow modification

Probe:
5 paired runs

Result:
Base      = 68.2
Candidate = 75.4

Behavior:
unsupported_claims
7.2 → 2.4

Cost:
+8%

Decision:
ACCEPT

Transfer:
held-out = +3.9
OOD = +2.1

Experience:
exp_001

Confidence:
HIGH
```

这才是这个项目真正有研究价值的“可审计证据链”。

---

# 65. 最终原则

整个项目只需要记住一句话：

> **不要让 Agent 自己“觉得自己变好了”；让 Agent 提出假设，让独立的 Evaluation Runtime 用受控实验判断它是否真的变好了。**

进一步：

> **不要因为一个 Task 上变好了，就认为它学到了通用能力；必须经过多任务、Held-out 和 OOD 验证，才能把它提升为可复用 Experience。**

最终形成：

```text
Observe
  ↓
Diagnose
  ↓
Hypothesize
  ↓
Experiment
  ↓
Measure
  ↓
Validate
  ↓
Remember
  ↓
Transfer
  ↓
Repeat
```

这就是 `agent_dev/multi-agent` 从普通 Multi-Agent 项目升级为 **Adaptive Agent Runtime** 的核心。
