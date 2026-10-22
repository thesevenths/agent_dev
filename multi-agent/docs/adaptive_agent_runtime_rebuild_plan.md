# Adaptive Agent Runtime：`agent_dev/multi-agent` 完整改造方案

> 目标：把现有 Multi-Agent 系统从“能完成任务的 Agent”升级为“能够观察执行、诊断失败、实验性修改自身 Harness、沉淀经验并验证跨任务泛化”的 **Adaptive Agent Runtime**。
>
> 核心思想：**不是自动改 Prompt，而是证据驱动地优化 Agent Harness。**

---

## 1. 项目定位

### 1.1 当前项目

现有 `agent_dev/multi-agent` 可以理解为：

```text
User Task
   ↓
Planner
   ↓
Multi-Agent
   ↓
Tools / Memory
   ↓
Reviewer
   ↓
Final Answer
```

它解决的是：

> 如何通过 Planner / Executor / Reviewer / Memory 等组件完成一个复杂任务。

### 1.2 改造后的项目

目标变成：

```text
                    User Task
                        │
                        ▼
                 Adaptive Runtime
                        │
          ┌─────────────┴─────────────┐
          │                           │
          ▼                           ▼
     Agent Runtime              Improvement Runtime
          │                           │
 Planner / Agents / Tools       Trace / Diagnosis
 Memory / Workflow              Proposal / Experiment
          │                       Comparison
          ▼                           │
       Execution                       │
          │                           ▼
          └───────────────→ Experience Library
                                    │
                                    ▼
                              Future Tasks
```

一句话定位：

> **An evidence-driven Adaptive Agent Runtime that recursively improves agent harnesses across tasks.**

---

# 2. 为什么值得做

Google Research 的 Harness² 把 Agent Harness 定义为模型周围的 Prompt、Tools、Memory、Skills、Orchestration 等系统，并通过 Propose → Probe → Compose 进行递归优化。

但后续 RRSI 和 ModularRSI 进一步暴露并处理了两个问题：

1. Harness 在固定任务上持续优化容易 overfit；
2. 单个任务的一条 trajectory 可能把“任务偶然性”误认为“系统性缺陷”。

因此本项目不应该做成 Harness² 的简单 clone，而应该加入：

- Failure Diagnosis
- Component-level Evolution
- Cross-task Validation
- Transfer Measurement
- Cost / Robustness
- Regularized Evolution
- Experience Pruning

这样项目具有更完整的研究闭环。

---

# 3. 核心研究问题

项目围绕以下 6 个问题展开：

### RQ1：Agent 为什么失败？

从 execution trace 中自动定位：

```text
Planning Failure
Tool Failure
Evidence Failure
Context Failure
Delegation Failure
Memory Failure
Workflow Failure
Task Completion Failure
```

### RQ2：应该改什么？

不是所有问题都修改 Prompt。

系统需要判断：

```text
Prompt
Workflow
Tool
Tool Guidance
Memory
Subagent
Skill
Script
Context Management
```

哪个组件最可能是根因。

### RQ3：怎么验证修改有效？

没有 Ground Truth 时：

```text
Hypothesis
   ↓
Expected Behavior
   ↓
Base vs Candidate
   ↓
Behavioral Evidence
```

如果有可靠 verifier，则优先使用 verifier。

因此评价体系应该是：

```text
Ground Truth / Verifier
        ↓
优先使用

没有 Verifier
        ↓
Behavioral Evidence
        ↓
作为 fallback
```

### RQ4：一次改动是否真的有效？

通过 controlled experiment：

```text
Base Harness
Candidate A
Candidate B
Candidate C
```

尽量一次只改变一个主要变量，避免 attribution problem。

### RQ5：当前 Task 的改进能否泛化？

不能：

```text
Task A 改好了
↓
直接保存
↓
所有任务使用
```

而应该：

```text
Task A
 ↓
Lesson
 ↓
Task B
 ↓
Task C
 ↓
Task D
 ↓
Transfer Evidence
```

### RQ6：持续自我改进会不会 overfit？

需要：

- Edit Budget
- Cost Constraint
- Failed Hypothesis History
- Candidate Critic
- Pruning
- Exploration
- Held-out Tasks
- Cross-task Transfer

---

# 4. 总体架构

```text
                         ┌───────────────────────┐
                         │   Adaptive Runtime    │
                         └───────────┬───────────┘
                                     │
                      ┌──────────────┴──────────────┐
                      │                             │
                      ▼                             ▼
              Agent Runtime                 Improvement Runtime
                      │                             │
          ┌───────────┼───────────┐       ┌─────────┼─────────┐
          ▼           ▼           ▼       ▼         ▼         ▼
       Planner     Agents       Tools   Trace   Diagnosis  Memory
          │           │           │       │         │         │
          └───────────┼───────────┘       └─────────┼─────────┘
                      ▼                             ▼
                  Execution                    Hypothesis
                      │                             │
                      ▼                             ▼
                    Trace                       Proposer
                                                    │
                                  ┌─────────────────┼────────────────┐
                                  ▼                 ▼                ▼
                               Prompt            Tool            Workflow
                               Candidate         Candidate       Candidate
                                  │                 │                │
                                  └─────────────────┼────────────────┘
                                                    ▼
                                                  Probe
                                                    │
                                                    ▼
                                              Comparator
                                                    │
                                    ┌───────────────┼───────────────┐
                                    ▼               ▼               ▼
                                 Behavior         Cost          Robustness
                                    │               │               │
                                    └───────────────┼───────────────┘
                                                    ▼
                                                 Composer
                                                    │
                                                    ▼
                                            Improved Harness
                                                    │
                                                    ▼
                                                 Reflector
                                                    │
                                                    ▼
                                          Experience Library
                                                    │
                                                    ▼
                                          Cross-task Retrieval
```

---

# 5. Harness 的可编辑组件

建议统一抽象为：

```text
Harness
├── prompt
├── workflow
├── agents
├── tools
├── tool_guidance
├── memory
├── skills
├── scripts
└── context
```

## 5.1 Prompt

解决：

> Agent 思考方式、任务规范、输出格式不合理。

例：

```text
原：
分析 BTC。

改：
分析 BTC 时必须分别检查：
Trend
Volatility
Liquidity
Funding
Open Interest
Market Regime
Risk
```

---

## 5.2 Workflow

解决：

> 执行流程本身不合理。

例：

```text
原：

Research
 ↓
Answer
```

改为：

```text
Research
 ↓
Evidence Verification
 ↓
Analysis
 ↓
Reviewer
 ↓
Answer
```

---

## 5.3 Tool

解决：

> Agent 缺少必要能力。

例如：

```text
get_price()
get_volume()
get_funding_rate()
get_open_interest()
```

---

## 5.4 Tool Guidance

解决：

> Agent 有工具，但不知道什么时候使用。

例如：

```text
分析 Crypto Market Regime 前必须获取：

price
volume
funding
open_interest
```

---

## 5.5 Subagents

解决：

> 单 Agent 任务过载。

例如：

```text
             Planner
                 │
      ┌──────────┼──────────┐
      ▼          ▼          ▼
   Macro       Market      Risk
   Agent       Agent       Agent
      │          │          │
      └──────────┼──────────┘
                 ▼
             Synthesizer
```

---

## 5.6 Memory

解决：

> 过去已经发现的问题在后续任务中重复发生。

例如：

```text
Lesson:
不要仅根据新闻标题判断市场 Regime。
```

---

## 5.7 Skill

解决：

> 某类任务重复出现。

例如：

```text
quant_market_analysis/
research_report/
source_verification/
```

---

## 5.8 Script

解决：

> 确定性、重复性的程序任务。

例如：

```text
download_data.py
calculate_factor.py
backtest.py
```

---

# 6. Trace：整个系统的基础

第一阶段最重要的不是 Self-Improvement，而是把执行过程记录完整。

建议 Trace 至少包含：

```json
{
  "task_id": "...",
  "harness_version": "v17",
  "agent": "...",
  "step": 12,
  "action": "...",
  "tool": "...",
  "tool_input": "...",
  "tool_output": "...",
  "observation": "...",
  "decision": "...",
  "error": "...",
  "tokens": 1200,
  "latency_ms": 2300
}
```

任务级统计：

```json
{
  "tool_calls": 37,
  "tool_failures": 2,
  "sources": 18,
  "primary_sources": 9,
  "claims": 32,
  "verified_claims": 25,
  "unsupported_claims": 4,
  "review_findings": 8,
  "revisions": 3,
  "tokens": 42000,
  "latency": 83
}
```

**没有 Trace，后面的 Diagnosis 和 Evolution 都会变成猜测。**

---

# 7. Failure Diagnosis

这是项目最重要的新模块之一。

输入：

```text
Task
+
Execution Trace
+
Final Result
+
Reviewer Findings
```

输出：

```json
{
  "failure": "unsupported_claims",
  "evidence": [
    "8 claims have no supporting source"
  ],
  "root_cause": "missing_evidence_verification",
  "target_component": "workflow",
  "confidence": 0.86
}
```

重点：

> Diagnosis 不负责修改系统，只负责回答“为什么失败”。

---

# 8. Hypothesis / Proposer

Diagnosis 后再提出修改假设。

例如：

```json
{
  "problem": "unsupported_claims",

  "hypothesis":
    "增加独立证据验证阶段可以减少未经支持的结论",

  "target_component": "workflow",

  "edit": {
    "insert": "evidence_verifier"
  },

  "expected_behavior": [
    "increase_verified_claims",
    "decrease_unsupported_claims"
  ]
}
```

关键原则：

> **每一个 Candidate 必须包含 Hypothesis。**

不要允许：

```text
“我感觉这样可能更好。”
```

而应该：

```text
Hypothesis
+
Target Component
+
Expected Behavior
+
Actual Evidence
```

---

# 9. Controlled Experiment

候选修改：

```text
Base
Candidate A
Candidate B
Candidate C
```

建议第一阶段：

> 一次只修改一个主要组件。

例如：

```text
A = Prompt modification
B = Workflow modification
C = Tool Guidance modification
```

不要第一版就：

```text
Prompt
+
Tool
+
Memory
+
Workflow
+
Subagent
```

否则无法知道效果来源。

---

# 10. Probe

所有 Candidate 在相同条件下运行：

```text
same task
same model
same tools
same budget
same seed where possible
```

记录：

```text
Trace A
Trace B
Trace C
```

然后进行对比。

---

# 11. Behavioral Comparator

没有 Ground Truth 时，不要假装知道“答案正确”。

比较：

```text
Expected Behavior
        ↓
Observed Behavior
        ↓
Behavioral Difference
```

例如：

```text
                     Base    Candidate

Primary sources        3          8
Cross validation       1          7
Unsupported claims     9          3
Reviewer findings      8          4
```

那么可以判断：

> Candidate 明显实现了预期行为变化。

但必须记录：

```text
Behavior Improvement
≠
Proven Task Correctness
```

---

# 12. Evaluation Hierarchy

评价系统采用优先级：

```text
Level 1
可靠 Ground Truth / Verifier
        ↓
Level 2
Benchmark / Automated Judge
        ↓
Level 3
Behavioral Evidence
        ↓
Level 4
LLM Reflection
```

原则：

> **能用可靠 verifier，就不要只用 behavioral evidence。**

Behavioral evidence 是开放世界中的 fallback，而不是永远替代 Ground Truth。

---

# 13. Multi-dimensional Evaluation

不要只有一个 Score。

建议：

```text
Task Quality
Behavior Evidence
Transfer
Cost
Robustness
Regression
```

例如：

```text
Candidate A

Task Quality       +8%
Behavior           +30%
Cost               +15%
Transfer           +2%
Robustness         +5%
```

Candidate B：

```text
Task Quality       +6%
Behavior           +20%
Cost               -5%
Transfer           +18%
Robustness         +12%
```

长期而言，Candidate B 可能更值得采用。

---

# 14. Composer

Composer 根据：

```text
Candidates
+
Execution Evidence
+
Diagnosis
+
Evaluation
```

产生：

```text
Improved Harness
```

同时记录：

```json
{
  "accepted_candidate": "B",
  "reason": "...",
  "evidence": "...",
  "score_delta": 0.12,
  "cost_delta": -0.05
}
```

---

# 15. Experience Library

不要把 Memory 设计成普通聊天记忆。

应该设计成：

# Improvement / Experience Library

例如：

```text
Lesson #001
────────────────────
Problem:
unsupported claims

Modification:
Evidence Verification

Scope:
Research / Analysis

Evidence:
5 tasks

Success:
4

Failure:
1

Confidence:
0.82
```

---

# 16. Experience 的泛化

经验不能：

```text
Task A
↓
直接复制给 Task B
```

而应该：

```text
Task A
 ↓
Lesson
 ↓
Applicability
 ↓
Task B
 ↓
Validation
 ↓
Task C
 ↓
Validation
```

例如：

```text
Lesson #001

Task A    +18%
Task B    +12%
Task C     +9%
Task D    +11%

Transfer confidence = High
```

这时才能认为：

> 该经验可能是 Generalizable Mechanism。

---

# 17. Task-specific vs Generalizable

项目必须专门区分：

```text
Task-specific improvement
```

和：

```text
Generalizable improvement
```

例如：

```text
Task A:
+20%

Task B:
+1%

Task C:
-2%
```

说明：

```text
可能是 Task-specific
```

而：

```text
Task A:
+10%

Task B:
+9%

Task C:
+8%

Task D:
+11%
```

才说明：

```text
可能是 Generalizable
```

建议定义：

```text
Transfer Gain
=
平均 held-out task improvement
```

---

# 18. Cross-task Validation

建议把任务分成：

```text
Evolution Set
Held-out Set
```

甚至：

```text
Domain A
Domain B
Domain C
```

改进只允许看到 Evolution Set。

最终在：

```text
Held-out tasks
```

验证。

这样才能真正研究：

> Agent Harness 是否学到了机制，而不是背题。

---

# 19. Overfitting Protection

这是第二阶段必须加入的模块。

参考 RRSI 的思想，加入：

### 19.1 Edit Budget

限制一次 Candidate 能修改多少个独立组件。

---

### 19.2 Failed Hypothesis History

如果：

```text
Hypothesis X
```

已经失败，不允许 Proposer 原样重复。

---

### 19.3 Candidate Critic

在真正运行之前检查：

```text
是否存在 benchmark-specific hack？
是否只是针对某个 task？
是否泄漏答案？
是否增加不合理复杂度？
```

---

### 19.4 Noise-adjusted Threshold

如果：

```text
Base = 70 ± 3
Candidate = 71
```

不要认为：

```text
+1 = improvement
```

因为可能只是 evaluation noise。

---

### 19.5 Cost Rule

如果：

```text
quality +1%
tokens +100%
```

应该拒绝。

---

### 19.6 Pruning

如果某组件：

```text
长期没有贡献
```

降低优先级或删除。

---

# 20. Exploration

如果系统一直优化：

```text
Prompt
Prompt
Prompt
Prompt
```

说明搜索空间坍缩。

需要：

```text
Prompt
Workflow
Tool
Memory
Subagent
Skill
Context
```

之间进行 exploration。

例如：

```text
当前 10 次：
Prompt 改动占 90%

→ Exploration signal

下一轮：
Tool / Workflow / Memory
获得更高探索概率
```

---

# 21. Git / Versioning

强烈建议 Harness 使用 Git 管理。

结构：

```text
harness/
├── prompt/
├── workflow/
├── tools/
├── agents/
├── memory/
├── skills/
└── config/
```

每个 Candidate：

```text
branch: candidate/<id>
```

Accepted：

```text
merge → main
```

这样每次 Self-Improvement 都可以：

```text
git diff
git log
git revert
```

而且研究结果可复现。

---

# 22. 推荐代码结构

建议最终：

```text
adaptive_runtime/
│
├── runtime/
│   ├── executor.py
│   ├── trace.py
│   └── context.py
│
├── diagnosis/
│   ├── failure_analyzer.py
│   ├── root_cause.py
│   └── hypothesis.py
│
├── evolution/
│   ├── proposer.py
│   ├── candidate.py
│   ├── probe.py
│   ├── comparator.py
│   └── composer.py
│
├── harness/
│   ├── prompt.yaml
│   ├── workflow.yaml
│   ├── tools.yaml
│   ├── agents.yaml
│   ├── memory.yaml
│   └── skills/
│
├── experience/
│   ├── lessons.py
│   ├── retrieval.py
│   ├── confidence.py
│   └── pruning.py
│
├── evaluation/
│   ├── verifier.py
│   ├── behavior.py
│   ├── cost.py
│   ├── robustness.py
│   └── transfer.py
│
└── experiments/
    ├── runs/
    ├── configs/
    └── reports/
```

---

# 23. 与本仓库 `multi-agent` 的关系

不要重写。

采用：

```text
现有 Multi-Agent
        ↓
保留
        ↓
作为 Agent Runtime
        ↓
外面增加 Adaptive Runtime
```

也就是说：

```text
                    Adaptive Runtime
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
        ▼                                   ▼
Improvement System                  Existing Multi-Agent
        │                                   │
Diagnosis / Proposal                  Planner
Experiment / Compare                  Researcher
Experience / Transfer                 Executor
        │                              Reviewer
        └───────────────┬───────────────────┘
                        ▼
                     Trace
```

---

# 24. 第一阶段不要做太大

MVP 只实现：

```text
Task
 ↓
Base Harness
 ↓
Trace
 ↓
Diagnosis
 ↓
2 Candidates
 ↓
Probe
 ↓
Behavior Comparison
 ↓
Choose
 ↓
Save Lesson
```

只支持 3 个可修改组件：

```text
Prompt
Workflow
Tool Guidance
```

不要第一版就实现：

```text
Plugin
Memory Evolution
Agent Topology Evolution
Script Evolution
```

---

# 25. MVP 示例

任务：

> 分析 BTC 当前市场状态。

Base：

```text
Research
 ↓
Analysis
 ↓
Answer
```

执行发现：

```text
unsupported claims = 8
```

Diagnosis：

```text
Evidence verification missing
```

产生：

```text
Candidate A:
Prompt 修改

Candidate B:
Workflow 增加 Evidence Verification
```

执行：

```text
Base:
unsupported = 8

A:
unsupported = 5

B:
unsupported = 2
```

选择：

```text
B
```

保存：

```text
Lesson #001

Research tasks with factual claims
should include evidence verification.
```

下一个任务：

> 分析 ETH 市场。

自动检索 Lesson #001。

然后：

```text
Base:
unsupported = 7

With Lesson:
unsupported = 3
```

于是：

```text
Transfer Evidence = positive
```

这就是一个完整的 Adaptive Runtime demo。

---

# 26. 第二阶段

加入：

```text
Tool evolution
Memory evolution
Subagent evolution
Skill evolution
Script evolution
```

同时加入：

```text
Cost
Robustness
Regression
```

---

# 27. 第三阶段

加入：

```text
Held-out tasks
Cross-domain transfer
Model transfer
Overfitting detection
Pruning
Exploration
```

最终实验：

```text
Base Agent
vs
Adaptive Agent
```

比较：

```text
Task Quality
Cost
Latency
Tool Calls
Generalization
Robustness
```

---

# 28. 第四阶段：研究级版本

研究重点可以变成：

### Experiment 1

```text
Prompt-only Evolution
vs
Component-level Evolution
```

### Experiment 2

```text
Single-task Experience
vs
Cross-task Experience
```

### Experiment 3

```text
No Diagnosis
vs
Diagnosis-guided Evolution
```

### Experiment 4

```text
No Transfer Validation
vs
Cross-task Validation
```

### Experiment 5

```text
Unregularized RSI
vs
Regularized RSI
```

### Experiment 6

```text
Behavior-only Selection
vs
Verifier + Behavior Hybrid
```

这些实验非常适合形成 GitHub README、技术博客甚至论文式报告。

---

# 29. 与 Harness² / RRSI / ModularRSI 的关系

不要把项目宣传成：

> “我复现了 Harness²。”

更好的定位是：

```text
Harness²
    ↓
Propose / Probe / Compose

RRSI
    ↓
Regularized Evolution

ModularRSI
    ↓
Modular + Cross-task Generalization

本项目
    ↓
Failure Diagnosis
+
Component-level Evolution
+
Behavior / Verifier Hybrid
+
Cross-task Transfer
+
Cost / Robustness
+
Regularized Evolution
```

因此可以定位成：

> **Adaptive Agent Runtime：evidence-driven, diagnosis-guided, cross-task recursive optimization of agent harnesses.**

---

# 30. 最终目标架构

最终希望形成：

```text
                           USER TASK
                               │
                               ▼
                       ┌───────────────┐
                       │ Task Analyzer │
                       └───────┬───────┘
                               │
                               ▼
                     Experience Retrieval
                               │
                               ▼
                      Current Harness
                               │
                               ▼
                         Agent Runtime
                               │
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
                     Component Selection
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
           Prompt           Workflow          Tool
              │                │                │
              └────────────────┼────────────────┘
                               ▼
                         Candidate Harnesses
                               │
                               ▼
                              Probe
                               │
                               ▼
                         Evidence Collector
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
             Quality        Behavior         Cost
                │              │              │
                └──────────────┼──────────────┘
                               ▼
                           Comparator
                               │
                               ▼
                            Composer
                               │
                               ▼
                       Improved Harness
                               │
                               ▼
                         Held-out Tasks
                               │
                               ▼
                         Transfer Test
                               │
                               ▼
                       Experience Library
                               │
                               ▼
                       Future Tasks
```

---

# 31. 最终评价指标

建议 README 最终展示：

| 指标 | 含义 |
|---|---|
| Task Success | 任务完成率 |
| Behavioral Gain | 预期行为改善 |
| Transfer Gain | 跨任务收益 |
| OOD Gain | 未见任务收益 |
| Cost Delta | Token / API 成本变化 |
| Latency Delta | 延迟变化 |
| Tool Efficiency | Tool 使用效率 |
| Regression Rate | 其他任务退化比例 |
| Lesson Reuse Rate | 经验被成功复用比例 |
| Lesson Survival | 经验长期有效性 |
| Edit Efficiency | 每次 Harness 修改带来的收益 |
| Overfit Gap | Evolution vs Held-out 差距 |

核心指标：

```text
Generalization Gap
=
Evolution Gain - Held-out Gain
```

越小越好。

---

# 32. 最终项目一句话

> **让 Agent 不仅会执行任务，还能够观察自己的执行过程，定位失败原因，针对 Harness 的不同组件提出可验证的修改，通过 controlled experiments 选择有效机制，并把经过跨任务验证的经验持续沉淀下来，同时控制 self-improvement 的成本和 overfitting。**

---

# 33. 实施优先级

## P0 —— 必须做

```text
[1] Trace
[2] Diagnosis
[3] Hypothesis
[4] Candidate
[5] Probe
[6] Behavioral Comparator
[7] Harness Versioning
```

## P1 —— 核心能力

```text
[8] Experience Library
[9] Cross-task Retrieval
[10] Transfer Evaluation
[11] Cost Evaluation
[12] Regression Detection
```

## P2 —— 研究能力

```text
[13] Edit Budget
[14] Failed Hypothesis History
[15] Candidate Critic
[16] Pruning
[17] Exploration
[18] Held-out Evaluation
```

## P3 —— 高级研究方向

```text
[19] Modular Evolution
[20] Cross-domain Transfer
[21] Cross-model Transfer
[22] Plugin / Skill Evolution
[23] Automated Experiment Analysis
```

---

# 34. 最终建议

不要一次把所有东西写完。

推荐路线：

```text
现有 agent_dev/multi-agent
            │
            ▼
       P0: Trace
            │
            ▼
       P0: Diagnosis
            │
            ▼
       P0: Candidate
            │
            ▼
       P0: A/B Probe
            │
            ▼
       P0: Comparator
            │
            ▼
       P1: Experience
            │
            ▼
       P1: Transfer
            │
            ▼
       P2: Regularization
            │
            ▼
       P3: Research System
```

**第一版的核心不是“Agent 自动变得多聪明”，而是证明一个最小闭环：**

```text
Failure
  ↓
Diagnosis
  ↓
Hypothesis
  ↓
Controlled Experiment
  ↓
Evidence
  ↓
Harness Improvement
  ↓
Reusable Lesson
  ↓
New Task
  ↓
Transfer
```

只要这个闭环真正跑通，后面增加 Tool / Memory / Subagent / Skill / Script 等组件就是工程扩展，而不是重新设计系统。

---

## 参考方向

- Google Research — Harness²：Recursive Agent Harnessing for an Open World
- Google Research — RRSI：Regularized Recursive Self-Improvement of Agent Harnesses
- ModularRSI：Modular and Generalizable Recursive Harness Self-Improvement
- Recursive Harness Self-Improvement（RHI）
- PluginRSI：Reusable Plugin-based Harness Evolution
