# ADR-006: Independent Evaluation and Holdout Isolation

- Status: Accepted
- Date: 2026-11-18

## Context
capability_bench 已有四臂/四组与 family-level clustered CI；holdout 隔离散落在
`rsi_bench.split_tasks`、`promotion.py` family/task disjoint、`HoldoutAccessVerifier` chain。
无统一 BenchmarkAdapter；外部 benchmark（ScienceAgentBench/FIRE-Bench/DISCERN 等）未接入。

## Decision
- 统一 case/family/GT schema 与 holdout policy（复用现有字段，不重写 benchmark）；
- 统计聚类单元 = Research Family；≥8 独立 family、≥3 paired seeds 为目标，样本不足 → INCONCLUSIVE，不降阈；
- 外部 benchmark 通过 `BenchmarkAdapter` 接口按需接入：保留原任务/评分，记录访问/许可状态；
  adapter 存在 ≠ 通过；无法访问记 BLOCKED，不伪造。
- 独立人类专家盲评不可由 LLM-as-judge 替代；缺失 → L2/L4 记 INCONCLUSIVE。

## Alternatives
1. 用 LLM 自评替代专家 — 拒绝：违反 Evidence over Self-Evaluation。
2. 假设外部 benchmark 可访问并填结果 — 拒绝：科研诚信。
3. 适配器 + 诚实阻塞（采用）。

## Consequences
- 正面：评测可复现、可审计、不可伪造 PASS。
- 代价：部分 gate 将长期 INCONCLUSIVE 直到外部资源到位。
- 迁移风险：低。
