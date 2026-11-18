# ADR-002: Research Program and Round Lifecycle

- Status: Accepted
- Date: 2026-11-18

## Context
M7 `DiscoveryCoordinator` 是单轮引擎：16 stage、磁盘状态、resume；一次 `run()` 即终态。
M6 `loop/controller.py` 是多轮 driver，但驱动 dual-layer evolution，不调用 discovery，
也不跨轮继承 ClaimEvidenceGraph / VerifiedResearchMemory。

## Decision
新增薄外层 `ResearchProgramDriver`（不修改 DiscoveryCoordinator）：
- ResearchProgram = 有界研究目标 + 多轮 + 预算/审批策略；
- ResearchRound = 一次 DiscoveryCoordinator.run() + 独立 ID/verdict/memory commit；
- 下一轮规划基于上一轮真实 artifacts（claim verdict、memory、失败），非硬编码脚本；
- program/round 状态机区分科学 INCONCLUSIVE 与工程 FAILED。

## Alternatives
1. 改 DiscoveryCoordinator 内部循环 — 拒绝：破坏单轮幂等/resume，违反"reuse as single-round engine"。
2. 复用 loop/controller 直接驱动 — 拒绝：需注入 discovery+graph+memory 且语义不同（evolution 轮 vs research 轮）。
3. 薄 program 层（采用）。

## Consequences
- 正面：多轮可追踪、可恢复、轮间经验可继承、预算可控。
- 代价：新增 program state 持久化（复用 RuntimeState 的 journal/checkpoint，不新建 Runtime）。
- 迁移风险：中；需保证 program 层不改单轮语义。
