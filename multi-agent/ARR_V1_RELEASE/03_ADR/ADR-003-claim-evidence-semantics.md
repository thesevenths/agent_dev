# ADR-003: Claim and Evidence Semantics

- Status: Accepted
- Date: 2026-11-18

## Context
M7.6 发现真实缺陷：selection/replication verdict 为 `INCONCLUSIVE` 时，
`discovery_coordinator.py:867-872` 的二元表达式把其写成 `REFUTES` 边，
与 claim 自身的 `INCONCLUSIVE` 状态矛盾，且 `graph.contradictions()` 会把该边当反证，
进而阻塞 memory `_claim_gate`。

## Decision
三态显式映射：
- selection: `SUPPORTS→SUPPORTS`、`REFUTES→REFUTES`、`INCONCLUSIVE→无 support/refute 边`；
- replication: `SUPPORTS→REPLICATES`、`REFUTES→REFUTES`、`INCONCLUSIVE→无 support/refute 边`；
- 保留 Evidence 节点与所有 terminal 事件；不确定不得转成证伪。
新增中性边类型 `INCONCLUSIVE_EVIDENCE`（若需表达"有证据但不足"），默认不建边。

## Alternatives
1. 把 INCONCLUSIVE 当 REFUTES — 拒绝：科学上错误，污染证据图。
2. 不建边也不记 Evidence — 拒绝：丢失可审计痕迹。
3. 中性边/无 effect 边 + Evidence 保留（采用）。

## Consequences
- 正面：verdict 与 claim 状态一致；memory 晋升门不被不确定证据污染。
- 代价：需扩展 graph EDGE_TYPES 并加测试断言。
- 迁移风险：低；历史图不重写，新规则只作用于新边。
