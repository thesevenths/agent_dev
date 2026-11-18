# ADR-004: Verified Memory Lifecycle

- Status: Accepted
- Date: 2026-11-18

## Context
`VerifiedResearchMemory` 已有 candidate/verified 两级 tier、hash-chain ledger、
promote 验证链（证据磁盘重读、claim gate）。但任务书要求六类记忆：
Candidate / Verified / Failed Experiment / Methodology / Limitation / Contradiction；
现有实现中 failed 仅为 verification_status、limitation 仅为字段、methodology/contradiction 缺失。

## Decision
保留两级 tier 与 append-only ledger 不变；新增结构化 `memory_class` 字段与六类枚举，
methodology/contradiction 作为一等记录类型（含 evidence refs、conditions、supersession）。
生命周期扩展为 CANDIDATE→EVIDENCE_ATTACHED→UNDER_VERIFICATION→VERIFIED / CONTRADICTED /
DEPRECATED / REVOKED。INCONCLUSIVE 永不晋升为 Verified finding（保留为 CANDIDATE/历史经验）。

## Alternatives
1. 新建第二个记忆系统 — 拒绝：违反复用，破坏现有 SHA 链。
2. 用字段模拟六类 — 拒绝：无类型安全、检索无法按类过滤。
3. 扩展现有 schema（采用）。

## Consequences
- 正面：类别可检索、负迁移可测量、生命周期可审计。
- 代价：schema 追加字段（默认值兼容旧记录）。
- 迁移风险：低；历史记录按默认类映射。
