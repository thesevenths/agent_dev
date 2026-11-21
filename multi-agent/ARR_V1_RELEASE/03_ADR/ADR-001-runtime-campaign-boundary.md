# ADR-001: Runtime and Campaign Execution Boundary

- Status: Accepted
- Date: 2026-11-18

## Context
M5 `ResearchService`/`CampaignRunner` 执行 ML `ExperimentPlan`（hypothesis/candidate/paired-seed trials）；
M7 DomainPack 执行科学实验（arms/seeds/splits）。两者契约不同，强行合并需重写两者。

## Decision
保留 `FormalRuntimeAdapter` 作为唯一生产执行 seam：复用 M5 `RuntimeState`（journal/reservation/claim）
+ M6 `HarnessLoader`（SHA 绑定）+ 真实 DomainPack 引擎。不新建 Runtime/Campaign，不合并两套抽象。
在 adapter 内增加持久 campaign 状态机（REQUIRED states）与原子 claim/重复 guard（见 WP-01）。

## Alternatives
1. 把 M7 实验改写为 ExperimentPlan 走 CampaignRunner — 拒绝：架构重写，违反任务书。
2. 新建独立 Runtime — 拒绝：违反"no parallel Runtime"。
3. 薄适配层（采用）— 最小、可测、复用现有原语。

## Consequences
- 正面：生产链路可审计、SHA 绑定、预算可结算、无需返工 M5/M6。
- 代价：adapter 需自行保证 exactly-once 诚实（at-most-once dispatch claim + backend 幂等声明）。
- 迁移风险：低；均为追加式改动，不改 M5/M6 契约。

## P0 后续（v1.0 内处理）
dispatch 持久化状态机 + 原子 claim + 重复/并发 guard + result 结构化持久化（P0-1/P0-3）。
