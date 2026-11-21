# ADR-005: Methodology RSI Contract

- Status: Accepted
- Date: 2026-11-18

## Context
`candidate_generation.generate_candidate` 已基于真实失败计数诊断瓶颈，但改进方案来自
预注册固定库 `IMPROVEMENT_LIBRARY`，且 `CandidateRecord` 生成后由调用方自行保存，
无 method-candidate registry。两个现存 registry（rrsi EvolutionRegistry、evolution CandidateRegistry）
分别是 promotion 记录与 domain-model candidate，均不匹配。

## Decision
新增 `MethodRegistry`（薄封装复用 `EvolutionRegistry.record` 协议 + 冻结 artifact 磁盘重读）：
- `ResearchMethod` = method_id/version/parent_sha/component_config/config_sha/限制/评估史/promotion_status；
- MethodCandidate 必须含 source_failure_refs / diagnosis / expected_improvement / falsification /
  side_effects / evaluation_protocol / holdout_definition / rollback_conditions；
- 程序化验证 evidence refs 与 config，不信任 LLM 生成的 SHA；
- 保留 M6 promotion/adoption/rollback 作为晋级链；holdout 不参与 candidate 生成/选择；
- NO_CLEAR_BOTTLENECK 时不生成 Candidate（合法行为，不等于 RSI 成功）。

## Alternatives
1. 直接改 prompt 当 RSI — 拒绝：违反"Experimental Improvement over Prompt Modification"。
2. 新建独立 RSI 引擎 — 拒绝：复用 M6。
3. MethodRegistry + 复用晋级链（采用）。

## Consequences
- 正面：改进可追溯、可回滚、可跨轮累积。
- 代价：需 registry 持久化与冻结 artifact。
- 迁移风险：中；保持 M6 契约不变。
