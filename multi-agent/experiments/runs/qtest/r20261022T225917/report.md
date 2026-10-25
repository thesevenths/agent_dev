# 单轮报告 r20261022T225917

- 时间：20261022T230007
- 阶段：P0（能看见/能量/能拒绝）
- 基线：`base/v1`（33c0e64）
- 诊断主导模式：`COMPLETENESS_FAILURE`（UNKNOWN 0%）

## 候选

- target：`AGENT_MAX_ITERATIONS_CODE`（L1）
- 改动：`60` → `80`
- patch_hash：`1719431c4ab121e6` / base_commit：`33c0e64` / judge_version：`1ef704c6ebac`
- 假设：AGENT_MAX_ITERATIONS_CODE 60→80 可缓解 COMPLETENESS_FAILURE（已是全局 15 的 4 倍仍 28 次失败 → 预期无效，仅作证伪）
- 来源：exploitation（针对主导失败模式）

## 演化约束

- dormant 组件跳过：`无`
- exploration：未触发（single_level）

## 证据（配对）

```
配对 n=32  mean(Δ)=+38.10  sd(Δ)=0.00  δ=0.00  p=0.0000
cost: base=0 cand=0 ΔC=+0.0%
regression_rate=0.00 missing=0 timeout=0
```

- 判定：**ACCEPT**（VALIDATED）
- 行为指标 ΔB 仅供归因，不参与判定

| 任务 | n | mean(Δ) | sd |
|---|---|---|---|
| REP-001 | 4 | +38.10 | 0.00 |
| REP-002 | 4 | +38.10 | 0.00 |
| REP-003 | 4 | +38.10 | 0.00 |
| REP-004 | 4 | +38.10 | 0.00 |
| REP-005 | 4 | +38.10 | 0.00 |
| REP-006 | 4 | +38.10 | 0.00 |
| REP-007 | 4 | +38.10 | 0.00 |
| REP-008 | 4 | +38.10 | 0.00 |

## 判据明细

| 检查 | 通过 | 值 |
|---|---|---|
| evidence_complete | ✓ | `{'missing': 0, 'timeout': 0}` |
| min_tasks | ✓ | `{'n_tasks': 8, 'need': 2}` |
| contract_floor | ✓ | `{'A_base': 0.6667, 'A_cand': 1.0, 'floor': 0.6167}` |
| judge_stable | ✓ | `{'unstable_ratio': 0.0, 'max': 0.2}` |
| plan_not_undercut | ✓ | `{'undercut_rate': 0.0}` |
| regression_rate | ✓ | `{'rate': 0.0, 'max': 0.15, 'n_tasks': 8}` |
| primary_gain | ✓ | `{'n': 8, 'mean': 38.1, 'delta_crit': 0.0, 'p': 0.0}` |
| cost_rule | ✓ | `{'dC': 0.0, 'allowance': 0.6215, 'dQ': 38.1}` |

## 成本

- tokens：0
- 实际配对数：32 / 计划 4
- 停止原因：无
