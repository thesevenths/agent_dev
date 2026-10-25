# 单轮报告 r20261022T190341

- 时间：20261022T190355
- 阶段：P0（能看见/能量/能拒绝）
- 基线：`base/v1`（33c0e64）
- 诊断主导模式：`UNKNOWN`（UNKNOWN 100%）

## 候选

- target：`AGENT_MAX_ITERATIONS`（L1）
- 改动：`15` → `25`
- patch_hash：`5b4b10a8e722f3af` / base_commit：`33c0e64` / judge_version：`7a755ce7260b`
- 假设：AGENT_MAX_ITERATIONS 15→25 可缓解 UNKNOWN（爬虫类任务的全局 ReAct 预算；未单独配置时落在这里）
- 来源：exploitation（针对主导失败模式）

## 演化约束

- dormant 组件跳过：`无`
- exploration：未触发（single_level）

## 证据（配对）

```
配对 n=8  mean(Δ)=+46.67  sd(Δ)=0.00  δ=0.00  p=0.0000
cost: base=0 cand=0 ΔC=+0.0%
regression_rate=0.00 missing=0 timeout=0
```

- 判定：**ACCEPT**（VALIDATED）
- 行为指标 ΔB 仅供归因，不参与判定

| 任务 | n | mean(Δ) | sd |
|---|---|---|---|
| REP-001 | 2 | +46.67 | 0.00 |
| REP-002 | 2 | +46.67 | 0.00 |
| REP-003 | 2 | +46.67 | 0.00 |
| REP-004 | 2 | +46.67 | 0.00 |

## 判据明细

| 检查 | 通过 | 值 |
|---|---|---|
| evidence_complete | ✓ | `{'missing': 0, 'timeout': 0}` |
| contract_floor | ✓ | `{'A_base': 0.6667, 'A_cand': 1.0, 'floor': 0.6167}` |
| judge_stable | ✓ | `{'unstable_ratio': 0.0, 'max': 0.2}` |
| regression_rate | ✓ | `{'rate': 0.0, 'max': 0.15, 'n_tasks': 4}` |
| primary_gain | ✓ | `{'n': 8, 'mean': 46.67, 'delta_crit': 0.0, 'p': 0.0}` |
| cost_rule | ✓ | `{'dC': 0.0, 'allowance': 0.75, 'dQ': 46.67}` |

## 成本

- tokens：0
- 实际配对数：8 / 计划 2
- 停止原因：无
