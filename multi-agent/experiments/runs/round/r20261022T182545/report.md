# 单轮报告 r20261022T182545

- 时间：20261022T182603
- 阶段：P0（能看见/能量/能拒绝）
- 诊断主导模式：`COMPLETION_FAILURE`（UNKNOWN 0%）

## 候选

- target：`AGENT_MAX_ITERATIONS_CODE`
- 改动：`60` → `80`
- 假设：AGENT_MAX_ITERATIONS_CODE 60→80 可缓解 COMPLETION_FAILURE（已是全局 15 的 4 倍仍 28 次失败 → 预期无效，仅作证伪）

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
