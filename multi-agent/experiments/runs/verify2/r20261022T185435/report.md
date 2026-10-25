# 单轮报告 r20261022T185435

- 时间：20261022T185446
- 阶段：P0（能看见/能量/能拒绝）
- 基线：`base/v1`（33c0e64）
- 诊断主导模式：`UNKNOWN`（UNKNOWN 100%）

## 候选

- target：`AGENT_MAX_ITERATIONS_CODE`（L1）
- 改动：`60` → `100`
- patch_hash：`aede6cca674680a6` / base_commit：`33c0e64` / judge_version：`7a755ce7260b`
- 假设：AGENT_MAX_ITERATIONS_CODE 60→100 可缓解 UNKNOWN（已是全局 15 的 4 倍仍 28 次失败 → 预期无效，仅作证伪）
- 来源：exploitation（针对主导失败模式）

## 演化约束

- dormant 组件跳过：`无`
- exploration：未触发（single_level）

## 证据（配对）

```
配对 n=4  mean(Δ)=-26.66  sd(Δ)=0.00  δ=0.00  p=0.0000
cost: base=0 cand=0 ΔC=+0.0%
regression_rate=1.00 missing=0 timeout=0
```

- 判定：**REJECT**（regression）
- 行为指标 ΔB 仅供归因，不参与判定

| 任务 | n | mean(Δ) | sd |
|---|---|---|---|
| REP-001 | 2 | -26.66 | 0.00 |
| REP-002 | 2 | -26.66 | 0.00 |

## 判据明细

| 检查 | 通过 | 值 |
|---|---|---|
| stage1_stop_loss | ✗ | `{'mean_delta': -26.66}` |

## 成本

- tokens：0
- 实际配对数：4 / 计划 2
- 停止原因：stop_loss
