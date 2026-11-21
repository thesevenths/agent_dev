# ARR v1.0 — 08 Memory Transfer Report

## 1. 四臂对照（复用 M7.3，GT-based 修复后）

| Arm | repeat_failure_accuracy (GT) | memory_usage | harmful |
|---|---:|---:|---:|
| OFF | 0.70 | – | – |
| ON | 0.10 | 1.00 | 0.00 |
| SHUFFLED | 0.70 | 0.00 | 0.30 |
| EQUAL_TOKEN | 0.50 | – | – |

结论：Memory ON **未改善** grounded accuracy（0.10 < OFF 0.70）；SHUFFLED 出现 30% harmful usage。
不宣称 Memory 有效，verdict INCONCLUSIVE（专家未审）。

## 2. v1.0 新增归因（WP-06）

- `memory_attribution.py` 七状态：retrieved / injected / cited / applied / changed_decision /
  improved_outcome / caused_harm；明确"检索≠使用≠改善"。
- 真实 3 轮 program 中：r2 注入 1 条历史 memory ref、r3 注入 2 条，逐轮递增（来自 `program_state.json` 实证）。
- 负迁移测量：wrong-domain / stale / contradictory memory usage 由 GT 分类统计。

## 3. 六类记忆（WP-05）
CANDIDATE / VERIFIED / FAILED_EXPERIMENT / METHODOLOGY / LIMITATION / CONTRADICTION；
INCONCLUSIVE 不晋升 Verified finding（保持 CANDIDATE/历史经验）。
