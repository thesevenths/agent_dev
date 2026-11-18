# ARR v1.0 — 10 Benchmark Report

## 1. 内部 benchmark（复用并修复）

| Benchmark | 关键指标 | verdict |
|---|---|---|
| M7.2 Scientific Judgment | novelty acc 1.0 但 evidence sufficiency 0.64；多主结局门控；n=14<30 | INCONCLUSIVE |
| M7.3 Memory | GT-based accuracy；ON 0.10 < OFF 0.70；四臂 | INCONCLUSIVE |
| M7.4 RSI | NO_CLEAR_BOTTLENECK；无 Candidate | INCONCLUSIVE |
| M7.5 Integrated | family=7<8；C/D CI 下界 0<MME；四组 A/B/C/D | INCONCLUSIVE |

统计：family 作为聚类单元，`_clustered_bootstrap` 整族重采样 CI；不把 seeds 当独立 family。

## 2. 外部 benchmark（WP-08）

- `external_benchmarks.py` 提供 `BenchmarkAdapter` + GenericJsonl adapter + coverage matrix。
- 6 个目标（ScienceAgentBench / FIRE-Bench / DISCERN / ResearchClawBench /
  SCI-AGENT-EXAM / SciAgentArena）全部 `data_present=false` → status BLOCKED。
- 明确声明：adapter 存在 ≠ benchmark PASS；无真实数据/许可不填结果。

## 3. 专家评审
无独立人类专家盲评（B-01）；LLM-as-judge 仅为诊断，不替代。L2/L4 因此不可 PASS。
