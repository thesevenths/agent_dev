# Benchmark 报告

- 判分口径：`1ef704c6ebac`
- 轮数：3（ACCEPT 3 / REJECT 0 / SKIP 0）
- ΔQ 序列：[38.1, 38.1, 38.1]
- holdout Q 序列：[53.33, 53.33, 53.33]（drift +0.00）

## 结论

- 跑完 3 轮：ACCEPT 3 / REJECT 0 / SKIP 0
- ΔQ 序列：['+38.1', '+38.1', '+38.1']
- 至少有一个改动真正过了噪声门 —— 闭环能产出有效结论
- holdout 平稳（drift=+0.00），未 promote 期间没有泄漏或漂移

## 逐轮

| 轮 | dominant | 候选 | L | 决策 | n | ΔQ | δ | p | tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | UNKNOWN | `AGENT_PARSE_TRIES` | 1 | ACCEPT/VALIDATED | 4 | +38.10 | 0.00 | 0.000 | 0 |
| 2 | UNKNOWN | `AGENT_PARSE_TRIES` | 1 | ACCEPT/VALIDATED | 4 | +38.10 | 0.00 | 0.000 | 0 |
| 3 | UNKNOWN | `AGENT_REPLAN_ARTIFACT_MAX` | 1 | ACCEPT/VALIDATED | 4 | +38.10 | 0.00 | 0.000 | 0 |

## 怎么读这份报告

1. **ΔQ 序列**才是「有没有在变强」的证据。全为正且逐步过门 = 有效；全在噪声内 = 搜索空间或诊断信号已经不够了。
2. **holdout drift** 在未 promote 时应≈0。若明显漂移，先查泄漏，别急着庆祝提升。
3. **单轮 ACCEPT 不等于累积提升**。累积要靠 promote 换基线，再重跑本 benchmark 对比。
