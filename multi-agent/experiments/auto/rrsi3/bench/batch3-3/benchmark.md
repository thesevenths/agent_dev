# Benchmark 报告

- 判分口径：`1ef704c6ebac`
- 轮数：1（ACCEPT 0 / REJECT 1 / SKIP 0）
- ΔQ 序列：[0.2871]
- holdout Q 序列：[86.02]（drift +0.00）

## 结论

- 跑完 1 轮：ACCEPT 0 / REJECT 1 / SKIP 0
- ΔQ 序列（evolve 集）：['+0.3']
- 诊断输入来源：`round_traces`
- 没有任何改动过门：要么搜索空间已穷举，要么诊断信号不足（看 UNKNOWN 占比）
- holdout 平稳（drift=+0.00），未 promote 期间没有泄漏或漂移

## 逐轮

| 轮 | dominant | 候选 | L | 决策 | n | ΔQ | δ | p | holdout ΔQ | 诊断源 | tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | PLANNING_FAILURE | `AGENT_REPLAN_ARTIFACT_MAX` | 1 | REJECT/within_noise | 13 | +0.29 | 2.85 | 0.830 | - | round_traces(4) | 24,459,277 |

- 第 1 轮备注：第 1 轮：用 base 臂 bootstrap 扫描作为诊断输入（不读人工日志）

## 怎么读这份报告

1. **ΔQ 序列**才是「有没有在变强」的证据。全为正且逐步过门 = 有效；全在噪声内 = 搜索空间或诊断信号已经不够了。
2. **holdout ΔQ** 是唯一能算「泛化」的一列：它没参与诊断也没参与提案。未 promote 时 holdout 均值应平稳，若它自己往上走，先查泄漏。
3. **单轮 ACCEPT 不等于累积提升**。累积要靠 promote 换基线（人工或 auto），再重跑本 benchmark 对比。
4. **诊断源必须是 `round_traces`**。出现 `manual_log` 说明那一轮读的是别人跑的日志，它的 dominant 与本轮探针无因果关系。
