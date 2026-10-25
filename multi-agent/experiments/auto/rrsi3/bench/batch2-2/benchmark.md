# Benchmark 报告

- 判分口径：`1ef704c6ebac`
- 轮数：1（ACCEPT 1 / REJECT 0 / SKIP 0）
- ΔQ 序列：[3.7526]
- holdout Q 序列：[82.4]（drift +0.00）

## 结论

- 跑完 1 轮：ACCEPT 1 / REJECT 0 / SKIP 0
- ΔQ 序列（evolve 集）：['+3.8']
- ΔQ 序列（holdout 配对）：['-2.9']
- 诊断输入来源：`round_traces`
- 至少有一个改动真正过了噪声门 —— 闭环能产出有效结论
- base 版本 1→2（自动 promote），此时 holdout 上升才算泛化证据

## 逐轮

| 轮 | dominant | 候选 | L | 决策 | n | ΔQ | δ | p | holdout ΔQ | 诊断源 | tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | EVIDENCE_FAILURE | `AGENT_MAX_ITERATIONS_CODE` | 1 | ACCEPT/VALIDATED | 13 | +3.75 | 3.06 | 0.020 | -2.85/10 | round_traces(4) | 27,999,207 |

- 第 1 轮备注：第 1 轮：用 base 臂 bootstrap 扫描作为诊断输入（不读人工日志）; 已自动换基线并落地：base/v2

## 怎么读这份报告

1. **ΔQ 序列**才是「有没有在变强」的证据。全为正且逐步过门 = 有效；全在噪声内 = 搜索空间或诊断信号已经不够了。
2. **holdout ΔQ** 是唯一能算「泛化」的一列：它没参与诊断也没参与提案。未 promote 时 holdout 均值应平稳，若它自己往上走，先查泄漏。
3. **单轮 ACCEPT 不等于累积提升**。累积要靠 promote 换基线（人工或 auto），再重跑本 benchmark 对比。
4. **诊断源必须是 `round_traces`**。出现 `manual_log` 说明那一轮读的是别人跑的日志，它的 dominant 与本轮探针无因果关系。
