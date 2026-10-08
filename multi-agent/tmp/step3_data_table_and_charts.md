# Step 3 — 数据整理与可视化（机构卖出 Top 5 + NDX 走势）

AS_OF: 整理执行 2026-10-08 09:25（本地）；NDX 数据为 2026-10-07 收盘（17:15 EDT）

## 0. 重要口径声明（务必先读）
- **"2026-10-07 当日机构卖出 Top 5" 无公开权威数据**（Step 2 已确认：SEC 13F 为季度报告，不披露逐日卖出明细）。
- 本步骤的"机构卖出 Top 5"采用**代理口径**：MarketBeat 披露的 **QQQM（Invesco NASDAQ 100 ETF）近 24 个月累计卖出金额最大的 5 家机构**。该口径为**累计、ETF 层面、非 10-07 单日**，仅作为可视化与结构化展示的近似参考，**不可解读为 10-07 当日机构卖出**。
- 所有图表标题均已标注"非 2026-10-07 当日数据"，避免误导。

## 1. 机构卖出 Top 5 结构化表格（代理口径：QQQM 近24个月累计）
| 排名 | 机构 | 卖出规模 (USD) | 备注 |
| --- | --- | --- | --- |
| 1 | Northwestern Mutual Wealth Management Co. | $3.51M | 近24个月累计卖出最多 |
| 2 | Clal Insurance Enterprises Holdings Ltd | $2.38M | 以色列保险集团 |
| 3 | Bank of Nova Scotia (Scotiabank) | $721.32K | 加拿大银行 |
| 4 | Mizuho Bank Ltd. | $715K | 日本银行 |
| 5 | Morgan Stanley | $453.55K | 美国投行 |

> 数据来源：MarketBeat — Invesco NASDAQ 100 ETF (QQQM) Institutional Ownership。
> 口径：近 24 个月累计卖出（13F 汇总），**非 2026-10-07 单日**。

### 可视化：机构卖出规模对比（柱状图）
![机构卖出规模对比](chart_institutional_selling_top5.png)

**图表解读**：卖出规模呈明显"头部集中"——第 1 名（Northwestern Mutual，$3.51M）约为第 5 名（Morgan Stanley，$453K）的 7.7 倍；前 2 家合计占 5 家总额的约 81%。

## 2. NDX 2026-10-07 走势（Step 1 数据，供上下文）
| 指标 | 数值 |
| --- | --- |
| 前收盘 (10-06) | 31,224.69 |
| 开盘 (10-07) | 30,976.25 |
| 当日最低 | 30,904.46 |
| 当日最高 | 31,170.12 |
| 收盘 (10-07) | **31,160.08** |
| 涨跌 / 涨跌幅 | -64.61 / **-0.21%** |
| 成交量 | 978,915,741 |

### 可视化：NDX 日内走势（柱状图）
![NDX 2026-10-07 日内走势](chart_ndx_2026-10-07.png)

**图表解读**：10-07 低开（30,976.25，低于前收盘约 0.79%），盘中探底 30,904.46 后持续回升，尾盘收于 31,160.08，仅较前收盘微跌 0.21%，呈"低开高走、收复失地"形态，印证"跌了一点"的小幅回调定性。

## 3. 产出文件清单
| 文件 | 说明 |
| --- | --- |
| `E:\agent_dev\multi-agent\tmp\chart_institutional_selling_top5.png` | 机构卖出规模对比柱状图（代理口径） |
| `E:\agent_dev\multi-agent\tmp\chart_ndx_2026-10-07.png` | NDX 2026-10-07 日内走势柱状图 |
| `E:\agent_dev\multi-agent\tmp\step3_data_table_and_charts.md` | 本文件（结构化表格 + 图表） |

## 4. 给下游（Step 4 报告）的提示
- 报告须**显著标注**"机构卖出 Top 5 为代理口径（QQQM 近24个月累计），非 10-07 当日数据"，避免用户误读为当日机构抛售。
- 可补充说明：若需真正的"10-07 当日机构卖出 Top 5"，需付费实时 institutional flow 数据源（Bloomberg / Refinitiv / Nasdaq Data Link）。
