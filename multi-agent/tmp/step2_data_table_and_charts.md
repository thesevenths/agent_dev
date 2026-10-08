# Step 2 — 数据整理与可视化（2026-10-07 NDX "低开高走"）

**AS_OF: 数据整理 2026-10-08 09:41（本地）；个股数据为 2026-10-07 收盘（17:15 EDT）**

---

## 一、"高走"的标的层面驱动：NDX 成分股领涨 Top 5（2026-10-07 收盘）

| 排名 | 公司 | 代码 | 收盘价 | 涨跌 | 涨跌幅 | 板块 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Micron Technology | MU | 1,088.00 | +42.44 | **+4.06%** | 半导体/存储 |
| 2 | Amgen | AMGN | 413.08 | +10.48 | +2.60% | 医疗 |
| 3 | Intuit | INTU | 297.24 | +7.43 | +2.56% | 软件 |
| 4 | Intuitive Surgical | ISRG | 414.52 | +9.76 | +2.41% | 医疗/器械 |
| 5 | Sandisk | SNDK | 1,692.42 | +31.96 | +1.92% | 存储 |

![NDX 成分股领涨 Top 5](chart_ndx_top_gainers_2026-10-07.png)

> **数据来源**：Slickcharts Nasdaq 100 Gainers（2026-10-07 收盘）。
> **解读**：领涨以**半导体/存储（MU +4.06%、SNDK +1.92%）**与**医疗（AMGN、ISRG）**为主，叠加软件（INTU）与存储（SNDK）。多数成分股收涨，是指数"低开高走"收复失地的标的层面支撑。

### 当日市场领涨板块（含非 NDX 成分，供板块背景）
| 公司 | 代码 | 涨跌幅 | 板块 |
| --- | --- | --- | --- |
| Constellation Energy | CEG | +7.00% | 电力/核电 |
| Ciena | CIEN | +11.46% | 光通信/网络设备 |
| Vistra Corp. | VST | +10.27% | 电力 |
| Marvell Technology | MRVL | +5.46% | 半导体 |
| Corning | GLW | +5.43% | 光通信/玻璃 |

> **解读**：当日**电力/核电（CEG、VST）与光通信/网络设备（CIEN、GLW、MRVL）**领涨，反映 AI 数据中心电力与网络基础设施主题活跃，为纳指科技板块提供买盘。

---

## 二、机构买入 Top 5（⚠️ 代理口径：QQQM 近24个月累计，非 10-07 当日数据）

> **核心结论**：**"2026-10-07 当日机构买入 Top 5" 无公开权威数据**（SEC 13F 为季度报告，不披露逐日买入明细）。
> 采用**代理口径**：MarketBeat 披露的 **QQQM（Invesco NASDAQ 100 ETF）近 24 个月累计买入金额最大的 5 家机构**（近24个月共 1,223 家机构买入 QQQM）。
> **口径说明**：① 近 24 个月累计买入（13F 汇总），**非 10-07 单日**；② 买入标的统一为 **QQQM（ETF）**，非对某只 NDX 成分股的直接买入；③ 13F **不披露买入动机**，故"可能原因"均标注"未披露"。

| 排名 | 机构名称 | 买入标的 | 买入规模 (USD) | 机构类型 | 可能原因 |
| --- | --- | --- | --- | --- | --- |
| 1 | Strategic Financial Concepts LLC | QQQM | $6.20M | 财富管理/资管 | 未披露 |
| 2 | Invesco Ltd. | QQQM | $3.70M | 基金管理人（ETF发行方） | 未披露 |
| 3 | Migdal Insurance & Financial Holdings | QQQM | $3.02M | 保险（以色列） | 未披露 |
| 4 | Nolim (National Mutual Insurance Fed. of Ag. Coops) | QQQM | $2.74M | 保险（以色列） | 未披露 |
| 5 | Raymond James Financial Inc. | QQQM | $2.66M | 财富管理/券商 | 未披露 |

![机构买入规模 Top 5（代理口径）](chart_institutional_buying_top5.png)

> **数据来源**：MarketBeat — Invesco NASDAQ 100 ETF (QQQM) Institutional Ownership。
> **集中度**：Top 1（Strategic Financial Concepts）占 Top 5 合计（$18.32M）的 **33.8%**；Top 2 合计占 **58.4%**，头部集中度较高。
> **机构类型分布**：财富管理/资管 2 家、保险 2 家、基金管理人 1 家——以**保险与财富管理**类机构为主。

---

## 三、数据可得性与口径提示
- **个股领涨/放量数据**：2026-10-07 收盘口径，可解释"高走"的标的层面驱动。
- **机构买入 Top 5**：为**代理口径（24个月累计）**，**不可解读为 10-07 当日机构买入**；若需真正当日机构买入，需付费实时 institutional flow 数据源（Bloomberg / Refinitiv / Nasdaq Data Link）。

## 四、产出文件
- `E:\agent_dev\multi-agent\tmp\step2_data_table_and_charts.md`（本文件）
- `E:\agent_dev\multi-agent\tmp\chart_ndx_top_gainers_2026-10-07.png`（领涨个股柱状图）
- `E:\agent_dev\multi-agent\tmp\chart_institutional_buying_top5.png`（机构买入规模对比柱状图）
