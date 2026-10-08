# Step 2 — 2026-10-07 纳斯达克100成分股 机构卖出 Top 5 检索结果

AS_OF: 检索执行时间 2026-10-08 09:24（本地）；所涉数据为 2026-10-07 交易日

## 结论（重要 — 数据可得性限制）
**无法提供"2026-10-07 当日"机构（共同基金/对冲基金/养老金）卖出金额/股数排名前 5 的机构名单。**

原因：
1. **机构持仓/买卖的权威披露渠道是 SEC Form 13F**，其性质为**季度**报告（季度结束后 45 天内提交），且**不披露逐日交易、不披露卖出明细**，只披露季末持仓快照。因此**不存在"某一具体交易日（10-07）机构卖出 Top 5"的公开权威数据**。
2. 检索到的 13F 相关数据（HedgeTrace、MarketBeat 等）均为 **Q3 2026 季度持仓**（filing date 约 2026-10-06），是季度末持仓，**不是 10-07 当日卖出**。
3. 检索到的"机构卖出"新闻（如 Yahoo Finance "Institutional investors reveal cautious approach to tech favorites"）为**季度 13F 汇总**的定性描述，非单日数据。
4. 单日机构净买卖（block trades / dark pool / 13F 之外的实时流）通常需付费数据源（如 Bloomberg、Refinitiv、Nasdaq Data Link 的实时 institutional flow），公开搜索无法获取到"10-07 当日机构卖出 Top 5"的精确名单。

## 检索到的相关（但非当日）参考信息
| 来源 | 内容 | 时间口径 | 是否满足"10-07 当日机构卖出 Top 5" |
| --- | --- | --- | --- |
| MarketBeat — QQQM Institutional Ownership | 近 24 个月卖出 QQQM 最多的机构：Northwestern Mutual Wealth Mgmt ($3.51M)、Clal Insurance ($2.38M)、Bank of Nova Scotia ($721K)、Mizuho Bank ($715K)、Morgan Stanley ($453K) 等 | 近 24 个月累计（13F 汇总） | 否（非单日、非个股、为 ETF 累计） |
| HedgeTrace | Q3 2026 13F 持仓（filing 2026-10-06）：Private Capital Advisors、Legacy Private Trust、SEGRA Capital 等 | Q3 2026 季度末 | 否（季度持仓，非卖出、非单日） |
| Yahoo Finance 新闻 | 机构对科技股"谨慎/观望"，13F 显示净买卖大致平衡（data center 板块 24.3% 买 / 24.3% 卖） | Q3 2026 13F 汇总 | 否（定性、季度、非 Top5 名单） |

## 建议（供下游/用户）
- 若目标是"10-07 当日机构卖出 Top 5"，公开免费渠道**无法精确满足**；需付费实时 institutional flow 数据源。
- 可替代的、可公开获取的近似口径：
  a) **10-07 当日 NDX 成分股中跌幅/成交量最大的个股**（可作为"被抛售标的"的代理指标）；
  b) **Q3 2026 13F 中减持幅度最大的机构**（季度口径，非单日）；
  c) 10-07 当日**大宗交易/暗池**公开披露（若有）。
- 建议向用户澄清：是否接受"季度 13F 减持 Top 5"或"当日跌幅/放量 Top 个股"作为替代口径。

## 数据来源
- MarketBeat: https://www.marketbeat.com/stocks/NASDAQ/QQQM/institutional-ownership
- HedgeTrace: https://hedgetrace.com
- Yahoo Finance: https://finance.yahoo.com/markets/stocks/articles/institutional-investors-reveal-cautious-approach-215416925.html
- SEC 13F FAQ: https://www.sec.gov/rules-regulations/staff-guidance/frequently-asked-questions-about-form-13f

## 备注
- 本步骤未获得"10-07 当日机构卖出 Top 5"的精确名单；已如实记录数据可得性限制与替代口径建议，交由 Supervisor/用户决策。
