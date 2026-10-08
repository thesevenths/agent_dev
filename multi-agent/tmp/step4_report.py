# -*- coding: utf-8 -*-
"""Step 4: BTC vs NDX 3-5年投资价值对比分析报告（含图表）"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

OUT = r"E:\agent_dev\multi-agent\tmp"

def setup_font():
    for f in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei"]:
        if any(f.lower() in x.name.lower() for x in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break
    plt.rcParams["axes.unicode_minus"] = False
setup_font()

NDX_C = "#1f77b4"
BTC_C = "#f7931a"

def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return p

# 图1: 累计回报对比
fig, ax = plt.subplots(figsize=(8, 5))
cats = ["1年", "3年(年化)", "5年(年化)"]
ndx = [25.59, 28.43, 22.5]
btc = [-33.0, 175.0, 200.0]
x = np.arange(len(cats)); w = 0.35
b1 = ax.bar(x - w/2, ndx, w, label="NDX (QQQ)", color=NDX_C)
b2 = ax.bar(x + w/2, btc, w, label="BTC", color=BTC_C)
for b in list(b1) + list(b2):
    ax.annotate(f"{b.get_height():+.0f}%", (b.get_x()+b.get_width()/2, b.get_height()),
                ha="center", va="bottom" if b.get_height() >= 0 else "top", fontsize=9)
ax.axhline(0, color="grey", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(cats)
ax.set_ylabel("回报 (%)")
ax.set_title("NDX vs BTC 历史累计回报对比\n(NDX 为年化口径；BTC 3Y/5Y 为估算区间中值)", fontsize=11)
ax.legend(); ax.grid(axis="y", alpha=0.3)
p1 = save(fig, "chart1_returns.png")

# 图2: 风险指标对比
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
ax = axes[0]
ax.bar(["NDX", "BTC"], [22.5, 50], color=[NDX_C, BTC_C], width=0.5)
for i, v in enumerate([22.5, 50]):
    ax.text(i, v+1, f"~{v:.0f}%", ha="center", fontsize=10)
ax.set_title("年化波动率 (代理)", fontsize=10); ax.set_ylabel("%"); ax.grid(axis="y", alpha=0.3)
ax = axes[1]
ax.bar(["NDX", "BTC"], [-34, -51], color=[NDX_C, BTC_C], width=0.5)
for i, v in enumerate([-34, -51]):
    ax.text(i, v-3, f"~{v:.0f}%", ha="center", fontsize=10)
ax.set_title("最大回撤 (近5年/当前周期)", fontsize=10); ax.set_ylabel("%"); ax.grid(axis="y", alpha=0.3)
fig.suptitle("NDX vs BTC 风险指标对比", fontsize=11)
fig.tight_layout()
p2 = save(fig, "chart2_risk.png")

# 图3: BTC 机构价格预测区间
fig, ax = plt.subplots(figsize=(8.5, 5))
years = [2026, 2027, 2028, 2029, 2030]
cons = [75000, 90000, 120000, 150000, 140000]
neut = [100000, 150000, 200000, 250000, 240000]
opt = [120000, 200000, 300000, 400000, 380000]
ax.fill_between(years, cons, opt, color=BTC_C, alpha=0.18, label="保守-乐观区间")
ax.plot(years, neut, "o-", color=BTC_C, lw=2, label="中性")
ax.plot(years, cons, "s--", color="#888", lw=1.2, label="保守")
ax.plot(years, opt, "^--", color="#c0392b", lw=1.2, label="乐观")
ax.axhline(84199, color=NDX_C, ls=":", lw=1.5)
ax.text(2026.05, 86000, "当前价 $84,199 (2026-10-07)", color=NDX_C, fontsize=9)
ax.set_ylabel("BTC 价格 (USD)"); ax.set_xticks(years)
ax.set_title("BTC 机构价格预测区间 2026-2030\n(CryptoRank / CoinStats，定性区间)", fontsize=11)
ax.legend(); ax.grid(alpha=0.3)
p3 = save(fig, "chart3_btc_forecast.png")

# 图4: 相关性结构
fig, ax = plt.subplots(figsize=(7, 4.5))
labels = ["BTC-NDX\n20日(短期)", "BTC-NDX\n长期", "BTC-S&P500\n(11年最低)"]
vals = [-0.43, 0.80, 0.10]
colors = ["#c0392b", NDX_C, "#888"]
bars = ax.barh(labels, vals, color=colors)
ax.axvline(0, color="black", lw=1)
for b, v in zip(bars, vals):
    ax.text(v + (0.03 if v >= 0 else -0.03), b.get_y()+b.get_height()/2,
            f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=10)
ax.set_xlim(-0.6, 1.0); ax.set_xlabel("相关系数")
ax.set_title("NDX 与 BTC 相关性结构 (2026-10)\n短期负相关 / 长期强相关", fontsize=11)
ax.grid(axis="x", alpha=0.3)
p4 = save(fig, "chart4_correlation.png")

# 图5: 分情景预期回报
fig, ax = plt.subplots(figsize=(8.5, 4.8))
scen = ["熊市", "中性", "牛市"]
ndx_scen = [5, 12, 18]
btc_scen = [-10, 25, 45]
x = np.arange(len(scen)); w = 0.35
b1 = ax.bar(x - w/2, ndx_scen, w, label="NDX (年化, 定性)", color=NDX_C)
b2 = ax.bar(x + w/2, btc_scen, w, label="BTC (年化, 定性)", color=BTC_C)
for b in list(b1)+list(b2):
    ax.annotate(f"{b.get_height():+.0f}%", (b.get_x()+b.get_width()/2, b.get_height()),
                ha="center", va="bottom" if b.get_height()>=0 else "top", fontsize=9)
ax.axhline(0, color="grey", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(scen)
ax.set_ylabel("3-5年年化预期回报 (%)")
ax.set_title("NDX vs BTC 分情景预期回报 (定性估算)\n熊市=高利率/衰退；中性=盈利增长+减半周期；牛市=AI扩散+机构采用", fontsize=10)
ax.legend(); ax.grid(axis="y", alpha=0.3)
p5 = save(fig, "chart5_scenarios.png")

print("CHARTS:", p1, p2, p3, p4, p5)

report = """# 比特币 (BTC) vs 纳斯达克100 (NDX)：未来 3-5 年投资价值对比分析

**AS_OF: 报告生成 2026-10-08 10:56（本地）；NDX/BTC 价格数据为 2026-10-07 收盘；估值/利率数据为 2026-10-02~05 区间；机构预测为 2026-10-03~07 区间**

> **数据口径声明**：本报告综合 Step 1（基础数据）与 Step 2（风险/收益驱动）的检索结果。所有数据来自公开来源（Yahoo Finance、GuruFocus、Nasdaq 官方、Russell/Vanguard/Wells Fargo 2026 展望、CryptoRank、CoinStats、Binance Research、JPMorgan 监管文件）。**年化波动率、最大回撤、BTC 3Y/5Y 回报、机构价格预测为代理/估算/定性数据**，已逐项标注，引用时须注意口径。

---

## 一、分析背景

本报告对比 **比特币 (BTC)** 与 **纳斯达克100 (NDX)** 在 **未来 3-5 年** 的投资价值，从 **风险** 与 **潜在涨幅** 两个维度综合评估，并给出分情景结论与配置建议。

**当前时点关键背景（2026-10）：**
- **NDX**：QQQ $757.73（2026-10-07），YTD +24.08%，1Y +25.59%，处于 AI 驱动的高位；P/E ~29-31（合理偏高）。
- **BTC**：$84,199（2026-10-07），1Y 约 -33%，处于 2025-10 峰值 $126,198 后的周期下行/底部区域；下次减半 2028-04。
- **宏观**：10Y 美债 **5.28%**（2002-05-15 以来最高），Fed 大概率维持利率不变；高实际利率同时压制两者估值。
- **结构性变化**：BTC 与 NDX 短期相关性降至 **-0.43**（负相关），长期仍 **0.8**；BTC 从"被动风险资产"转为"前瞻性价格发现机制"。

---

## 二、数据概览

### 2.1 基础数据对比（截至 2026-10-07）

| 维度 | NDX (QQQ) | BTC | 数据口径 |
| --- | --- | --- | --- |
| 当前价格 | $757.73 | $84,199.14 | 精确（10-07 收盘） |
| 1 年回报 | +25.59% | 约 -33% | NDX 精确 / BTC 估算 |
| 3 年回报（年化） | +28.43% | 约 +150%~200% | NDX 精确 / BTC 估算 |
| 5 年回报（年化） | 约 +22%~23% | 约 +150%~250% | NDX 精确 / BTC 估算 |
| 估值 | P/E ~29-31（长期均值 27.86） | 无 P/E（市值 ~$1.67T 估算） | NDX 精确 / BTC 估算 |
| 年化波动率 | 约 20%-25% | 约 40%-60% | **均为代理** |
| 最大回撤（近5年/当前周期） | 约 -33%~-35%（2022） | 约 -50%~-52%（2025 周期） | **均为代理** |
| Beta | 1.26 | 无（高贝塔） | NDX 精确 |
| 关键周期 | 无（持续盈利增长） | 4 年减半（下次 2028-04） | BTC 精确 |

![累计回报对比](chart1_returns.png)

> **解读**：NDX 过去 1-5 年回报稳健（年化 ~22-28%），波动可控；BTC 长期回报弹性更大（3Y/5Y 估算 +150%~250%），但 1 年深度回撤（-33%），波动为 NDX 的 2-3 倍。

---

## 三、风险维度对比

### 3.1 风险指标可视化

![风险指标对比](chart2_risk.png)

> **解读**：BTC 年化波动率（~50%）约为 NDX（~22.5%）的 2 倍多；当前周期最大回撤（~-51%）也显著深于 NDX 近 5 年（~-34%）。**BTC 是典型的高波动、高回撤资产。**

### 3.2 NDX 核心风险（未来 3-5 年）

| 风险类别 | 具体内容 | 严重度 |
| --- | --- | --- |
| **估值风险** | P/E ~29-31 高于长期均值 27.86；"expensive valuations" 被列为核心风险 | 中高 |
| **AI 集中度** | 前十大持仓占 46.84%，AI 主题集中度过高；互联网泡沫重演担忧 | 中高 |
| **利率风险** | 10Y 美债 5.28%（2002 年以来最高）压制成长股估值；利率波动 | 高 |
| **宏观/地缘** | 伊朗冲突、通胀反弹、政治不确定性、关税 | 中 |
| **结构性** | 盈利领导权收窄至 AI 龙头；未来回报依赖盈利增长而非估值扩张 | 中 |

### 3.3 BTC 核心风险（未来 3-5 年）

| 风险类别 | 具体内容 | 严重度 |
| --- | --- | --- |
| **利率/宏观** | 高实际利率（10Y 5.28%）、强美元、全球衰退、ETF 流出（熊市情景） | 高 |
| **监管** | "unresolved regulatory and macro uncertainty remains its biggest risk to the pace of adoption" | 高 |
| **周期** | 2025-10 峰值 $126,198 已过，当前处于周期下行/底部区域；矿工经济承压 | 中高 |
| **本金损失** | JPMorgan 结构化票据：IBIT 跌 >30% 可能损失全部本金 | 中（针对杠杆产品） |
| **机构/流动性** | 机构需求减弱、强制卖出、估值倍数压缩（熊市情景） | 中 |

### 3.4 风险维度小结

- **NDX**：风险主要来自**估值偏高 + AI 集中度 + 高实际利率**，属于"高位成长股"的系统性风险；回撤可控（~-34%），但估值压缩空间有限。
- **BTC**：风险来自**高实际利率 + 监管不确定性 + 周期位置**，波动与回撤显著更大（~-51%），且存在本金损失尾部风险（杠杆产品）。
- **利率敏感性**：NDX 对实际利率**高度敏感**（成长股估值）；BTC 已从"被动反应"转为"前瞻性定价"，短期部分解耦，但高实际利率仍是长期利空。

---

## 四、涨幅维度对比

### 4.1 NDX 收益驱动（未来 3-5 年）

| 驱动 | 内容 | 确定性 |
| --- | --- | --- |
| **AI 生产力 J 曲线** | 生成式 AI 向生产力转化，提升营收与利润率 | 中高 |
| **AI 资本开支** | AI buildout 支撑实际 GDP >2% | 中高 |
| **盈利增长** | "Earnings growth, rather than valuation expansion, is likely to drive equity returns" | 高 |
| **AI 扩散** | AI 从少数龙头向全行业扩散 | 中 |
| **宏观支撑** | GDP >2%、宽松金融条件、财富效应、健康劳动力市场 | 中高 |

**机构展望**（Russell/Vanguard/Wells Fargo）："solid returns, risk skews to the upside"——未来回报靠**盈利增长**而非估值扩张。

### 4.2 BTC 收益驱动（未来 3-5 年）

| 驱动 | 内容 | 确定性 |
| --- | --- | --- |
| **减半周期** | 下次减半 2028-04（日发行 450→225 BTC），2027-2028 预期周期峰值 | 中高 |
| **机构采用** | 现货 ETF 流入、结构化产品、机构提前 6-12 个月建仓 | 中高 |
| **储备资产** | 弱美元、低实际收益率、组合分散价值（Fidelity） | 中 |
| **供应稀缺** | 固定 2100 万上限 + 减半驱动稀缺 | 高 |

**机构价格预测**（CryptoRank / CoinStats，定性区间）：

![BTC 机构价格预测](chart3_btc_forecast.png)

| 年份 | 保守 | 中性 | 乐观 |
| --- | --- | --- | --- |
| 2026 | $75,000 | $100,000 | $120,000+ |
| 2027 | $90,000 | $150,000 | $200,000+ |
| 2028 | $120,000 | $200,000 | $300,000+ |
| 2029 | $150,000 | $250,000 | $400,000+ |
| 2030 | $140,000 | $240,000 | $380,000 |

> **解读**：BTC 涨幅弹性来自**减半周期 + 机构采用 + 供应稀缺**，中性情景 2030 达 $240,000（较当前 +185%），乐观情景 $380,000（+350%）。但**预测为定性区间**，不确定性极高。

### 4.3 涨幅维度小结

- **NDX**：涨幅来自**盈利增长**（确定性较高），年化预期 ~12-18%（中性-牛市），但估值已偏高，上行空间受估值压缩制约。
- **BTC**：涨幅来自**减半周期 + 机构采用**（弹性大但确定性低），中性情景年化 ~25%，牛市可达 ~45%，但熊市可能负回报。

---

## 五、相关性结构（组合视角）

![相关性结构](chart4_correlation.png)

| 指标 | 数值 | 含义 |
| --- | --- | --- |
| BTC-NDX 20 日相关系数 | **-0.43** | 短期反向运动（不同驱动） |
| BTC-NDX 长期相关性 | **0.8** | 长期仍为风险资产，同涨同跌 |
| BTC-S&P 500 相关性 | **11 年最低**（2015 年以来） | BTC 与美股短期解耦 |
| BTC-全球宽松广度指数 | +0.21（ETF 前）→ **-0.778**（2026） | 结构性反转 |

> **关键发现**：BTC 与 NDX **短期负相关**（-0.43）但**长期强相关**（0.8）。这意味着：
> - **短期**：两者可作为**对冲/分散**工具（不同驱动因素）。
> - **长期**：两者仍为**风险资产**，系统性风险（高利率/衰退）下会同跌。
> - **结构性变化**：ETF 机构化后，BTC 从"被动风险资产"转为"前瞻性价格发现机制"（机构提前 6-12 个月建仓）。

---

## 六、分情景综合评估（未来 3-5 年）

![分情景预期回报](chart5_scenarios.png)

| 情景 | 宏观假设 | NDX 年化预期 | BTC 年化预期 | 风险等级 |
| --- | --- | --- | --- | --- |
| **熊市** | 高利率持续 + 全球衰退 + AI 泡沫破裂 | ~+5%（估值压缩） | ~-10%（周期底部） | 高 |
| **中性** | 盈利增长 + 减半周期 + 机构采用 | ~+12% | ~+25% | 中 |
| **牛市** | AI 扩散 + 弱美元 + 储备资产化 | ~+18% | ~+45% | 中高 |

> **注**：以上为**定性估算**（基于机构展望与预测区间），非精确数据。

### 6.1 风险-涨幅综合评分

| 维度 | NDX | BTC | 优势方 |
| --- | --- | --- | --- |
| **风险（低=优）** | 波动 ~22.5%，回撤 ~-34% | 波动 ~50%，回撤 ~-51% | **NDX** |
| **涨幅弹性（高=优）** | 年化 ~12-18% | 年化 ~25-45%（中性-牛市） | **BTC** |
| **确定性（高=优）** | 盈利增长驱动，确定性高 | 减半周期+机构采用，确定性中 | **NDX** |
| **利率敏感性（低=优）** | 高（成长股估值） | 中（短期解耦） | **BTC**（短期） |
| **分散价值** | 基准资产 | 短期负相关（-0.43），可对冲 | **BTC**（短期） |
| **尾部风险（低=优）** | 估值压缩 | 本金损失（杠杆）+ 监管 | **NDX** |

---

## 七、结论与配置建议

### 7.1 核心结论

1. **NDX 是"稳健成长"选择**：风险可控（波动 ~22.5%、回撤 ~-34%），涨幅来自盈利增长（确定性高），但估值偏高（P/E ~29-31）制约上行空间。**适合追求稳健、低波动的核心配置。**
2. **BTC 是"高弹性卫星"选择**：涨幅弹性大（中性年化 ~25%、牛市 ~45%），但风险显著更高（波动 ~50%、回撤 ~-51%），且存在监管与周期尾部风险。**适合风险承受能力强、追求弹性的卫星配置。**
3. **两者短期负相关（-0.43）**：在组合中可起到**分散/对冲**作用，但长期仍为风险资产（相关 0.8），系统性风险下会同跌。
4. **高实际利率（10Y 5.28%）是共同利空**：NDX 估值受压，BTC 长期承压；但 BTC 短期已部分解耦。

### 7.2 配置建议（3-5 年视角）

| 投资者类型 | 建议配置 | 理由 |
| --- | --- | --- |
| **保守型** | NDX 80-90% + BTC 0-5% | 追求稳健，BTC 仅作极小卫星 |
| **平衡型** | NDX 60-70% + BTC 10-20% | 兼顾稳健与弹性，利用短期负相关分散 |
| **进取型** | NDX 40-50% + BTC 25-40% | 追求弹性，承受高波动与回撤 |

> **配置要点**：
> - **NDX 为核心**：盈利增长驱动，确定性高，适合作为组合基石。
> - **BTC 为卫星**：减半周期（2028-04）+ 机构采用提供弹性，但须控制仓位（≤20-40%），避免杠杆产品（本金损失风险）。
> - **再平衡**：利用两者短期负相关，定期再平衡可捕捉波动收益。
> - **风险对冲**：高利率/衰退情景下，两者会同跌，需搭配债券/现金对冲系统性风险。

### 7.3 关键监测指标

| 资产 | 监测指标 | 触发信号 |
| --- | --- | --- |
| NDX | P/E（>35 警惕）、AI 资本开支、10Y 美债 | P/E 突破 35 或 10Y >6% 减仓 |
| BTC | 减半日期（2028-04）、ETF 流入、监管框架 | ETF 持续流出或监管收紧减仓 |
| 共同 | 10Y 美债、美元指数、全球 GDP | 10Y >6% 或衰退信号降低风险敞口 |

---

## 八、数据口径与免责声明

### 8.1 数据口径标注

| 数据项 | 口径 | AS_OF |
| --- | --- | --- |
| NDX/BTC 价格 | **精确**（10-07 收盘） | 2026-10-07 |
| NDX 1Y/3Y 回报、P/E | **精确** | 2026-10-02~07 |
| NDX 年化波动率、最大回撤 | **代理**（基于 Beta 与官方年度回报估算） | 2026-10-07 |
| BTC 3Y/5Y 回报、市值 | **估算**（基于历史价格锚点） | 2026-10-07 |
| BTC 年化波动率、最大回撤 | **代理**（基于历史周期） | 2026-10-07 |
| 10Y 美债 5.28% | **精确** | 2026-10-02 |
| BTC-NDX 相关系数 | **精确**（20 日 -0.43 / 长期 0.8） | 2026-10-03~04 |
| 机构价格预测（2026-2030） | **定性区间**（CryptoRank/CoinStats） | 2026-10-03~07 |
| 分情景预期回报 | **定性估算** | 2026-10-08 |

### 8.2 免责声明

- 本报告基于公开来源数据，**不构成投资建议**。
- **代理/估算/定性数据**（波动率、最大回撤、BTC 3Y/5Y 回报、机构预测、分情景回报）存在不确定性，引用时须注意口径。
- 市场有风险，投资需谨慎。过往表现不代表未来收益。

---

## 九、图表索引

| 图表 | 文件 | 内容 |
| --- | --- | --- |
| 图1 | `chart1_returns.png` | NDX vs BTC 历史累计回报对比 |
| 图2 | `chart2_risk.png` | 年化波动率 & 最大回撤对比 |
| 图3 | `chart3_btc_forecast.png` | BTC 机构价格预测区间 2026-2030 |
| 图4 | `chart4_correlation.png` | NDX 与 BTC 相关性结构 |
| 图5 | `chart5_scenarios.png` | 分情景预期回报对比 |

---

*报告生成：2026-10-08 10:56（本地）| 数据 AS_OF：2026-10-07 收盘（价格）/ 2026-10-02~05（估值/利率）/ 2026-10-03~07（机构预测）*
"""

rp = os.path.join(OUT, "report_btc_vs_ndx_3_5yr_2026-10-08.md")
with open(rp, "w", encoding="utf-8") as f:
    f.write(report)
print("REPORT:", rp)
print("DONE")
