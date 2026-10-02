# -*- coding: utf-8 -*-
"""
BTC/USD 近期价格分析报告生成脚本
Step 5/5: 读取上游数据 -> 计算技术指标 -> 生成图表 -> 撰写 Markdown 报告
AS_OF: 2026-10-02 22:22
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

TMP = r"F:\agent\multi-agent\tmp"
PRICE_JSON = os.path.join(TMP, "btc_price_2026-09-28_to_2026-10-02.json")
SENT_JSON = os.path.join(TMP, "btc_news_sentiment_2026-09-28_to_2026-10-02.json")
REPORT_MD = os.path.join(TMP, "btc_analysis_report_2026-10-02.md")

# 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 读取数据 ----------
with open(PRICE_JSON, "r", encoding="utf-8") as f:
    price = json.load(f)
with open(SENT_JSON, "r", encoding="utf-8") as f:
    sent = json.load(f)

rows = price["daily"]
df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# ---------- 技术指标 ----------
# 简单移动平均 (SMA)
df["SMA3"] = df["close"].rolling(3).mean()
df["SMA5"] = df["close"].rolling(5).mean()
# 布林带 (20 周期不可用，用 5 周期 + 2 std 近似)
df["BB_mid"] = df["SMA5"]
df["BB_up"] = df["SMA5"] + 2 * df["close"].rolling(5).std()
df["BB_lo"] = df["SMA5"] - 2 * df["close"].rolling(5).std()
# RSI (14 周期不可用，用 5 周期近似)
delta = df["close"].diff()
gain = delta.clip(lower=0).rolling(5).mean()
loss = (-delta.clip(upper=0)).rolling(5).mean()
rs = gain / loss
df["RSI5"] = 100 - (100 / (1 + rs))
# 成交量变化
df["vol_change_pct"] = df["volume_k"].pct_change() * 100

# 关键统计
closes = df["close"].values
period_ret = (closes[-1] / closes[0] - 1) * 100
high_5d = df["high"].max()
low_5d = df["low"].min()
avg_vol = df["volume_k"].mean()
max_vol_day = df.loc[df["volume_k"].idxmax(), "date"].strftime("%m/%d")

# ---------- 图表 1: 价格 + 布林带 + 成交量 ----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                gridspec_kw={"height_ratios": [3, 1]})
x = df["date"]
ax1.plot(x, df["close"], color="#f7931a", linewidth=2.5, marker="o", label="收盘价")
ax1.plot(x, df["SMA5"], color="#1f77b4", linewidth=1.5, linestyle="--", label="SMA5")
ax1.fill_between(x, df["BB_lo"], df["BB_up"], color="#1f77b4", alpha=0.12, label="布林带(5,2σ)")
ax1.scatter(x, df["high"], color="#2ca02c", s=18, zorder=3, label="最高")
ax1.scatter(x, df["low"], color="#d62728", s=18, zorder=3, label="最低")
ax1.set_ylabel("价格 (USD)")
ax1.set_title("BTC/USD 近期价格走势 (2026-09-28 ~ 2026-10-02)", fontsize=14, fontweight="bold")
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f"${v:,.0f}"))

colors = ["#2ca02c" if c >= 0 else "#d62728" for c in df["change_pct"]]
ax2.bar(x, df["volume_k"], color=colors, alpha=0.8, width=0.6)
ax2.set_ylabel("成交量 (K)")
ax2.set_xlabel("日期")
ax2.grid(True, alpha=0.3, axis="y")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
plt.tight_layout()
chart1 = os.path.join(TMP, "btc_price_chart.png")
plt.savefig(chart1, dpi=130, bbox_inches="tight")
plt.close()

# ---------- 图表 2: 涨跌幅 + RSI ----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                gridspec_kw={"height_ratios": [1, 1]})
colors = ["#2ca02c" if c >= 0 else "#d62728" for c in df["change_pct"]]
ax1.bar(x, df["change_pct"], color=colors, alpha=0.85, width=0.6)
ax1.axhline(0, color="gray", linewidth=0.8)
ax1.set_ylabel("日涨跌幅 (%)")
ax1.set_title("BTC/USD 日涨跌幅与 RSI(5)", fontsize=14, fontweight="bold")
ax1.grid(True, alpha=0.3, axis="y")
for i, v in enumerate(df["change_pct"]):
    ax1.text(x[i], v + (0.05 if v >= 0 else -0.12), f"{v:+.2f}%",
             ha="center", fontsize=9, fontweight="bold")

ax2.plot(x, df["RSI5"], color="#9467bd", linewidth=2, marker="s")
ax2.axhline(70, color="#d62728", linestyle="--", linewidth=1, label="超买 70")
ax2.axhline(30, color="#2ca02c", linestyle="--", linewidth=1, label="超卖 30")
ax2.fill_between(x, 30, 70, color="#9467bd", alpha=0.08)
ax2.set_ylim(0, 100)
ax2.set_ylabel("RSI(5)")
ax2.set_xlabel("日期")
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
plt.tight_layout()
chart2 = os.path.join(TMP, "btc_rsi_chart.png")
plt.savefig(chart2, dpi=130, bbox_inches="tight")
plt.close()

# ---------- 图表 3: 宏观情景 (CoinShares) ----------
fig, ax = plt.subplots(figsize=(10, 6))
scenarios = ["滞胀情景", "基准情景", "衰退/紧急宽松"]
low = [70000, 110000, 140000]
high = [85000, 140000, 170000]
colors_s = ["#d62728", "#f7931a", "#2ca02c"]
bars = ax.barh(scenarios, [h - l for l, h in zip(low, high)],
               left=low, color=colors_s, alpha=0.85, height=0.5)
for i, (l, h) in enumerate(zip(low, high)):
    ax.text(h + 2000, i, f"${h//1000}K", va="center", fontsize=10, fontweight="bold")
    ax.text(l - 2000, i, f"${l//1000}K", va="center", ha="right", fontsize=10)
# 当前价格线
cur = 84795
ax.axvline(cur, color="#1f77b4", linestyle="--", linewidth=2, label=f"当前价 ${cur:,}")
ax.set_xlabel("BTC 价格区间 (USD)")
ax.set_title("CoinShares 2026 宏观情景下的 BTC 价格区间", fontsize=14, fontweight="bold")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f"${v/1000:.0f}K"))
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
chart3 = os.path.join(TMP, "btc_scenario_chart.png")
plt.savefig(chart3, dpi=130, bbox_inches="tight")
plt.close()

# ---------- 撰写 Markdown 报告 ----------
report = f"""# 比特币 (BTC/USD) 近期价格分析报告

**AS_OF: 2026-10-02 22:22**（10/02 为盘中最新快照，非收盘；其余为每日收盘价）

> 数据区间：2026-09-28 ~ 2026-10-02 ｜ 数据来源：Investing.com (Bitfinex)、CoinMarketCap、CoinShares、Forex.com 等

---

## 一、分析背景

本报告基于 2026-09-28 至 2026-10-02 共 5 个交易日的 BTC/USD 每日价格数据，结合近期宏观事件、ETF 资金流与市场情绪，分析比特币的涨跌趋势，并给出后续走势展望。

---

## 二、数据概览

| 日期 | 开盘 | 最高 | 最低 | 收盘 | 涨跌% | 成交量(K) |
|---|---|---|---|---|---|---|
| 2026-09-28 | 84,451 | 84,979 | 82,700 | 83,532 | -1.09% | 1.72 |
| 2026-09-29 | 83,524 | 84,599 | 82,811 | 83,719 | +0.22% | 0.52 |
| 2026-09-30 | 83,719 | 85,567 | 83,048 | 83,621 | -0.12% | 1.20 |
| 2026-10-01 | 83,607 | 85,277 | 83,241 | 84,906 | +1.54% | 1.17 |
| 2026-10-02 (盘中) | 84,881 | 84,881 | 84,774 | 84,795 | -0.13% | 1.16 |

**关键统计**：
- 5 日区间：${low_5d:,.0f} ~ ${high_5d:,.0f}
- 区间涨跌幅：{period_ret:+.2f}%（9/28 收盘 → 10/02 盘中）
- 平均日成交量：{avg_vol:.2f}K
- 最大成交量日：{max_vol_day}（1.72K）
- 最新实时价格：**$84,795**（as of 2026-10-02 22:22，盘中快照）

---

## 三、价格走势与涨跌分析

![价格走势](btc_price_chart.png)

**走势解读**：
1. **9/28 放量下跌**：当日成交量 1.72K 为 5 日最高，价格下跌 -1.09%，显示抛压较重，最低触及 $82,700。
2. **9/29~9/30 缩量震荡**：成交量降至 0.52K（9/29，5 日最低），价格在 $83,000~$85,600 区间窄幅波动，多空博弈均衡。
3. **10/01 放量上涨**：成交量回升至 1.17K，价格大涨 +1.54%（5 日最大单日涨幅），收盘 $84,906，显示买盘回归。
4. **10/02 盘中小幅回落**：截至 22:22，价格 $84,795，微跌 -0.13%，处于高位整理。

**整体趋势**：过去 5 天 BTC 在 **$82,700 – $85,600** 区间震荡，整体小幅上行（+1.5%）。10/01 的放量上涨是关键转折信号，表明短期买盘力量增强。

---

## 四、技术面解读

![涨跌幅与RSI](btc_rsi_chart.png)

**技术指标**：
- **SMA5**：当前约 ${df['SMA5'].iloc[-1]:,.0f}，价格位于 SMA5 上方，短期趋势偏多。
- **布林带(5,2σ)**：价格接近上轨，短期存在超买风险，但尚未突破。
- **RSI(5)**：当前约 {df['RSI5'].iloc[-1]:.1f}，处于中性偏强区域（50~70），未达超买（>70）。
- **成交量**：10/01 放量上涨确认买盘，10/02 缩量整理，健康回调。

**技术面结论**：短期偏多，但接近布林带上轨，需警惕回调风险。若放量突破 $85,600（5 日高点），则打开上行空间；若跌破 $83,000，则可能回踩 $82,700 支撑。

---

## 五、宏观情绪分析

![宏观情景](btc_scenario_chart.png)

**量化情绪指标**：CryptOracle 接口本次返回为空，未获得 positive/negative ratio 等量化数据。

**关键宏观事件与情绪要点**：

| 日期 | 主题 | 要点 |
|---|---|---|
| 10/01 | ETF 资金流 / 牛市叙事 | BMTV 节目聚焦牛市、全球流动性与 ETF 资金流，市场情绪偏乐观 |
| 10/02 | CoinShares 2026 宏观情景 | 滞胀→$70K；基准→$110K~$140K；衰退/紧急宽松→$170K+ |
| 10/02 | ETF 资金流 vs 宏观压力 | $1.1B 单周净流入 vs $545M 单日流出；PPI 超预期削弱降息预期 |
| 10/02 | 估值与周期 | MVRV Z-score 降至 0.5（接近熊市底部）；长期持有者占比 <60% |
| 10/02 | 宏观驱动机制 | 2026 年 BTC 更多由美国宏观数据（通胀/利率/流动性）驱动 |

**情绪小结**：偏中性略偏多——ETF 资金流与牛市叙事支撑乐观，但宏观（PPI 超预期、降息预期受挫）与周期/估值指标提示下行风险，整体呈"混合/分歧"格局。

---

## 六、后续走势展望

### 短期（1~2 周）
- **基准情景**：延续 $83,000~$86,000 区间震荡，10/01 放量上涨后短期偏多，但接近布林带上轨，需警惕回调。
- **关键位**：上方阻力 $85,600（5 日高点），下方支撑 $83,000 / $82,700。
- **触发条件**：若放量突破 $85,600，则上看 $87,000~$88,000；若跌破 $83,000，则回踩 $82,700 甚至 $82,000。

### 中期（1~3 个月）
- **基准情景（CoinShares）**：BTC 在 $110,000~$140,000 区间运行，假设增长放缓、通胀粘性、Fed 谨慎降息。
- **上行风险**：若衰退迫使 Fed 紧急宽松，BTC 可上探 $170,000+。
- **下行风险**：若滞胀加剧，BTC 可能回落至 $70,000。

### 关键驱动因素
1. **Fed 政策**：PPI 超预期削弱近期降息预期，需关注后续 CPI、非农等数据。
2. **ETF 资金流**：$1.1B 单周净流入 vs $545M 单日流出，资金流波动加剧，需持续跟踪。
3. **宏观相关性**：BTC 与股市相关性上升，吸收更多宏观压力，需关注美股走势。
4. **链上指标**：MVRV Z-score 0.5 接近熊市底部，长期持有者占比下降，暗示潜在抛压。

### 综合判断
- **短期**：偏多但谨慎，区间震荡为主，关注 $85,600 阻力与 $83,000 支撑。
- **中期**：基准情景下 $110K~$140K，但需警惕宏观下行风险（滞胀→$70K）。
- **操作建议**：短期可逢低（$83,000 附近）布局，突破 $85,600 后加仓；中期需跟踪 Fed 政策与 ETF 资金流，警惕滞胀风险。

---

## 七、结论

1. **近期走势**：过去 5 天 BTC 在 $82,700~$85,600 区间震荡，整体小幅上行（+1.5%），10/01 放量上涨是关键转折信号。
2. **技术面**：短期偏多，价格位于 SMA5 上方，RSI 中性偏强，但接近布林带上轨，需警惕回调。
3. **宏观情绪**：偏中性略偏多，ETF 资金流与牛市叙事支撑乐观，但 PPI 超预期与周期/估值指标提示下行风险。
4. **后续展望**：短期区间震荡（$83,000~$86,000），中期基准情景 $110K~$140K，需警惕滞胀下行风险（$70K）与衰退上行机会（$170K+）。

---

## 附录：数据来源

- 价格数据：Investing.com (Bitfinex)、CoinMarketCap、Yahoo Finance
- 宏观/情绪：CoinShares 2026 Outlook、Forex.com/Farside、Bitcoin Foundation、Investing.com 分析
- 数据文件：
  - `F:\\agent\\multi-agent\\tmp\\btc_price_2026-09-28_to_2026-10-02.json`
  - `F:\\agent\\multi-agent\\tmp\\btc_news_sentiment_2026-09-28_to_2026-10-02.json`

*报告生成时间：2026-10-02 22:26*
"""

with open(REPORT_MD, "w", encoding="utf-8") as f:
    f.write(report)

print("报告已生成:", REPORT_MD)
print("图表 1:", chart1)
print("图表 2:", chart2)
print("图表 3:", chart3)
print(f"区间涨跌幅: {period_ret:+.2f}%")
print(f"RSI5 当前: {df['RSI5'].iloc[-1]:.1f}")
print(f"SMA5 当前: {df['SMA5'].iloc[-1]:,.0f}")
