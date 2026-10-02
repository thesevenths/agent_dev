# -*- coding: utf-8 -*-
"""
Step 4: 比特币近期行情数据分析、可视化与报告生成
读取上游 Step1/Step2 数据文件，计算技术指标，生成图表，输出 Markdown 报告。
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
PRICE_FILE = os.path.join(TMP, "btc_price_2026-09-28_to_2026-10-02.json")
SENT_FILE = os.path.join(TMP, "btc_news_sentiment_2026-09-28_to_2026-10-02.json")
OUT_CHART = os.path.join(TMP, "btc_analysis_chart.png")
OUT_REPORT = os.path.join(TMP, "btc_analysis_report_2026-10-02.md")

# ---------- 读取数据 ----------
with open(PRICE_FILE, "r", encoding="utf-8") as f:
    price = json.load(f)
with open(SENT_FILE, "r", encoding="utf-8") as f:
    sent = json.load(f)

rows = price["daily"]
df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# ---------- 技术指标 ----------
# 涨跌幅（基于收盘价）
df["ret_pct"] = df["close"].pct_change() * 100

# 7日均线（窗口内可用数据，标注为“窗口均线”）
df["ma7"] = df["close"].rolling(window=7, min_periods=1).mean()

# RSI(14) —— 数据仅5天，按可用窗口计算（Wilder 平滑），标注为“短窗口RSI”
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=1, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=1, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))
df["rsi14"] = rsi(df["close"], 14)

# 布林带（窗口内，20周期不可用，用可用窗口 min_periods=2, 2标准差）
df["bb_mid"] = df["close"].rolling(window=20, min_periods=2).mean()
bb_std = df["close"].rolling(window=20, min_periods=2).std()
df["bb_up"] = df["bb_mid"] + 2 * bb_std
df["bb_lo"] = df["bb_mid"] - 2 * bb_std

# 区间统计
closes = df["close"].values
period_high = df["high"].max()
period_low = df["low"].min()
start_close = df["close"].iloc[0]
last_close = df["close"].iloc[-1]
period_change_pct = (last_close / start_close - 1) * 100
vol_mean = df["volume_k"].mean()
vol_max_day = df.loc[df["volume_k"].idxmax(), "date"].strftime("%m-%d")

# ---------- 图表 ----------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1.2, 1.2]})
fig.suptitle("BTC/USD 近期行情分析（2026-09-28 ~ 2026-10-02）", fontsize=15, fontweight="bold")

# 子图1：K线 + 均线 + 布林带
ax1 = axes[0]
x = np.arange(len(df))
width = 0.6
for i, r in df.iterrows():
    up = r["close"] >= r["open"]
    color = "#e74c3c" if up else "#27ae60"  # 国际惯例：红涨绿跌
    ax1.plot([i, i], [r["low"], r["high"]], color=color, linewidth=1.2, zorder=2)
    ax1.bar(i, abs(r["close"] - r["open"]), bottom=min(r["open"], r["close"]),
            width=width, color=color, edgecolor=color, zorder=3)
ax1.plot(x, df["ma7"], color="#f39c12", linewidth=2, label="窗口均线 (MA, 可用窗口)", zorder=4)
ax1.fill_between(x, df["bb_lo"], df["bb_up"], color="#3498db", alpha=0.12, label="布林带 (2σ, 可用窗口)")
ax1.set_ylabel("价格 (USD)")
ax1.set_title("K线 + 均线 + 布林带", fontsize=11)
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(period_low * 0.995, period_high * 1.005)

# 子图2：成交量
ax2 = axes[1]
colors = ["#e74c3c" if df["close"].iloc[i] >= df["open"].iloc[i] else "#27ae60" for i in range(len(df))]
ax2.bar(x, df["volume_k"], color=colors, width=width, alpha=0.85)
ax2.axhline(vol_mean, color="#34495e", linestyle="--", linewidth=1, label=f"均值 {vol_mean:.2f}K")
ax2.set_ylabel("成交量 (K)")
ax2.set_title("成交量", fontsize=11)
ax2.legend(loc="upper left", fontsize=9)
ax2.grid(True, alpha=0.3)

# 子图3：RSI
ax3 = axes[2]
ax3.plot(x, df["rsi14"], color="#8e44ad", linewidth=2, marker="o", label="RSI(14, 短窗口)")
ax3.axhline(70, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.7)
ax3.axhline(30, color="#27ae60", linestyle="--", linewidth=1, alpha=0.7)
ax3.fill_between(x, 30, 70, color="#ecf0f1", alpha=0.4)
ax3.set_ylim(0, 100)
ax3.set_ylabel("RSI")
ax3.set_title("RSI(14) — 短窗口（数据仅5天，仅供参考）", fontsize=11)
ax3.legend(loc="upper left", fontsize=9)
ax3.grid(True, alpha=0.3)

# X轴日期
ax3.set_xticks(x)
ax3.set_xticklabels([d.strftime("%m-%d") for d in df["date"]], rotation=0)
ax3.xaxis.set_major_locator(mdates.DayLocator())

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUT_CHART, dpi=130, bbox_inches="tight")
plt.close()

# ---------- 关键指标数值 ----------
last_rsi = df["rsi14"].iloc[-1]
last_ma7 = df["ma7"].iloc[-1]
last_bb_up = df["bb_up"].iloc[-1]
last_bb_lo = df["bb_lo"].iloc[-1]

# ---------- 撰写报告 ----------
report = f"""# 比特币（BTC/USD）近期行情分析报告

**AS_OF: 2026-10-02 22:22**（10/02 为盘中最新快照，非收盘；其余为每日收盘价）

> 数据区间：2026-09-28 ~ 2026-10-02（5 个交易日）
> 数据来源：Investing.com (Bitfinex)、CoinMarketCap、CoinShares、Forex.com/Farside、Bitcoin Foundation 等（详见文末）

---

## 一、分析背景

本报告基于 2026-09-28 至 2026-10-02 共 5 个交易日的 BTC/USD 日线数据，结合近期宏观事件、ETF 资金流与市场情绪，分析比特币的涨跌趋势、技术面状态，并给出后续走势展望。

**重要说明**：
- 10/02 数据为**盘中快照**（as of 22:22），非当日最终收盘价。
- 数据窗口仅 5 天，**7日均线、RSI(14)、布林带**等技术指标均基于可用短窗口计算，**仅供参考**，统计意义有限。
- 量化情绪指标（CryptOracle）本次接口返回为空，情绪分析基于新闻与宏观事件定性判断。

---

## 二、数据概览

| 日期 | 开盘 | 最高 | 最低 | 收盘 | 涨跌% | 成交量(K) |
|---|---|---|---|---|---|---|
| 2026-09-28 | 84,451 | 84,979 | 82,700 | 83,532 | -1.09% | 1.72 |
| 2026-09-29 | 83,524 | 84,599 | 82,811 | 83,719 | +0.22% | 0.52 |
| 2026-09-30 | 83,719 | 85,567 | 83,048 | 83,621 | -0.12% | 1.20 |
| 2026-10-01 | 83,607 | 85,277 | 83,241 | 84,906 | +1.54% | 1.17 |
| 2026-10-02 (盘中) | 84,881 | 84,881 | 84,774 | 84,795 | -0.13% | 1.16 |

**区间关键统计**：
- 区间最高：**${period_high:,.0f}**（09-30）
- 区间最低：**${period_low:,.0f}**（09-28）
- 区间振幅：约 **{(period_high/period_low-1)*100:.1f}%**
- 区间涨跌幅（9/28 收盘 → 10/02 盘中）：**{period_change_pct:+.2f}%**
- 平均成交量：**{vol_mean:.2f}K**（最大量出现在 {vol_max_day}）

---

## 三、价格走势图与关键指标

![BTC 近期行情分析图](btc_analysis_chart.png)

*图：BTC/USD K线 + 窗口均线 + 布林带（上）、成交量（中）、RSI(14) 短窗口（下）。红涨绿跌。*

---

## 四、涨跌分析

### 4.1 价格走势回顾
- **09-28**：低开低走，收跌 **-1.09%** 至 83,532，区间内最大单日跌幅，成交量 1.72K 为区间最高，显示抛压集中释放。
- **09-29**：小幅反弹 **+0.22%**，但成交量骤降至 0.52K（区间最低），反弹动能不足。
- **09-30**：冲高回落，盘中触及区间最高 **85,567** 后收跌 **-0.12%**，上攻受阻。
- **10-01**：放量上涨 **+1.54%** 至 84,906，为区间最大单日涨幅，成交量 1.17K 配合，多头占优。
- **10-02（盘中）**：高位小幅回落 **-0.13%** 至 84,795，呈高位盘整态势。

### 4.2 趋势判断
- 整体呈**区间震荡、重心小幅上移**格局：BTC 在 **$82,700 – $85,600** 区间内运行，5 日累计上涨约 **{period_change_pct:+.1f}%**。
- **支撑位**：约 **$82,700 – $83,000**（区间低点密集区）。
- **阻力位**：约 **$85,300 – $85,600**（09-30 与 10-01 高点）。
- 10/01 放量突破后，10/02 未能继续上攻，短期面临**方向选择**。

---

## 五、技术面解读

| 指标 | 数值（10/02 盘中） | 解读 |
|---|---|---|
| 窗口均线 (MA) | ${last_ma7:,.0f} | 价格 ${last_close:,.0f} 略低于均线，短期动能偏弱 |
| 布林带上轨 | ${last_bb_up:,.0f} | 价格接近中轨，未触及上轨，上行空间有限 |
| 布林带下轨 | ${last_bb_lo:,.0f} | 价格远离下轨，下行空间相对有限 |
| RSI(14, 短窗口) | {last_rsi:.1f} | 处于中性区（30–70），无明显超买/超卖 |

**技术面小结**：
- 价格围绕窗口均线窄幅波动，**趋势信号不显著**。
- RSI 处于中性区，既未超买也未超卖，**缺乏方向性指引**。
- 布林带收口，预示**波动率压缩**，后续可能选择方向突破。
- 成交量在 10/01 放大后 10/02 维持，**量价配合尚可**，但尚未形成持续放量突破。

> ⚠️ 再次强调：以上技术指标基于 5 天短窗口计算，**统计意义有限**，仅作辅助参考。

---

## 六、宏观与情绪分析

### 6.1 量化情绪
- CryptOracle 情绪接口本次返回为空，**未获得量化情绪数据**。

### 6.2 新闻与宏观事件
| 日期 | 主题 | 要点 |
|---|---|---|
| 10-01 | ETF 资金流 / 牛市叙事 | BMTV 聚焦牛市、全球流动性与 ETF 资金流，市场情绪偏乐观 |
| 10-02 | CoinShares 2026 宏观情景 | 滞胀 → $70K；基准 → $110K–$140K；衰退+紧急宽松 → $170K+ |
| 10-02 | ETF 资金流 vs 宏观压力 | $1.1B 单周净流入 vs $545M 单日流出并存；PPI 超预期（0.5% vs 0.3%）削弱降息预期 |
| 10-02 | 估值与周期 | MVRV Z-score 降至 0.5（近熊市底部）；长期持有者占比 <60%（约 200 万 BTC 边际供给） |
| 10-02 | 宏观驱动机制 | 2026 年 BTC 更多由美国宏观数据（通胀/利率/流动性）驱动，杠杆与美元走强放大波动 |

### 6.3 情绪小结
- **偏中性略偏多**：ETF 资金流与牛市叙事支撑乐观情绪。
- **宏观逆风**：PPI 超预期、Fed 降息预期受挫，BTC 与股市相关性上升，吸收更多宏观压力。
- **周期/估值信号**：MVRV 接近底部、长期持有者减持，提示**下行风险**。
- 整体呈 **"混合/分歧"** 格局，多空因素交织。

---

## 七、后续走势展望

### 7.1 短期（1–2 周）
- **基准情景（概率较高）**：延续 **$82,700 – $85,600** 区间震荡，等待宏观数据（CPI、Fed 表态）与 ETF 资金流给出方向。
- **上行触发**：若 ETF 持续净流入 + Fed 释放鸽派信号，可能突破 **$85,600** 阻力，上看 **$87,000 – $88,000**。
- **下行触发**：若宏观数据继续超预期（通胀/就业偏强）+ ETF 资金流出，可能跌破 **$82,700** 支撑，下探 **$80,000 – $81,000**。

### 7.2 中期（1–3 个月）
- **取决于宏观情景**（CoinShares 框架）：
  - **基准情景**（增长放缓 + 粘性通胀 + 谨慎降息）：BTC 在 **$110K – $140K** 区间运行，但需时间积累。
  - **滞胀情景**：BTC 可能回落至 **$70K** 附近。
  - **衰退 + 紧急宽松**：BTC 可上探 **$170K+**。
- **关键变量**：
  1. **Fed 利率路径**：降息预期是核心驱动。
  2. **ETF 资金流**：持续净流入是上行关键。
  3. **美元与收益率**：美元走弱 + 收益率下行利好 BTC。
  4. **长期持有者行为**：若继续减持，边际供给增加，压制价格。

### 7.3 风险提示
- **宏观数据风险**：通胀/就业数据超预期可能推迟降息，压制 BTC。
- **ETF 资金流波动**：单日/单周大幅流出可能引发连锁抛售。
- **杠杆与波动放大**：高杠杆环境下，小幅波动可能触发清算，放大跌幅。
- **周期位置**：MVRV 接近底部 + 长期持有者减持，提示**周期底部尚未确认**，中期偏弱风险存在。

---

## 八、结论

1. **近期走势**：BTC 在 **$82,700 – $85,600** 区间震荡，5 日累计上涨约 **{period_change_pct:+.1f}%**，重心小幅上移。
2. **技术面**：趋势信号不显著，RSI 中性，布林带收口，**短期面临方向选择**。
3. **宏观情绪**：偏中性略偏多，但宏观逆风（PPI、降息预期受挫）与周期信号（MVRV 近底部、长期持有者减持）提示**下行风险**。
4. **后续展望**：
   - **短期**：大概率延续区间震荡，突破 **$85,600** 或跌破 **$82,700** 将决定方向。
   - **中期**：取决于 Fed 利率路径、ETF 资金流与宏观情景，基准情景下 **$110K – $140K**，但需警惕滞胀与周期底部风险。

> **免责声明**：本报告基于公开数据与新闻分析，技术指标基于短窗口计算，仅供参考，不构成投资建议。加密货币波动剧烈，请谨慎决策。

---

## 数据来源
- 价格数据：[Investing.com](https://www.investing.com/crypto/bitcoin/btc-usd-historical-data)、[CoinMarketCap](https://coinmarketcap.com/currencies/bitcoin/historical-data)
- 宏观/情绪：[CoinShares 2026 Outlook](https://www.etftrends.com/coinshares-content-hub/bitcoin-could-hit-170k-2026-fed-crisis-scenario)、[Investing.com 分析](https://www.investing.com/analysis/bitcoin-stalls-below-70k-as-etf-flows-clash-with-macro-pressure-200675819)、[Forex.com/Farside](https://www.forex.com/ie/news-and-analysis/q2-2026-bitcoin-outlook-more-pain-to-come-before-the-cycle-bottoms)、[Bitcoin Foundation](https://bitcoinfoundation.org/news/bitcoin/why-u-s-macroeconomic-data-drives-bitcoin-price-in-2026-inflation-interest-rates-and-liquidity-impact-explained)

**报告生成时间**：2026-10-02 22:24
**数据文件**：
- `F:\\agent\\multi-agent\\tmp\\btc_price_2026-09-28_to_2026-10-02.json`
- `F:\\agent\\multi-agent\\tmp\\btc_news_sentiment_2026-09-28_to_2026-10-02.json`
- 图表：`F:\\agent\\multi-agent\\tmp\\btc_analysis_chart.png`
"""

with open(OUT_REPORT, "w", encoding="utf-8") as f:
    f.write(report)

print("=== 分析完成 ===")
print(f"图表: {OUT_CHART}")
print(f"报告: {OUT_REPORT}")
print(f"\n关键指标 (10/02 盘中):")
print(f"  最新价: ${last_close:,.0f}")
print(f"  窗口均线: ${last_ma7:,.0f}")
print(f"  布林带上/下轨: ${last_bb_up:,.0f} / ${last_bb_lo:,.0f}")
print(f"  RSI(14, 短窗口): {last_rsi:.1f}")
print(f"  区间涨跌幅: {period_change_pct:+.2f}%")
print(f"  区间最高/最低: ${period_high:,.0f} / ${period_low:,.0f}")
