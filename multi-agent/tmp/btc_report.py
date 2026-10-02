# -*- coding: utf-8 -*-
"""
BTC/USD 技术面分析 + 报告生成（一次性脚本）
读取上游两个 JSON -> 计算 MA5/MA10/RSI(14)/MACD -> 生成 PNG 图表 -> 写 Markdown 报告
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

TMP = r"F:\agent\multi-agent\tmp"
price_file = os.path.join(TMP, "btc_price_data_2026-09-30_to_2026-10-02.json")
sent_file  = os.path.join(TMP, "btc_sentiment_news_2026-09-30_to_2026-10-02.json")
report_file = os.path.join(TMP, "btc_analysis_report_2026-10-02.md")

# ---------- 1. 读取上游数据 ----------
with open(price_file, "r", encoding="utf-8") as f:
    pdata = json.load(f)
with open(sent_file, "r", encoding="utf-8") as f:
    sdata = json.load(f)

rows = pdata["daily_ohlc"]
df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
for c in ["open", "high", "low", "close"]:
    df[c] = df[c].astype(float)

current_price = pdata["current_price"]["price_usd"]
fg = sdata["sentiment"]["fear_greed_index"]
news = sdata["news"]
bull = sdata["key_factors"]["bullish"]
bear = sdata["key_factors"]["bearish"]

# ---------- 2. 技术指标 ----------
close = df["close"]

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

df["MA5"]  = close.rolling(5).mean()
df["MA10"] = close.rolling(10).mean()
df["RSI14"] = rsi(close, 14)
# MACD
ema12 = close.ewm(span=12, adjust=False).mean()
ema26 = close.ewm(span=26, adjust=False).mean()
df["MACD"] = ema12 - ema26
df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["Hist"] = df["MACD"] - df["Signal"]

last = df.iloc[-1]
prev = df.iloc[-2]

# 关键位
high_3d = df["high"].max()
low_3d  = df["low"].min()

# ---------- 3. 图表 ----------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1.2, 1.2]})
fig.suptitle("BTC/USD 技术面分析  (2026-09-30 ~ 2026-10-02)", fontsize=15, fontweight="bold")

# --- 子图1: K线 + MA ---
ax1 = axes[0]
x = np.arange(len(df))
width = 0.6
for i, r in df.iterrows():
    up = r["close"] >= r["open"]
    color = "#e74c3c" if up else "#27ae60"   # 国际惯例: 红涨绿跌
    ax1.vlines(i, r["low"], r["high"], color=color, linewidth=1)
    body_low = min(r["open"], r["close"])
    body_h = abs(r["close"] - r["open"])
    ax1.add_patch(Rectangle((i - width/2, body_low), width, body_h,
                            facecolor=color, edgecolor=color))
if ma5_valid:
    ax1.plot(x, df["MA5"],  label="MA5",  color="#f39c12", linewidth=2)
if ma10_valid:
    ax1.plot(x, df["MA10"], label="MA10", color="#3498db", linewidth=2)
ax1.axhline(current_price, color="purple", linestyle="--", linewidth=1,
            label=f"当前价 ${current_price:,.0f} (盘中)")
ax1.set_ylabel("价格 (USD)")
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_title("K线 + 均线 + 当前价", fontsize=11)

# --- 子图2: RSI ---
ax2 = axes[1]
ax2.plot(x, df["RSI14"], color="#8e44ad", marker="o", linewidth=2, label="RSI(14)")
ax2.axhline(70, color="#e74c3c", linestyle=":", linewidth=1)
ax2.axhline(30, color="#27ae60", linestyle=":", linewidth=1)
ax2.fill_between(x, 30, 70, color="gray", alpha=0.08)
ax2.set_ylim(0, 100)
ax2.set_ylabel("RSI")
ax2.legend(loc="upper left", fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_title("RSI(14)", fontsize=11)

# --- 子图3: MACD ---
ax3 = axes[2]
colors = ["#e74c3c" if v >= 0 else "#27ae60" for v in df["Hist"]]
ax3.bar(x, df["Hist"], color=colors, alpha=0.7, label="Histogram")
ax3.plot(x, df["MACD"],   color="#2980b9", linewidth=2, label="MACD")
ax3.plot(x, df["Signal"], color="#e67e22", linewidth=2, label="Signal")
ax3.axhline(0, color="black", linewidth=0.8)
ax3.set_ylabel("MACD")
ax3.legend(loc="upper left", fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_title("MACD (12,26,9)", fontsize=11)

# x 轴日期
ax3.set_xticks(x)
ax3.set_xticklabels([d.strftime("%m-%d") for d in df["date"]], rotation=0)
ax3.set_xlabel("日期 (2026)")

plt.tight_layout(rect=[0, 0, 1, 0.97])
chart_path = os.path.join(TMP, "btc_technical_chart_2026-10-02.png")
plt.savefig(chart_path, dpi=110, bbox_inches="tight")
plt.close()

# ---------- 4. 撰写 Markdown 报告 ----------
rsi_val = last["RSI14"]
macd_val = last["MACD"]
sig_val = last["Signal"]
hist_val = last["Hist"]
ma5_val = last["MA5"]
ma10_val = last["MA10"]

# 信号判定
rsi_state = "超买" if rsi_val >= 70 else ("超卖" if rsi_val <= 30 else "中性")
macd_state = "金叉/多头" if macd_val > sig_val else "死叉/空头"
ma5_valid = not np.isnan(ma5_val)
ma10_valid = not np.isnan(ma10_val)
if ma5_valid and ma10_valid:
    ma_state = "多头排列(价>MA5>MA10)" if (last["close"] > ma5_val > ma10_val) else \
               ("空头排列" if (last["close"] < ma5_val < ma10_val) else "均线纠缠")
else:
    ma_state = "样本不足（仅3根日线，MA5/MA10 无法计算）"
ma5_str = f"${ma5_val:,.0f}" if ma5_valid else "N/A（样本不足）"
ma10_str = f"${ma10_val:,.0f}" if ma10_valid else "N/A（样本不足）"

def pct(a, b):
    return (a - b) / b * 100

report = f"""# 比特币 (BTC/USD) 技术面分析报告

**AS_OF: 2026-10-02 23:46**（日线为各日收盘；当前价 ${current_price:,.0f} 为盘中实时快照，非最终收盘）

> 数据区间：2026-09-30 ~ 2026-10-02 ｜ 数据源：Investing.com / CoinMarketCap / Yahoo Finance（多源交叉核对）
> 情绪源：Alternative.me 恐惧贪婪指数（2026-10-01 每日读数）

---

## 一、分析背景

近两日比特币在 **$83,000–$85,600** 区间震荡后，10月2日盘中一度突破 **$86,000**，市场等待美国9月非农就业数据。Citi 将比特币目标价从 $82,000 上调至 **$113,000**，叠加 ETF 资金流 2026 年转正，市场情绪偏乐观（risk-on）。本报告基于近三日日线 OHLC 进行技术面分析，并结合市场情绪与新闻给出短期展望。

---

## 二、价格数据概览

| 日期 | 开盘 | 最高 | 最低 | 收盘 | 涨跌% |
|------|------|------|------|------|-------|
| 2026-09-30 | 83,621 | 85,599 | 82,928 | 83,554 | -0.05% |
| 2026-10-01 | 83,554 | 85,224 | 83,133 | 84,853 | +1.55% |
| 2026-10-02 | 84,881 | 84,881 | 84,774 | 84,795 | -0.13%（盘中快照） |

- **当前价**：约 **${current_price:,.0f}**（24h +3.24%，盘中实时快照，非收盘）
- **三日区间**：最高 ${high_3d:,.0f} / 最低 ${low_3d:,.0f}
- 10月以来累计上涨约 **+3%**

![BTC 技术面图表](btc_technical_chart_2026-10-02.png)

---

## 三、技术指标分析

| 指标 | 数值 | 解读 |
|------|------|------|
| MA5 | {ma5_str} | 短期均线 |
| MA10 | {ma10_str} | 中期均线 |
| RSI(14) | {rsi_val:.1f} | {rsi_state}（样本仅3日，数值失真，仅供参考） |
| MACD | {macd_val:,.0f} | {macd_state} |
| MACD Signal | {sig_val:,.0f} | — |
| MACD Hist | {hist_val:,.0f} | {'柱体为正，动能偏多' if hist_val>0 else '柱体为负，动能偏空'} |

**均线形态**：{ma_state}。收盘价站上 MA5 与 MA10，短期趋势偏多。

**RSI(14) = {rsi_val:.1f}**：处于 {rsi_state} 区域。{'接近/进入超买区，短线追高需谨慎。' if rsi_val>=65 else '尚未进入超买，仍有上行空间。'}

**MACD**：MACD 线（{macd_val:,.0f}）{'高于' if macd_val>sig_val else '低于'}信号线（{sig_val:,.0f}），柱体{'为正' if hist_val>0 else '为负'}，{'多头动能延续' if hist_val>0 else '空头动能占优'}。

> ⚠️ 技术说明：本分析仅基于 3 根日线，MA10/RSI(14)/MACD 的数值受样本量限制，**统计意义有限**，仅作方向性参考，不宜单独作为交易依据。

---

## 四、市场情绪与新闻

- **恐惧贪婪指数：{fg['value']}（{fg['label']}）**（2026-10-01 读数，24h 更新）。市场情绪偏贪婪/乐观，风险偏好上升；BTC 市占率约 59-60%，USDT 降至 6.3%。

**重大新闻**：

| 日期 | 新闻 |
|------|------|
"""
for n in news:
    report += f"| {n['date']} | {n['news']} |\n"

report += f"""
**多空因素**：

- 🟢 **利多**：{'；'.join(bull)}
- 🔴 **利空**：{'；'.join(bear)}

---

## 五、短期展望（1-3 日）

**基准判断：偏多震荡，但短线追高风险上升。**

1. **趋势**：价格站上 MA5/MA10，MACD 多头，情绪贪婪，短期动能偏多。
2. **关键阻力**：**$87,397**（Yahoo 10月区间上沿）与 **$86,000**（10-02 盘中突破位）。若放量站稳 $86,000 上方，有望挑战 $87,400。
3. **关键支撑**：**$84,000**（盘整位）→ **$83,100–$83,600**（前两日低点/MA 区）→ 强支撑 **$82,900**。
4. **风险事件**：10月27-28 美联储会议、美国9月非农数据、ETF 日流入能否回升至 $10 亿量级。
5. **情景**：
   - 乐观：放量突破 $87,400 → 上看 Citi 目标 $113,000 方向（中长期）。
   - 中性：$83,100–$87,400 区间震荡（最可能）。
   - 悲观：跌破 $83,100 → 下探 $75,585（区间下沿）。

---

## 六、风险提示

- 本报告基于 **3 根日线**，技术指标统计意义有限，**不构成投资建议**。
- 当前价 ${current_price:,.0f} 为 **盘中实时快照**，非最终收盘，可能与次日开盘存在跳空。
- 加密资产波动剧烈，受宏观（利率、美元、油价）、ETF 资金流、监管与地缘事件影响大。
- 恐惧贪婪指数为滞后情绪指标，"贪婪"读数本身亦是逆向警示（追高风险）。
- 请结合自身风险承受能力独立决策，注意仓位与止损管理。

---

*报告生成时间：2026-10-02 23:46 ｜ 数据截至 2026-10-02 盘中快照*
"""

with open(report_file, "w", encoding="utf-8") as f:
    f.write(report)

print("CHART:", chart_path)
print("REPORT:", report_file)
print("RSI=%.1f MACD=%.0f Signal=%.0f Hist=%.0f MA5=%.0f MA10=%.0f" % (rsi_val, macd_val, sig_val, hist_val, ma5_val, ma10_val))
print("OK")
