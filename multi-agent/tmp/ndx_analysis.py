# -*- coding: utf-8 -*-
"""
Nasdaq-100 (^NDX) 技术分析 + 国庆节后(2026-10-08起)涨跌概率估算
输入: F:\agent\multi-agent\tmp\ndx_recent_data.json (上游 step1)
      F:\agent\multi-agent\tmp\ndx_market_sentiment_news.json (上游 step2)
输出: F:\agent\multi-agent\tmp\ndx_analysis_result.json
      F:\agent\multi-agent\tmp\ndx_analysis_charts.png
AS_OF: 2026-10-01 16:00 (US Eastern close, 最近确认交易日)
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager

# ---------- 中文字体 (强制重建缓存) ----------
font_manager._load_fontmanager(try_read_cache=False)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DengXian", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 读取数据 ----------
with open(r"F:\agent\multi-agent\tmp\ndx_recent_data.json", encoding="utf-8") as fp:
    raw = json.load(fp)

rows = raw["recent_trading_days"]
# 按日期升序排列
rows = sorted(rows, key=lambda r: r["date"])
df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date").sort_index()

close = df["close"].astype(float)
high = df["high"].astype(float)
low = df["low"].astype(float)
open_ = df["open"].astype(float)

# ---------- 技术指标 ----------
def sma(s, n): return s.rolling(n).mean()
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

# RSI (Wilder)
def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    ru = up.ewm(alpha=1/n, adjust=False).mean()
    rd = dn.ewm(alpha=1/n, adjust=False).mean()
    rs = ru / rd
    return 100 - 100/(1+rs)

# MACD
def macd(s, fast=12, slow=26, signal=9):
    line = ema(s, fast) - ema(s, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist

df["SMA5"] = sma(close, 5)
df["SMA10"] = sma(close, 10)
df["RSI14"] = rsi(close, 14)
df["MACD"], df["MACD_sig"], df["MACD_hist"] = macd(close)

# 布林带 (20,2)
df["BB_mid"] = sma(close, 20)
bb_std = close.rolling(20).std()
df["BB_up"] = df["BB_mid"] + 2*bb_std
df["BB_lo"] = df["BB_mid"] - 2*bb_std

last = df.iloc[-1]
prev = df.iloc[-2]

# ---------- 概率估算 (多因子加权) ----------
# 1) 趋势因子: 收盘价 vs SMA5/SMA10
trend_score = 0.5
if last["close"] > last["SMA5"]: trend_score += 0.25
if last["close"] > last["SMA10"]: trend_score += 0.25
# 近5日动量
mom5 = (close.iloc[-1]/close.iloc[-6]-1)
trend_score += 0.1 if mom5 > 0 else -0.1

# 2) RSI 因子: 超买(>70)看跌, 超卖(<30)看涨, 中性50-70偏多
rsi_v = last["RSI14"]
if rsi_v >= 70: rsi_score = 0.35
elif rsi_v >= 60: rsi_score = 0.55
elif rsi_v >= 45: rsi_score = 0.5
elif rsi_v >= 30: rsi_score = 0.45
else: rsi_score = 0.65

# 3) MACD 因子: 柱状图>0 且 金叉 偏多
macd_score = 0.5
if last["MACD"] > last["MACD_sig"]: macd_score += 0.2
if last["MACD_hist"] > 0: macd_score += 0.1
if last["MACD_hist"] > prev["MACD_hist"]: macd_score += 0.1  # 动能增强
macd_score = min(max(macd_score, 0.1), 0.9)

# 4) 市场情绪因子 (来自 step2: mixed-to-cautiously-bullish)
# 4个利多 vs 4个利空, 整体偏多但受限 -> 0.58
sentiment_score = 0.58

# 加权: 趋势0.35, RSI0.2, MACD0.2, 情绪0.25
tech_prob = 0.35*trend_score + 0.20*rsi_score + 0.20*macd_score + 0.25*sentiment_score
tech_prob = min(max(tech_prob, 0.05), 0.95)

# 5) 历史波动率 -> 节后5日收益分布 (对数正态近似)
ret = np.log(close/close.shift(1)).dropna()
mu = ret.mean()
sigma = ret.std()
# 节后5个交易日累计收益的均值/标准差
horizon = 5
mu5 = mu * horizon
sigma5 = sigma * np.sqrt(horizon)
# 上涨概率 = P(累计收益>0) = 1 - Phi(-mu5/sigma5)
from math import erf, sqrt
def norm_cdf(x): return 0.5*(1+erf(x/sqrt(2)))
stat_prob = norm_cdf(mu5/sigma5)

# 综合概率 = 技术/情绪 与 统计 的均值
final_up_prob = 0.5*tech_prob + 0.5*stat_prob
final_down_prob = 1 - final_up_prob

# 节后5日预测区间 (基于对数正态)
base = close.iloc[-1]
up_95 = base*np.exp(mu5 + 1.96*sigma5)
dn_95 = base*np.exp(mu5 - 1.96*sigma5)
up_1sd = base*np.exp(mu5 + sigma5)
dn_1sd = base*np.exp(mu5 - sigma5)

# ---------- 打印关键指标 ----------
print("="*60)
print(f"AS_OF: 2026-10-01 16:00 (US Eastern close)")
print(f"最新收盘: {last['close']:.2f}  日涨跌: {last['change_pct']:+.2f}%")
print(f"SMA5: {last['SMA5']:.2f}  SMA10: {last['SMA10']:.2f}")
print(f"RSI14: {rsi_v:.1f}")
print(f"MACD: {last['MACD']:.2f}  Signal: {last['MACD_sig']:.2f}  Hist: {last['MACD_hist']:.2f}")
print(f"近5日动量: {mom5*100:+.2f}%")
print(f"日波动率(年化): {sigma*np.sqrt(252)*100:.1f}%")
print("-"*60)
print(f"技术/情绪上涨概率: {tech_prob*100:.1f}%")
print(f"统计(波动率)上涨概率: {stat_prob*100:.1f}%")
print(f"综合上涨概率: {final_up_prob*100:.1f}%  下跌概率: {final_down_prob*100:.1f}%")
print(f"节后5日 95%区间: [{dn_95:.0f}, {up_95:.0f}]")
print(f"节后5日 1σ区间: [{dn_1sd:.0f}, {up_1sd:.0f}]")
print("="*60)

# ---------- 可视化 ----------
fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True,
                         gridspec_kw={"height_ratios":[3,1,1,1]})
fig.suptitle("Nasdaq-100 (^NDX) 技术分析 — 国庆节后(2026-10-08起)涨跌概率估算",
             fontsize=14, fontweight="bold")
fig.text(0.5, 0.955, "AS_OF: 2026-10-01 16:00 (US Eastern close, 最近确认交易日)",
         ha="center", fontsize=9, color="gray")

x = df.index

# 1) 价格 + 均线 + 布林带
ax = axes[0]
ax.plot(x, close, color="#1f77b4", lw=2, marker="o", ms=4, label="收盘价")
ax.plot(x, df["SMA5"], color="#ff7f0e", lw=1.5, label="SMA5")
ax.plot(x, df["SMA10"], color="#2ca02c", lw=1.5, label="SMA10")
ax.fill_between(x, df["BB_lo"], df["BB_up"], color="#1f77b4", alpha=0.12, label="布林带(20,2)")
# 节后预测区间 (从10-01往后5个交易日, 用工作日近似)
future_dates = pd.bdate_range(start="2026-10-05", periods=6)
ax.plot(future_dates, [base]*len(future_dates), color="gray", ls="--", lw=1, alpha=0.6)
ax.fill_between(future_dates,
                [dn_95]*len(future_dates), [up_95]*len(future_dates),
                color="#d62728", alpha=0.15, label="节后5日95%预测区间")
ax.fill_between(future_dates,
                [dn_1sd]*len(future_dates), [up_1sd]*len(future_dates),
                color="#ff7f0e", alpha=0.25, label="节后5日1σ预测区间")
ax.set_ylabel("指数点位")
ax.legend(loc="upper left", fontsize=8, ncol=3)
ax.grid(alpha=0.3)
ax.set_title("价格 / 均线 / 布林带 / 节后预测区间", fontsize=10)

# 2) RSI
ax = axes[1]
ax.plot(x, df["RSI14"], color="#9467bd", lw=1.8, marker="o", ms=3)
ax.axhline(70, color="red", ls="--", lw=1, alpha=0.6)
ax.axhline(30, color="green", ls="--", lw=1, alpha=0.6)
ax.axhline(50, color="gray", ls=":", lw=1, alpha=0.5)
ax.fill_between(x, 30, 70, color="gray", alpha=0.05)
ax.set_ylim(0, 100)
ax.set_ylabel("RSI(14)")
ax.grid(alpha=0.3)
ax.set_title(f"RSI(14) = {rsi_v:.1f}", fontsize=10)

# 3) MACD
ax = axes[2]
colors = ["#d62728" if v>0 else "#2ca02c" for v in df["MACD_hist"]]
ax.bar(x, df["MACD_hist"], color=colors, alpha=0.6, width=0.6)
ax.plot(x, df["MACD"], color="#1f77b4", lw=1.5, label="MACD")
ax.plot(x, df["MACD_sig"], color="#ff7f0e", lw=1.5, label="Signal")
ax.axhline(0, color="gray", lw=0.8)
ax.set_ylabel("MACD")
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.3)
ax.set_title("MACD(12,26,9)", fontsize=10)

# 4) 涨跌概率
ax = axes[3]
ax.axis("off")
prob_text = (
    f"国庆节后(2026-10-08起) 5个交易日涨跌概率估算\n\n"
    f"  技术/情绪上涨概率: {tech_prob*100:.1f}%\n"
    f"  统计(波动率)上涨概率: {stat_prob*100:.1f}%\n"
    f"  ─────────────────────────────\n"
    f"  综合上涨概率: {final_up_prob*100:.1f}%   下跌概率: {final_down_prob*100:.1f}%\n\n"
    f"  节后5日 95%预测区间: {dn_95:.0f} ~ {up_95:.0f}\n"
    f"  节后5日 1σ预测区间: {dn_1sd:.0f} ~ {up_1sd:.0f}\n"
    f"  基准点位(10-01收盘): {base:.2f}"
)
ax.text(0.02, 0.5, prob_text, transform=ax.transAxes, fontsize=11,
        verticalalignment="center", family="monospace",
        bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))
ax.set_title("涨跌概率估算结果", fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(r"F:\agent\multi-agent\tmp\ndx_analysis_charts.png", dpi=130, bbox_inches="tight")
print("图表已保存: F:\\agent\\multi-agent\\tmp\\ndx_analysis_charts.png")

# ---------- 保存结果 JSON ----------
result = {
    "as_of": "2026-10-01 16:00 (US Eastern close, 最近确认交易日)",
    "latest": {"date": "2026-10-01", "close": float(last["close"]), "change_pct": float(last["change_pct"])},
    "indicators": {
        "SMA5": round(float(last["SMA5"]),2),
        "SMA10": round(float(last["SMA10"]),2),
        "RSI14": round(float(rsi_v),1),
        "MACD": round(float(last["MACD"]),2),
        "MACD_signal": round(float(last["MACD_sig"]),2),
        "MACD_hist": round(float(last["MACD_hist"]),2),
        "mom_5d_pct": round(float(mom5*100),2),
        "daily_vol_annualized_pct": round(float(sigma*np.sqrt(252)*100),1),
    },
    "probability": {
        "tech_sentiment_up": round(float(tech_prob),3),
        "stat_vol_up": round(float(stat_prob),3),
        "final_up": round(float(final_up_prob),3),
        "final_down": round(float(final_down_prob),3),
        "horizon_trading_days": horizon,
        "post_holiday_95_range": [round(float(dn_95),0), round(float(up_95),0)],
        "post_holiday_1sd_range": [round(float(dn_1sd),0), round(float(up_1sd),0)],
        "base_close": float(base),
    },
    "factor_scores": {
        "trend": round(float(trend_score),3),
        "rsi": round(float(rsi_score),3),
        "macd": round(float(macd_score),3),
        "sentiment": round(float(sentiment_score),3),
    },
    "chart": "F:\\agent\\multi-agent\\tmp\\ndx_analysis_charts.png",
}
with open(r"F:\agent\multi-agent\tmp\ndx_analysis_result.json", "w", encoding="utf-8") as fp:
    json.dump(result, fp, ensure_ascii=False, indent=2)
print("结果已保存: F:\\agent\\multi-agent\\tmp\\ndx_analysis_result.json")
