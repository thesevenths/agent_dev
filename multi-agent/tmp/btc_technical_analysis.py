# -*- coding: utf-8 -*-
"""
Step 3/5: 比特币技术面分析与可视化
基于上游抓取的 BTC/USD 日线 OHLC 数据，计算短期趋势指标
(MA5/MA10、RSI、MACD、布林带、支撑/阻力)，并生成价格走势图与关键指标图表。

AS_OF: 2026-10-02 23:45 (local)
数据说明: 2026-10-02 为盘中快照(非最终收盘)。
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import matplotlib.font_manager as fm

# ---------- 中文字体 ----------
def setup_chinese_font():
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                  "Source Han Sans SC", "WenQuanYi Micro Hei", "Arial Unicode MS"]
    available = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            plt.rcParams["font.sans-serif"] = [c]
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt.rcParams["font.sans-serif"][0]

font_used = setup_chinese_font()
print("使用字体:", font_used)

OUT = r"F:\agent\multi-agent\tmp"

# ---------- 读取上游数据 ----------
with open(r"F:\agent\multi-agent\tmp\btc_price_data_2026-09-30_to_2026-10-02.json", encoding="utf-8") as f:
    price = json.load(f)

rows = price["daily_ohlc"]
df = pd.DataFrame(rows)[["date", "open", "high", "low", "close", "change_pct"]]
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# 当前盘中价(用于标注)
cur_price = price["current_price"]["price_usd"]

# ---------- 技术指标 ----------
def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(close, period=20, num_std=2):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower

# 样本仅3个交易日：MA5/MA10/RSI14/布林带(20) 样本不足，无法有效计算。
# 改用可计算的短周期指标：MA3、RSI(3)。MACD 用 EMA 递推可产生近似值(标注为近似)。
df["MA3"] = df["close"].rolling(3).mean()
df["RSI3"] = rsi(df["close"], 3)
df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd(df["close"])
# 布林带(3,2) 作为短周期波动带替代
df["BB_upper"], df["BB_mid"], df["BB_lower"] = bollinger(df["close"], period=3, num_std=2)

# 支撑/阻力(基于近3日)
support = df["low"].min()
resistance = df["high"].max()
# 关键心理位
key_levels = {
    "支撑(近3日最低)": round(support, 0),
    "阻力(近3日最高)": round(resistance, 0),
    "潜在区间上沿(新闻)": 87397,
    "Citi 目标价": 113000,
}

# 最新指标值
last = df.iloc[-1]
indicators = {
    "MA5": round(last["MA5"], 2) if not np.isnan(last["MA5"]) else None,
    "MA10": round(last["MA10"], 2) if not np.isnan(last["MA10"]) else None,
    "RSI14": round(last["RSI14"], 2) if not np.isnan(last["RSI14"]) else None,
    "MACD": round(last["MACD"], 2) if not np.isnan(last["MACD"]) else None,
    "MACD_signal": round(last["MACD_signal"], 2) if not np.isnan(last["MACD_signal"]) else None,
    "MACD_hist": round(last["MACD_hist"], 2) if not np.isnan(last["MACD_hist"]) else None,
    "BB_upper": round(last["BB_upper"], 2) if not np.isnan(last["BB_upper"]) else None,
    "BB_mid": round(last["BB_mid"], 2) if not np.isnan(last["BB_mid"]) else None,
    "BB_lower": round(last["BB_lower"], 2) if not np.isnan(last["BB_lower"]) else None,
}

# ---------- 图1: 价格走势图 (K线 + MA + 布林带 + 当前价) ----------
fig, ax = plt.subplots(figsize=(11, 6.5))
width = 0.55
for i, r in df.iterrows():
    up = r["close"] >= r["open"]
    color = "#e74c3c" if up else "#27ae60"  # 国际惯例: 红涨绿跌
    ax.vlines(r["date"], r["low"], r["high"], color=color, lw=1.2)
    ax.bar(r["date"], r["open"] - r["close"], width=width,
           bottom=min(r["open"], r["close"]), color=color, edgecolor=color)

# MA 线
ax.plot(df["date"], df["MA5"], label="MA5", color="#f39c12", lw=2, marker="o", ms=4)
ax.plot(df["date"], df["MA10"], label="MA10", color="#2980b9", lw=2, marker="s", ms=4)
# 布林带
ax.plot(df["date"], df["BB_upper"], label="布林上轨", color="#9b59b6", lw=1, ls="--", alpha=0.7)
ax.plot(df["date"], df["BB_lower"], label="布林下轨", color="#9b59b6", lw=1, ls="--", alpha=0.7)
ax.fill_between(df["date"], df["BB_lower"], df["BB_upper"], color="#9b59b6", alpha=0.08)

# 当前盘中价
ax.axhline(cur_price, color="#16a085", lw=1.5, ls=":", label=f"当前盘中价 ${cur_price:,.0f}")
# 关键位
ax.axhline(resistance, color="#e74c3c", lw=1, ls=":", alpha=0.6)
ax.axhline(support, color="#27ae60", lw=1, ls=":", alpha=0.6)
ax.text(df["date"].iloc[-1], resistance, f" 阻力 ${resistance:,.0f}", color="#e74c3c", fontsize=9, va="bottom")
ax.text(df["date"].iloc[-1], support, f" 支撑 ${support:,.0f}", color="#27ae60", fontsize=9, va="top")

ax.set_title("BTC/USD 价格走势图 (2026-09-30 ~ 2026-10-02)\n含 MA5/MA10、布林带、支撑/阻力  |  AS_OF: 2026-10-02 23:45", fontsize=12)
ax.set_ylabel("价格 (USD)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
fig.savefig(f"{OUT}\\btc_price_chart.png", dpi=130)
plt.close(fig)

# ---------- 图2: RSI ----------
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.plot(df["date"], df["RSI14"], color="#8e44ad", lw=2, marker="o", ms=5, label="RSI(14)")
ax.axhline(70, color="#e74c3c", lw=1, ls="--", alpha=0.7)
ax.axhline(30, color="#27ae60", lw=1, ls="--", alpha=0.7)
ax.fill_between(df["date"], 70, 100, color="#e74c3c", alpha=0.06)
ax.fill_between(df["date"], 0, 30, color="#27ae60", alpha=0.06)
ax.set_ylim(0, 100)
ax.set_title(f"RSI(14)  最新: {indicators['RSI14']}  |  AS_OF: 2026-10-02 23:45", fontsize=11)
ax.set_ylabel("RSI")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
fig.savefig(f"{OUT}\\btc_rsi_chart.png", dpi=130)
plt.close(fig)

# ---------- 图3: MACD ----------
fig, ax = plt.subplots(figsize=(11, 3.6))
colors = ["#e74c3c" if h >= 0 else "#27ae60" for h in df["MACD_hist"].fillna(0)]
ax.bar(df["date"], df["MACD_hist"].fillna(0), width=0.5, color=colors, alpha=0.6, label="MACD 柱")
ax.plot(df["date"], df["MACD"].fillna(0), color="#2980b9", lw=2, label="MACD 线")
ax.plot(df["date"], df["MACD_signal"].fillna(0), color="#f39c12", lw=2, label="Signal 线")
ax.axhline(0, color="gray", lw=0.8)
ax.set_title(f"MACD(12,26,9)  最新: MACD={indicators['MACD']}  Signal={indicators['MACD_signal']}  |  AS_OF: 2026-10-02 23:45", fontsize=11)
ax.set_ylabel("MACD")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
fig.savefig(f"{OUT}\\btc_macd_chart.png", dpi=130)
plt.close(fig)

# ---------- 输出指标 JSON ----------
result = {
    "asset": "BTC/USD",
    "step": "技术面分析与可视化",
    "as_of": "2026-10-02 23:45 (local)",
    "data_note": "2026-10-02 为盘中快照(非最终收盘)；样本仅3个交易日，MA10/RSI14/MACD/布林带(20)因样本不足为近似/部分NaN，解读需谨慎。",
    "price_series": df[["date", "open", "high", "low", "close", "change_pct"]].assign(
        date=lambda d: d["date"].dt.strftime("%Y-%m-%d")).to_dict("records"),
    "current_price": cur_price,
    "indicators_latest": indicators,
    "key_levels": key_levels,
    "charts": {
        "price": f"{OUT}\\btc_price_chart.png",
        "rsi": f"{OUT}\\btc_rsi_chart.png",
        "macd": f"{OUT}\\btc_macd_chart.png",
    },
}
with open(f"{OUT}\\btc_technical_indicators.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("\n===== 最新技术指标 =====")
for k, v in indicators.items():
    print(f"{k}: {v}")
print("\n===== 关键位 =====")
for k, v in key_levels.items():
    print(f"{k}: {v}")
print("\n图表已保存:")
for k, v in result["charts"].items():
    print(" ", v)
