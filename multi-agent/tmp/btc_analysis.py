# -*- coding: utf-8 -*-
"""
BTC/USD 近期价格数据分析与可视化
Step 3/5: 计算涨跌幅、趋势指标（7日均线、RSI等），生成价格走势图和关键指标图表
数据源: F:\agent\multi-agent\tmp\btc_price_2026-09-28_to_2026-10-02.json
AS_OF: 2026-10-02 22:22 (Oct 02 为盘中快照)
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

# ---------- 读取数据 ----------
with open(r"F:\agent\multi-agent\tmp\btc_price_2026-09-28_to_2026-10-02.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data["daily"])
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# ---------- 指标计算 ----------
# 涨跌幅（基于收盘价）
df["ret_pct"] = df["close"].pct_change() * 100

# 7日均线（数据仅5天，MA7 用可用窗口计算，标注为"可用窗口均线"）
df["ma7"] = df["close"].rolling(window=7, min_periods=1).mean()
# 5日均线
df["ma5"] = df["close"].rolling(window=5, min_periods=1).mean()

# RSI (Wilder)
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=1, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=1, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# 数据仅5天，RSI14 无法稳定计算，改用 RSI5 作为短期参考
df["rsi5"] = rsi(df["close"], 5)

# 布林带 (20日, 2std) - 数据不足，用可用窗口
df["bb_mid"] = df["close"].rolling(window=20, min_periods=1).mean()
bb_std = df["close"].rolling(window=20, min_periods=1).std()
df["bb_up"] = df["bb_mid"] + 2 * bb_std
df["bb_low"] = df["bb_mid"] - 2 * bb_std

# 区间统计
period_high = df["high"].max()
period_low = df["low"].min()
start_close = df["close"].iloc[0]
end_close = df["close"].iloc[-1]
total_change = (end_close / start_close - 1) * 100

print("=== 关键指标 ===")
print(f"区间最高: {period_high:,.0f}  区间最低: {period_low:,.0f}")
print(f"起始收盘(9/28): {start_close:,.0f}  最新(10/02盘中): {end_close:,.0f}")
print(f"区间累计涨跌: {total_change:+.2f}%")
print(f"最新 RSI5: {df['rsi5'].iloc[-1]:.1f}")
print(f"最新 MA5: {df['ma5'].iloc[-1]:,.0f}")
print()
print(df[["date","close","ret_pct","ma5","rsi5"]].to_string(index=False))

# ---------- 可视化 ----------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1.2, 1.2]})
fig.suptitle("BTC/USD 近期价格走势与关键指标  (2026-09-28 ~ 2026-10-02)",
             fontsize=15, fontweight="bold")
fig.text(0.5, 0.965, "AS_OF: 2026-10-02 22:22  |  10/02 为盘中快照，非收盘",
         ha="center", fontsize=9, color="gray")

# --- 子图1: K线 + 均线 ---
ax1 = axes[0]
width = 0.6
for i, row in df.iterrows():
    color = "#e74c3c" if row["close"] >= row["open"] else "#27ae60"
    # 影线
    ax1.plot([i, i], [row["low"], row["high"]], color=color, linewidth=1.2, zorder=2)
    # 实体
    bottom = min(row["open"], row["close"])
    height = abs(row["close"] - row["open"])
    ax1.bar(i, height, bottom=bottom, width=width, color=color,
            edgecolor=color, zorder=3)
    ax1.text(i, row["high"] + 300, f"{row['high']:,.0f}", ha="center", fontsize=7, color="#555")
    ax1.text(i, row["low"] - 600, f"{row['low']:,.0f}", ha="center", fontsize=7, color="#555")

ax1.plot(df.index, df["ma5"], color="#f39c12", linewidth=2, label="MA5 (可用窗口)", zorder=4)
ax1.plot(df.index, df["ma7"], color="#3498db", linewidth=2, linestyle="--", label="MA7 (可用窗口)", zorder=4)

# 标注最新价
ax1.axhline(end_close, color="#8e44ad", linewidth=1, linestyle=":", alpha=0.7)
ax1.text(len(df)-0.4, end_close, f" 最新 {end_close:,.0f}", color="#8e44ad", fontsize=9, va="bottom")

ax1.set_ylabel("价格 (USD)")
ax1.set_title("K线 + 均线", fontsize=11)
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.5, len(df)-0.5)

# --- 子图2: 涨跌幅柱状图 ---
ax2 = axes[1]
colors = ["#e74c3c" if v >= 0 else "#27ae60" for v in df["ret_pct"].fillna(0)]
bars = ax2.bar(df.index, df["ret_pct"].fillna(0), color=colors, width=width, alpha=0.8)
for i, v in enumerate(df["ret_pct"].fillna(0)):
    ax2.text(i, v + (0.05 if v >= 0 else -0.15), f"{v:+.2f}%", ha="center", fontsize=8)
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_ylabel("涨跌幅 (%)")
ax2.set_title("每日涨跌幅", fontsize=11)
ax2.grid(True, alpha=0.3, axis="y")

# --- 子图3: RSI5 ---
ax3 = axes[2]
ax3.plot(df.index, df["rsi5"], color="#9b59b6", linewidth=2, marker="o", markersize=5)
ax3.axhline(70, color="#e74c3c", linewidth=1, linestyle="--", alpha=0.6, label="超买 70")
ax3.axhline(30, color="#27ae60", linewidth=1, linestyle="--", alpha=0.6, label="超卖 30")
ax3.fill_between(df.index, 30, 70, alpha=0.08, color="gray")
for i, v in enumerate(df["rsi5"]):
    if not np.isnan(v):
        ax3.text(i, v + 2, f"{v:.0f}", ha="center", fontsize=8, color="#9b59b6")
ax3.set_ylabel("RSI (5)")
ax3.set_title("RSI (5日, 短期参考)", fontsize=11)
ax3.set_ylim(0, 100)
ax3.legend(loc="upper left", fontsize=8)
ax3.grid(True, alpha=0.3)

# X轴日期
tick_labels = [d.strftime("%m-%d") for d in df["date"]]
plt.xticks(df.index, tick_labels)
plt.xlabel("日期 (2026)")

plt.tight_layout(rect=[0, 0, 1, 0.95])
chart_path = r"F:\agent\multi-agent\tmp\btc_price_chart_2026-09-28_to_2026-10-02.png"
plt.savefig(chart_path, dpi=130, bbox_inches="tight")
plt.close()
print(f"\n图表已保存: {chart_path}")

# ---------- 保存指标数据 ----------
out = {
    "as_of": "2026-10-02 22:22",
    "period_high": float(period_high),
    "period_low": float(period_low),
    "start_close_0928": float(start_close),
    "end_close_1002_intraday": float(end_close),
    "total_change_pct": round(float(total_change), 2),
    "latest_rsi5": round(float(df["rsi5"].iloc[-1]), 1),
    "latest_ma5": round(float(df["ma5"].iloc[-1]), 0),
    "daily": [
        {
            "date": r["date"].strftime("%Y-%m-%d"),
            "close": float(r["close"]),
            "ret_pct": None if pd.isna(r["ret_pct"]) else round(float(r["ret_pct"]), 2),
            "ma5": round(float(r["ma5"]), 0),
            "rsi5": None if pd.isna(r["rsi5"]) else round(float(r["rsi5"]), 1),
        }
        for _, r in df.iterrows()
    ],
    "chart_path": chart_path,
}
with open(r"F:\agent\multi-agent\tmp\btc_indicators_2026-09-28_to_2026-10-02.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("指标已保存: F:\\agent\\multi-agent\\tmp\\btc_indicators_2026-09-28_to_2026-10-02.json")
