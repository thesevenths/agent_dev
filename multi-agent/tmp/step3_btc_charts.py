# -*- coding: utf-8 -*-
"""
Step 3 — 数据整理与可视化
将 BTC 价格/涨跌 与 美联储利率 关联数据整理为结构化表格，并生成可视化图表：
  1) BTC 价格走势（近 6 个交易日，含 10-07 高亮）
  2) 利率-BTC 关系图（联邦基金利率目标区间 与 BTC 价格 双轴对照）
  3) 10-07 当日 OHLC 涨跌结构图
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import pandas as pd
from datetime import datetime

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"E:\agent_dev\multi-agent\tmp"

# ----------------------------------------------------------------------
# 数据（来源：Step1 / Step2 上游文件，AS_OF 见各文件）
# ----------------------------------------------------------------------
# BTC 近 6 个交易日收盘价（Yahoo Finance BTC-USD）
btc = pd.DataFrame({
    "date": ["2026-09-30", "2026-10-01", "2026-10-02", "2026-10-04", "2026-10-05", "2026-10-07"],
    "close": [83553.85, 84853.10, 84497.21, 86480.30, 85786.59, 84199.14],
})
btc["date"] = pd.to_datetime(btc["date"])
btc["chg_pct"] = btc["close"].pct_change() * 100

# 10-07 当日 OHLC
ohlc = {"open": 85546.23, "high": 85570.70, "low": 83802.94, "close": 84199.14}
prev_close = 85786.59

# 美联储联邦基金利率目标区间（中值）与 BTC 价格 对照（事件驱动）
fed = pd.DataFrame({
    "date": ["2026-07-29", "2026-09-16", "2026-10-07"],
    "ff_mid": [3.625, 3.875, 3.875],   # 目标区间中值（%）
    "event": ["维持 3.50–3.75%", "加息25bp→3.75–4.00%", "（预期10月再加息）"],
    "btc": [84497.21, 77293.66, 84199.14],
})
fed["date"] = pd.to_datetime(fed["date"])

# ----------------------------------------------------------------------
# 图 1：BTC 价格走势（近 6 个交易日，高亮 10-07）
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.5))
colors = ["#2e7d32" if c >= 0 else "#c62828" for c in btc["chg_pct"].fillna(0)]
ax.plot(btc["date"], btc["close"], marker="o", color="#1565c0", lw=2.2, zorder=3)
for i, row in btc.iterrows():
    ax.annotate(f"${row['close']:,.0f}", (row["date"], row["close"]),
                textcoords="offset points", xytext=(0, 12), ha="center", fontsize=8.5, color="#333")
# 高亮 10-07
last = btc.iloc[-1]
ax.scatter([last["date"]], [last["close"]], s=160, color="#c62828", zorder=4, edgecolor="white", lw=1.5)
ax.annotate("10-07 收盘\n$84,199 (-1.85%)", (last["date"], last["close"]),
            textcoords="offset points", xytext=(-95, -45), ha="center", fontsize=9,
            color="#c62828", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#c62828"))
ax.set_title("比特币(BTC)价格走势 — 近6个交易日（2026-09-30 ~ 2026-10-07）", fontsize=13, fontweight="bold")
ax.set_ylabel("BTC 价格 (USD)")
ax.set_xlabel("日期")
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax.set_ylim(82000, 88000)
fig.text(0.99, 0.01, "AS_OF: 2026-10-08 09:56（本地）；数据源 Yahoo Finance BTC-USD", ha="right", fontsize=7, color="#888")
plt.tight_layout()
plt.savefig(f"{OUT}\\chart_btc_price_trend.png", dpi=130)
plt.close()

# ----------------------------------------------------------------------
# 图 2：利率-BTC 关系图（双轴：联邦基金利率中值 vs BTC 价格）
# ----------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(10, 5.5))
ax2 = ax1.twinx()
ax1.plot(fed["date"], fed["ff_mid"], marker="s", color="#e65100", lw=2.5, label="联邦基金利率中值(%)", zorder=4)
ax2.plot(fed["date"], fed["btc"], marker="o", color="#1565c0", lw=2.2, ls="--", label="BTC 价格(USD)", zorder=3)
for i, row in fed.iterrows():
    ax1.annotate(f"{row['ff_mid']:.3f}%\n{row['event']}", (row["date"], row["ff_mid"]),
                 textcoords="offset points", xytext=(0, 14), ha="center", fontsize=8, color="#e65100")
    ax2.annotate(f"${row['btc']:,.0f}", (row["date"], row["btc"]),
                 textcoords="offset points", xytext=(0, -18), ha="center", fontsize=8, color="#1565c0")
ax1.set_title("美联储利率 与 BTC 价格 对照（事件驱动）", fontsize=13, fontweight="bold")
ax1.set_ylabel("联邦基金利率中值 (%)", color="#e65100")
ax2.set_ylabel("BTC 价格 (USD)", color="#1565c0")
ax1.set_xlabel("日期")
ax1.tick_params(axis="y", labelcolor="#e65100")
ax2.tick_params(axis="y", labelcolor="#1565c0")
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax1.set_ylim(3.4, 4.3)
ax2.set_ylim(70000, 90000)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
fig.text(0.99, 0.01, "AS_OF: 2026-10-08 09:56（本地）；利率源 FOMC 决议，BTC 源 Yahoo Finance", ha="right", fontsize=7, color="#888")
plt.tight_layout()
plt.savefig(f"{OUT}\\chart_fed_rate_vs_btc.png", dpi=130)
plt.close()

# ----------------------------------------------------------------------
# 图 3：10-07 当日 OHLC 涨跌结构（K线式）
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5.5))
x = 0.5
w = 0.4
# 影线
ax.plot([x, x], [ohlc["low"], ohlc["high"]], color="#333", lw=1.5, zorder=3)
# 实体（开→收，下跌为红）
body_lo = min(ohlc["open"], ohlc["close"])
body_h = abs(ohlc["open"] - ohlc["close"])
ax.add_patch(Rectangle((x - w/2, body_lo), w, body_h, facecolor="#c62828", edgecolor="#7f0000", zorder=4))
# 前收参考线
ax.axhline(prev_close, color="#1565c0", ls="--", lw=1.2, zorder=2)
ax.text(0.62, prev_close, f"前收 $85,786.59", color="#1565c0", fontsize=8.5, va="bottom")
# 标注
ax.text(x + 0.06, ohlc["high"], f"高 ${ohlc['high']:,.0f}", fontsize=8.5, va="center")
ax.text(x + 0.06, ohlc["low"], f"低 ${ohlc['low']:,.0f}", fontsize=8.5, va="center")
ax.text(x + 0.06, ohlc["open"], f"开 ${ohlc['open']:,.0f}", fontsize=8.5, va="center")
ax.text(x + 0.06, ohlc["close"], f"收 ${ohlc['close']:,.0f}  (-1.85%)", fontsize=8.5, va="center", color="#c62828", fontweight="bold")
ax.set_xlim(0, 1.4)
ax.set_ylim(83000, 86500)
ax.set_xticks([])
ax.set_title("2026-10-07 BTC 当日 OHLC 结构（高开低走）", fontsize=12, fontweight="bold")
ax.set_ylabel("BTC 价格 (USD)")
ax.grid(True, alpha=0.3, axis="y")
fig.text(0.99, 0.01, "AS_OF: 2026-10-08 09:56（本地）；数据源 Yahoo Finance BTC-USD", ha="right", fontsize=7, color="#888")
plt.tight_layout()
plt.savefig(f"{OUT}\\chart_btc_1007_ohlc.png", dpi=130)
plt.close()

print("Charts saved:")
print(f"{OUT}\\chart_btc_price_trend.png")
print(f"{OUT}\\chart_fed_rate_vs_btc.png")
print(f"{OUT}\\chart_btc_1007_ohlc.png")
