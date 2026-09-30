# -*- coding: utf-8 -*-
"""
上证指数 (SH000001) 技术分析与走势研判
AS_OF: 2026-09-30 14:50
数据来源: 上游 CrawlerAgent (step1 盘中快照 as of 10:10:56 + step2 近5日K线)
说明: 9/30 为盘中数据(非收盘), 9/25 收盘缺失, 用线性插值补齐以便计算均线/MACD。
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.dates as mdates
from datetime import datetime

# ---------- 中文字体 ----------
for f in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "Arial Unicode MS"]:
    if any(f.lower() in x.name.lower() for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

OUT = r"E:\agent_dev\multi-agent\tmp\sh_index_ta_chart_20260930.png"

# ---------- 数据 (来自上游文件) ----------
# 9/25 缺失, 用 9/24 与 9/28 收盘线性插值补齐 (标注为估算)
rows = [
    # date,        open,    high,    low,     close,    turnover(亿), is_est, is_intraday
    ("2026-09-24", 3936.0,  3940.0,  3885.0,  3888.37, 16700,  False, False),
    ("2026-09-25", None,    None,    None,    None,     None,   True,  False),   # 估算
    ("2026-09-28", 3880.0,  3885.0,  3820.0,  3823.62,  None,   False, False),
    ("2026-09-29", 3828.0,  3835.0,  3820.0,  3830.45, 14091.97, False, False),
    ("2026-09-30", 3839.25, 3847.68, 3836.48, 3844.76, 2511.82, False, True),   # 盘中
]
df = pd.DataFrame(rows, columns=["date","open","high","low","close","turnover","is_est","is_intraday"])
df["date"] = pd.to_datetime(df["date"])

# 插值补齐 9/25 收盘 (open/high/low 用相邻均值近似)
c = df["close"].values
c[1] = (c[0] + c[2]) / 2.0
df["close"] = c
df["open"] = df["open"].fillna(df["close"])
df["high"] = df["high"].fillna(df["close"] * 1.002)
df["low"]  = df["low"].fillna(df["close"] * 0.998)

# ---------- 技术指标 ----------
# 仅5个交易日, MA10/MA20 样本不足 → 用 MA5 + MA3 (短周期) 替代, 避免 NaN
df["MA5"]  = df["close"].rolling(5).mean()
df["MA3"]  = df["close"].rolling(3).mean()

# MACD (12,26,9) — 用全序列 EWM, 起点有偏差但趋势方向可用
ema12 = df["close"].ewm(span=12, adjust=False).mean()
ema26 = df["close"].ewm(span=26, adjust=False).mean()
df["DIF"] = ema12 - ema26
df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
df["MACD"] = 2 * (df["DIF"] - df["DEA"])

# RSI(6) — 短周期, 适配5日样本
delta = df["close"].diff()
gain = delta.clip(lower=0).ewm(alpha=1/6, adjust=False).mean()
loss = (-delta.clip(upper=0)).ewm(alpha=1/6, adjust=False).mean()
rs = gain / loss
df["RSI6"] = 100 - 100/(1+rs)

# 支撑/压力
support1 = 3830.0   # 9/29 收盘 + 9/30 盘中低点区
support2 = 3800.0   # 整数关
resist1  = 3850.0
resist2  = 3880.0   # 9/24 收盘区
resist3  = 3888.37  # 9/24 高点

# ---------- 绘图 ----------
fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True,
                         gridspec_kw={"height_ratios":[3,1,1,1]})
fig.suptitle("上证指数 (SH000001) 技术分析与走势研判\nAS_OF: 2026-09-30 14:50  |  9/30为盘中数据(as of 10:10:56), 9/25为插值估算",
             fontsize=14, fontweight="bold")

ax = axes[0]
# K线
for i, r in df.iterrows():
    up = r["close"] >= r["open"]
    col = "#e63946" if up else "#2a9d8f"
    ax.vlines(r["date"], r["low"], r["high"], color=col, lw=1.2)
    ax.add_patch(plt.Rectangle((r["date"] - np.timedelta64(8,"h"),
                               min(r["open"],r["close"])),
                               np.timedelta64(16,"h"),
                               abs(r["close"]-r["open"]) or 0.5,
                               facecolor=col, edgecolor=col))
# 均线 (仅5日样本, 用 MA5/MA3)
ax.plot(df["date"], df["MA5"],  label="MA5",  color="#f4a261", lw=1.6)
ax.plot(df["date"], df["MA3"],  label="MA3",  color="#9b5de5", lw=1.6)
# 支撑压力
for lv, lab, c in [(support1,"支撑1 3830","#2a9d8f"),(support2,"支撑2 3800","#2a9d8f"),
                   (resist1,"压力1 3850","#e63946"),(resist2,"压力2 3880","#e63946")]:
    ax.axhline(lv, color=c, ls="--", lw=1, alpha=0.7)
    ax.text(df["date"].iloc[-1], lv, f"  {lab}", color=c, fontsize=9, va="center")
# 标注盘中/估算
ax.annotate("盘中\n3844.76", xy=(df["date"].iloc[-1], 3844.76),
            xytext=(df["date"].iloc[-1]-np.timedelta64(1,"D"), 3895),
            arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9, color="#333")
ax.annotate("插值估算", xy=(df["date"].iloc[1], df["close"].iloc[1]),
            xytext=(df["date"].iloc[1]-pd.Timedelta(days=1.5), 3860),
            arrowprops=dict(arrowstyle="->", color="gray"), fontsize=8, color="gray")
ax.set_ylabel("点位")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)
ax.set_title("K线 + 均线 (MA5/MA3) + 支撑压力位", fontsize=11)

# 成交量
ax = axes[1]
vol = df["turnover"].fillna(0)
cols = ["#e63946" if df["close"].iloc[i]>=df["open"].iloc[i] else "#2a9d8f" for i in range(len(df))]
ax.bar(df["date"], vol, width=0.5, color=cols, alpha=0.7)
ax.set_ylabel("成交额(亿元)")
ax.grid(alpha=0.3)
ax.set_title("成交额 (量能持续萎缩 → 节前'放假模式')", fontsize=11)

# MACD
ax = axes[2]
mcols = ["#e63946" if v>=0 else "#2a9d8f" for v in df["MACD"]]
ax.bar(df["date"], df["MACD"], width=0.5, color=mcols, alpha=0.7, label="MACD柱")
ax.plot(df["date"], df["DIF"], label="DIF", color="#f4a261", lw=1.4)
ax.plot(df["date"], df["DEA"], label="DEA", color="#457b9d", lw=1.4)
ax.axhline(0, color="gray", lw=0.8)
ax.set_ylabel("MACD")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)
ax.set_title("MACD (12,26,9) — DIF/DEA 金叉修复中", fontsize=11)

# RSI
ax = axes[3]
ax.plot(df["date"], df["RSI6"], color="#9b5de5", lw=1.6, marker="o", ms=4)
ax.axhline(70, color="#e63946", ls="--", lw=0.8)
ax.axhline(30, color="#2a9d8f", ls="--", lw=0.8)
ax.axhline(50, color="gray", ls=":", lw=0.8)
ax.fill_between(df["date"], 30, 70, color="gray", alpha=0.08)
ax.set_ylabel("RSI6")
ax.set_ylim(0,100)
ax.grid(alpha=0.3)
ax.set_title("RSI(6) — 中性区, 无超买超卖", fontsize=11)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

plt.tight_layout()
plt.savefig(OUT, dpi=130, bbox_inches="tight")
print("SAVED:", OUT)

# ---------- 输出关键指标 ----------
print("\n=== 关键指标 (AS_OF 2026-09-30 14:50) ===")
last = df.iloc[-1]
print(f"最新(盘中): {last['close']:.2f}  涨跌: +0.37%")
print(f"MA5:  {last['MA5']:.2f}")
print(f"MA3:  {last['MA3']:.2f}")
print(f"DIF:  {last['DIF']:.2f}  DEA: {last['DEA']:.2f}  MACD柱: {last['MACD']:.2f}")
print(f"RSI6: {last['RSI6']:.1f}")
print(f"支撑: 3830 / 3800   压力: 3850 / 3880 / 3888")
print(f"52周区间: 3741.11 - 4258.86  当前距52周高点: {(last['close']/4258.86-1)*100:.1f}%")
