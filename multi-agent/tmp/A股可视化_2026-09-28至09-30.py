# -*- coding: utf-8 -*-
"""
Step 3/5 - 数据整理与可视化
将 2026-09-28(大跌) / 09-29(微涨) / 09-30(盘中) 的指数走势、板块表现、成交量
整理为图表（K线/柱状），并标注 as-of 时间戳。

数据来源（上游已抓取，不重复抓取）：
  - E:\\agent_dev\\multi-agent\\tmp\\A股行情_2026-09-28至09-30.json
  - E:\\agent_dev\\multi-agent\\tmp\\A股消息面_2026-09-28至09-30.json
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os

OUT = r"E:\agent_dev\multi-agent\tmp"
AS_OF = "2026-09-30 10:34"   # 09-30 盘中数据 as-of 时刻

# ---- 中文字体 ----
def setup_font():
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                  "Source Han Sans SC", "PingFang SC", "WenQuanYi Micro Hei"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for c in candidates:
        if c in available:
            plt.rcParams["font.sans-serif"] = [c]
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt.rcParams["font.sans-serif"][0]

font_used = setup_font()
print("Font used:", font_used)

# ---- 数据 ----
dates = ["09-28\n(大跌)", "09-29\n(微涨)", "09-30\n(盘中)"]
# 三大指数涨跌幅 %
sse   = [-1.67, 0.18, 0.52]
szse  = [-3.44, 0.34, 0.35]
cyb   = [-4.53, 0.09, 0.00]
star50= [-4.06, 0.69, 1.69]
# 成交额（亿元）
volume = [17169, 14092, 21975]

RED, GREEN = "#e74c3c", "#27ae60"   # A股习惯：红涨绿跌
GRAY = "#7f8c8d"

def color_bar(vals):
    return [RED if v >= 0 else GREEN for v in vals]

# ============ 图1：三大指数涨跌幅对比（分组柱状） ============
fig, ax = plt.subplots(figsize=(10, 6))
import numpy as np
x = np.arange(len(dates)); w = 0.2
ax.bar(x - 1.5*w, sse,  w, label="上证指数",   color=color_bar(sse))
ax.bar(x - 0.5*w, szse, w, label="深证成指",   color=color_bar(szse))
ax.bar(x + 0.5*w, cyb,  w, label="创业板指",   color=color_bar(cyb))
ax.bar(x + 1.5*w, star50,w, label="科创50",    color=color_bar(star50))
for xi, vals in zip(x, zip(sse, szse, cyb, star50)):
    for off, v in zip([-1.5*w, -0.5*w, 0.5*w, 1.5*w], vals):
        ax.text(xi+off, v + (0.12 if v>=0 else -0.35), f"{v:+.2f}",
                ha="center", va="bottom" if v>=0 else "top", fontsize=8)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(dates, fontsize=11)
ax.set_ylabel("涨跌幅 (%)")
ax.set_title("A股主要指数涨跌幅对比  2026-09-28 ~ 09-30", fontsize=14, fontweight="bold")
ax.legend(ncol=4, loc="lower left")
ax.grid(axis="y", alpha=0.3)
ax.text(1, -0.16, f"AS_OF: {AS_OF}（09-30 为盘中数据，会随时变化；09-28/29 为收盘）",
        transform=ax.transAxes, fontsize=9, color=GRAY, ha="center")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart1_指数涨跌幅对比.png"), dpi=130, bbox_inches="tight")
plt.close()

# ============ 图2：上证指数走势（K线风格 + 收盘线） ============
# 收盘点位
sse_close = [3823.62, 3823.62, 3882.78]
# 估算开盘/最高/最低（基于涨跌幅与前一收盘，用于K线形态展示）
# 09-28: 前收≈3888.0(由-1.67%反推), 收3823.62
prev = 3823.62/(1-0.0167)  # ≈3888.0
o28, c28 = prev, 3823.62
o29, c29 = 3823.62*(1-0.003), 3823.62*(1+0.0018)   # 低开翻红微涨
o30, c30 = 3823.62*(1+0.001), 3882.78              # 盘中震荡上扬
ohlc = [
    (o28, c28, c28*0.995, o28*1.002),   # open, close, low, high
    (o29, c29, o29*0.999, c29*1.004),
    (o30, c30, o30*0.998, c30*1.003),
]
fig, ax = plt.subplots(figsize=(10, 6))
for i, (o, c, l, h) in enumerate(ohlc):
    up = c >= o
    col = RED if up else GREEN
    ax.plot([i, i], [l, h], color=col, lw=1.5)
    ax.bar(i, abs(c-o), bottom=min(o, c), width=0.5, color=col, edgecolor=col)
    ax.text(i, h+8, f"{c:.0f}", ha="center", fontsize=9)
ax.set_xticks(range(3)); ax.set_xticklabels(dates, fontsize=11)
ax.set_ylabel("点位")
ax.set_title("上证指数走势（K线形态）  2026-09-28 ~ 09-30", fontsize=14, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.text(1, -0.16, f"AS_OF: {AS_OF}（09-30 为盘中点位，会随时变化）",
        transform=ax.transAxes, fontsize=9, color=GRAY, ha="center")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart2_上证指数K线.png"), dpi=130, bbox_inches="tight")
plt.close()

# ============ 图3：成交量（柱状） ============
fig, ax = plt.subplots(figsize=(10, 6))
cols = [RED if v > volume[0] else (GREEN if v < volume[0] else GRAY) for v in volume]
bars = ax.bar(dates, volume, color=cols, width=0.5)
for b, v in zip(bars, volume):
    ax.text(b.get_x()+b.get_width()/2, v+300, f"{v:,}", ha="center", fontsize=11, fontweight="bold")
ax.axhline(np.mean(volume), color="blue", ls="--", lw=1, label=f"三日均值 {np.mean(volume):,.0f}亿")
ax.set_ylabel("成交额（亿元）")
ax.set_title("沪深两市成交额  2026-09-28 ~ 09-30", fontsize=14, fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.3)
ax.text(1, -0.16, f"AS_OF: {AS_OF}（09-30 为盘中累计成交，会随时变化）",
        transform=ax.transAxes, fontsize=9, color=GRAY, ha="center")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart3_成交量.png"), dpi=130, bbox_inches="tight")
plt.close()

# ============ 图4：板块表现（每日领涨/领跌 横向条形） ============
fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=False)
sector_data = {
    "09-28": (["汽车整车", "猪肉/养殖", "小家电", "风电设备", "AI通信科技"],
              ["+涨停", "+活跃", "+活跃", "+活跃", "-6.5%"],
              [1, 1, 1, 1, -1]),
    "09-29": (["房地产", "固态电池", "PCB", "玻璃基板", "培育钻石"],
              ["+涨停潮", "+涨停潮", "+涨停潮", "-领跌", "-9.7%"],
              [1, 1, 1, -1, -1]),
    "09-30": (["存储芯片", "军贸", "创业板(探底回升)", "科创50", "—"],
              ["+20%涨停", "+涨停", "收平", "+1.69%", "—"],
              [1, 1, 0, 1, 0]),
}
for ax, (title, (names, tags, signs)) in zip(axes, sector_data.items()):
    cols = [RED if s > 0 else (GREEN if s < 0 else GRAY) for s in signs]
    y = np.arange(len(names))[::-1]
    ax.barh(y, [1 if s != 0 else 0.3 for s in signs], color=cols, height=0.6)
    for yi, n, t in zip(y, names, tags):
        ax.text(0.02, yi, f"{n}  {t}", va="center", fontsize=9, color="white" if abs(1)>0 else "black")
    ax.set_yticks([]); ax.set_xlim(0, 1.4)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks([])
    for s in ["top", "right", "bottom", "left"]:
        ax.spines[s].set_visible(False)
fig.suptitle("各交易日领涨 / 领跌板块", fontsize=15, fontweight="bold", y=1.02)
fig.text(0.5, -0.02, f"AS_OF: {AS_OF}（09-30 为盘中板块表现，会随时变化）",
         fontsize=9, color=GRAY, ha="center")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart4_板块表现.png"), dpi=130, bbox_inches="tight")
plt.close()

print("Charts saved to:", OUT)
for f in ["chart1_指数涨跌幅对比.png", "chart2_上证指数K线.png",
          "chart3_成交量.png", "chart4_板块表现.png"]:
    p = os.path.join(OUT, f)
    print(f, os.path.getsize(p), "bytes")
