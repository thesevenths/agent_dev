import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np, os

cands = ["Microsoft YaHei","SimHei","Microsoft JhengHei","PingFang SC","Noto Sans CJK SC","Source Han Sans SC","WenQuanYi Micro Hei"]
avail = {f.name for f in fm.fontManager.ttflist}
chosen = next((c for c in cands if c in avail), None)
print("chosen font:", chosen)
if chosen:
    plt.rcParams["font.family"] = chosen
plt.rcParams["axes.unicode_minus"] = False

out = r"E:\agent_dev\multi-agent\tmp"

# ---- Chart 1: Today vs Yesterday ----
fig, axes = plt.subplots(1, 2, figsize=(11,4.6))
days = ["昨日\n2026-09-29","今日\n2026-09-30"]
close = [3830.45, 3842.19]
chg = [0.18, 0.31]
colors = ["#8aa0b8", "#d64545"]
ax = axes[0]
bars = ax.bar(days, close, color=colors, width=0.5)
ax.set_title("上证指数收盘价对比 (点)", fontsize=13, fontweight="bold")
ax.set_ylim(3800, 3860)
for b,v in zip(bars, close):
    ax.text(b.get_x()+b.get_width()/2, v+1, f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("收盘点位"); ax.grid(axis="y", alpha=0.3)
ax = axes[1]
bars = ax.bar(days, chg, color=colors, width=0.5)
ax.set_title("涨跌幅对比 (%)", fontsize=13, fontweight="bold")
ax.axhline(0, color="gray", lw=0.8)
for b,v in zip(bars, chg):
    ax.text(b.get_x()+b.get_width()/2, v+0.005, f"+{v:.2f}%", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("涨跌幅 (%)"); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out,"chart1_today_vs_yesterday.png"), dpi=130, bbox_inches="tight"); plt.close()

# ---- Chart 2: Historical probability by source ----
fig, ax = plt.subplots(figsize=(10,5.2))
periods = ["2000-2011\n财新","2010-2023\n招商/华金","2015-2024\n东财Choice","2016-2025\n证券日报/Wind","2016-2025\n澎湃·郭施亮"]
up = [60, 64.3, 70, 70, 60]
down = [40, 35.7, 30, 30, 40]
x = np.arange(len(periods))
ax.bar(x, up, color="#d64545", label="上涨概率", width=0.6)
ax.bar(x, down, bottom=up, color="#5b8def", label="下跌概率", width=0.6)
for i,(u,d) in enumerate(zip(up,down)):
    ax.text(i, u/2, f"{u:.1f}%", ha="center", va="center", color="white", fontweight="bold")
    ax.text(i, u+d/2, f"{d:.1f}%", ha="center", va="center", color="white", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(periods, fontsize=9)
ax.set_ylim(0,100); ax.set_ylabel("概率 (%)")
ax.set_title("历年国庆节后第一个交易日 上证指数涨跌概率 (多口径)", fontsize=13, fontweight="bold")
ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out,"chart2_history_probability.png"), dpi=130, bbox_inches="tight"); plt.close()

# ---- Chart 3: Consensus donut ----
fig, ax = plt.subplots(figsize=(5.5,5.5))
ax.pie([65,35], colors=["#d64545","#5b8def"], startangle=90,
       autopct="%1.0f%%", pctdistance=0.75,
       textprops={"color":"white","fontweight":"bold","fontsize":14},
       wedgeprops={"width":0.42,"edgecolor":"white"})
ax.text(0,0.08,"综合口径", ha="center", fontsize=12, color="#555")
ax.text(0,-0.12,"涨 60~70%", ha="center", fontsize=13, fontweight="bold", color="#d64545")
ax.set_title("国庆后首个交易日 涨跌概率(综合)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(out,"chart3_consensus_donut.png"), dpi=130, bbox_inches="tight"); plt.close()

print("charts:", [f for f in os.listdir(out) if f.endswith(".png")])
