# -*- coding: utf-8 -*-
"""
Step 4: 纳斯达克100 (NDX/QQQ) 资金流向数据分析与可视化
基于 Step1(资金流) / Step2(机构vs散户持仓) / Step3(宏观驱动) 的已获取数据。
AS_OF: 2026-10-07 17:28
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"F:\agent\multi-agent\tmp"
os.makedirs(OUT, exist_ok=True)

# ============================================================
# 1. 数据（来自 Step1/2/3 已获取，单位：亿美元）
# ============================================================
years = [2021, 2022, 2023, 2024, 2025]

# 年度净资金流（QQQ 代理，Step1）
net_flow = {2021: 280, 2022: -150, 2023: 210, 2024: 236, 2025: 244}
# 年度 AUM 变化（Step1）
aum_change = {2021: 320, 2022: -210, 2023: 380, 2024: 350, 2025: 315}
# NDX 年度涨跌幅（Step3）
ndx_return = {2021: 21.4, 2022: -33.1, 2023: 54.5, 2024: 25.7, 2025: 18.0}
# 联邦基金利率（年末，Step3）
fed_rate = {2021: 0.50, 2022: 4.50, 2023: 5.50, 2024: 4.25, 2025: 3.50}
# NVDA P/E TTM（年初，Step3）
nvda_pe = {2021: 74.6, 2022: 74.6, 2023: 80.7, 2024: 80.5, 2025: 46.6}

# 机构 vs 散户 资金方向（定性→量化指数，Step2）
investor_net = {
    "被动巨头(贝莱德/先锋/道富)": [1.0, 0.3, 1.0, 1.0, 1.0],
    "量化(文艺复兴等)":           [0.8, 0.2, 0.9, 1.0, 1.0],
    "散户(零佣金ETF)":            [1.0, -0.3, 0.9, 1.0, 1.0],
    "主动/宏观(桥水等)":          [0.7, -0.6, 0.6, 0.5, -0.8],
}

df = pd.DataFrame({
    "年份": years,
    "净资金流": [net_flow[y] for y in years],
    "AUM变化": [aum_change[y] for y in years],
    "NDX涨跌幅%": [ndx_return[y] for y in years],
    "联邦基金利率%": [fed_rate[y] for y in years],
    "NVDA_PE": [nvda_pe[y] for y in years],
})
print(df.to_string(index=False))

# 累计净资金流
df["累计净资金流"] = df["净资金流"].cumsum()
print("\n累计净资金流:", df["累计净资金流"].tolist())

# ============================================================
# 图1: 年度净资金流柱状图 + 累计折线（双轴）+ 关键节点标注
# ============================================================
fig, ax1 = plt.subplots(figsize=(11, 6.5))
colors = ["#2e7d32" if v >= 0 else "#c62828" for v in df["净资金流"]]
bars = ax1.bar(df["年份"].astype(str), df["净资金流"], color=colors, width=0.55, alpha=0.85, label="年度净资金流")
for b, v in zip(bars, df["净资金流"]):
    ax1.text(b.get_x() + b.get_width()/2, v + (12 if v >= 0 else -28), f"{v:+.0f}",
             ha="center", fontsize=10, fontweight="bold")
ax1.axhline(0, color="gray", lw=0.8)
ax1.set_ylabel("年度净资金流（亿美元）", fontsize=11)
ax1.set_xlabel("年份", fontsize=11)
ax1.set_title("纳斯达克100 (QQQ) 年度净资金流与累计趋势 (2021–2025)", fontsize=13, fontweight="bold")

ax2 = ax1.twinx()
ax2.plot(df["年份"].astype(str), df["累计净资金流"], color="#1565c0", marker="o", lw=2.2, label="累计净资金流")
for x, v in zip(df["年份"].astype(str), df["累计净资金流"]):
    ax2.text(x, v + 18, f"{v:+.0f}", ha="center", fontsize=9, color="#1565c0")
ax2.set_ylabel("累计净资金流（亿美元）", fontsize=11, color="#1565c0")

# 关键节点标注
ax1.annotate("2021 散户牛市\n零佣金+科技叙事", xy=("2021", 280), xytext=("2021", 360),
             fontsize=8.5, ha="center", color="#2e7d32",
             arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=0.8))
ax1.annotate("2022 激进加息+俄乌\nNDX -33% 阶段性流出", xy=("2022", -150), xytext=("2022", -230),
             fontsize=8.5, ha="center", color="#c62828",
             arrowprops=dict(arrowstyle="->", color="#c62828", lw=0.8))
ax1.annotate("2023 AI叙事回归\nNDX +54.5%", xy=("2023", 210), xytext=("2023", 300),
             fontsize=8.5, ha="center", color="#2e7d32",
             arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=0.8))
ax1.annotate("2024 开启降息\n净流入236亿", xy=("2024", 236), xytext=("2024", 330),
             fontsize=8.5, ha="center", color="#2e7d32",
             arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=0.8))

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
ax1.set_ylim(-300, 420)
ax2.set_ylim(-100, 1100)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart1_annual_net_flow.png"), dpi=130)
plt.close()
print("Saved chart1_annual_net_flow.png")

# ============================================================
# 图2: 机构 vs 散户 资金方向（分组柱状图）
# ============================================================
inv_df = pd.DataFrame(investor_net, index=years)
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(years))
w = 0.2
for i, col in enumerate(inv_df.columns):
    vals = inv_df[col].values
    bars = ax.bar(x + i*w, vals, w, label=col)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + (0.04 if v >= 0 else -0.12),
                f"{v:+.1f}", ha="center", fontsize=8)
ax.axhline(0, color="gray", lw=0.8)
ax.set_xticks(x + w*1.5)
ax.set_xticklabels(years)
ax.set_ylabel("净资金方向指数（+买入 / -卖出）", fontsize=11)
ax.set_title("机构 vs 散户 资金方向对比 (2021–2025)\n被动/量化/散户=净买入；主动宏观(桥水)2025阶段性卖出",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="upper left")
ax.set_ylim(-1.2, 1.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart2_institutional_vs_retail.png"), dpi=130)
plt.close()
print("Saved chart2_institutional_vs_retail.png")

# ============================================================
# 图3: 宏观驱动因素 vs 资金流（利率 + NVDA P/E 双轴 + 资金流柱）
# ============================================================
fig, ax1 = plt.subplots(figsize=(11, 6.5))
colors = ["#2e7d32" if v >= 0 else "#c62828" for v in df["净资金流"]]
ax1.bar(df["年份"].astype(str), df["净资金流"], color=colors, width=0.5, alpha=0.5, label="年度净资金流")
ax1.set_ylabel("年度净资金流（亿美元）", fontsize=11)
ax1.set_xlabel("年份", fontsize=11)
ax1.set_title("宏观驱动因素 vs 资金流向 (2021–2025)\n利率下行 + AI盈利消化估值 = 净流入顺风", fontsize=13, fontweight="bold")

ax2 = ax1.twinx()
ax2.plot(df["年份"].astype(str), df["联邦基金利率%"], color="#e65100", marker="s", lw=2.2, label="联邦基金利率%")
ax2.plot(df["年份"].astype(str), df["NVDA_PE"], color="#6a1b9a", marker="^", lw=2.2, label="NVDA P/E (TTM)")
ax2.set_ylabel("利率% / P/E", fontsize=11)
for x, v in zip(df["年份"].astype(str), df["联邦基金利率%"]):
    ax2.text(x, v + 0.3, f"{v:.1f}%", ha="center", fontsize=8, color="#e65100")
for x, v in zip(df["年份"].astype(str), df["NVDA_PE"]):
    ax2.text(x, v + 2, f"{v:.0f}x", ha="center", fontsize=8, color="#6a1b9a")

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)
ax1.set_ylim(-300, 420)
ax2.set_ylim(0, 95)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart3_macro_drivers.png"), dpi=130)
plt.close()
print("Saved chart3_macro_drivers.png")

# ============================================================
# 图4: 代表性机构 2025 Q2 持仓方向（Step2）
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5.5))
inst = ["贝莱德\nBlackRock", "桥水\nBridgewater", "文艺复兴\nRenaissance", "道富\nState Street"]
blackrock = [1.0, 1.0, 1.0, 1.0]
bridgewater = [-0.92, -0.8, -0.5, -0.3]
ren = [1.81, 6.99, 3.27, 1.0]
state = [0.43, 0.43, 0.43, 0.43]

x = np.arange(len(inst))
w = 0.2
ax.bar(x - 1.5*w, blackrock, w, label="贝莱德(净买入)", color="#1565c0")
ax.bar(x - 0.5*w, bridgewater, w, label="桥水(2025减持)", color="#c62828")
ax.bar(x + 0.5*w, ren, w, label="文艺复兴(激进加仓)", color="#2e7d32")
ax.bar(x + 1.5*w, state, w, label="道富(+43%)", color="#6a1b9a")
ax.axhline(0, color="gray", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(inst, fontsize=10)
ax.set_ylabel("持仓变化方向（+增持 / -减持）", fontsize=11)
ax.set_title("代表性机构 2025 Q2 持仓方向 (13F)\n桥水减持92% vs 文艺复兴/贝莱德加仓", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.set_ylim(-1.2, 7.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart4_institutional_holdings.png"), dpi=130)
plt.close()
print("Saved chart4_institutional_holdings.png")

# ============================================================
# 汇总统计
# ============================================================
print("\n=== 汇总统计 ===")
print(f"5年累计净资金流: {df['净资金流'].sum():+.0f} 亿美元")
print(f"5年累计AUM变化: {df['AUM变化'].sum():+.0f} 亿美元")
print(f"净流入年份: {[y for y in years if net_flow[y] > 0]}")
print(f"净流出年份: {[y for y in years if net_flow[y] < 0]}")
print(f"NDX 5年累计涨幅: {sum(ndx_return.values()):+.1f}%")
print("\nAll charts saved to", OUT)
