# -*- coding: utf-8 -*-
"""
Step 3 — NDX vs BTC 量化对比与可视化
基于 Step 1 / Step 2 检索数据，计算风险（波动率、最大回撤、下行风险）
与涨幅（历史年化、估值隐含空间）两维度对比，生成对比图表。

AS_OF: 2026-10-08 10:53（本地）
数据源: Step 1 (step1_ndx_btc_base_data.md) + Step 2 (step2_risk_return_drivers.md)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch

# ---------- 中文字体 ----------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"E:\agent_dev\multi-agent\tmp"
os.makedirs(OUT, exist_ok=True)

# =====================================================================
# 一、基础数据（来自 Step 1 / Step 2，已标注代理/估算）
# =====================================================================
# --- NDX (QQQ) ---
NDX = {
    "price": 757.73,            # QQQ 2026-10-07 收盘
    "ret_1y": 25.59,            # 精确
    "ret_3y_ann": 28.43,        # 精确（年化）
    "ret_5y_ann": 22.9,         # Nasdaq 官方 5Y 年化
    "pe": 30.45,                # worldperatio 2026-10-02
    "pe_long_avg": 27.86,       # GuruFocus
    "pe_5y_avg": 30.49,
    "beta": 1.26,
    "vol_lo": 20.0, "vol_hi": 25.0, "vol_mid": 22.5,   # 代理
    "mdd_lo": -35.0, "mdd_hi": -33.0, "mdd_mid": -34.0, # 代理（2022）
}
# --- BTC ---
BTC = {
    "price": 84199.14,          # 2026-10-07 收盘
    "ret_1y": -33.0,            # 估算
    "ret_3y_cum_lo": 150.0, "ret_3y_cum_hi": 200.0, "ret_3y_cum_mid": 175.0,  # 估算（累计）
    "ret_5y_cum_lo": 150.0, "ret_5y_cum_hi": 250.0, "ret_5y_cum_mid": 200.0,  # 估算（累计）
    "vol_lo": 40.0, "vol_hi": 60.0, "vol_mid": 50.0,   # 代理
    "mdd_lo": -52.0, "mdd_hi": -50.0, "mdd_mid": -51.0, # 代理（2025 周期）
    "peak": 126198.0,           # 2025-10-06
    "trough": 60074.0,          # 2026-02
}

# =====================================================================
# 二、派生指标计算
# =====================================================================
# 2.1 BTC 累计回报 → 年化
def cum_to_ann(cum_pct, years):
    """累计回报(%) → 年化(%)"""
    return ((1 + cum_pct / 100) ** (1 / years) - 1) * 100

btc_3y_ann_lo = cum_to_ann(BTC["ret_3y_cum_lo"], 3)
btc_3y_ann_hi = cum_to_ann(BTC["ret_3y_cum_hi"], 3)
btc_3y_ann_mid = cum_to_ann(BTC["ret_3y_cum_mid"], 3)
btc_5y_ann_lo = cum_to_ann(BTC["ret_5y_cum_lo"], 5)
btc_5y_ann_hi = cum_to_ann(BTC["ret_5y_cum_hi"], 5)
btc_5y_ann_mid = cum_to_ann(BTC["ret_5y_cum_mid"], 5)

# 2.2 风险调整收益（夏普近似 = 年化收益 / 年化波动率，无风险利率取 0 简化）
def sharpe(ann_ret, vol):
    return ann_ret / vol

# 2.3 估值隐含空间（NDX）：当前 P/E 回归长期均值的隐含估值压缩/扩张
#     若盈利不变，P/E 从 30.45 → 27.86 的隐含价格变化
ndx_pe_reversion = (NDX["pe_long_avg"] / NDX["pe"] - 1) * 100  # 负值=估值压缩风险
ndx_pe_5y = (NDX["pe_5y_avg"] / NDX["pe"] - 1) * 100

# 2.4 BTC 机构预测隐含涨幅（2030 中性 $240,000 / 乐观 $380,000）
btc_2030_neutral = 240000
btc_2030_optim = 380000
btc_2030_neutral_upside = (btc_2030_neutral / BTC["price"] - 1) * 100
btc_2030_optim_upside = (btc_2030_optim / BTC["price"] - 1) * 100
btc_2030_neutral_ann = cum_to_ann(btc_2030_neutral_upside, 4)  # 2026→2030 ≈ 4 年
btc_2030_optim_ann = cum_to_ann(btc_2030_optim_upside, 4)

# 2.5 下行风险（最大回撤幅度，绝对值）
ndx_mdd_abs = abs(NDX["mdd_mid"])
btc_mdd_abs = abs(BTC["mdd_mid"])

# 2.6 波动率倍数
vol_ratio = BTC["vol_mid"] / NDX["vol_mid"]

print("=" * 60)
print("派生指标计算结果")
print("=" * 60)
print(f"BTC 3Y 年化: {btc_3y_ann_lo:.1f}% ~ {btc_3y_ann_hi:.1f}% (中值 {btc_3y_ann_mid:.1f}%)")
print(f"BTC 5Y 年化: {btc_5y_ann_lo:.1f}% ~ {btc_5y_ann_hi:.1f}% (中值 {btc_5y_ann_mid:.1f}%)")
print(f"NDX P/E 回归长期均值隐含: {ndx_pe_reversion:.1f}% (估值压缩风险)")
print(f"NDX P/E 回归 5Y 均值隐含: {ndx_pe_5y:.1f}%")
print(f"BTC 2030 中性隐含涨幅: {btc_2030_neutral_upside:.0f}% (年化 {btc_2030_neutral_ann:.1f}%)")
print(f"BTC 2030 乐观隐含涨幅: {btc_2030_optim_upside:.0f}% (年化 {btc_2030_optim_ann:.1f}%)")
print(f"波动率倍数 (BTC/NDX): {vol_ratio:.1f}x")
print(f"夏普近似 (NDX 5Y): {sharpe(NDX['ret_5y_ann'], NDX['vol_mid']):.2f}")
print(f"夏普近似 (BTC 5Y 中值): {sharpe(btc_5y_ann_mid, BTC['vol_mid']):.2f}")

# =====================================================================
# 三、图表生成
# =====================================================================
C_NDX = "#1f77b4"   # 蓝
C_BTC = "#ff7f0e"   # 橙
C_GRAY = "#6c757d"

# ---------- 图 1：风险维度对比（波动率 + 最大回撤）----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("风险维度对比：NDX vs BTC（年化波动率 & 最大回撤）", fontsize=15, fontweight="bold")

# 1a. 年化波动率（区间）
ax = axes[0]
x = np.arange(2)
width = 0.5
vol_ndx = NDX["vol_mid"]
vol_btc = BTC["vol_mid"]
vol_ndx_err = [[NDX["vol_mid"] - NDX["vol_lo"], [NDX["vol_hi"] - NDX["vol_mid"]]]]
vol_btc_err = [[BTC["vol_mid"] - BTC["vol_lo"], [BTC["vol_hi"] - BTC["vol_mid"]]]]
bars = ax.bar(x, [vol_ndx, vol_btc], width, color=[C_NDX, C_BTC], alpha=0.85,
              yerr=[[NDX["vol_mid"]-NDX["vol_lo"], BTC["vol_mid"]-BTC["vol_lo"]],
                    [NDX["vol_hi"]-NDX["vol_mid"], BTC["vol_hi"]-BTC["vol_mid"]]],
              capsize=8, error_kw=dict(elinewidth=2, ecolor="black"))
ax.set_xticks(x)
ax.set_xticklabels(["NDX (QQQ)", "BTC"], fontsize=12)
ax.set_ylabel("年化波动率 (%)", fontsize=11)
ax.set_title("年化波动率（代理区间）", fontsize=12)
ax.set_ylim(0, 70)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
for i, (lo, hi, mid) in enumerate([(NDX["vol_lo"], NDX["vol_hi"], vol_ndx),
                                    (BTC["vol_lo"], BTC["vol_hi"], vol_btc)]):
    ax.text(i, hi + 2, f"{lo:.0f}–{hi:.0f}%", ha="center", fontsize=10, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.text(0.5, -0.18, f"BTC 波动率约为 NDX 的 {vol_ratio:.1f} 倍", transform=ax.transAxes,
        ha="center", fontsize=10, color=C_GRAY)

# 1b. 最大回撤（区间）
ax = axes[1]
mdd_ndx = NDX["mdd_mid"]
mdd_btc = BTC["mdd_mid"]
bars = ax.bar(x, [mdd_ndx, mdd_btc], width, color=[C_NDX, C_BTC], alpha=0.85,
              yerr=[[abs(NDX["mdd_lo"]-NDX["mdd_mid"]), abs(BTC["mdd_lo"]-BTC["mdd_mid"])],
                    [abs(NDX["mdd_hi"]-NDX["mdd_mid"]), abs(BTC["mdd_hi"]-BTC["mdd_mid"])]],
              capsize=8, error_kw=dict(elinewidth=2, ecolor="black"))
ax.set_xticks(x)
ax.set_xticklabels(["NDX (QQQ)", "BTC"], fontsize=12)
ax.set_ylabel("最大回撤 (%)", fontsize=11)
ax.set_title("最大回撤（代理区间，近 5 年）", fontsize=12)
ax.set_ylim(-65, 5)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
for i, (lo, hi, mid) in enumerate([(NDX["mdd_lo"], NDX["mdd_hi"], mdd_ndx),
                                    (BTC["mdd_lo"], BTC["mdd_hi"], mdd_btc)]):
    ax.text(i, mid - 3, f"{lo:.0f}–{hi:.0f}%", ha="center", fontsize=10, fontweight="bold", color="white")
ax.grid(axis="y", alpha=0.3)
ax.text(0.5, -0.18, "NDX: 2022 熊市 | BTC: 2025 周期峰值→底部", transform=ax.transAxes,
        ha="center", fontsize=10, color=C_GRAY)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart1_risk_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("图 1 已保存: chart1_risk_comparison.png")

# ---------- 图 2：涨幅维度对比（历史年化回报）----------
fig, ax = plt.subplots(figsize=(11, 6))
fig.suptitle("涨幅维度对比：历史年化回报（NDX vs BTC）", fontsize=15, fontweight="bold")

categories = ["1 年回报", "3 年年化", "5 年年化"]
x = np.arange(len(categories))
width = 0.35

# NDX（精确/官方）
ndx_vals = [NDX["ret_1y"], NDX["ret_3y_ann"], NDX["ret_5y_ann"]]
# BTC（估算区间，取中值）
btc_vals = [BTC["ret_1y"], btc_3y_ann_mid, btc_5y_ann_mid]

bars1 = ax.bar(x - width/2, ndx_vals, width, label="NDX (QQQ)", color=C_NDX, alpha=0.85)
bars2 = ax.bar(x + width/2, btc_vals, width, label="BTC", color=C_BTC, alpha=0.85)

# BTC 误差棒（3Y/5Y 区间）—— 形状 (2, n)：[下偏差, 上偏差]
btc_err = np.array([
    [0, 0],  # 1Y 无区间
    [btc_3y_ann_mid - btc_3y_ann_lo, btc_3y_ann_hi - btc_3y_ann_mid],
    [btc_5y_ann_mid - btc_5y_ann_lo, btc_5y_ann_hi - btc_5y_ann_mid],
])
ax.errorbar(x + width/2, btc_vals, yerr=btc_err, fmt="none", color="black", capsize=6, elinewidth=2)

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12)
ax.set_ylabel("年化回报 (%)", fontsize=11)
ax.axhline(0, color="black", linewidth=0.8)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=11)

# 数值标注
for i, v in enumerate(ndx_vals):
    ax.text(x[i] - width/2, v + (2 if v >= 0 else -6), f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold", color=C_NDX)
for i, v in enumerate(btc_vals):
    ax.text(x[i] + width/2, v + (2 if v >= 0 else -6), f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold", color=C_BTC)

ax.text(0.5, -0.22, "NDX 为精确/官方数据；BTC 3Y/5Y 为估算区间（误差棒），1Y 为估算值",
        transform=ax.transAxes, ha="center", fontsize=9, color=C_GRAY)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart2_return_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("图 2 已保存: chart2_return_comparison.png")

# ---------- 图 3：雷达图（风险-收益综合画像）----------
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
fig.suptitle("NDX vs BTC 风险-收益综合画像（雷达图）", fontsize=15, fontweight="bold")

# 维度（归一化到 0-100，方向统一为"越高越好"）
# 1. 年化收益（5Y）: 越高越好
# 2. 风险调整收益（夏普近似）: 越高越好
# 3. 低波动率: 波动率越低越好 → 100 - vol
# 4. 低回撤: 回撤越小越好 → 100 - |mdd|
# 5. 估值吸引力: NDX 用 P/E 相对长期均值（越低越好）; BTC 用周期位置（当前底部区域=高分）
# 6. 上行空间（隐含）: 越高越好

# 归一化函数
def norm(val, lo, hi, invert=False):
    v = (val - lo) / (hi - lo) * 100
    return 100 - v if invert else v

# 5Y 年化收益
r1_ndx = norm(NDX["ret_5y_ann"], 0, 60)
r1_btc = norm(btc_5y_ann_mid, 0, 60)
# 夏普近似
r2_ndx = norm(sharpe(NDX["ret_5y_ann"], NDX["vol_mid"]), 0, 2.5)
r2_btc = norm(sharpe(btc_5y_ann_mid, BTC["vol_mid"]), 0, 2.5)
# 低波动率（invert）
r3_ndx = norm(NDX["vol_mid"], 0, 60, invert=True)
r3_btc = norm(BTC["vol_mid"], 0, 60, invert=True)
# 低回撤（invert）
r4_ndx = norm(ndx_mdd_abs, 0, 60, invert=True)
r4_btc = norm(btc_mdd_abs, 0, 60, invert=True)
# 估值吸引力: NDX P/E 30.45 vs 长期 27.86 → 偏高=低分; BTC 当前 $84k 处于周期底部区域=高分
r5_ndx = norm(NDX["pe"], 11.66, 38.3, invert=True)  # P/E 历史区间
r5_btc = 75.0  # 当前处于周期底部区域（$84k vs 峰值 $126k），估值吸引力中等偏高
# 上行空间（隐含）: NDX 盈利驱动（保守 ~15%）; BTC 2030 中性隐含 ~184%
r6_ndx = norm(15.0, 0, 200)
r6_btc = norm(btc_2030_neutral_upside, 0, 200)

labels = ["5Y 年化收益", "风险调整收益\n(夏普近似)", "低波动率", "低回撤", "估值吸引力", "上行空间\n(隐含)"]
ndx_scores = [r1_ndx, r2_ndx, r3_ndx, r4_ndx, r5_ndx, r6_ndx]
btc_scores = [r1_btc, r2_btc, r3_btc, r4_btc, r5_btc, r6_btc]

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
ndx_scores += ndx_scores[:1]
btc_scores += btc_scores[:1]
angles += angles[:1]

ax.plot(angles, ndx_scores, "o-", linewidth=2, label="NDX (QQQ)", color=C_NDX)
ax.fill(angles, ndx_scores, alpha=0.25, color=C_NDX)
ax.plot(angles, btc_scores, "o-", linewidth=2, label="BTC", color=C_BTC)
ax.fill(angles, btc_scores, alpha=0.25, color=C_BTC)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8, color=C_GRAY)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
ax.text(0.5, -0.15, "各维度归一化到 0-100（越高越好）；BTC 估值吸引力基于周期位置（当前底部区域）",
        transform=ax.transAxes, ha="center", fontsize=9, color=C_GRAY)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart3_radar.png"), dpi=150, bbox_inches="tight")
plt.close()
print("图 3 已保存: chart3_radar.png")

# ---------- 图 4：BTC 回撤曲线（2025 周期峰值→底部→当前）----------
fig, ax = plt.subplots(figsize=(11, 6))
fig.suptitle("BTC 当前周期回撤曲线（2025-10 峰值 → 2026-02 底部 → 当前）", fontsize=14, fontweight="bold")

# 关键价格点（来自 Step 1）
dates = ["2025-10-06\n(峰值)", "2025-12\n(年末)", "2026-02\n(底部)", "2026-09-12\n(1Y 参考)", "2026-10-07\n(当前)"]
prices = [126198, 87000, 60074, 77293, 84199]
# 回撤（相对峰值）
drawdown = [(p / 126198 - 1) * 100 for p in prices]

ax.plot(range(len(dates)), prices, "o-", color=C_BTC, linewidth=2.5, markersize=10, label="BTC 价格")
ax.fill_between(range(len(dates)), prices, 126198, alpha=0.15, color=C_BTC)
ax2 = ax.twinx()
ax2.plot(range(len(dates)), drawdown, "s--", color="#d62728", linewidth=2, markersize=8, label="回撤幅度")
ax2.fill_between(range(len(dates)), drawdown, 0, alpha=0.1, color="#d62728")

ax.set_xticks(range(len(dates)))
ax.set_xticklabels(dates, fontsize=10)
ax.set_ylabel("BTC 价格 (USD)", fontsize=11, color=C_BTC)
ax2.set_ylabel("回撤幅度 (%)", fontsize=11, color="#d62728")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.set_ylim(40000, 140000)
ax2.set_ylim(-60, 10)
ax.grid(axis="y", alpha=0.3)

# 标注
ax.annotate(f"峰值 ${126198:,}", xy=(0, 126198), xytext=(0.3, 130000),
            fontsize=10, fontweight="bold", color=C_BTC,
            arrowprops=dict(arrowstyle="->", color=C_BTC))
ax.annotate(f"底部 ${60074:,}\n(回撤 -52.4%)", xy=(2, 60074), xytext=(2.3, 50000),
            fontsize=10, fontweight="bold", color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728"))
ax.annotate(f"当前 ${84199:,}\n(回撤 -33.2%)", xy=(4, 84199), xytext=(3.2, 95000),
            fontsize=10, fontweight="bold", color=C_BTC,
            arrowprops=dict(arrowstyle="->", color=C_BTC))

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)
ax.text(0.5, -0.2, "峰值 2025-10-06 $126,198（减半后 534 天，史上最短）；底部 2026-02 $60,074；当前 2026-10-07 $84,199",
        transform=ax.transAxes, ha="center", fontsize=9, color=C_GRAY)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart4_btc_drawdown.png"), dpi=150, bbox_inches="tight")
plt.close()
print("图 4 已保存: chart4_btc_drawdown.png")

# ---------- 图 5：风险-收益散点图（波动率 vs 年化收益，气泡=回撤）----------
fig, ax = plt.subplots(figsize=(10, 7))
fig.suptitle("风险-收益散点图：年化波动率 vs 5Y 年化回报", fontsize=15, fontweight="bold")

# NDX
ax.scatter(NDX["vol_mid"], NDX["ret_5y_ann"], s=abs(NDX["mdd_mid"])*8, color=C_NDX,
           alpha=0.7, edgecolors="black", linewidths=2, zorder=5)
ax.annotate("NDX (QQQ)\n波动 22.5% | 5Y 年化 22.9% | 回撤 -34%",
            xy=(NDX["vol_mid"], NDX["ret_5y_ann"]), xytext=(NDX["vol_mid"]+3, NDX["ret_5y_ann"]+3),
            fontsize=10, fontweight="bold", color=C_NDX,
            arrowprops=dict(arrowstyle="->", color=C_NDX))

# BTC
ax.scatter(BTC["vol_mid"], btc_5y_ann_mid, s=abs(BTC["mdd_mid"])*8, color=C_BTC,
           alpha=0.7, edgecolors="black", linewidths=2, zorder=5)
ax.annotate("BTC\n波动 50% | 5Y 年化 ~42% | 回撤 -51%",
            xy=(BTC["vol_mid"], btc_5y_ann_mid), xytext=(BTC["vol_mid"]-15, btc_5y_ann_mid+5),
            fontsize=10, fontweight="bold", color=C_BTC,
            arrowprops=dict(arrowstyle="->", color=C_BTC))

# 无差异线（45 度，波动率=收益）
ax.plot([0, 70], [0, 70], ":", color=C_GRAY, linewidth=1, label="波动率 = 收益（无差异线）")

ax.set_xlabel("年化波动率 (%)", fontsize=12)
ax.set_ylabel("5Y 年化回报 (%)", fontsize=12)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.set_xlim(0, 70)
ax.set_ylim(0, 60)
ax.grid(alpha=0.3)
ax.legend(fontsize=10)
ax.text(0.5, -0.15, "气泡大小 = 最大回撤幅度；BTC 位于右上方（高波动、高收益），NDX 位于左下方（低波动、中收益）",
        transform=ax.transAxes, ha="center", fontsize=9, color=C_GRAY)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart5_scatter.png"), dpi=150, bbox_inches="tight")
plt.close()
print("图 5 已保存: chart5_scatter.png")

print("\n全部 5 张图表生成完毕。")
