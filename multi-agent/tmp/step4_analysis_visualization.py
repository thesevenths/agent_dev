"""
Step 4/6: 数据分析与可视化
基于 Step 1-3 的数据，生成纳斯达克100资金流向分析图表

AS_OF: 2026-10-07 20:42
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Set font for Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

output_dir = r"F:\agent\multi-agent\tmp"

# ============================================================
# CHART 1: QQQ & QQQM Multi-Window Net Flows
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))

windows = ['5日', '1个月', '3个月', '6个月', '1年', '3年', '5年', '10年']
qqq_flows = [0.9005, -6.5, 10.68, 21.52, 22.59, 66.04, 82.02, 121.82]
qqqm_flows = [0.23011, 2.61, 9.21, 20.19, 28.64, 59.62, 70.85, 72.26]

x = np.arange(len(windows))
width = 0.35

bars1 = ax.bar(x - width/2, qqq_flows, width, label='QQQ (Invesco QQQ Trust)', color='#2196F3', alpha=0.85)
bars2 = ax.bar(x + width/2, qqqm_flows, width, label='QQQM (Invesco NASDAQ 100 ETF)', color='#FF9800', alpha=0.85)

for bar in bars1:
    h = bar.get_height()
    ax.annotate(f'${h:.1f}B' if abs(h) >= 1 else f'${h*1000:.0f}M',
                xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 3 if h >= 0 else -12), textcoords="offset points",
                ha='center', va='bottom' if h >= 0 else 'top', fontsize=9, fontweight='bold')
for bar in bars2:
    h = bar.get_height()
    ax.annotate(f'${h:.1f}B' if abs(h) >= 1 else f'${h*1000:.0f}M',
                xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 3 if h >= 0 else -12), textcoords="offset points",
                ha='center', va='bottom' if h >= 0 else 'top', fontsize=9, fontweight='bold')

ax.set_xlabel('时间窗口', fontsize=12)
ax.set_ylabel('净资金流 (十亿美元)', fontsize=12)
ax.set_title('QQQ & QQQM 多窗口净资金流向\n(数据截至 2026-10-05)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(windows, fontsize=11)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.legend(fontsize=11, loc='upper left')
ax.grid(axis='y', alpha=0.3)

ax.annotate('⚠ 1个月净流出\n$6.5B', xy=(1 - width/2, -6.5), xytext=(2.5, -12),
            arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, color='red', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'chart1_qqq_qqqm_flows.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Chart 1 saved.")

# ============================================================
# CHART 2: ETF Industry Macro Context (H1 2026)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

categories = ['股票ETF', '固定收益ETF', '商品/另类']
h1_2026 = [680, 292, -12.4]
h1_2025 = [340, 150, 79.5]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, h1_2026, width, label='H1 2026', color='#4CAF50', alpha=0.85)
bars2 = ax.bar(x + width/2, h1_2025, width, label='H1 2025 (估算)', color='#9E9E9E', alpha=0.7)

for bar in bars1:
    h = bar.get_height()
    ax.annotate(f'${h:.0f}B', xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar in bars2:
    h = bar.get_height()
    ax.annotate(f'${h:.0f}B', xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

ax.set_xlabel('ETF类别', fontsize=12)
ax.set_ylabel('净资金流 (十亿美元)', fontsize=12)
ax.set_title('2026 H1 ETF行业净资金流 vs 2025 H1\n(总流入 $1.0T，创历史纪录，+86% YoY)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'chart2_etf_industry_h1_2026.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Chart 2 saved.")

# ============================================================
# CHART 3: Mag 7 Institutional Sentiment (Q2 2026 13F)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))

stocks = ['GOOGL', 'NVDA', 'AAPL', 'META', 'AMZN', 'MSFT']
sentiment = [3, 2, 1, 2, 1, -1]
colors = ['#4CAF50' if s > 0 else '#F44336' if s < 0 else '#9E9E9E' for s in sentiment]

bars = ax.barh(stocks, sentiment, color=colors, alpha=0.85, height=0.6)

annotations = {
    'GOOGL': 'Berkshire 7倍加仓\nThird Point/TCI 新开',
    'NVDA': 'BlackRock 第一大持仓\nCoatue/Appaloosa 加仓',
    'AAPL': 'BlackRock 5.0% 权重\nDruckenmiller 卖出',
    'META': 'Third Point 新开\nBlackRock 1.5%',
    'AMZN': 'Coatue +49%\nBridgewater 大幅减持',
    'MSFT': 'Druckenmiller/TCI 退出\nMS/BNP/UBS 降级'
}

for i, (stock, s) in enumerate(zip(stocks, sentiment)):
    x_pos = s + 0.15 if s >= 0 else s - 0.15
    ha = 'left' if s >= 0 else 'right'
    ax.annotate(annotations[stock], xy=(s, i), xytext=(x_pos, i),
                textcoords='data', ha=ha, va='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

ax.set_xlabel('机构情绪得分 (+买入 / -卖出)', fontsize=12)
ax.set_title('Mag 7 机构持仓情绪 (Q2 2026 13F)\n数据截至 2026-06-30', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlim(-3, 5)
ax.grid(axis='x', alpha=0.3)

green_patch = mpatches.Patch(color='#4CAF50', label='偏多')
red_patch = mpatches.Patch(color='#F44336', label='偏空')
ax.legend(handles=[green_patch, red_patch], fontsize=11, loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'chart3_mag7_institutional_sentiment.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Chart 3 saved.")

# ============================================================
# CHART 4: AI Infrastructure Weight by Fund
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

funds = ['Coatue', 'Appaloosa', 'Atreides', 'Soros', 'Duquesne', 'Baillie Gifford', 'H&H (段永平)', 'Berkshire', 'Pershing Square', 'Himalaya (李录)']
weights = [68, 44, 28, 19, 17, 8, 8, 0, 0, 0]
colors = ['#F44336' if w >= 40 else '#FF9800' if w >= 15 else '#4CAF50' if w > 0 else '#9E9E9E' for w in weights]

bars = ax.barh(funds, weights, color=colors, alpha=0.85, height=0.6)

for i, w in enumerate(weights):
    ax.annotate(f'{w}%', xy=(w, i), xytext=(5, 0), textcoords='offset points',
                ha='left', va='center', fontsize=10, fontweight='bold')

ax.set_xlabel('AI基础设施持仓权重 (%)', fontsize=12)
ax.set_title('Q2 2026 主要基金 AI 基础设施持仓权重\n(半导体/数据中心/电力，合计 $536B 美股持仓)', fontsize=13, fontweight='bold')
ax.set_xlim(0, 80)
ax.grid(axis='x', alpha=0.3)

red_patch = mpatches.Patch(color='#F44336', label='激进 (≥40%)')
orange_patch = mpatches.Patch(color='#FF9800', label='中等 (15-40%)')
green_patch = mpatches.Patch(color='#4CAF50', label='保守 (<15%)')
gray_patch = mpatches.Patch(color='#9E9E9E', label='不参与 (0%)')
ax.legend(handles=[red_patch, orange_patch, green_patch, gray_patch], fontsize=10, loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'chart4_ai_infra_weight_by_fund.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Chart 4 saved.")

# ============================================================
# CHART 5: Key Events Timeline 2026
# ============================================================
fig, ax = plt.subplots(figsize=(14, 8))

events = [
    ("2026-01", "ETF行业Q1净流入超$500B\n主动管理ETF $245B (+70% YoY)", 'bull'),
    ("2026-04-06", "BlackRock (BLK) 提交\nNasdaq 100 ETF申请", 'neutral'),
    ("2026-04-07", "State Street (STT) 提交\nNasdaq 100 ETF申请", 'neutral'),
    ("2026-05-01", "Nasdaq-100 更新方法论\n(Fast Entry + 季度再平衡)", 'bull'),
    ("2026-06-12", "SpaceX IPO\n($135/股, ~$75B)", 'bull'),
    ("2026-06-17", "ETF行业YTD净流入\n突破 $1.0万亿", 'bull'),
    ("2026-07-07", "SpaceX 纳入\nNasdaq-100/QQQ\n(权重~1.25-1.35%)", 'bull'),
    ("2026-07-底", "QQQ 单日$5.7B\n创纪录流出", 'bear'),
    ("2026-09-16", "Fed 加息25bp\n10Y美债破5%", 'bear'),
    ("2026-10-05", "MSFT 多家投行降级\n(MS/BNP/UBS)", 'bear'),
    ("2026-10-28", "下次FOMC会议\n(当前利率4.00%)", 'neutral'),
]

y_positions = np.arange(len(events))[::-1]
for i, (date, desc, sentiment) in enumerate(events):
    y = y_positions[i]
    color = '#4CAF50' if sentiment == 'bull' else '#F44336' if sentiment == 'bear' else '#2196F3'
    ax.scatter(0, y, color=color, s=200, zorder=5, edgecolors='white', linewidths=2)
    ax.annotate(f'{date}\n{desc}', xy=(0, y), xytext=(0.02, y),
                textcoords='data', ha='left', va='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.15, edgecolor=color))

ax.set_xlim(-0.05, 0.6)
ax.set_ylim(-0.5, len(events) - 0.5)
ax.set_yticks([])
ax.set_xticks([])
ax.set_title('2026年纳斯达克100关键事件时间线', fontsize=14, fontweight='bold', pad=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.axvline(x=0, color='gray', linewidth=2, alpha=0.5)

green_patch = mpatches.Patch(color='#4CAF50', label='利好/流入')
red_patch = mpatches.Patch(color='#F44336', label='利空/流出')
blue_patch = mpatches.Patch(color='#2196F3', label='中性/结构性')
ax.legend(handles=[green_patch, red_patch, blue_patch], fontsize=11, loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'chart5_2026_timeline.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Chart 5 saved.")

# ============================================================
# CHART 6: Mag 7 Performance 2026 (as of 7/13)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

mag7 = ['AMAT', 'LRCX', 'AAPL', 'NVDA', 'GOOGL', 'META', 'AMZN', 'MSFT']
perf = [124.3, 93.0, 28.8, 15.0, 12.0, 8.0, 5.0, -20.4]
colors = ['#4CAF50' if p > 0 else '#F44336' for p in perf]

bars = ax.barh(mag7, perf, color=colors, alpha=0.85, height=0.6)

for i, p in enumerate(perf):
    x_pos = p + 2 if p >= 0 else p - 2
    ha = 'left' if p >= 0 else 'right'
    ax.annotate(f'{p:+.1f}%', xy=(p, i), xytext=(x_pos, i),
                textcoords='data', ha=ha, va='center', fontsize=10, fontweight='bold')

ax.set_xlabel('2026年涨跌幅 (%) (截至 2026-07-13)', fontsize=12)
ax.set_title('Mag 7 及半导体股 2026 年表现\n(数据源: Morningstar, 截至 2026-07-13)', fontsize=13, fontweight='bold')
ax.axvline(x=0, color='black', linewidth=0.8)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'chart6_mag7_performance_2026.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Chart 6 saved.")

print("\nAll 6 charts generated successfully!")
print(f"Output directory: {output_dir}")
