"""
Step 3/4 — 生成英伟达回购分析可视化图表
基于 Step 1/2 检索数据（AS_OF: 股价截至 2026-10-06 收盘）
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 输出目录
OUT_DIR = r"E:\agent_dev\multi-agent\tmp"

# ============================================================
# 数据（来自 Step 1/2 检索，AS_OF: 2026-10-06 收盘）
# ============================================================

# 股价数据（2026-09-17 至 2026-10-06）
stock_data = {
    '2026-09-17': 219.34,
    '2026-09-18': 222.27,
    '2026-09-21': 227.38,
    '2026-09-22': 228.87,
    '2026-09-23': 225.51,
    '2026-09-24': 224.58,
    '2026-09-25': 225.07,
    '2026-09-28': 228.86,  # 回购公告日
    '2026-09-29': 227.21,
    '2026-09-30': 228.38,
    '2026-10-01': 230.86,
    '2026-10-02': 233.95,
    '2026-10-05': 238.90,
    '2026-10-06': 239.24,
}

# 回购历史（财年）
buyback_fy = {
    'FY2025': 34.0,
    'FY2026': 40.4,
    'FY2027 H1': 39.8,
}

# 回购授权历史
auth_timeline = {
    '2026-04-26': 38.5,   # FY2027 Q1 末剩余授权
    '2026-05-18': 118.5,  # +$80B 后
    '2026-09-28': 235.0,  # +$150B 后（总剩余）
}

# 市值历史（年末）
market_cap_history = {
    '2019': 144.0,
    '2020': 323.2,
    '2021': 735.9,
    '2022': 359.5,
    '2023': 1220.0,
    '2024': 3290.0,
    '2025': 4530.0,
    '2026': 5770.0,  # 2026-10-06
}

# 现金流数据
cash_flow = {
    'FY2027 H1 经营现金流': 74.4,
    'FY2027 H1 自由现金流': 70.0,
    'FY2027 H1 股东回报': 46.1,
    '现金及等价物 (2026-07)': 56.6,
}

# ============================================================
# 图 1: 股价走势（回购公告前后）
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
dates = [datetime.strptime(d, '%Y-%m-%d') for d in stock_data.keys()]
prices = list(stock_data.values())

ax.plot(dates, prices, 'b-o', linewidth=2, markersize=6, label='NVDA 收盘价')
ax.axvline(x=datetime(2026, 9, 28), color='red', linestyle='--', linewidth=2, label='回购公告日 (2026-09-28)')
ax.fill_between(dates, prices, alpha=0.1, color='blue')

# 标注关键价格
ax.annotate(f'公告日: $228.86', xy=(datetime(2026, 9, 28), 228.86),
            xytext=(datetime(2026, 9, 20), 235),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red')
ax.annotate(f'最新: $239.24 (+4.5%)', xy=(datetime(2026, 10, 6), 239.24),
            xytext=(datetime(2026, 9, 25), 242),
            arrowprops=dict(arrowstyle='->', color='green'),
            fontsize=10, color='green')

ax.set_title('英伟达 (NVDA) 股价走势 — 回购公告前后 (AS_OF: 2026-10-06)', fontsize=14, fontweight='bold')
ax.set_xlabel('日期', fontsize=12)
ax.set_ylabel('收盘价 (USD)', fontsize=12)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'chart3_1_stock_price.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ chart3_1_stock_price.png 已生成")

# ============================================================
# 图 2: 回购规模对比（财年）
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
fy_labels = list(buyback_fy.keys())
fy_values = list(buyback_fy.values())
colors = ['#4472C4', '#ED7D31', '#70AD47']

bars = ax.bar(fy_labels, fy_values, color=colors, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, fy_values):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'${val}B', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_title('英伟达财年回购规模 (AS_OF: 2026-10-06)', fontsize=14, fontweight='bold')
ax.set_xlabel('财年', fontsize=12)
ax.set_ylabel('回购金额 (十亿美元)', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'chart3_2_buyback_fy.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ chart3_2_buyback_fy.png 已生成")

# ============================================================
# 图 3: 回购授权增长时间线
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
auth_dates = [datetime.strptime(d, '%Y-%m-%d') for d in auth_timeline.keys()]
auth_values = list(auth_timeline.values())

ax.plot(auth_dates, auth_values, 'g-o', linewidth=3, markersize=10, label='剩余授权 (十亿美元)')
ax.fill_between(auth_dates, auth_values, alpha=0.2, color='green')

# 标注关键节点
for date, val in zip(auth_dates, auth_values):
    ax.annotate(f'${val}B', xy=(date, val), xytext=(0, 10),
                textcoords='offset points', ha='center', fontsize=11, fontweight='bold')

ax.axvline(x=datetime(2026, 9, 28), color='red', linestyle='--', linewidth=2, label='+$150B 公告 (2026-09-28)')
ax.axvline(x=datetime(2026, 5, 18), color='orange', linestyle='--', linewidth=2, label='+$80B 公告 (2026-05-18)')

ax.set_title('英伟达回购授权增长时间线 (AS_OF: 2026-10-06)', fontsize=14, fontweight='bold')
ax.set_xlabel('日期', fontsize=12)
ax.set_ylabel('剩余授权 (十亿美元)', fontsize=12)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'chart3_3_auth_timeline.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ chart3_3_auth_timeline.png 已生成")

# ============================================================
# 图 4: 市值增长历史
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
years = list(market_cap_history.keys())
caps = list(market_cap_history.values())

ax.plot(years, caps, 'b-o', linewidth=2, markersize=8, label='市值 (十亿美元)')
ax.fill_between(years, caps, alpha=0.1, color='blue')

for x, y in zip(years, caps):
    ax.annotate(f'${y}B', xy=(x, y), xytext=(0, 10),
                textcoords='offset points', ha='center', fontsize=9)

ax.set_title('英伟达市值增长历史 (AS_OF: 2026-10-06)', fontsize=14, fontweight='bold')
ax.set_xlabel('年份', fontsize=12)
ax.set_ylabel('市值 (十亿美元)', fontsize=12)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'chart3_4_market_cap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ chart3_4_market_cap.png 已生成")

# ============================================================
# 图 5: 现金流支撑（FY2027 H1）
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
cf_labels = list(cash_flow.keys())
cf_values = list(cash_flow.values())
colors = ['#4472C4', '#ED7D31', '#70AD47', '#FFC000']

bars = ax.barh(cf_labels, cf_values, color=colors, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, cf_values):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2.,
            f'${val}B', ha='left', va='center', fontsize=11, fontweight='bold')

ax.set_title('英伟达 FY2027 H1 现金流与股东回报 (AS_OF: 2026-10-06)', fontsize=14, fontweight='bold')
ax.set_xlabel('金额 (十亿美元)', fontsize=12)
ax.set_ylabel('指标', fontsize=12)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'chart3_5_cash_flow.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ chart3_5_cash_flow.png 已生成")

# ============================================================
# 图 6: 回购规模 vs 市值（占比）
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
categories = ['新增授权 $150B', '总剩余授权 $235B', 'FY2027 H1 回购 $39.8B']
market_cap = 5770.0  # 十亿美元
percentages = [150/market_cap*100, 235/market_cap*100, 39.8/market_cap*100]
colors = ['#4472C4', '#ED7D31', '#70AD47']

bars = ax.bar(categories, percentages, color=colors, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, percentages):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_title('英伟达回购规模 vs 市值占比 (AS_OF: 2026-10-06, 市值 $5.77T)', fontsize=14, fontweight='bold')
ax.set_xlabel('指标', fontsize=12)
ax.set_ylabel('占市值比例 (%)', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'chart3_6_buyback_vs_mcap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ chart3_6_buyback_vs_mcap.png 已生成")

print("\n✅ 所有 6 张图表已生成至:", OUT_DIR)
