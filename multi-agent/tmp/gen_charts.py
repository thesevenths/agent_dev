import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

def find_chinese_font():
    candidates = ['Microsoft YaHei','SimHei','Microsoft JhengHei','Noto Sans CJK SC','WenQuanYi Zen Hei','SimSun']
    available = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            return c
    for f in fm.fontManager.ttflist:
        if any(k in f.name for k in ['Hei','CJK','YaHei','SimSun','Song']):
            return f.name
    return None

font = find_chinese_font()
print("Using font:", font)
if font:
    plt.rcParams['font.family'] = font
plt.rcParams['axes.unicode_minus'] = False

out_dir = r"E:\agent_dev\multi-agent\tmp"

# Chart 1: index daily change
fig, ax = plt.subplots(figsize=(9,5.5))
idx = ['上证指数','深证成指','创业板指','科创50','北证50']
chg = [0.31, -0.11, -0.23, -2.51, 0.70]
colors = ['#d62728' if c>=0 else '#2ca02c' for c in chg]
bars = ax.bar(idx, chg, color=colors)
ax.axhline(0, color='gray', lw=0.8)
for b,c in zip(bars,chg):
    ax.text(b.get_x()+b.get_width()/2, c + (0.08 if c>=0 else -0.18), f'{c:+.2f}%', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('涨跌幅 (%)')
ax.set_title('2026-09-30 A股主要指数当日涨跌幅（收盘）', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir,'chart1_index_change.png'), dpi=110)
plt.close()

# Chart 2: sector performance
fig, ax = plt.subplots(figsize=(9,5.5))
sectors = ['医药生物','美容护理','食品饮料','银行','钢铁','农林牧渔','通信','机械设备','计算机','电子']
s_chg = [2.73,1.70,1.68,1.47,1.24,1.02,-0.62,-0.90,-0.98,-2.38]
s_colors = ['#d62728' if c>=0 else '#2ca02c' for c in s_chg]
bars = ax.barh(sectors[::-1], s_chg[::-1], color=s_colors[::-1])
ax.axvline(0, color='gray', lw=0.8)
for b,c in zip(bars, s_chg[::-1]):
    ax.text(c + (0.05 if c>=0 else -0.05), b.get_y()+b.get_height()/2, f'{c:+.2f}%', va='center', ha='left' if c>=0 else 'right', fontsize=10)
ax.set_xlabel('涨跌幅 (%)')
ax.set_title('2026-09-30 申万行业涨跌幅（领涨/领跌）', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir,'chart2_sector.png'), dpi=110)
plt.close()

# Chart 3: historical post-holiday probability
fig, ax = plt.subplots(figsize=(9,5.5))
labels = ['节后首日上涨\n(近十年2016-2025)','节后5日正收益\n(近十年)','节后一周上涨\n(中信建投,剔除18/24)']
probs = [70, 60, 62.5]
bars = ax.bar(labels, probs, color=['#d62728','#ff7f0e','#1f77b4'])
ax.axhline(50, color='gray', ls='--', lw=1)
ax.text(2.4, 50.5, '50% 中性线', fontsize=9, color='gray')
for b,p in zip(bars,probs):
    ax.text(b.get_x()+b.get_width()/2, p+1.5, f'{p}%', ha='center', fontsize=12, fontweight='bold')
ax.set_ylabel('上涨概率 (%)')
ax.set_ylim(0,80)
ax.set_title('国庆后首个交易日 A股上涨概率（历史统计）', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir,'chart3_probability.png'), dpi=110)
plt.close()

print("Charts saved:", [f for f in os.listdir(out_dir) if f.startswith('chart')])
