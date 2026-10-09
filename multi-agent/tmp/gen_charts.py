import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
out_dir = r'E:\agent_dev\multi-agent\tmp\charts'
os.makedirs(out_dir, exist_ok=True)

# ============ Chart 1: ML algorithm evolution ============
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
stages = [
    ("计量经济学\nEconometric", "线性/统计模型\n(公司名 XTX 由来)", "#4C72B0"),
    ("树模型\nTrees (GBDT)", "XGBoost / LightGBM\n表格数据强基线", "#55A868"),
    ("神经网络\nNeural Nets", "LSTM / MLP\n时序建模", "#8172B2"),
    ("现代深度学习\nDeep Learning", "大规模 GPU 训练\n25,000+ GPU", "#CCB974"),
    ("基础模型\nFoundation Models", "金融时序预训练\nXTX AI Lab 方向", "#C44E52"),
]
n = len(stages); bw = 1.55
for i, (title, sub, color) in enumerate(stages):
    x = 0.4 + i*1.95
    box = FancyBboxPatch((x, 2.2), bw, 2.6, boxstyle="round,pad=0.08",
                         linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.9)
    ax.add_patch(box)
    ax.text(x+bw/2, 3.9, title, ha='center', va='center', fontsize=10.5, color='white', fontweight='bold')
    ax.text(x+bw/2, 2.9, sub, ha='center', va='center', fontsize=8.5, color='white')
    if i < n-1:
        ax.annotate('', xy=(x+bw+0.32, 3.5), xytext=(x+bw+0.02, 3.5),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=2))
ax.text(5, 5.4, "XTX Markets 技术栈演进路线（公开信息）", ha='center', fontsize=14, fontweight='bold')
ax.text(5, 1.3, "演进方向：从可解释统计模型 → 数据驱动深度模型 → 大规模基础模型\n核心逻辑：用更多算力 + 更多数据，换取更精细的价格预测",
        ha='center', fontsize=9.5, color='#444', style='italic')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'chart1_algo_evolution.png'), dpi=130, bbox_inches='tight')
plt.close()

# ============ Chart 2: Strategy architecture ============
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis('off')
def box(x, y, w, h, title, sub, color, fs=10):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                       linewidth=1.6, edgecolor=color, facecolor=color, alpha=0.92)
    ax.add_patch(b)
    ax.text(x+w/2, y+h*0.68, title, ha='center', va='center', fontsize=fs, color='white', fontweight='bold')
    ax.text(x+w/2, y+h*0.3, sub, ha='center', va='center', fontsize=8, color='white')
box(0.3, 4.6, 2.6, 1.8, "数据层", "订单簿 / tick / 情绪\n1万亿+ 数据点/日\n1EB 存储", "#4C72B0")
box(3.6, 4.6, 2.6, 1.8, "模型层", "ML 价格预测\n53,000+ 工具\n25,000+ GPU", "#55A868")
box(6.9, 4.6, 2.6, 1.8, "执行层", "低延迟执行\n滑点/成本最小化\n数百万笔/日", "#8172B2")
box(10.0, 4.6, 1.7, 1.8, "风控层", "波动率目标\n仓位管理", "#CCB974")
for x0 in [2.9, 6.2, 9.5]:
    ax.annotate('', xy=(x0+0.7, 5.5), xytext=(x0, 5.5),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2.2))
box(3.6, 1.6, 4.7, 1.6, "输出：价格预测 (Price Forecasts)", "覆盖 股票 / 固收 / 外汇 / 商品 / 加密\n用于自营交易 + 向机构客户提供流动性", "#C44E52")
ax.annotate('', xy=(4.9, 3.2), xytext=(4.9, 4.6), arrowprops=dict(arrowstyle='->', color='#333', lw=2.2))
ax.annotate('', xy=(8.2, 3.2), xytext=(8.2, 4.6), arrowprops=dict(arrowstyle='->', color='#333', lw=2.2))
box(0.3, 0.2, 11.4, 1.0, "", "", "#333333")
ax.text(6, 0.7, "盈利来源：极弱信号(51-52%准确率) × 海量下注(数百万笔) × 复利  →  检测微小价差累积利润",
        ha='center', va='center', fontsize=10.5, color='white', fontweight='bold')
ax.annotate('', xy=(6, 1.6), xytext=(6, 3.2), arrowprops=dict(arrowstyle='->', color='#333', lw=2.2))
ax.text(6, 6.6, "XTX Markets 策略架构示意图（基于公开信息重构）", ha='center', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'chart2_architecture.png'), dpi=130, bbox_inches='tight')
plt.close()

# ============ Chart 3: Profitability logic - weak signal x volume ============
fig, ax = plt.subplots(figsize=(10, 5.5))
# Simulate: 52% accuracy, many trades, compounding edge
np.random.seed(42)
n_trades = 10000
p = 0.52
wins = np.random.binomial(1, p, n_trades)
# each win +1 unit, each loss -1 unit (simplified)
pnl = np.cumsum(np.where(wins==1, 1, -1))
ax.plot(pnl, color='#55A868', lw=1.2, label='52% 准确率（略偏对）')
# random 50% baseline
np.random.seed(7)
wins50 = np.random.binomial(1, 0.5, n_trades)
pnl50 = np.cumsum(np.where(wins50==1, 1, -1))
ax.plot(pnl50, color='#999999', lw=1.0, alpha=0.7, label='50% 准确率（纯随机）')
ax.axhline(0, color='#333', lw=0.8, ls='--')
ax.set_xlabel('交易笔数（累计）', fontsize=11)
ax.set_ylabel('累计盈亏（单位）', fontsize=11)
ax.set_title('极弱信号 × 海量下注 × 复利：52% vs 50% 的长期分化', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.text(0.02, 0.95, "单笔优势极小，但 10,000 笔后\n统计优势显著累积（示意，非真实回测）",
        transform=ax.transAxes, fontsize=9, va='top', color='#444',
        bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8))
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'chart3_weak_signal.png'), dpi=130, bbox_inches='tight')
plt.close()

# ============ Chart 4: Feasibility - what retail can/cannot replicate ============
fig, ax = plt.subplots(figsize=(11, 6))
ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis('off')
# Two columns
ax.add_patch(FancyBboxPatch((0.4, 0.4), 4.3, 6.6, boxstyle="round,pad=0.1",
             linewidth=1.6, edgecolor='#55A868', facecolor='#eaf5ec'))
ax.add_patch(FancyBboxPatch((5.3, 0.4), 4.3, 6.6, boxstyle="round,pad=0.1",
             linewidth=1.6, edgecolor='#C44E52', facecolor='#fbeaea'))
ax.text(2.55, 6.6, "个人【能】模仿", ha='center', fontsize=13, fontweight='bold', color='#2e7d32')
ax.text(7.45, 6.6, "个人【难/无法】模仿", ha='center', fontsize=13, fontweight='bold', color='#b71c1c')
can = ["算法本身\n(XGBoost/LSTM/\nTransformer/ensemble)",
       "方法论\n(因子选择/训练/\n系统化执行)",
       "波动率目标化\n与仓位管理",
       "模型组合\n(portfolio of models)",
       "学习路径\n(Ernie Chan 系列)"]
cannot = ["数据规模\n(1EB 存储/原始\n订单流)",
          "算力规模\n(25,000+ GPU/\n€1bn 数据中心)",
          "执行/成本\n(低延迟/滑点/\n交易成本)",
          "做空/杠杆\n(Reg SHO/保证金/\n借券限制)",
          "市场中性能力\n(个人多只能\nlong-only)"]
for i, t in enumerate(can):
    ax.text(2.55, 5.7 - i*1.15, "[可]  " + t, ha='center', va='center', fontsize=9.5, color='#1b5e20')
for i, t in enumerate(cannot):
    ax.text(7.45, 5.7 - i*1.15, "[难]  " + t, ha='center', va='center', fontsize=9.5, color='#7f1d1d')
ax.text(5, 0.15, "结论：差距是结构性的（数据+算力+执行+成本），而非智力/算法",
        ha='center', fontsize=11, fontweight='bold', color='#333')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'chart4_feasibility.png'), dpi=130, bbox_inches='tight')
plt.close()

print("ALL CHARTS DONE")
for f in sorted(os.listdir(out_dir)):
    print(f, os.path.getsize(os.path.join(out_dir, f)), "bytes")
