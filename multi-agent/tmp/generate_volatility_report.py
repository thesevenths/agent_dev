"""
Step 3/3: 生成波动率交易策略报告
- 三种波动率对比折线图
- 买卖信号示例数值计算表
- 业务原因说明
- 是否属于量化投资思路的讨论
- 中文 Markdown 报告，含 AS_OF 与数据溯源声明
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 加载数据
# ============================================================

print("=" * 80)
print("Step 3/3: 生成波动率交易策略报告")
print("=" * 80)

# 加载波动率指标数据
vol_df = pd.read_csv(r'E:\agent_dev\multi-agent\tmp\volatility_metrics.csv', parse_dates=['date'])
print(f"波动率指标数据加载完成: {len(vol_df)} 条记录")

# 分离 QQQ 和 BTC 数据
qqq_vol = vol_df[vol_df['asset'] == 'QQQ'].copy()
btc_vol = vol_df[vol_df['asset'] == 'BTC-USD'].copy()

print(f"QQQ 数据: {len(qqq_vol)} 条")
print(f"BTC 数据: {len(btc_vol)} 条")

# ============================================================
# 2. 生成三种波动率对比折线图
# ============================================================

print("\n生成三种波动率对比折线图...")

# 创建图形
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

# QQQ 波动率对比
ax1 = axes[0]
ax1.plot(qqq_vol['date'], qqq_vol['rv_20d'], label='20日已实现波动率 (RV)', color='#1f77b4', linewidth=1.5)
ax1.plot(qqq_vol['date'], qqq_vol['rv_60d'], label='60日已实现波动率 (RV)', color='#ff7f0e', linewidth=1.5)
ax1.plot(qqq_vol['date'], qqq_vol['garch_annualized_vol'], label='GARCH(1,1) 年化波动率', color='#2ca02c', linewidth=1.5, linestyle='--')
ax1.axhline(y=0.20, color='#d62728', linestyle=':', linewidth=2, label='假设 IV = 20%')
ax1.set_title('QQQ 三种波动率指标对比 (Illustrative)', fontsize=14, fontweight='bold')
ax1.set_ylabel('年化波动率', fontsize=12)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

# BTC 波动率对比
ax2 = axes[1]
ax2.plot(btc_vol['date'], btc_vol['rv_20d'], label='20日已实现波动率 (RV)', color='#1f77b4', linewidth=1.5)
ax2.plot(btc_vol['date'], btc_vol['rv_60d'], label='60日已实现波动率 (RV)', color='#ff7f0e', linewidth=1.5)
ax2.plot(btc_vol['date'], btc_vol['garch_annualized_vol'], label='GARCH(1,1) 年化波动率', color='#2ca02c', linewidth=1.5, linestyle='--')
ax2.axhline(y=0.50, color='#d62728', linestyle=':', linewidth=2, label='假设 IV = 50%')
ax2.set_title('BTC-USD 三种波动率指标对比 (Illustrative)', fontsize=14, fontweight='bold')
ax2.set_ylabel('年化波动率', fontsize=12)
ax2.set_xlabel('日期', fontsize=12)
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

plt.tight_layout()
chart_path = r'E:\agent_dev\multi-agent\tmp\volatility_comparison_chart.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"图表已保存: {chart_path}")

# ============================================================
# 3. 生成买卖信号示例
# ============================================================

print("\n生成买卖信号示例...")

# 定义信号规则
def generate_signals(df, asset_name):
    """
    基于波动率指标生成买卖信号
    规则:
    1. RV > 阈值 (如 30%): 高波动，考虑减仓或做空
    2. IV > RV: 市场预期波动大于实际波动，做多波动率 (买入期权)
    3. GARCH 预测波动率上升: 预期波动加剧，谨慎操作
    """
    df = df.copy()
    
    # 信号 1: RV 阈值信号
    rv_threshold = 0.30  # 30% 年化波动率
    df['signal_rv_high'] = df['rv_20d'] > rv_threshold
    df['signal_rv_low'] = df['rv_20d'] < 0.15  # 15% 低波动
    
    # 信号 2: IV vs RV 信号
    df['iv_rv_spread'] = df['iv_assumed'] - df['rv_20d']
    df['signal_iv_high'] = df['iv_rv_spread'] > 0.10  # IV 比 RV 高 10 个百分点
    
    # 信号 3: GARCH 波动率变化
    df['garch_vol_change'] = df['garch_annualized_vol'].diff()
    df['signal_garch_rising'] = df['garch_vol_change'] > 0.02  # GARCH 波动率上升 2 个百分点
    
    # 综合信号
    df['composite_signal'] = 'NEUTRAL'
    df.loc[df['signal_rv_high'], 'composite_signal'] = 'REDUCE'  # 高波动减仓
    df.loc[df['signal_rv_low'], 'composite_signal'] = 'INCREASE'  # 低波动加仓
    df.loc[df['signal_iv_high'], 'composite_signal'] = 'BUY_VOL'  # 做多波动率
    
    return df

# 生成 QQQ 信号
qqq_signals = generate_signals(qqq_vol, 'QQQ')

# 生成 BTC 信号
btc_signals = generate_signals(btc_vol, 'BTC-USD')

# 提取最新信号
qqq_latest_signal = qqq_signals.iloc[-1]
btc_latest_signal = btc_signals.iloc[-1]

print(f"\nQQQ 最新信号 (截至 {qqq_latest_signal['date'].date()}):")
print(f"  20日 RV: {qqq_latest_signal['rv_20d']:.4f} ({qqq_latest_signal['rv_20d']*100:.2f}%)")
print(f"  假设 IV: {qqq_latest_signal['iv_assumed']:.4f} ({qqq_latest_signal['iv_assumed']*100:.2f}%)")
print(f"  IV-RV 价差: {qqq_latest_signal['iv_rv_spread']:.4f}")
print(f"  综合信号: {qqq_latest_signal['composite_signal']}")

print(f"\nBTC 最新信号 (截至 {btc_latest_signal['date'].date()}):")
print(f"  20日 RV: {btc_latest_signal['rv_20d']:.4f} ({btc_latest_signal['rv_20d']*100:.2f}%)")
print(f"  假设 IV: {btc_latest_signal['iv_assumed']:.4f} ({btc_latest_signal['iv_assumed']*100:.2f}%)")
print(f"  IV-RV 价差: {btc_latest_signal['iv_rv_spread']:.4f}")
print(f"  综合信号: {btc_latest_signal['composite_signal']}")

# 统计信号分布
print(f"\nQQQ 信号分布:")
print(qqq_signals['composite_signal'].value_counts())

print(f"\nBTC 信号分布:")
print(btc_signals['composite_signal'].value_counts())

# ============================================================
# 4. 生成 Markdown 报告
# ============================================================

print("\n生成 Markdown 报告...")

# 获取最新数据 (使用带信号的 DataFrame)
qqq_latest = qqq_signals.iloc[-1]
btc_latest = btc_signals.iloc[-1]

# 计算一些统计指标
qqq_rv_20d_mean = qqq_vol['rv_20d'].mean()
qqq_rv_20d_max = qqq_vol['rv_20d'].max()
qqq_rv_20d_min = qqq_vol['rv_20d'].min()

btc_rv_20d_mean = btc_vol['rv_20d'].mean()
btc_rv_20d_max = btc_vol['rv_20d'].max()
btc_rv_20d_min = btc_vol['rv_20d'].min()

# 生成报告内容
report_content = f"""# 波动率交易策略分析报告

**AS_OF: 2026-10-08 18:19**

## ⚠️ 重要声明

**数据性质**: 本报告基于 **Illustrative（示例性）** 数据生成，包含大量年度均值填充：
- **QQQ**: 仅 18 条真实日线数据 (2.5%)，705 条为年度平均股价填充
- **BTC**: 仅 51 条真实日线数据 (5%)，960 条为年度平均价格填充

**计算局限性**:
- 已实现波动率: 年度均值填充导致波动率严重低估，无法反映真实市场波动
- GARCH 模型: 基于填充数据拟合，参数估计不具统计显著性
- 隐含波动率: 使用假设值 (QQQ: 20%, BTC: 50%)，非市场实际值

**使用建议**:
- ✅ 适用于: 方法论演示、教学示例、策略框架展示
- ❌ 不适用于: 实际交易决策、回测验证、绩效评估

**数据溯源**:
- 源数据: `E:\\agent_dev\\multi-agent\\tmp\\step1_nasdaq_btc_historical_data.md`
- 日线数据: `E:\\agent_dev\\multi-agent\\tmp\\qqq_daily.csv`, `E:\\agent_dev\\multi-agent\\tmp\\btc_daily.csv`
- 波动率指标: `E:\\agent_dev\\multi-agent\\tmp\\volatility_metrics.csv`

---

## 1. 分析背景

本报告旨在演示如何使用三种波动率指标（已实现波动率、GARCH 模型、隐含波动率）指导纳斯达克 100 指数 (QQQ) 和比特币 (BTC-USD) 的买卖决策。

### 1.1 三种波动率指标简介

| 指标 | 定义 | 数据来源 | 特点 |
| :--- | :--- | :--- | :--- |
| **已实现波动率 (RV)** | 历史收益率的滚动标准差 × √252 年化 | 历史价格数据 | 反映过去实际波动，滞后指标 |
| **GARCH(1,1)** | 自回归条件异方差模型，预测条件方差 | 历史收益率序列 | 捕捉波动聚集效应，预测未来波动 |
| **隐含波动率 (IV)** | 期权价格反推的波动率 (Black-Scholes) | 期权市场价格 | 反映市场对未来波动的预期，前瞻指标 |

### 1.2 策略逻辑

1. **RV 阈值信号**: 当 20 日 RV 超过 30% 时，市场波动加剧，考虑减仓或对冲
2. **IV-RV 价差信号**: 当 IV 显著高于 RV 时，市场预期波动大于实际波动，可做多波动率 (买入期权)
3. **GARCH 预测信号**: 当 GARCH 预测波动率上升时，预期未来波动加剧，谨慎操作

---

## 2. 数据概览

### 2.1 数据范围

| 资产 | 日期范围 | 总行数 | 真实数据 | 填充数据 |
| :--- | :--- | :--- | :--- | :--- |
| **QQQ** | 2024-01-01 至 2026-10-07 | 723 | 18 (2.5%) | 705 (97.5%) |
| **BTC-USD** | 2024-01-01 至 2026-10-07 | 1011 | 51 (5.0%) | 960 (95.0%) |

### 2.2 最新数据 (截至 2026-10-07)

| 指标 | QQQ | BTC-USD |
| :--- | :--- | :--- |
| **收盘价** | {qqq_latest['close']:.2f} | {btc_latest['close']:.2f} |
| **20日已实现波动率** | {qqq_latest['rv_20d']:.4f} ({qqq_latest['rv_20d']*100:.2f}%) | {btc_latest['rv_20d']:.4f} ({btc_latest['rv_20d']*100:.2f}%) |
| **60日已实现波动率** | {qqq_latest['rv_60d']:.4f} ({qqq_latest['rv_60d']*100:.2f}%) | {btc_latest['rv_60d']:.4f} ({btc_latest['rv_60d']*100:.2f}%) |
| **GARCH 年化波动率** | {qqq_latest['garch_annualized_vol']:.4f} ({qqq_latest['garch_annualized_vol']*100:.2f}%) | {btc_latest['garch_annualized_vol']:.4f} ({btc_latest['garch_annualized_vol']*100:.2f}%) |
| **假设 IV** | {qqq_latest['iv_assumed']:.4f} ({qqq_latest['iv_assumed']*100:.2f}%) | {btc_latest['iv_assumed']:.4f} ({btc_latest['iv_assumed']*100:.2f}%) |

### 2.3 波动率统计

| 统计量 | QQQ 20日 RV | BTC 20日 RV |
| :--- | :--- | :--- |
| **均值** | {qqq_rv_20d_mean:.4f} ({qqq_rv_20d_mean*100:.2f}%) | {btc_rv_20d_mean:.4f} ({btc_rv_20d_mean*100:.2f}%) |
| **最大值** | {qqq_rv_20d_max:.4f} ({qqq_rv_20d_max*100:.2f}%) | {btc_rv_20d_max:.4f} ({btc_rv_20d_max*100:.2f}%) |
| **最小值** | {qqq_rv_20d_min:.4f} ({qqq_rv_20d_min*100:.2f}%) | {btc_rv_20d_min:.4f} ({btc_rv_20d_min*100:.2f}%) |

---

## 3. 三种波动率对比分析

### 3.1 波动率对比图表

![三种波动率对比](volatility_comparison_chart.png)

**图表说明**:
- **蓝色实线**: 20 日已实现波动率 (RV)，反映近期市场实际波动
- **橙色实线**: 60 日已实现波动率 (RV)，反映中期趋势性波动
- **绿色虚线**: GARCH(1,1) 年化波动率，基于模型预测的条件波动率
- **红色点线**: 假设隐含波动率 (IV)，QQQ 为 20%，BTC 为 50%

### 3.2 关键观察

#### QQQ
1. **RV 特征**: 由于年度均值填充，大部分时间 RV 接近 0%，仅在真实数据区间 (2025-12-24 至 2026-03-05) 出现波动
2. **GARCH 特征**: GARCH 波动率从初始值 20.40% 逐渐收敛至长期水平 11.78%，反映波动率均值回复特性
3. **IV 对比**: 假设 IV (20%) 高于 GARCH 长期波动率 (11.78%)，表明市场可能高估了 QQQ 的波动风险

#### BTC-USD
1. **RV 特征**: BTC 的 RV 波动较大，20 日 RV 在 0% 至 51.8% 之间波动，反映加密货币的高波动特性
2. **GARCH 特征**: GARCH 波动率从初始值 16.87% 逐渐收敛，但在真实数据区间出现显著波动
3. **IV 对比**: 假设 IV (50%) 远高于 GARCH 长期波动率 (16.87%)，表明市场对 BTC 的波动预期显著高于历史实际波动

---

## 4. 买卖信号示例

### 4.1 信号规则

| 信号类型 | 触发条件 | 操作建议 | 业务逻辑 |
| :--- | :--- | :--- | :--- |
| **高波动减仓** | 20日 RV > 30% | 减仓或对冲 | 市场波动加剧，风险上升，降低敞口 |
| **低波动加仓** | 20日 RV < 15% | 加仓或持有 | 市场波动较低，风险可控，可增加敞口 |
| **做多波动率** | IV - RV > 10% | 买入期权 | 市场预期波动大于实际波动，期权被低估 |
| **GARCH 预警** | GARCH 波动率上升 > 2% | 谨慎操作 | 模型预测未来波动加剧，提前防范 |

### 4.2 最新信号 (截至 2026-10-07)

#### QQQ

| 指标 | 数值 | 信号 |
| :--- | :--- | :--- |
| **20日 RV** | {qqq_latest['rv_20d']:.4f} ({qqq_latest['rv_20d']*100:.2f}%) | {'⚠️ 高波动' if qqq_latest['rv_20d'] > 0.30 else '✅ 低波动'} |
| **假设 IV** | {qqq_latest['iv_assumed']:.4f} ({qqq_latest['iv_assumed']*100:.2f}%) | - |
| **IV-RV 价差** | {qqq_latest['iv_rv_spread']:.4f} ({qqq_latest['iv_rv_spread']*100:.2f}%) | {'📈 做多波动率' if qqq_latest['iv_rv_spread'] > 0.10 else '⏸️ 中性'} |
| **GARCH 波动率** | {qqq_latest['garch_annualized_vol']:.4f} ({qqq_latest['garch_annualized_vol']*100:.2f}%) | - |
| **综合信号** | - | **{qqq_latest['composite_signal']}** |

**业务解释**:
- QQQ 的 20 日 RV 为 {qqq_latest['rv_20d']*100:.2f}%，{'超过' if qqq_latest['rv_20d'] > 0.30 else '低于'} 30% 阈值，{'建议减仓或对冲' if qqq_latest['rv_20d'] > 0.30 else '可考虑加仓或持有'}
- 假设 IV (20%) {'高于' if qqq_latest['iv_rv_spread'] > 0 else '低于'} 20 日 RV ({qqq_latest['rv_20d']*100:.2f}%)，价差为 {qqq_latest['iv_rv_spread']*100:.2f}%，{'市场可能高估波动，可考虑卖出期权' if qqq_latest['iv_rv_spread'] < 0 else '市场预期波动大于实际波动，可考虑买入期权'}

#### BTC-USD

| 指标 | 数值 | 信号 |
| :--- | :--- | :--- |
| **20日 RV** | {btc_latest['rv_20d']:.4f} ({btc_latest['rv_20d']*100:.2f}%) | {'⚠️ 高波动' if btc_latest['rv_20d'] > 0.30 else '✅ 低波动'} |
| **假设 IV** | {btc_latest['iv_assumed']:.4f} ({btc_latest['iv_assumed']*100:.2f}%) | - |
| **IV-RV 价差** | {btc_latest['iv_rv_spread']:.4f} ({btc_latest['iv_rv_spread']*100:.2f}%) | {'📈 做多波动率' if btc_latest['iv_rv_spread'] > 0.10 else '⏸️ 中性'} |
| **GARCH 波动率** | {btc_latest['garch_annualized_vol']:.4f} ({btc_latest['garch_annualized_vol']*100:.2f}%) | - |
| **综合信号** | - | **{btc_latest['composite_signal']}** |

**业务解释**:
- BTC 的 20 日 RV 为 {btc_latest['rv_20d']*100:.2f}%，{'超过' if btc_latest['rv_20d'] > 0.30 else '低于'} 30% 阈值，{'建议减仓或对冲' if btc_latest['rv_20d'] > 0.30 else '可考虑加仓或持有'}
- 假设 IV (50%) {'高于' if btc_latest['iv_rv_spread'] > 0 else '低于'} 20 日 RV ({btc_latest['rv_20d']*100:.2f}%)，价差为 {btc_latest['iv_rv_spread']*100:.2f}%，{'市场可能高估波动，可考虑卖出期权' if btc_latest['iv_rv_spread'] < 0 else '市场预期波动大于实际波动，可考虑买入期权'}

### 4.3 信号分布统计

#### QQQ 信号分布

| 信号类型 | 出现次数 | 占比 |
| :--- | :--- | :--- |
"""

# 添加 QQQ 信号分布
for signal, count in qqq_signals['composite_signal'].value_counts().items():
    pct = count / len(qqq_signals) * 100
    report_content += f"| {signal} | {count} | {pct:.2f}% |\n"

report_content += """
#### BTC 信号分布

| 信号类型 | 出现次数 | 占比 |
| :--- | :--- | :--- |
"""

# 添加 BTC 信号分布
for signal, count in btc_signals['composite_signal'].value_counts().items():
    pct = count / len(btc_signals) * 100
    report_content += f"| {signal} | {count} | {pct:.2f}% |\n"

report_content += f"""
---

## 5. 业务原因说明

### 5.1 为什么使用三种波动率指标？

1. **已实现波动率 (RV)**:
   - **优点**: 基于历史数据，计算简单，直观反映过去实际波动
   - **缺点**: 滞后指标，无法预测未来波动
   - **应用场景**: 评估近期市场风险，设置止损阈值

2. **GARCH(1,1) 模型**:
   - **优点**: 捕捉波动聚集效应，可预测未来条件波动率
   - **缺点**: 模型假设较强，参数估计对数据质量敏感
   - **应用场景**: 预测未来波动，动态调整仓位

3. **隐含波动率 (IV)**:
   - **优点**: 反映市场对未来波动的预期，前瞻指标
   - **缺点**: 依赖期权市场流动性，可能包含投机成分
   - **应用场景**: 评估期权定价是否合理，寻找波动率交易机会

### 5.2 为什么组合使用？

- **互补性**: RV 反映过去，GARCH 预测未来，IV 反映市场预期，三者结合可全面评估波动风险
- **交叉验证**: 当三种指标一致时，信号更可靠；当指标冲突时，需进一步分析
- **风险管理**: 通过多指标监控，可提前识别波动率突变，降低尾部风险

### 5.3 具体数值计算示例

#### 示例 1: QQQ 高波动减仓信号

**假设场景**: 2026-01-05，QQQ 20 日 RV = 95.01%

**计算过程**:
1. 计算 20 日对数收益率标准差: σ_20d = 0.0598
2. 年化: RV_20d = 0.0598 × √252 = 0.9501 (95.01%)
3. 判断: 95.01% > 30% 阈值 → 触发高波动减仓信号

**业务解释**:
- 20 日 RV 高达 95.01%，表明近期市场波动剧烈
- 可能原因: 宏观经济数据发布、美联储政策变化、地缘政治事件
- 操作建议: 减仓 50% 或买入看跌期权对冲，降低组合波动风险

#### 示例 2: BTC 做多波动率信号

**假设场景**: 2026-09-21，BTC 20 日 RV = 50.62%，假设 IV = 50%

**计算过程**:
1. 计算 20 日对数收益率标准差: σ_20d = 0.0318
2. 年化: RV_20d = 0.0318 × √252 = 0.5062 (50.62%)
3. IV-RV 价差: 50% - 50.62% = -0.62%

**业务解释**:
- IV (50%) 略低于 RV (50.62%)，价差为 -0.62%
- 表明市场预期波动略低于实际波动，期权可能被高估
- 操作建议: 可考虑卖出期权 (如卖出跨式组合)，赚取波动率溢价

#### 示例 3: GARCH 波动率预警

**假设场景**: 2025-12-03，BTC GARCH 年化波动率从 39.41% 上升至 38.40%

**计算过程**:
1. GARCH(1,1) 模型: σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1)
2. 参数: ω = 0.00000564, α = 0.10, β = 0.85
3. 预测: σ²(t+1) = 0.00000564 + 0.10 × ε²(t) + 0.85 × σ²(t)
4. 年化波动率: √(σ²(t+1) × 252)

**业务解释**:
- GARCH 波动率从 39.41% 下降至 38.40%，波动率回落
- 表明市场波动正在收敛，风险降低
- 操作建议: 可逐步恢复仓位，增加 BTC 敞口

---

## 6. 是否属于量化投资思路？

### 6.1 量化投资的核心特征

1. **数据驱动**: 基于历史数据和统计模型，而非主观判断
2. **规则明确**: 交易信号由明确的数学规则生成，可重复、可验证
3. **系统化**: 策略执行系统化，减少人为情绪干扰
4. **可回测**: 策略可在历史数据上回测，评估绩效

### 6.2 本策略的量化特征

| 特征 | 本策略表现 | 是否符合 |
| :--- | :--- | :--- |
| **数据驱动** | 基于历史价格数据计算 RV、GARCH、IV | ✅ 符合 |
| **规则明确** | 信号规则明确 (RV > 30% 减仓，IV-RV > 10% 做多波动率) | ✅ 符合 |
| **系统化** | 信号生成自动化，可程序化执行 | ✅ 符合 |
| **可回测** | 可在历史数据上回测策略绩效 | ✅ 符合 (但数据质量限制回测有效性) |

### 6.3 结论

**本策略属于量化投资思路**，具体表现为:

1. **波动率因子策略**: 利用波动率指标作为交易信号，是典型的量化因子策略
2. **统计套利**: 通过 IV-RV 价差寻找期权定价偏差，属于统计套利范畴
3. **风险管理**: 使用 GARCH 模型预测波动率，动态调整仓位，属于量化风险管理

**局限性**:
- 数据质量: 当前数据为 Illustrative，包含大量填充，无法用于实际交易
- 模型假设: GARCH 模型假设较强，可能不适用于所有市场状态
- 交易成本: 未考虑交易成本、滑点、流动性等因素
- 过拟合风险: 信号阈值 (30%, 15%, 10%) 需通过历史数据优化，避免过拟合

### 6.4 改进建议

1. **数据改进**:
   - 获取完整的 2024-2026 年日线数据
   - 获取真实的期权隐含波动率数据 (CBOE, Deribit)
   - 增加更多资产类别 (如 SPY, ETH) 进行对比

2. **模型改进**:
   - 使用更复杂的 GARCH 模型 (如 GJR-GARCH, EGARCH) 捕捉不对称效应
   - 引入机器学习模型 (如 LSTM) 预测波动率
   - 结合宏观变量 (如 VIX, 利率) 增强预测能力

3. **策略改进**:
   - 优化信号阈值，使用历史数据回测确定最优参数
   - 引入仓位管理规则 (如凯利公式) 优化资金分配
   - 增加止损止盈规则，控制尾部风险
   - 考虑交易成本和滑点，评估策略净收益

4. **风险管理**:
   - 监控模型失效风险 (如市场结构变化)
   - 设置最大回撤限制，避免极端损失
   - 定期重新校准模型参数

---

## 7. 结论

### 7.1 主要发现

1. **波动率差异**: BTC 的波动率显著高于 QQQ，20 日 RV 均值分别为 {btc_rv_20d_mean*100:.2f}% 和 {qqq_rv_20d_mean*100:.2f}%
2. **IV-RV 价差**: 假设 IV 均高于 RV，表明市场可能高估了波动风险，存在卖出期权的套利机会
3. **GARCH 特性**: 两种资产的 GARCH 波动率均呈现均值回复特性，长期波动率分别为 11.78% (QQQ) 和 16.87% (BTC)

### 7.2 策略建议

1. **QQQ**: 当前 20 日 RV 为 {qqq_latest['rv_20d']*100:.2f}%，{'建议减仓' if qqq_latest['rv_20d'] > 0.30 else '可考虑加仓'}；IV-RV 价差为 {qqq_latest['iv_rv_spread']*100:.2f}%，{'可考虑卖出期权' if qqq_latest['iv_rv_spread'] < 0 else '可考虑买入期权'}
2. **BTC**: 当前 20 日 RV 为 {btc_latest['rv_20d']*100:.2f}%，{'建议减仓' if btc_latest['rv_20d'] > 0.30 else '可考虑加仓'}；IV-RV 价差为 {btc_latest['iv_rv_spread']*100:.2f}%，{'可考虑卖出期权' if btc_latest['iv_rv_spread'] < 0 else '可考虑买入期权'}

### 7.3 风险提示

⚠️ **重要**: 本报告基于 Illustrative 数据生成，所有计算结果仅用于方法论演示，**不可用于实际交易决策**。

**主要风险**:
1. **数据风险**: 年度均值填充导致波动率严重失真
2. **模型风险**: GARCH 模型假设可能不适用于所有市场状态
3. **市场风险**: 波动率可能突然飙升，超出模型预测范围
4. **流动性风险**: 期权市场流动性不足，可能导致交易成本上升

**后续步骤**:
1. 获取完整真实数据，重新计算所有波动率指标
2. 进行完整的策略回测，评估历史绩效
3. 优化信号阈值和仓位管理规则
4. 考虑交易成本和滑点，评估策略净收益
5. 小资金实盘测试，验证策略有效性

---

## 附录

### A. 数据文件清单

| 文件 | 路径 | 说明 |
| :--- | :--- | :--- |
| 源数据 | `E:\\agent_dev\\multi-agent\\tmp\\step1_nasdaq_btc_historical_data.md` | 原始历史数据 |
| QQQ 日线 | `E:\\agent_dev\\multi-agent\\tmp\\qqq_daily.csv` | QQQ 日线数据 (723 条) |
| BTC 日线 | `E:\\agent_dev\\multi-agent\\tmp\\btc_daily.csv` | BTC 日线数据 (1011 条) |
| 波动率指标 | `E:\\agent_dev\\multi-agent\\tmp\\volatility_metrics.csv` | 三种波动率指标 (1734 条) |
| 数据质量说明 | `E:\\agent_dev\\multi-agent\\tmp\\data_quality_notes.md` | 数据质量与局限性说明 |
| 波动率图表 | `E:\\agent_dev\\multi-agent\\tmp\\volatility_comparison_chart.png` | 三种波动率对比图表 |

### B. 方法论参考

1. **已实现波动率**: 滚动窗口标准差 × √252 年化
2. **GARCH(1,1)**: σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1)
3. **Black-Scholes 期权定价**: C = S·N(d1) - K·e^(-rT)·N(d2)
4. **隐含波动率**: 通过 Black-Scholes 模型反推，使用二分法求解

### C. 免责声明

本报告仅供教育和研究目的，不构成任何投资建议。投资者应根据自身情况独立判断，并承担投资风险。

**报告生成时间**: 2026-10-08 18:19
**数据截至**: 2026-10-07
**报告作者**: CodeAgent (Step 3/3)
"""

# 保存报告
report_path = r'E:\agent_dev\multi-agent\tmp\volatility_trading_report.md'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_content)

print(f"\n✅ 报告已生成: {report_path}")
print(f"   图表: {chart_path}")
print(f"   报告大小: {len(report_content)} 字符")

# ============================================================
# 5. 输出摘要
# ============================================================

print("\n" + "=" * 80)
print("Step 3/3 完成: 波动率交易策略报告")
print("=" * 80)
print(f"""
生成的文件:
1. 报告: {report_path}
2. 图表: {chart_path}

报告内容:
- 分析背景与三种波动率指标简介
- 数据概览与最新数据
- 三种波动率对比分析 (含图表)
- 买卖信号示例 (含数值计算)
- 业务原因说明
- 是否属于量化投资思路的讨论
- 结论与风险提示

⚠️ 重要声明: 所有数据为 Illustrative，仅用于方法论演示，不可用于实际交易决策。
""")
