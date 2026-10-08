import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 从 Step 1 数据中提取 QQQ 和 BTC 的日线数据
# ============================================================

# QQQ 数据（从 step1_nasdaq_btc_historical_data.md 提取）
qqq_data = {
    'date': ['2026-03-05','2026-03-04','2026-03-03','2026-03-02','2026-02-27','2026-02-26',
             '2026-01-29','2026-01-28','2026-01-27','2026-01-26','2026-01-23','2026-01-22',
             '2026-01-02','2025-12-31','2025-12-30','2025-12-29','2025-12-26','2025-12-24'],
    'close': [608.91, 610.75, 601.58, 608.09, 607.29, 609.24,
              629.43, 633.22, 631.13, 625.46, 622.72, 620.76,
              613.12, 614.31, 619.43, 620.87, 623.89, 623.93]
}

# BTC 数据（从 step1_nasdaq_btc_historical_data.md 提取）
btc_data = {
    'date': ['2026-10-07','2026-10-06','2026-10-05','2026-10-04','2026-10-03','2026-10-02',
             '2026-10-01','2026-09-30','2026-09-29','2026-09-28','2026-09-27','2026-09-26',
             '2026-09-25','2026-09-24','2026-09-23','2026-09-22','2026-09-21','2026-09-20',
             '2026-09-19','2026-09-18','2026-09-17','2026-09-16','2026-09-15','2026-09-14',
             '2026-09-13','2026-01-03','2026-01-02','2026-01-01','2025-12-31','2025-12-30',
             '2025-12-29','2025-12-28','2025-12-27','2025-12-26','2025-12-25','2025-12-24',
             '2025-12-23','2025-12-22','2025-12-21','2025-12-20','2025-12-19','2025-12-04',
             '2025-12-03','2025-12-02','2025-12-01','2025-11-30','2025-11-29','2025-11-28',
             '2025-11-27','2025-11-26','2025-11-25'],
    'close': [84068, 85615, 85738, 86511, 84750, 84518,
              84906, 83621, 83719, 83532, 84455, 84375,
              84071, 84355, 84340, 86194, 86617, 81191,
              81227, 80880, 76350, 76140, 75580, 78180,
              76800, 90603, 89945, 88732, 87509, 88430,
              87138, 87836, 87802, 87301, 87235, 87612,
              87414, 88490, 88622, 88344, 88103, 92142,
              93528, 91350, 86322, 90394, 90852, 90919,
              91285, 90518, 87342]
}

# 创建 DataFrame
qqq_df = pd.DataFrame(qqq_data)
btc_df = pd.DataFrame(btc_data)

# 按日期排序
qqq_df = qqq_df.sort_values('date').reset_index(drop=True)
btc_df = btc_df.sort_values('date').reset_index(drop=True)

# 计算日收益率
qqq_df['return'] = qqq_df['close'].pct_change()
btc_df['return'] = btc_df['close'].pct_change()

print("QQQ 数据概览:")
print(qqq_df.head(10))
print(f"\nQQQ 数据点数: {len(qqq_df)}")
print(f"QQQ 最新收盘价: {qqq_df['close'].iloc[-1]}")

print("\nBTC 数据概览:")
print(btc_df.head(10))
print(f"\nBTC 数据点数: {len(btc_df)}")
print(f"BTC 最新收盘价: {btc_df['close'].iloc[-1]}")

# ============================================================
# 2. 计算已实现波动率 (RV)
# ============================================================

def calculate_rv(returns, window=20):
    """计算滚动已实现波动率（年化）"""
    rv = returns.rolling(window=window).std() * np.sqrt(252)
    return rv

# QQQ RV
qqq_df['RV_20d'] = calculate_rv(qqq_df['return'], window=20)
qqq_df['RV_10d'] = calculate_rv(qqq_df['return'], window=10)

# BTC RV
btc_df['RV_20d'] = calculate_rv(btc_df['return'], window=20)
btc_df['RV_10d'] = calculate_rv(btc_df['return'], window=10)

print("\nQQQ RV 最新值:")
print(f"RV_20d: {qqq_df['RV_20d'].iloc[-1]*100:.2f}%")
print(f"RV_10d: {qqq_df['RV_10d'].iloc[-1]*100:.2f}%")

print("\nBTC RV 最新值:")
print(f"RV_20d: {btc_df['RV_20d'].iloc[-1]*100:.2f}%")
print(f"RV_10d: {btc_df['RV_10d'].iloc[-1]*100:.2f}%")

# ============================================================
# 3. 拟合 GARCH(1,1) 模型
# ============================================================

def fit_garch(returns, omega_init=0.0001, alpha_init=0.1, beta_init=0.85):
    """
    简化版 GARCH(1,1) 拟合（使用矩估计法）
    实际应用中应使用 MLE 或 QMLE
    """
    n = len(returns)
    # 使用矩估计初始化
    var = np.var(returns)
    omega = var * (1 - alpha_init - beta_init)
    
    # 迭代估计
    sigma2 = np.zeros(n)
    sigma2[0] = var
    
    for t in range(1, n):
        eps2 = returns[t-1]**2
        sigma2[t] = omega + alpha_init * eps2 + beta_init * sigma2[t-1]
    
    # 计算预测波动率
    sigma = np.sqrt(sigma2)
    
    return sigma, omega, alpha_init, beta_init

# QQQ GARCH
qqq_returns = qqq_df['return'].dropna().values
qqq_garch_sigma, qqq_omega, qqq_alpha, qqq_beta = fit_garch(qqq_returns)

# BTC GARCH
btc_returns = btc_df['return'].dropna().values
btc_garch_sigma, btc_omega, btc_alpha, btc_beta = fit_garch(btc_returns)

print(f"\nQQQ GARCH 参数: omega={qqq_omega:.6f}, alpha={qqq_alpha:.2f}, beta={qqq_beta:.2f}")
print(f"QQQ GARCH 最新波动率: {qqq_garch_sigma[-1]*100:.2f}% (日), {qqq_garch_sigma[-1]*np.sqrt(252)*100:.2f}% (年化)")

print(f"\nBTC GARCH 参数: omega={btc_omega:.6f}, alpha={btc_alpha:.2f}, beta={btc_beta:.2f}")
print(f"BTC GARCH 最新波动率: {btc_garch_sigma[-1]*100:.2f}% (日), {btc_garch_sigma[-1]*np.sqrt(252)*100:.2f}% (年化)")

# ============================================================
# 4. 隐含波动率 (IV) - 使用假设数据
# ============================================================

# QQQ IV 假设（基于 VIX 和期权市场）
qqq_iv_30d = 0.20  # 20% 年化
qqq_iv_60d = 0.22  # 22% 年化

# BTC IV 假设（基于 Deribit 期权市场）
btc_iv_30d = 0.50  # 50% 年化
btc_iv_60d = 0.55  # 55% 年化

print(f"\nQQQ IV 假设: 30d={qqq_iv_30d*100:.1f}%, 60d={qqq_iv_60d*100:.1f}%")
print(f"BTC IV 假设: 30d={btc_iv_30d*100:.1f}%, 60d={btc_iv_60d*100:.1f}%")

# ============================================================
# 5. 波动率目标仓位管理 (Volatility Targeting)
# ============================================================

def volatility_targeting(current_vol, target_vol, max_position=1.0):
    """
    波动率目标仓位管理
    仓位系数 = 目标波动率 / 当前波动率
    """
    if current_vol == 0:
        return 1.0
    position = min(target_vol / current_vol, max_position)
    return position

# QQQ 波动率目标
qqq_target_vol = 0.20  # 20% 年化目标
qqq_current_vol = qqq_garch_sigma[-1] * np.sqrt(252)  # GARCH 年化波动率
qqq_position = volatility_targeting(qqq_current_vol, qqq_target_vol)

print(f"\nQQQ 波动率目标仓位:")
print(f"目标波动率: {qqq_target_vol*100:.1f}%")
print(f"当前 GARCH 年化波动率: {qqq_current_vol*100:.2f}%")
print(f"仓位系数: {qqq_position:.2f} (即 {qqq_position*100:.0f}% 仓位)")

# BTC 波动率目标
btc_target_vol = 0.40  # 40% 年化目标（BTC 波动更大，目标更高）
btc_current_vol = btc_garch_sigma[-1] * np.sqrt(252)  # GARCH 年化波动率
btc_position = volatility_targeting(btc_current_vol, btc_target_vol)

print(f"\nBTC 波动率目标仓位:")
print(f"目标波动率: {btc_target_vol*100:.1f}%")
print(f"当前 GARCH 年化波动率: {btc_current_vol*100:.2f}%")
print(f"仓位系数: {btc_position:.2f} (即 {btc_position*100:.0f}% 仓位)")

# ============================================================
# 6. 波动率均值回归交易
# ============================================================

def volatility_percentile(rv_series, current_rv):
    """计算当前 RV 在历史中的分位数"""
    percentile = (rv_series < current_rv).sum() / len(rv_series)
    return percentile

# QQQ 波动率分位数（使用 RV_10d，因为数据点不足 20 个）
qqq_rv_series = qqq_df['RV_10d'].dropna()
qqq_current_rv = qqq_rv_series.iloc[-1]
qqq_vol_percentile = volatility_percentile(qqq_rv_series, qqq_current_rv)

print(f"\nQQQ 波动率均值回归:")
print(f"当前 RV_10d: {qqq_current_rv*100:.2f}%")
print(f"历史分位数: {qqq_vol_percentile*100:.1f}%")

# BTC 波动率分位数
btc_rv_series = btc_df['RV_20d'].dropna()
btc_current_rv = btc_rv_series.iloc[-1]
btc_vol_percentile = volatility_percentile(btc_rv_series, btc_current_rv)

print(f"\nBTC 波动率均值回归:")
print(f"当前 RV_20d: {btc_current_rv*100:.2f}%")
print(f"历史分位数: {btc_vol_percentile*100:.1f}%")

# ============================================================
# 7. IV-RV 价差交易
# ============================================================

def iv_rv_spread(iv, rv):
    """计算 IV-RV 价差"""
    return iv - rv

# QQQ IV-RV 价差
qqq_iv_rv_spread = iv_rv_spread(qqq_iv_30d, qqq_current_rv)
print(f"\nQQQ IV-RV 价差:")
print(f"IV_30d: {qqq_iv_30d*100:.1f}%")
print(f"RV_10d: {qqq_current_rv*100:.2f}%")
print(f"价差: {qqq_iv_rv_spread*100:.2f}%")
if qqq_iv_rv_spread > 0:
    print("结论: IV > RV，市场恐慌溢价，适合卖出期权（做空 IV）")
else:
    print("结论: IV < RV，市场低估风险，适合买入期权（做多 IV）")

# BTC IV-RV 价差
btc_iv_rv_spread = iv_rv_spread(btc_iv_30d, btc_current_rv)
print(f"\nBTC IV-RV 价差:")
print(f"IV_30d: {btc_iv_30d*100:.1f}%")
print(f"RV_20d: {btc_current_rv*100:.2f}%")
print(f"价差: {btc_iv_rv_spread*100:.2f}%")
if btc_iv_rv_spread > 0:
    print("结论: IV > RV，市场恐慌溢价，适合卖出期权（做空 IV）")
else:
    print("结论: IV < RV，市场低估风险，适合买入期权（做多 IV）")

# ============================================================
# 8. 生成图表
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: QQQ 价格与 RV
ax1 = axes[0, 0]
ax1.plot(qqq_df['date'], qqq_df['close'], 'b-', label='QQQ 收盘价', linewidth=2)
ax1.set_title('QQQ 价格与已实现波动率', fontsize=12)
ax1.set_xlabel('日期')
ax1.set_ylabel('价格')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax1_twin = ax1.twinx()
ax1_twin.plot(qqq_df['date'], qqq_df['RV_20d']*100, 'r--', label='RV_20d (年化)', linewidth=1.5)
ax1_twin.set_ylabel('RV (%)', color='r')
ax1_twin.tick_params(axis='y', labelcolor='r')

# 图2: BTC 价格与 RV
ax2 = axes[0, 1]
ax2.plot(btc_df['date'], btc_df['close'], 'b-', label='BTC 收盘价', linewidth=2)
ax2.set_title('BTC 价格与已实现波动率', fontsize=12)
ax2.set_xlabel('日期')
ax2.set_ylabel('价格')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax2_twin = ax2.twinx()
ax2_twin.plot(btc_df['date'], btc_df['RV_20d']*100, 'r--', label='RV_20d (年化)', linewidth=1.5)
ax2_twin.set_ylabel('RV (%)', color='r')
ax2_twin.tick_params(axis='y', labelcolor='r')

# 图3: QQQ GARCH 波动率
ax3 = axes[1, 0]
garch_dates = qqq_df['date'].iloc[1:]  # 去掉第一个 NaN
ax3.plot(garch_dates, qqq_garch_sigma*100, 'g-', label='GARCH 日波动率', linewidth=2)
ax3.set_title('QQQ GARCH 波动率预测', fontsize=12)
ax3.set_xlabel('日期')
ax3.set_ylabel('波动率 (%)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 图4: BTC GARCH 波动率
ax4 = axes[1, 1]
garch_dates_btc = btc_df['date'].iloc[1:]  # 去掉第一个 NaN
ax4.plot(garch_dates_btc, btc_garch_sigma*100, 'g-', label='GARCH 日波动率', linewidth=2)
ax4.set_title('BTC GARCH 波动率预测', fontsize=12)
ax4.set_xlabel('日期')
ax4.set_ylabel('波动率 (%)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('E:\\agent_dev\\multi-agent\\tmp\\volatility_charts.png', dpi=150, bbox_inches='tight')
print("\n图表已保存: E:\\agent_dev\\multi-agent\\tmp\\volatility_charts.png")

# ============================================================
# 9. 保存计算结果
# ============================================================

results = {
    'qqq': {
        'latest_close': float(qqq_df['close'].iloc[-1]),
        'rv_20d': float(qqq_current_rv),
        'rv_10d': float(qqq_df['RV_10d'].iloc[-1]),
        'garch_daily_vol': float(qqq_garch_sigma[-1]),
        'garch_annual_vol': float(qqq_current_vol),
        'garch_params': {'omega': float(qqq_omega), 'alpha': float(qqq_alpha), 'beta': float(qqq_beta)},
        'iv_30d': float(qqq_iv_30d),
        'iv_60d': float(qqq_iv_60d),
        'vol_target': float(qqq_target_vol),
        'position': float(qqq_position),
        'vol_percentile': float(qqq_vol_percentile),
        'iv_rv_spread': float(qqq_iv_rv_spread)
    },
    'btc': {
        'latest_close': float(btc_df['close'].iloc[-1]),
        'rv_20d': float(btc_current_rv),
        'rv_10d': float(btc_df['RV_10d'].iloc[-1]),
        'garch_daily_vol': float(btc_garch_sigma[-1]),
        'garch_annual_vol': float(btc_current_vol),
        'garch_params': {'omega': float(btc_omega), 'alpha': float(btc_alpha), 'beta': float(btc_beta)},
        'iv_30d': float(btc_iv_30d),
        'iv_60d': float(btc_iv_60d),
        'vol_target': float(btc_target_vol),
        'position': float(btc_position),
        'vol_percentile': float(btc_vol_percentile),
        'iv_rv_spread': float(btc_iv_rv_spread)
    }
}

with open('E:\\agent_dev\\multi-agent\\tmp\\volatility_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n计算结果已保存: E:\\agent_dev\\multi-agent\\tmp\\volatility_results.json")
print("\n=== 计算完成 ===")
