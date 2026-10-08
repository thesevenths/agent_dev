import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 从 Step 1 文件中提取数据
# ============================================================

# QQQ 数据（从 step1 文件中提取的近期日线数据）
qqq_data = {
    'date': ['2026-03-05','2026-03-04','2026-03-03','2026-03-02','2026-02-27','2026-02-26',
             '2026-01-29','2026-01-28','2026-01-27','2026-01-26','2026-01-23','2026-01-22',
             '2026-01-02','2025-12-31','2025-12-30','2025-12-29','2025-12-26','2025-12-24'],
    'close': [608.91, 610.75, 601.58, 608.09, 607.29, 609.24,
              629.43, 633.22, 631.13, 625.46, 622.72, 620.76,
              613.12, 614.31, 619.43, 620.87, 623.89, 623.93]
}

# BTC 数据（从 step1 文件中提取的近期日线数据）
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
              87138, 87802, 87301, 87235, 87235, 87612,
              87414, 88490, 88622, 88344, 88103, 92142,
              93528, 91350, 86322, 90394, 90852, 90919,
              91285, 90518, 87342]
}

# 创建 DataFrame
df_qqq = pd.DataFrame(qqq_data)
df_qqq['date'] = pd.to_datetime(df_qqq['date'])
df_qqq = df_qqq.sort_values('date').reset_index(drop=True)

df_btc = pd.DataFrame(btc_data)
df_btc['date'] = pd.to_datetime(df_btc['date'])
df_btc = df_btc.sort_values('date').reset_index(drop=True)

print("QQQ 数据范围:", df_qqq['date'].min(), "至", df_qqq['date'].max(), "共", len(df_qqq), "条")
print("BTC 数据范围:", df_btc['date'].min(), "至", df_btc['date'].max(), "共", len(df_btc), "条")

# ============================================================
# 2. 计算已实现波动率 (Realized Volatility)
# ============================================================

def calculate_realized_volatility(prices, window=20, annualize=252):
    """
    计算已实现波动率 (Realized Volatility)
    
    公式:
    RV = std(daily_returns, window) * sqrt(annualize)
    
    其中 daily_returns = ln(P_t / P_{t-1})
    """
    log_returns = np.log(prices / prices.shift(1))
    rv = log_returns.rolling(window=window).std() * np.sqrt(annualize)
    return rv

# QQQ 已实现波动率
df_qqq['log_return'] = np.log(df_qqq['close'] / df_qqq['close'].shift(1))
df_qqq['RV_20d'] = df_qqq['log_return'].rolling(window=20).std() * np.sqrt(252)
df_qqq['RV_60d'] = df_qqq['log_return'].rolling(window=60).std() * np.sqrt(252)

# BTC 已实现波动率（使用365天年化，因为BTC 24/7交易）
df_btc['log_return'] = np.log(df_btc['close'] / df_btc['close'].shift(1))
df_btc['RV_20d'] = df_btc['log_return'].rolling(window=20).std() * np.sqrt(365)
df_btc['RV_60d'] = df_btc['log_return'].rolling(window=60).std() * np.sqrt(365)

print("\n=== QQQ 已实现波动率 ===")
print(df_qqq[['date', 'close', 'RV_20d', 'RV_60d']].tail(10).to_string())

print("\n=== BTC 已实现波动率 ===")
print(df_btc[['date', 'close', 'RV_20d', 'RV_60d']].tail(10).to_string())

# ============================================================
# 3. GARCH(1,1) 模型估计（手动实现）
# ============================================================

def fit_garch_manual(returns, annualize=252):
    """
    手动拟合 GARCH(1,1) 模型
    
    模型:
    σ_t² = ω + α·ε_{t-1}² + β·σ_{t-1}²
    
    使用最大似然估计
    """
    from scipy.optimize import minimize
    
    # 初始参数
    omega0 = np.var(returns) * 0.05
    alpha0 = 0.1
    beta0 = 0.85
    
    def neg_log_likelihood(params):
        omega, alpha, beta = params
        
        # 检查参数约束
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)
        
        for t in range(1, n):
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        
        # 对数似然
        ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + returns**2 / sigma2)
        return -ll
    
    # 优化
    result = minimize(neg_log_likelihood, [omega0, alpha0, beta0], 
                     method='Nelder-Mead', 
                     bounds=[(1e-8, None), (0, 0.5), (0, 0.95)])
    
    omega, alpha, beta = result.x
    
    # 条件方差预测
    sigma2 = np.zeros(len(returns))
    sigma2[0] = np.var(returns)
    
    for t in range(1, len(returns)):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
    
    # 1步预测
    cond_var_1d = omega + alpha * returns[-1]**2 + beta * sigma2[-1]
    cond_vol_1d = np.sqrt(cond_var_1d) * np.sqrt(annualize)
    
    # 5步预测
    sigma2_5d = sigma2[-1]
    for i in range(5):
        sigma2_5d = omega + alpha * returns[-1]**2 + beta * sigma2_5d
    cond_vol_5d = np.sqrt(sigma2_5d) * np.sqrt(annualize)
    
    # 长期方差
    long_var = omega / (1 - alpha - beta)
    long_vol = np.sqrt(long_var) * np.sqrt(annualize)
    
    return {
        'omega': omega,
        'alpha': alpha,
        'beta': beta,
        'alpha_plus_beta': alpha + beta,
        'cond_vol_1d': cond_vol_1d,
        'cond_vol_5d': cond_vol_5d,
        'long_vol': long_vol
    }

# QQQ GARCH
print("\n=== QQQ GARCH(1,1) 模型 ===")
qqq_returns = df_qqq['log_return'].dropna()
qqq_garch = fit_garch_manual(qqq_returns, annualize=252)
print(f"ω (omega) = {qqq_garch['omega']:.6f}")
print(f"α (alpha) = {qqq_garch['alpha']:.4f}")
print(f"β (beta)  = {qqq_garch['beta']:.4f}")
print(f"α + β     = {qqq_garch['alpha_plus_beta']:.4f}")
print(f"1日条件波动率 (年化) = {qqq_garch['cond_vol_1d']*100:.2f}%")
print(f"5日条件波动率 (年化) = {qqq_garch['cond_vol_5d']*100:.2f}%")
print(f"长期波动率 (年化)    = {qqq_garch['long_vol']*100:.2f}%")

# BTC GARCH
print("\n=== BTC GARCH(1,1) 模型 ===")
btc_returns = df_btc['log_return'].dropna()
btc_garch = fit_garch_manual(btc_returns, annualize=365)
print(f"ω (omega) = {btc_garch['omega']:.6f}")
print(f"α (alpha) = {btc_garch['alpha']:.4f}")
print(f"β (beta)  = {btc_garch['beta']:.4f}")
print(f"α + β     = {btc_garch['alpha_plus_beta']:.4f}")
print(f"1日条件波动率 (年化) = {btc_garch['cond_vol_1d']*100:.2f}%")
print(f"5日条件波动率 (年化) = {btc_garch['cond_vol_5d']*100:.2f}%")
print(f"长期波动率 (年化)    = {btc_garch['long_vol']*100:.2f}%")

# ============================================================
# 4. 隐含波动率 (Implied Volatility) 示例
# ============================================================

# 从 step1 文件中获取的期权数据
qqq_iv = 0.20  # 20%
btc_iv = 0.50  # 50%

print("\n=== 隐含波动率 (IV) ===")
print(f"QQQ IV = {qqq_iv*100:.1f}%")
print(f"BTC IV = {btc_iv*100:.1f}%")

# ============================================================
# 5. 波动率阈值策略
# ============================================================

print("\n" + "="*60)
print("波动率阈值策略")
print("="*60)

# QQQ 策略
qqq_rv_20d = df_qqq['RV_20d'].iloc[-1]
qqq_rv_60d = df_qqq['RV_60d'].iloc[-1]
qqq_garch_vol = qqq_garch['cond_vol_1d']

print(f"\n【QQQ 当前状态】")
print(f"RV(20d) = {qqq_rv_20d*100:.2f}%")
print(f"RV(60d) = {qqq_rv_60d*100:.2f}%")
print(f"GARCH 1日条件波动率 = {qqq_garch_vol*100:.2f}%")
print(f"IV = {qqq_iv*100:.1f}%")

# 策略规则
qqq_rv_threshold_high = 0.25  # 25%
qqq_rv_threshold_low = 0.15   # 15%

if qqq_rv_20d > qqq_rv_threshold_high:
    qqq_signal = "减仓/买入看跌期权"
    qqq_reason = "RV(20d) 高于 25% 阈值，波动率偏高，建议降低风险敞口"
elif qqq_rv_20d < qqq_rv_threshold_low:
    qqq_signal = "加仓/买入看涨期权"
    qqq_reason = "RV(20d) 低于 15% 阈值，波动率偏低，建议增加风险敞口"
else:
    qqq_signal = "持有/观望"
    qqq_reason = "RV(20d) 在正常范围内，维持当前仓位"

print(f"信号: {qqq_signal}")
print(f"原因: {qqq_reason}")

# BTC 策略
btc_rv_20d = df_btc['RV_20d'].iloc[-1]
btc_rv_60d = df_btc['RV_60d'].iloc[-1]
btc_garch_vol = btc_garch['cond_vol_1d']

print(f"\n【BTC 当前状态】")
print(f"RV(20d) = {btc_rv_20d*100:.2f}%")
print(f"RV(60d) = {btc_rv_60d*100:.2f}%")
print(f"GARCH 1日条件波动率 = {btc_garch_vol*100:.2f}%")
print(f"IV = {btc_iv*100:.1f}%")

btc_rv_threshold_high = 0.80  # 80%
btc_rv_threshold_low = 0.40   # 40%

if btc_rv_20d > btc_rv_threshold_high:
    btc_signal = "减仓/买入看跌期权"
    btc_reason = "RV(20d) 高于 80% 阈值，波动率极高，建议大幅降低风险敞口"
elif btc_rv_20d < btc_rv_threshold_low:
    btc_signal = "加仓/买入看涨期权"
    btc_reason = "RV(20d) 低于 40% 阈值，波动率偏低，建议增加风险敞口"
else:
    btc_signal = "持有/观望"
    btc_reason = "RV(20d) 在正常范围内，维持当前仓位"

print(f"信号: {btc_signal}")
print(f"原因: {btc_reason}")

# ============================================================
# 6. 可视化
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# QQQ 价格与 RV
ax1 = axes[0, 0]
ax1.plot(df_qqq['date'], df_qqq['close'], 'b-', label='QQQ 收盘价')
ax1.set_title('QQQ 价格与已实现波动率')
ax1.set_xlabel('日期')
ax1.set_ylabel('价格')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax1b = ax1.twinx()
ax1b.plot(df_qqq['date'], df_qqq['RV_20d']*100, 'r--', label='RV(20d)')
ax1b.set_ylabel('RV(20d) %', color='r')
ax1b.legend(loc='upper left')

# BTC 价格与 RV
ax2 = axes[0, 1]
ax2.plot(df_btc['date'], df_btc['close'], 'b-', label='BTC 收盘价')
ax2.set_title('BTC 价格与已实现波动率')
ax2.set_xlabel('日期')
ax2.set_ylabel('价格')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax2b = ax2.twinx()
ax2b.plot(df_btc['date'], df_btc['RV_20d']*100, 'r--', label='RV(20d)')
ax2b.set_ylabel('RV(20d) %', color='r')
ax2b.legend(loc='upper left')

# QQQ RV 对比
ax3 = axes[1, 0]
ax3.plot(df_qqq['date'], df_qqq['RV_20d']*100, 'b-', label='RV(20d)')
ax3.plot(df_qqq['date'], df_qqq['RV_60d']*100, 'g-', label='RV(60d)')
ax3.axhline(y=qqq_rv_threshold_high*100, color='r', linestyle='--', label='高阈值 25%')
ax3.axhline(y=qqq_rv_threshold_low*100, color='g', linestyle='--', label='低阈值 15%')
ax3.set_title('QQQ 已实现波动率对比')
ax3.set_xlabel('日期')
ax3.set_ylabel('RV %')
ax3.legend()
ax3.grid(True, alpha=0.3)

# BTC RV 对比
ax4 = axes[1, 1]
ax4.plot(df_btc['date'], df_btc['RV_20d']*100, 'b-', label='RV(20d)')
ax4.plot(df_btc['date'], df_btc['RV_60d']*100, 'g-', label='RV(60d)')
ax4.axhline(y=btc_rv_threshold_high*100, color='r', linestyle='--', label='高阈值 80%')
ax4.axhline(y=btc_rv_threshold_low*100, color='g', linestyle='--', label='低阈值 40%')
ax4.set_title('BTC 已实现波动率对比')
ax4.set_xlabel('日期')
ax4.set_ylabel('RV %')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('E:\\agent_dev\\multi-agent\\tmp\\step2_volatility_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n图表已保存: E:\\agent_dev\\multi-agent\\tmp\\step2_volatility_analysis.png")
