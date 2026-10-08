import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 数据准备（从上游文件提取）
# ============================================================

# QQQ 日线数据
qqq_data = [
    ("2026-03-05", 607.40, 612.76, 602.26, 608.91, 89602400),
    ("2026-03-04", 604.16, 612.88, 603.43, 610.75, 70943900),
    ("2026-03-03", 596.33, 603.96, 591.87, 601.58, 97015500),
    ("2026-03-02", 598.86, 609.92, 597.99, 608.09, 75264600),
    ("2026-02-27", 602.98, 608.32, 602.19, 607.29, 68125200),
    ("2026-02-26", 615.59, 615.59, 603.98, 609.24, 96178900),
    ("2026-01-29", 632.65, 633.67, 618.27, 629.43, 79944000),
    ("2026-01-28", 635.46, 636.60, 631.81, 633.22, 50691700),
    ("2026-01-27", 628.91, 632.04, 627.34, 631.13, 38997200),
    ("2026-01-26", 623.21, 627.61, 622.12, 625.46, 35983000),
    ("2026-01-23", 619.73, 625.40, 618.65, 622.72, 43645800),
    ("2026-01-22", 622.35, 622.46, 617.78, 620.76, 42254800),
    ("2026-01-02", 620.06, 622.85, 610.15, 613.12, 61859200),
    ("2025-12-31", 619.65, 619.96, 614.05, 614.31, 40746500),
    ("2025-12-30", 619.84, 622.18, 619.22, 619.43, 31226800),
    ("2025-12-29", 620.10, 622.78, 618.73, 620.87, 32458300),
    ("2025-12-26", 624.66, 625.52, 623.14, 623.89, 28959800),
    ("2025-12-24", 621.99, 624.28, 621.72, 623.93, 18468700),
]

# BTC-USD 日线数据
btc_data = [
    ("2026-10-07", 85636, 85666, 83735, 84068),
    ("2026-10-06", 85734, 86605, 85119, 85615),
    ("2026-10-05", 86501, 86968, 84952, 85738),
    ("2026-10-04", 84749, 86783, 84710, 86511),
    ("2026-10-03", 84518, 85034, 84447, 84750),
    ("2026-10-02", 84881, 87147, 83848, 84518),
    ("2026-10-01", 83607, 85277, 83241, 84906),
    ("2026-09-30", 83719, 85567, 83048, 83621),
    ("2026-09-29", 83524, 84599, 82811, 83719),
    ("2026-09-28", 84451, 84979, 82700, 83532),
    ("2026-09-27", 84419, 85058, 84107, 84455),
    ("2026-09-26", 84071, 84381, 83760, 84375),
    ("2026-09-25", 84355, 85172, 83161, 84071),
    ("2026-09-24", 84352, 84834, 82911, 84355),
    ("2026-09-23", 86194, 87280, 83495, 84340),
    ("2026-09-22", 86608, 86672, 85106, 86194),
    ("2026-09-21", 81184, 87371, 80890, 86617),
    ("2026-09-20", 81227, 81498, 80112, 81191),
    ("2026-09-19", 80869, 81904, 80801, 81227),
    ("2026-09-18", 76350, 81390, 76210, 80880),
    ("2026-09-17", 76140, 77110, 75920, 76350),
    ("2026-09-16", 75580, 76500, 74910, 76140),
    ("2026-09-15", 78180, 78240, 74890, 75580),
    ("2026-09-14", 76800, 79590, 76350, 78180),
    ("2026-09-13", 77260, 77430, 76460, 76800),
    ("2026-01-03", 89945, 90680, 89328, 90603),
    ("2026-01-02", 88733, 90884, 88299, 89945),
    ("2026-01-01", 87508, 88803, 87399, 88732),
    ("2025-12-31", 88430, 89080, 87131, 87509),
    ("2025-12-30", 87134, 89298, 86736, 88430),
    ("2025-12-29", 87836, 90299, 86718, 87138),
    ("2025-12-28", 87799, 87987, 87395, 87836),
    ("2025-12-27", 87301, 87875, 87183, 87802),
    ("2025-12-26", 87236, 89459, 86628, 87301),
    ("2025-12-25", 87608, 88502, 86949, 87235),
    ("2025-12-24", 87404, 87957, 86412, 87612),
    ("2025-12-23", 88490, 88898, 86607, 87414),
    ("2025-12-22", 88621, 90502, 87908, 88490),
    ("2025-12-21", 88345, 89028, 87613, 88622),
    ("2025-12-20", 88102, 88497, 87925, 88344),
    ("2025-12-19", 85476, 89339, 85108, 88103),
    ("2025-12-04", 93454, 94038, 90976, 92142),
    ("2025-12-03", 91345, 94061, 91056, 93528),
    ("2025-12-02", 86323, 92317, 86202, 91350),
    ("2025-12-01", 90389, 90398, 83862, 86322),
    ("2025-11-30", 90838, 91965, 90394, 90394),
    ("2025-11-29", 90919, 91188, 90260, 90852),
    ("2025-11-28", 91285, 92969, 90257, 90919),
    ("2025-11-27", 90518, 91898, 90090, 91285),
    ("2025-11-26", 87346, 90581, 86317, 90518),
    ("2025-11-25", 88270, 88457, 86131, 87342),
]

# 构建 DataFrame
qqq_df = pd.DataFrame(qqq_data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
qqq_df['date'] = pd.to_datetime(qqq_df['date'])
qqq_df = qqq_df.sort_values('date').reset_index(drop=True)

btc_df = pd.DataFrame(btc_data, columns=['date', 'open', 'high', 'low', 'close'])
btc_df['date'] = pd.to_datetime(btc_df['date'])
btc_df = btc_df.sort_values('date').reset_index(drop=True)

# 计算对数收益率
qqq_df['log_return'] = np.log(qqq_df['close'] / qqq_df['close'].shift(1))
btc_df['log_return'] = np.log(btc_df['close'] / btc_df['close'].shift(1))

qqq_ret = qqq_df.dropna(subset=['log_return']).copy()
btc_ret = btc_df.dropna(subset=['log_return']).copy()

# ============================================================
# 2. 已实现波动率（Realized Volatility）
# ============================================================

def calculate_realized_volatility(returns, window=20, annualize=252):
    rv = returns.rolling(window=window).std() * np.sqrt(annualize)
    return rv

qqq_rv_20 = calculate_realized_volatility(qqq_ret['log_return'], window=20, annualize=252)
btc_rv_20 = calculate_realized_volatility(btc_ret['log_return'], window=20, annualize=365)
qqq_rv_60 = calculate_realized_volatility(qqq_ret['log_return'], window=60, annualize=252)
btc_rv_60 = calculate_realized_volatility(btc_ret['log_return'], window=60, annualize=365)

print("QQQ 20-day RV (annualized):")
print(qqq_rv_20.dropna())
print("\nBTC 20-day RV (annualized):")
print(btc_rv_20.dropna())

# ============================================================
# 3. GARCH(1,1) 模型
# ============================================================

def garch11_log_likelihood(params, returns):
    omega, alpha, beta = params
    if omega <= 0 or alpha <= 0 or beta <= 0 or alpha + beta >= 1:
        return 1e10
    n = len(returns)
    sigma2 = np.zeros(n)
    log_likelihood = 0
    sigma2[0] = np.var(returns)
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        log_likelihood -= 0.5 * (np.log(2 * np.pi) + np.log(sigma2[t]) + returns[t]**2 / sigma2[t])
    return -log_likelihood

def fit_garch11(returns):
    initial_guess = [np.var(returns) * 0.05, 0.05, 0.90]
    bounds = [(1e-8, None), (1e-8, 0.5), (0.1, 0.99)]
    result = minimize(garch11_log_likelihood, initial_guess, args=(returns,), 
                      method='Nelder-Mead', bounds=bounds)
    omega, alpha, beta = result.x
    n = len(returns)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(returns)
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
    return [omega, alpha, beta], sigma2

qqq_returns = qqq_ret['log_return'].values
qqq_garch_params, qqq_sigma2 = fit_garch11(qqq_returns)
print("\nQQQ GARCH(1,1) parameters:")
print(f"  omega (ω) = {qqq_garch_params[0]:.6f}")
print(f"  alpha (α) = {qqq_garch_params[1]:.4f}")
print(f"  beta (β)  = {qqq_garch_params[2]:.4f}")
print(f"  alpha + beta = {qqq_garch_params[1] + qqq_garch_params[2]:.4f}")

btc_returns = btc_ret['log_return'].values
btc_garch_params, btc_sigma2 = fit_garch11(btc_returns)
print("\nBTC GARCH(1,1) parameters:")
print(f"  omega (ω) = {btc_garch_params[0]:.6f}")
print(f"  alpha (α) = {btc_garch_params[1]:.4f}")
print(f"  beta (β)  = {btc_garch_params[2]:.4f}")
print(f"  alpha + beta = {btc_garch_params[1] + btc_garch_params[2]:.4f}")

qqq_garch_vol_annual = np.sqrt(qqq_sigma2[-1]) * np.sqrt(252)
btc_garch_vol_annual = np.sqrt(btc_sigma2[-1]) * np.sqrt(365)

print(f"\nQQQ GARCH predicted annualized vol: {qqq_garch_vol_annual:.4f} ({qqq_garch_vol_annual*100:.2f}%)")
print(f"BTC GARCH predicted annualized vol: {btc_garch_vol_annual:.4f} ({btc_garch_vol_annual*100:.2f}%)")

# ============================================================
# 4. 隐含波动率（Implied Volatility）
# ============================================================

qqq_iv = 0.20
btc_iv = 0.50

print(f"\nQQQ Implied Volatility: {qqq_iv*100:.2f}%")
print(f"BTC Implied Volatility: {btc_iv*100:.2f}%")

# ============================================================
# 5. 波动率比较与信号
# ============================================================

qqq_rv_latest = qqq_rv_20.dropna().iloc[-1] if not qqq_rv_20.dropna().empty else np.nan
btc_rv_latest = btc_rv_20.dropna().iloc[-1] if not btc_rv_20.dropna().empty else np.nan

print(f"\nLatest QQQ 20-day RV: {qqq_rv_latest:.4f} ({qqq_rv_latest*100:.2f}%)")
print(f"Latest BTC 20-day RV: {btc_rv_latest:.4f} ({btc_rv_latest*100:.2f}%)")

qqq_iv_rv_ratio = qqq_iv / qqq_rv_latest if not np.isnan(qqq_rv_latest) else np.nan
btc_iv_rv_ratio = btc_iv / btc_rv_latest if not np.isnan(btc_rv_latest) else np.nan

print(f"\nQQQ IV/RV ratio: {qqq_iv_rv_ratio:.4f}")
print(f"BTC IV/RV ratio: {btc_iv_rv_ratio:.4f}")

# ============================================================
# 6. 生成图表
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('波动率指标分析：QQQ vs BTC-USD', fontsize=14, fontweight='bold')

ax1 = axes[0, 0]
ax1.plot(qqq_ret['date'], qqq_ret['close'], 'b-', label='QQQ 收盘价', linewidth=1.5)
ax1.set_xlabel('日期')
ax1.set_ylabel('价格', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_title('QQQ 价格走势')
ax1.legend(loc='upper left')
ax1b = ax1.twinx()
ax1b.plot(qqq_ret['date'], qqq_rv_20, 'r--', label='20日已实现波动率', linewidth=1.5)
ax1b.set_ylabel('已实现波动率（年化）', color='r')
ax1b.tick_params(axis='y', labelcolor='r')
ax1b.legend(loc='upper right')

ax2 = axes[0, 1]
ax2.plot(btc_ret['date'], btc_ret['close'], 'g-', label='BTC 收盘价', linewidth=1.5)
ax2.set_xlabel('日期')
ax2.set_ylabel('价格', color='g')
ax2.tick_params(axis='y', labelcolor='g')
ax2.set_title('BTC-USD 价格走势')
ax2.legend(loc='upper left')
ax2b = ax2.twinx()
ax2b.plot(btc_ret['date'], btc_rv_20, 'r--', label='20日已实现波动率', linewidth=1.5)
ax2b.set_ylabel('已实现波动率（年化）', color='r')
ax2b.tick_params(axis='y', labelcolor='r')
ax2b.legend(loc='upper right')

ax3 = axes[1, 0]
ax3.plot(qqq_ret['date'], np.sqrt(qqq_sigma2) * np.sqrt(252), 'b-', label='QQQ GARCH波动率', linewidth=1.5)
ax3.plot(btc_ret['date'], np.sqrt(btc_sigma2) * np.sqrt(365), 'g-', label='BTC GARCH波动率', linewidth=1.5)
ax3.set_xlabel('日期')
ax3.set_ylabel('GARCH 条件波动率（年化）')
ax3.set_title('GARCH(1,1) 条件波动率')
ax3.legend()

ax4 = axes[1, 1]
categories = ['QQQ', 'BTC']
rv_values = [qqq_rv_latest, btc_rv_latest]
iv_values = [qqq_iv, btc_iv]
garch_values = [qqq_garch_vol_annual, btc_garch_vol_annual]
x = np.arange(len(categories))
width = 0.25
bars1 = ax4.bar(x - width, rv_values, width, label='已实现波动率 (RV)', color='steelblue')
bars2 = ax4.bar(x, iv_values, width, label='隐含波动率 (IV)', color='coral')
bars3 = ax4.bar(x + width, garch_values, width, label='GARCH预测波动率', color='seagreen')
ax4.set_xlabel('资产')
ax4.set_ylabel('年化波动率')
ax4.set_title('三种波动率指标比较')
ax4.set_xticks(x)
ax4.set_xticklabels(categories)
ax4.legend()
for bar in bars1:
    height = bar.get_height()
    ax4.annotate(f'{height*100:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax4.annotate(f'{height*100:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for bar in bars3:
    height = bar.get_height()
    ax4.annotate(f'{height*100:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(r'E:\agent_dev\multi-agent\tmp\volatility_analysis_charts.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nChart saved to E:\\agent_dev\\multi-agent\\tmp\\volatility_analysis_charts.png")
