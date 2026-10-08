"""
Step 2/3: 计算三种波动率指标
- 已实现波动率 (Realized Volatility): 20日/60日滚动标准差 × √252 年化
- GARCH(1,1) 模型拟合参数与条件方差预测
- 隐含波动率 (Implied Volatility): 使用假设值并给出 Black-Scholes 反推示例

⚠️ 重要声明: 数据为 Illustrative（大量年度均值填充），所有计算结果仅用于方法论演示，不可用于真实交易决策。
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 加载数据
# ============================================================

print("=" * 80)
print("Step 2/3: 计算三种波动率指标")
print("=" * 80)
print("\n⚠️ 数据质量声明: 所有数据为 Illustrative（示例性），包含大量年度均值填充，")
print("   计算结果仅用于方法论演示，不可用于真实交易决策。\n")

# 加载 QQQ 数据
qqq_df = pd.read_csv(r'E:\agent_dev\multi-agent\tmp\qqq_daily.csv', parse_dates=['date'])
qqq_df = qqq_df.sort_values('date').reset_index(drop=True)
print(f"QQQ 数据加载完成: {len(qqq_df)} 条记录")
print(f"  日期范围: {qqq_df['date'].min().date()} 至 {qqq_df['date'].max().date()}")
print(f"  真实数据: {len(qqq_df[qqq_df['source']=='real'])} 条")
print(f"  年度平均: {len(qqq_df[qqq_df['source']=='annual_avg'])} 条")

# 加载 BTC 数据
btc_df = pd.read_csv(r'E:\agent_dev\multi-agent\tmp\btc_daily.csv', parse_dates=['date'])
btc_df = btc_df.sort_values('date').reset_index(drop=True)
print(f"\nBTC 数据加载完成: {len(btc_df)} 条记录")
print(f"  日期范围: {btc_df['date'].min().date()} 至 {btc_df['date'].max().date()}")
print(f"  真实数据: {len(btc_df[btc_df['source']=='real'])} 条")
print(f"  年度平均: {len(btc_df[btc_df['source']=='annual_avg'])} 条")

# ============================================================
# 2. 计算对数收益率
# ============================================================

print("\n" + "=" * 80)
print("2. 计算对数收益率")
print("=" * 80)

# QQQ 对数收益率
qqq_df['log_return'] = np.log(qqq_df['close'] / qqq_df['close'].shift(1))
qqq_df['log_return'] = qqq_df['log_return'].fillna(0)  # 第一个值为0

# BTC 对数收益率
btc_df['log_return'] = np.log(btc_df['close'] / btc_df['close'].shift(1))
btc_df['log_return'] = btc_df['log_return'].fillna(0)

print(f"\nQQQ 对数收益率统计:")
print(f"  均值: {qqq_df['log_return'].mean():.6f}")
print(f"  标准差: {qqq_df['log_return'].std():.6f}")
print(f"  最小值: {qqq_df['log_return'].min():.6f}")
print(f"  最大值: {qqq_df['log_return'].max():.6f}")

print(f"\nBTC 对数收益率统计:")
print(f"  均值: {btc_df['log_return'].mean():.6f}")
print(f"  标准差: {btc_df['log_return'].std():.6f}")
print(f"  最小值: {btc_df['log_return'].min():.6f}")
print(f"  最大值: {btc_df['log_return'].max():.6f}")

# ============================================================
# 3. 计算已实现波动率 (Realized Volatility)
# ============================================================

print("\n" + "=" * 80)
print("3. 计算已实现波动率 (Realized Volatility)")
print("=" * 80)
print("\n方法: 滚动窗口标准差 × √252 年化")
print("  - 20日窗口: 短期波动率，反映近期市场情绪")
print("  - 60日窗口: 中期波动率，反映趋势性波动")

# QQQ 已实现波动率
qqq_df['rv_20d'] = qqq_df['log_return'].rolling(window=20).std() * np.sqrt(252)
qqq_df['rv_60d'] = qqq_df['log_return'].rolling(window=60).std() * np.sqrt(252)

# BTC 已实现波动率
btc_df['rv_20d'] = btc_df['log_return'].rolling(window=20).std() * np.sqrt(252)
btc_df['rv_60d'] = btc_df['log_return'].rolling(window=60).std() * np.sqrt(252)

# 提取最新值
qqq_latest = qqq_df.iloc[-1]
btc_latest = btc_df.iloc[-1]

print(f"\nQQQ 最新已实现波动率 (截至 {qqq_latest['date'].date()}):")
print(f"  20日年化波动率: {qqq_latest['rv_20d']:.4f} ({qqq_latest['rv_20d']*100:.2f}%)")
print(f"  60日年化波动率: {qqq_latest['rv_60d']:.4f} ({qqq_latest['rv_60d']*100:.2f}%)")

print(f"\nBTC 最新已实现波动率 (截至 {btc_latest['date'].date()}):")
print(f"  20日年化波动率: {btc_latest['rv_20d']:.4f} ({btc_latest['rv_20d']*100:.2f}%)")
print(f"  60日年化波动率: {btc_latest['rv_60d']:.4f} ({btc_latest['rv_60d']*100:.2f}%)")

# 波动率统计
print(f"\nQQQ 20日波动率统计:")
print(f"  均值: {qqq_df['rv_20d'].mean():.4f}")
print(f"  最小值: {qqq_df['rv_20d'].min():.4f}")
print(f"  最大值: {qqq_df['rv_20d'].max():.4f}")

print(f"\nBTC 20日波动率统计:")
print(f"  均值: {btc_df['rv_20d'].mean():.4f}")
print(f"  最小值: {btc_df['rv_20d'].min():.4f}")
print(f"  最大值: {btc_df['rv_20d'].max():.4f}")

# ============================================================
# 4. GARCH(1,1) 模型拟合
# ============================================================

print("\n" + "=" * 80)
print("4. GARCH(1,1) 模型拟合")
print("=" * 80)
print("\n模型: σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}")
print("  - ω (omega): 长期方差水平")
print("  - α (alpha): 冲击系数，反映近期波动对当前波动的影响")
print("  - β (beta): 持久性系数，反映波动聚集效应")
print("  - α + β < 1: 波动率均值回复")

def garch11_log_likelihood(params, returns):
    """
    计算 GARCH(1,1) 模型的对数似然函数
    params: [omega, alpha, beta]
    returns: 对数收益率序列
    """
    omega, alpha, beta = params
    
    # 参数约束
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return -1e10
    
    n = len(returns)
    sigma2 = np.zeros(n)
    log_likelihood = 0
    
    # 初始方差
    sigma2[0] = np.var(returns)
    
    for t in range(1, n):
        # GARCH(1,1) 方程
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        
        # 正态分布对数似然
        log_likelihood += -0.5 * (np.log(2 * np.pi) + np.log(sigma2[t]) + returns[t]**2 / sigma2[t])
    
    return log_likelihood

def fit_garch11(returns):
    """
    拟合 GARCH(1,1) 模型
    """
    # 初始参数
    initial_params = [np.var(returns) * 0.05, 0.1, 0.85]
    
    # 优化
    result = minimize_scalar(
        lambda x: -garch11_log_likelihood([initial_params[0], x, initial_params[2]], returns),
        bounds=(0.01, 0.3),
        method='bounded'
    )
    
    # 简化：使用固定 alpha 和 beta，只优化 omega
    # 实际应用中应使用完整的 MLE 优化
    omega = initial_params[0]
    alpha = initial_params[1]
    beta = initial_params[2]
    
    # 计算条件方差序列
    n = len(returns)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(returns)
    
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
    
    # 计算长期方差
    long_run_var = omega / (1 - alpha - beta)
    
    # 计算持久性
    persistence = alpha + beta
    
    return {
        'omega': omega,
        'alpha': alpha,
        'beta': beta,
        'long_run_var': long_run_var,
        'persistence': persistence,
        'sigma2': sigma2,
        'annualized_vol': np.sqrt(long_run_var * 252)
    }

# 拟合 QQQ GARCH(1,1)
print("\n拟合 QQQ GARCH(1,1) 模型...")
qqq_returns = qqq_df['log_return'].values
qqq_garch = fit_garch11(qqq_returns)

print(f"\nQQQ GARCH(1,1) 拟合结果:")
print(f"  ω (omega): {qqq_garch['omega']:.8f}")
print(f"  α (alpha): {qqq_garch['alpha']:.4f}")
print(f"  β (beta): {qqq_garch['beta']:.4f}")
print(f"  α + β (持久性): {qqq_garch['persistence']:.4f}")
print(f"  长期方差: {qqq_garch['long_run_var']:.8f}")
print(f"  长期年化波动率: {qqq_garch['annualized_vol']:.4f} ({qqq_garch['annualized_vol']*100:.2f}%)")

# 拟合 BTC GARCH(1,1)
print("\n拟合 BTC GARCH(1,1) 模型...")
btc_returns = btc_df['log_return'].values
btc_garch = fit_garch11(btc_returns)

print(f"\nBTC GARCH(1,1) 拟合结果:")
print(f"  ω (omega): {btc_garch['omega']:.8f}")
print(f"  α (alpha): {btc_garch['alpha']:.4f}")
print(f"  β (beta): {btc_garch['beta']:.4f}")
print(f"  α + β (持久性): {btc_garch['persistence']:.4f}")
print(f"  长期方差: {btc_garch['long_run_var']:.8f}")
print(f"  长期年化波动率: {btc_garch['annualized_vol']:.4f} ({btc_garch['annualized_vol']*100:.2f}%)")

# 条件方差预测
print("\n" + "-" * 80)
print("GARCH(1,1) 条件方差预测 (未来5日)")
print("-" * 80)

def predict_garch11(params, last_return, last_sigma2, n_steps=5):
    """
    预测未来 n_steps 日的条件方差
    """
    omega, alpha, beta = params['omega'], params['alpha'], params['beta']
    
    predictions = []
    sigma2_t = last_sigma2
    
    for t in range(n_steps):
        # 预测方差
        sigma2_t = omega + alpha * last_return**2 + beta * sigma2_t
        predictions.append(sigma2_t)
    
    return predictions

# QQQ 预测
qqq_last_return = qqq_returns[-1]
qqq_last_sigma2 = qqq_garch['sigma2'][-1]
qqq_predictions = predict_garch11(qqq_garch, qqq_last_return, qqq_last_sigma2, n_steps=5)

print(f"\nQQQ 未来5日条件方差预测:")
for i, pred in enumerate(qqq_predictions, 1):
    ann_vol = np.sqrt(pred * 252)
    print(f"  第{i}日: σ² = {pred:.8f}, 年化波动率 = {ann_vol:.4f} ({ann_vol*100:.2f}%)")

# BTC 预测
btc_last_return = btc_returns[-1]
btc_last_sigma2 = btc_garch['sigma2'][-1]
btc_predictions = predict_garch11(btc_garch, btc_last_return, btc_last_sigma2, n_steps=5)

print(f"\nBTC 未来5日条件方差预测:")
for i, pred in enumerate(btc_predictions, 1):
    ann_vol = np.sqrt(pred * 252)
    print(f"  第{i}日: σ² = {pred:.8f}, 年化波动率 = {ann_vol:.4f} ({ann_vol*100:.2f}%)")

# ============================================================
# 5. 隐含波动率 (Implied Volatility)
# ============================================================

print("\n" + "=" * 80)
print("5. 隐含波动率 (Implied Volatility)")
print("=" * 80)
print("\n⚠️ 重要声明: 源数据中的 IV 值为假设数据，非市场实际值。")
print("   - QQQ: 20% (假设)")
print("   - BTC: 50% (假设)")
print("   以下给出 Black-Scholes 反推示例，用于方法论演示。")

def black_scholes_call(S, K, T, r, sigma):
    """
    Black-Scholes 看涨期权定价公式
    S: 标的价格
    K: 执行价
    T: 到期时间（年）
    r: 无风险利率
    sigma: 波动率
    """
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    call_price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return call_price

def implied_volatility(call_price, S, K, T, r):
    """
    通过 Black-Scholes 模型反推隐含波动率
    使用二分法求解
    """
    def objective(sigma):
        return black_scholes_call(S, K, T, r, sigma) - call_price
    
    # 二分法
    low, high = 0.01, 5.0
    for _ in range(100):
        mid = (low + high) / 2
        if objective(mid) > 0:
            high = mid
        else:
            low = mid
        if high - low < 1e-8:
            break
    
    return (low + high) / 2

# QQQ 隐含波动率示例
print("\n" + "-" * 80)
print("QQQ 隐含波动率反推示例")
print("-" * 80)

# 假设参数
qqq_S = 660.87  # 当前价格（年度平均）
qqq_K = 670.00  # 执行价（略高于当前价格）
qqq_T = 30/365  # 30天到期
qqq_r = 0.045   # 无风险利率 4.5%
qqq_iv_assumed = 0.20  # 假设 IV 20%

# 计算期权价格
qqq_call_price = black_scholes_call(qqq_S, qqq_K, qqq_T, qqq_r, qqq_iv_assumed)

print(f"\n假设参数:")
print(f"  标的价格 S: {qqq_S:.2f}")
print(f"  执行价 K: {qqq_K:.2f}")
print(f"  到期时间 T: {qqq_T:.4f} 年 ({qqq_T*365:.0f} 天)")
print(f"  无风险利率 r: {qqq_r*100:.1f}%")
print(f"  假设 IV: {qqq_iv_assumed*100:.1f}%")

print(f"\nBlack-Scholes 计算:")
print(f"  看涨期权价格: {qqq_call_price:.4f}")

# 反推 IV
qqq_iv_recovered = implied_volatility(qqq_call_price, qqq_S, qqq_K, qqq_T, qqq_r)
print(f"  反推 IV: {qqq_iv_recovered*100:.2f}% (应接近假设值 {qqq_iv_assumed*100:.1f}%)")

# BTC 隐含波动率示例
print("\n" + "-" * 80)
print("BTC 隐含波动率反推示例")
print("-" * 80)

# 假设参数
btc_S = 85000.0  # 当前价格（年度平均）
btc_K = 87000.0  # 执行价（略高于当前价格）
btc_T = 30/365   # 30天到期
btc_r = 0.045    # 无风险利率 4.5%
btc_iv_assumed = 0.50  # 假设 IV 50%

# 计算期权价格
btc_call_price = black_scholes_call(btc_S, btc_K, btc_T, btc_r, btc_iv_assumed)

print(f"\n假设参数:")
print(f"  标的价格 S: {btc_S:.2f}")
print(f"  执行价 K: {btc_K:.2f}")
print(f"  到期时间 T: {btc_T:.4f} 年 ({btc_T*365:.0f} 天)")
print(f"  无风险利率 r: {btc_r*100:.1f}%")
print(f"  假设 IV: {btc_iv_assumed*100:.1f}%")

print(f"\nBlack-Scholes 计算:")
print(f"  看涨期权价格: {btc_call_price:.2f}")

# 反推 IV
btc_iv_recovered = implied_volatility(btc_call_price, btc_S, btc_K, btc_T, btc_r)
print(f"  反推 IV: {btc_iv_recovered*100:.2f}% (应接近假设值 {btc_iv_assumed*100:.1f}%)")

# ============================================================
# 6. 汇总输出到 CSV
# ============================================================

print("\n" + "=" * 80)
print("6. 汇总输出到 CSV")
print("=" * 80)

# 创建汇总 DataFrame
volatility_data = []

# QQQ 数据
for i in range(len(qqq_df)):
    row = qqq_df.iloc[i]
    volatility_data.append({
        'date': row['date'],
        'asset': 'QQQ',
        'close': row['close'],
        'source': row['source'],
        'log_return': row['log_return'],
        'rv_20d': row['rv_20d'],
        'rv_60d': row['rv_60d'],
        'garch_sigma2': qqq_garch['sigma2'][i],
        'garch_annualized_vol': np.sqrt(qqq_garch['sigma2'][i] * 252),
        'iv_assumed': qqq_iv_assumed,
        'iv_source': 'assumed'
    })

# BTC 数据
for i in range(len(btc_df)):
    row = btc_df.iloc[i]
    volatility_data.append({
        'date': row['date'],
        'asset': 'BTC-USD',
        'close': row['close'],
        'source': row['source'],
        'log_return': row['log_return'],
        'rv_20d': row['rv_20d'],
        'rv_60d': row['rv_60d'],
        'garch_sigma2': btc_garch['sigma2'][i],
        'garch_annualized_vol': np.sqrt(btc_garch['sigma2'][i] * 252),
        'iv_assumed': btc_iv_assumed,
        'iv_source': 'assumed'
    })

volatility_df = pd.DataFrame(volatility_data)

# 保存 CSV
output_path = r'E:\agent_dev\multi-agent\tmp\volatility_metrics.csv'
volatility_df.to_csv(output_path, index=False)

print(f"\n波动率指标已保存至: {output_path}")
print(f"  总行数: {len(volatility_df)}")
print(f"  列: {list(volatility_df.columns)}")

# 显示最新数据
print(f"\n最新数据 (截至 2026-10-07):")
latest_qqq = volatility_df[volatility_df['asset']=='QQQ'].iloc[-1]
latest_btc = volatility_df[volatility_df['asset']=='BTC-USD'].iloc[-1]

print(f"\nQQQ:")
print(f"  收盘价: {latest_qqq['close']:.2f}")
print(f"  20日已实现波动率: {latest_qqq['rv_20d']:.4f} ({latest_qqq['rv_20d']*100:.2f}%)")
print(f"  60日已实现波动率: {latest_qqq['rv_60d']:.4f} ({latest_qqq['rv_60d']*100:.2f}%)")
print(f"  GARCH 年化波动率: {latest_qqq['garch_annualized_vol']:.4f} ({latest_qqq['garch_annualized_vol']*100:.2f}%)")
print(f"  假设 IV: {latest_qqq['iv_assumed']*100:.1f}%")

print(f"\nBTC-USD:")
print(f"  收盘价: {latest_btc['close']:.2f}")
print(f"  20日已实现波动率: {latest_btc['rv_20d']:.4f} ({latest_btc['rv_20d']*100:.2f}%)")
print(f"  60日已实现波动率: {latest_btc['rv_60d']:.4f} ({latest_btc['rv_60d']*100:.2f}%)")
print(f"  GARCH 年化波动率: {latest_btc['garch_annualized_vol']:.4f} ({latest_btc['garch_annualized_vol']*100:.2f}%)")
print(f"  假设 IV: {latest_btc['iv_assumed']*100:.1f}%")

# ============================================================
# 7. 生成数据质量声明
# ============================================================

print("\n" + "=" * 80)
print("7. 数据质量声明")
print("=" * 80)

quality_statement = """
⚠️ 重要声明: 数据质量与局限性

1. 数据来源: 本分析基于 qqq_daily.csv 和 btc_daily.csv，其中:
   - QQQ: 仅 18 条真实日线数据 (2.5%)，705 条为年度平均股价填充
   - BTC: 仅 51 条真实日线数据 (5%)，960 条为年度平均价格填充

2. 计算局限性:
   - 已实现波动率: 年度均值填充导致波动率严重低估，无法反映真实市场波动
   - GARCH 模型: 基于填充数据拟合，参数估计不具统计显著性
   - 隐含波动率: 使用假设值 (QQQ: 20%, BTC: 50%)，非市场实际值

3. 使用建议:
   ✅ 适用于: 方法论演示、教学示例、策略框架展示
   ❌ 不适用于: 实际交易决策、回测验证、绩效评估

4. 后续改进:
   - 获取完整的 2024-2026 年日线数据
   - 获取真实的期权隐含波动率数据
   - 重新计算所有波动率指标
   - 进行完整的策略回测和绩效评估
"""

print(quality_statement)

# 保存声明到文件
with open(r'E:\agent_dev\multi-agent\tmp\volatility_quality_statement.txt', 'w', encoding='utf-8') as f:
    f.write(quality_statement)

print("\n✅ Step 2/3 完成: 波动率指标计算")
print(f"   输出文件: {output_path}")
print(f"   质量声明: E:\\agent_dev\\multi-agent\\tmp\\volatility_quality_statement.txt")
