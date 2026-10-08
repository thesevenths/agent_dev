import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 准备数据
# 纳斯达克100指数（^NDX）数据
ndx_data = {
    'date': ['2025-11-28', '2025-12-01', '2025-12-02', '2025-12-03', '2025-12-04', '2025-12-05', 
             '2025-12-18', '2025-12-19', '2025-12-22', '2025-12-23', '2025-12-24', '2025-12-26', 
             '2025-12-29', '2025-12-30', '2025-12-31', '2026-01-02', '2026-01-05', '2026-01-06', 
             '2026-02-25', '2026-02-26', '2026-02-27', '2026-03-02', '2026-03-03', '2026-03-04', '2026-03-05'],
    'close': [25434.89, 25342.85, 25555.86, 25606.54, 25581.70, 25692.05, 
              25019.37, 25346.18, 25461.70, 25587.83, 25656.15, 25644.39, 
              25525.56, 25462.56, 25249.85, 25206.17, 25401.32, 25639.71, 
              25329.04, 25034.37, 24960.04, 24992.60, 24720.08, 25093.68, 25020.41]
}

# 比特币（BTC-USD）数据
btc_data = {
    'date': ['2025-11-25', '2025-11-26', '2025-11-27', '2025-11-28', '2025-11-29', '2025-11-30', 
             '2025-12-01', '2025-12-02', '2025-12-03', '2025-12-19', '2025-12-20', '2025-12-21', 
             '2025-12-22', '2025-12-23', '2025-12-24', '2025-12-25', '2025-12-26', '2025-12-27', 
             '2025-12-28', '2025-12-29', '2025-12-30', '2025-12-31', '2026-01-01', '2026-01-02', 
             '2026-09-13', '2026-09-14', '2026-09-15', '2026-09-16', '2026-09-17', '2026-09-18', 
             '2026-09-19', '2026-09-20', '2026-09-21', '2026-09-22', '2026-09-23', '2026-09-24', 
             '2026-09-25', '2026-09-26', '2026-09-27', '2026-09-28', '2026-09-29', '2026-09-30', 
             '2026-10-01', '2026-10-02', '2026-10-03', '2026-10-04', '2026-10-05'],
    'close': [87341.89, 90518.37, 91285.38, 90919.27, 90851.76, 90394.31, 
              86321.57, 91350.20, 93527.80, 88103.38, 88344.00, 88621.75, 
              88490.02, 87414.00, 87611.96, 87234.74, 87301.43, 87802.16, 
              87835.84, 87138.14, 88430.13, 87508.83, 88731.98, 89944.70, 
              76800, 78180, 75580, 76140, 76350, 80880, 
              81230, 81160, 86590, 86200, 84380, 84390, 
              84090, 84420, 84460, 83460, 83640, 83560, 
              84850, 84500, 84740, 86510, 85950]
}

# 创建DataFrame
ndx_df = pd.DataFrame(ndx_data)
btc_df = pd.DataFrame(btc_data)

# 2. 计算收益率
ndx_df['return'] = ndx_df['close'].pct_change()
btc_df['return'] = btc_df['close'].pct_change()

# 3. 计算已实现波动率（20日滚动）
ndx_df['rv_20d'] = ndx_df['return'].rolling(window=20).std() * np.sqrt(252)
btc_df['rv_20d'] = btc_df['return'].rolling(window=20).std() * np.sqrt(365)

# 4. 设计波动率交易策略
# 策略规则：
# - 当波动率低于阈值（如20%）时，买入
# - 当波动率高于阈值（如30%）时，卖出

def volatility_strategy(df, low_threshold=0.20, high_threshold=0.30):
    """
    波动率交易策略
    :param df: 包含日期、收盘价、波动率的数据
    :param low_threshold: 低波动率阈值（买入）
    :param high_threshold: 高波动率阈值（卖出）
    :return: 包含策略信号的数据
    """
    df = df.copy()
    df['signal'] = 0  # 0: 无操作, 1: 买入, -1: 卖出
    
    # 策略逻辑
    for i in range(1, len(df)):
        if df['rv_20d'].iloc[i-1] < low_threshold:
            df['signal'].iloc[i] = 1  # 买入
        elif df['rv_20d'].iloc[i-1] > high_threshold:
            df['signal'].iloc[i] = -1  # 卖出
    
    return df

# 应用策略
ndx_strategy = volatility_strategy(ndx_df, low_threshold=0.15, high_threshold=0.25)
btc_strategy = volatility_strategy(btc_df, low_threshold=0.40, high_threshold=0.60)

# 5. 回测策略表现
def backtest_strategy(df, initial_capital=100000):
    """
    回测策略表现
    :param df: 包含策略信号的数据
    :param initial_capital: 初始资金
    :return: 回测结果
    """
    df = df.copy()
    df['position'] = 0  # 持仓：0=空仓, 1=满仓
    df['portfolio_value'] = initial_capital
    
    # 回测逻辑
    for i in range(1, len(df)):
        # 更新持仓
        if df['signal'].iloc[i] == 1:
            df['position'].iloc[i] = 1
        elif df['signal'].iloc[i] == -1:
            df['position'].iloc[i] = 0
        
        # 计算组合价值
        if df['position'].iloc[i] == 1:
            df['portfolio_value'].iloc[i] = df['portfolio_value'].iloc[i-1] * (1 + df['return'].iloc[i])
        else:
            df['portfolio_value'].iloc[i] = df['portfolio_value'].iloc[i-1]
    
    # 计算绩效指标
    df['cumulative_return'] = df['portfolio_value'] / initial_capital - 1
    df['max_drawdown'] = (df['portfolio_value'] - df['portfolio_value'].cummax()) / df['portfolio_value'].cummax()
    
    # 夏普比率
    risk_free_rate = 0.05
    excess_return = df['return'] - risk_free_rate / 252
    sharpe_ratio = np.sqrt(252) * excess_return.mean() / excess_return.std()
    
    return df, sharpe_ratio

# 回测
ndx_backtest, ndx_sharpe = backtest_strategy(ndx_strategy)
btc_backtest, btc_sharpe = backtest_strategy(btc_strategy)

# 6. 计算绩效指标
def calculate_performance_metrics(df, initial_capital=100000):
    """
    计算绩效指标
    :param df: 回测结果
    :param initial_capital: 初始资金
    :return: 绩效指标字典
    """
    total_return = df['portfolio_value'].iloc[-1] / initial_capital - 1
    max_drawdown = df['max_drawdown'].min()
    annualized_return = (1 + total_return) ** (252 / len(df)) - 1
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': 0  # 将在外部计算
    }

ndx_metrics = calculate_performance_metrics(ndx_backtest)
btc_metrics = calculate_performance_metrics(btc_backtest)
ndx_metrics['sharpe_ratio'] = ndx_sharpe
btc_metrics['sharpe_ratio'] = btc_sharpe

# 7. 可视化
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 纳斯达克100指数
axes[0, 0].plot(ndx_backtest['date'], ndx_backtest['close'], label='NDX Close', color='blue')
axes[0, 0].plot(ndx_backtest['date'], ndx_backtest['portfolio_value'] / 100000 * ndx_backtest['close'].iloc[0], 
                label='Strategy Portfolio', color='red', linestyle='--')
axes[0, 0].set_title('Nasdaq 100 Index: Price & Strategy Performance')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Price / Portfolio Value')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 比特币
axes[0, 1].plot(btc_backtest['date'], btc_backtest['close'], label='BTC Close', color='blue')
axes[0, 1].plot(btc_backtest['date'], btc_backtest['portfolio_value'] / 100000 * btc_backtest['close'].iloc[0], 
                label='Strategy Portfolio', color='red', linestyle='--')
axes[0, 1].set_title('Bitcoin: Price & Strategy Performance')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Price / Portfolio Value')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 纳斯达克100指数波动率
axes[1, 0].plot(ndx_backtest['date'], ndx_backtest['rv_20d'], label='20D Realized Volatility', color='green')
axes[1, 0].axhline(y=0.15, color='blue', linestyle='--', label='Low Threshold (15%)')
axes[1, 0].axhline(y=0.25, color='red', linestyle='--', label='High Threshold (25%)')
axes[1, 0].set_title('Nasdaq 100 Index: 20D Realized Volatility')
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('Volatility')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 比特币波动率
axes[1, 1].plot(btc_backtest['date'], btc_backtest['rv_20d'], label='20D Realized Volatility', color='green')
axes[1, 1].axhline(y=0.40, color='blue', linestyle='--', label='Low Threshold (40%)')
axes[1, 1].axhline(y=0.60, color='red', linestyle='--', label='High Threshold (60%)')
axes[1, 1].set_title('Bitcoin: 20D Realized Volatility')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Volatility')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('E:\\agent_dev\\multi-agent\\tmp\\step3_volatility_strategy_backtest.png', dpi=150, bbox_inches='tight')
plt.show()

# 8. 输出结果
print("=" * 80)
print("波动率交易策略回测结果")
print("=" * 80)

print("\n【纳斯达克100指数（^NDX）】")
print(f"总收益率: {ndx_metrics['total_return']:.2%}")
print(f"年化收益率: {ndx_metrics['annualized_return']:.2%}")
print(f"最大回撤: {ndx_metrics['max_drawdown']:.2%}")
print(f"夏普比率: {ndx_metrics['sharpe_ratio']:.2f}")

print("\n【比特币（BTC-USD）】")
print(f"总收益率: {btc_metrics['total_return']:.2%}")
print(f"年化收益率: {btc_metrics['annualized_return']:.2%}")
print(f"最大回撤: {btc_metrics['max_drawdown']:.2%}")
print(f"夏普比率: {btc_metrics['sharpe_ratio']:.2f}")

print("\n" + "=" * 80)
print("策略说明")
print("=" * 80)
print("策略规则：")
print("- 当20日已实现波动率低于低阈值时，买入")
print("- 当20日已实现波动率高于高阈值时，卖出")
print("- 纳斯达克100指数：低阈值15%，高阈值25%")
print("- 比特币：低阈值40%，高阈值60%")

print("\n业务原因：")
print("- 低波动率时期，市场相对稳定，适合买入")
print("- 高波动率时期，市场风险较高，适合卖出")
print("- 通过波动率阈值，实现低买高卖的策略")
