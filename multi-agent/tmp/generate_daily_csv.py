import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================
# 生成完整的 2024-01-01 至 2026-10-07 日线数据
# 策略：
# 1. 使用已提取的日线数据
# 2. 对于缺失的日期，使用年度汇总数据（平均股价）进行插值
# 3. 标注数据缺口
# ============================================================

# 已提取的 QQQ 日线数据
qqq_real = pd.DataFrame([
    {'date': '2026-03-05', 'close': 608.91},
    {'date': '2026-03-04', 'close': 610.75},
    {'date': '2026-03-03', 'close': 601.58},
    {'date': '2026-03-02', 'close': 608.09},
    {'date': '2026-02-27', 'close': 607.29},
    {'date': '2026-02-26', 'close': 609.24},
    {'date': '2026-01-29', 'close': 629.43},
    {'date': '2026-01-28', 'close': 633.22},
    {'date': '2026-01-27', 'close': 631.13},
    {'date': '2026-01-26', 'close': 625.46},
    {'date': '2026-01-23', 'close': 622.72},
    {'date': '2026-01-22', 'close': 620.76},
    {'date': '2026-01-02', 'close': 613.12},
    {'date': '2025-12-31', 'close': 614.31},
    {'date': '2025-12-30', 'close': 619.43},
    {'date': '2025-12-29', 'close': 620.87},
    {'date': '2025-12-26', 'close': 623.89},
    {'date': '2025-12-24', 'close': 623.93},
])
qqq_real['date'] = pd.to_datetime(qqq_real['date'])

# QQQ 年度汇总数据
qqq_annual = {
    2024: {'avg': 459.82, 'open': 397.25, 'high': 533.33, 'low': 391.03, 'close': 507.45},
    2025: {'avg': 546.13, 'open': 506.46, 'high': 633.45, 'low': 413.60, 'close': 612.86},
    2026: {'avg': 660.87, 'open': 611.67, 'high': 745.34, 'low': 557.67, 'close': 716.08},
}

# 生成 QQQ 完整日线数据
def generate_qqq_daily():
    dates = pd.date_range('2024-01-01', '2026-10-07', freq='B')  # 工作日
    qqq_full = pd.DataFrame({'date': dates})
    
    # 标记哪些日期有真实数据
    qqq_full['has_real_data'] = qqq_full['date'].isin(qqq_real['date'])
    
    # 对于有真实数据的日期，使用真实收盘价
    qqq_full = qqq_full.merge(qqq_real, on='date', how='left')
    
    # 对于没有真实数据的日期，使用年度平均股价
    qqq_full['close'] = qqq_full.apply(
        lambda row: row['close'] if pd.notna(row['close']) else qqq_annual[row['date'].year]['avg'],
        axis=1
    )
    
    # 标记数据来源
    qqq_full['source'] = qqq_full['has_real_data'].map({True: 'real', False: 'annual_avg'})
    
    return qqq_full[['date', 'close', 'source']]

qqq_full = generate_qqq_daily()
print("QQQ 完整日线数据:")
print(f"  日期范围: {qqq_full['date'].min()} 至 {qqq_full['date'].max()}")
print(f"  数据条数: {len(qqq_full)}")
print(f"  真实数据: {len(qqq_full[qqq_full['source']=='real'])} 条")
print(f"  年度平均: {len(qqq_full[qqq_full['source']=='annual_avg'])} 条")
print()
print(qqq_full.head(10))
print("...")
print(qqq_full.tail(10))

# 保存 QQQ CSV
qqq_full.to_csv(r'E:\agent_dev\multi-agent\tmp\qqq_daily.csv', index=False)
print("\nQQQ CSV 已保存至: E:\\agent_dev\\multi-agent\\tmp\\qqq_daily.csv")

# ============================================================
# 生成 BTC 完整日线数据
# ============================================================

# 已提取的 BTC 日线数据
btc_real = pd.DataFrame([
    {'date': '2026-10-07', 'close': 84068.0},
    {'date': '2026-10-06', 'close': 85615.0},
    {'date': '2026-10-05', 'close': 85738.0},
    {'date': '2026-10-04', 'close': 86511.0},
    {'date': '2026-10-03', 'close': 84750.0},
    {'date': '2026-10-02', 'close': 84518.0},
    {'date': '2026-10-01', 'close': 84906.0},
    {'date': '2026-09-30', 'close': 83621.0},
    {'date': '2026-09-29', 'close': 83719.0},
    {'date': '2026-09-28', 'close': 83532.0},
    {'date': '2026-09-27', 'close': 84455.0},
    {'date': '2026-09-26', 'close': 84375.0},
    {'date': '2026-09-25', 'close': 84071.0},
    {'date': '2026-09-24', 'close': 84355.0},
    {'date': '2026-09-23', 'close': 84340.0},
    {'date': '2026-09-22', 'close': 86194.0},
    {'date': '2026-09-21', 'close': 86617.0},
    {'date': '2026-09-20', 'close': 81191.0},
    {'date': '2026-09-19', 'close': 81227.0},
    {'date': '2026-09-18', 'close': 80880.0},
    {'date': '2026-09-17', 'close': 76350.0},
    {'date': '2026-09-16', 'close': 76140.0},
    {'date': '2026-09-15', 'close': 75580.0},
    {'date': '2026-09-14', 'close': 78180.0},
    {'date': '2026-09-13', 'close': 76800.0},
    {'date': '2026-01-03', 'close': 90603.0},
    {'date': '2026-01-02', 'close': 89945.0},
    {'date': '2026-01-01', 'close': 88732.0},
    {'date': '2025-12-31', 'close': 87509.0},
    {'date': '2025-12-30', 'close': 88430.0},
    {'date': '2025-12-29', 'close': 87138.0},
    {'date': '2025-12-28', 'close': 87836.0},
    {'date': '2025-12-27', 'close': 87802.0},
    {'date': '2025-12-26', 'close': 87301.0},
    {'date': '2025-12-25', 'close': 87235.0},
    {'date': '2025-12-24', 'close': 87612.0},
    {'date': '2025-12-23', 'close': 87414.0},
    {'date': '2025-12-22', 'close': 88490.0},
    {'date': '2025-12-21', 'close': 88622.0},
    {'date': '2025-12-20', 'close': 88344.0},
    {'date': '2025-12-19', 'close': 88103.0},
    {'date': '2025-12-04', 'close': 92142.0},
    {'date': '2025-12-03', 'close': 93528.0},
    {'date': '2025-12-02', 'close': 91350.0},
    {'date': '2025-12-01', 'close': 86322.0},
    {'date': '2025-11-30', 'close': 90394.0},
    {'date': '2025-11-29', 'close': 90852.0},
    {'date': '2025-11-28', 'close': 90919.0},
    {'date': '2025-11-27', 'close': 91285.0},
    {'date': '2025-11-26', 'close': 90518.0},
    {'date': '2025-11-25', 'close': 87342.0},
])
btc_real['date'] = pd.to_datetime(btc_real['date'])

# BTC 年度汇总数据
btc_annual = {
    2024: {'avg': 65000, 'open': 42000, 'high': 73000, 'low': 38000, 'close': 96556},
    2025: {'avg': 85000, 'open': 96556, 'high': 99473, 'low': 76210, 'close': 87509},
    2026: {'avg': 85000, 'open': 88733, 'high': 94061, 'low': 74890, 'close': 84068},
}

# 生成 BTC 完整日线数据
def generate_btc_daily():
    dates = pd.date_range('2024-01-01', '2026-10-07', freq='D')  # 每天
    btc_full = pd.DataFrame({'date': dates})
    
    # 标记哪些日期有真实数据
    btc_full['has_real_data'] = btc_full['date'].isin(btc_real['date'])
    
    # 对于有真实数据的日期，使用真实收盘价
    btc_full = btc_full.merge(btc_real, on='date', how='left')
    
    # 对于没有真实数据的日期，使用年度平均价格
    btc_full['close'] = btc_full.apply(
        lambda row: row['close'] if pd.notna(row['close']) else btc_annual[row['date'].year]['avg'],
        axis=1
    )
    
    # 标记数据来源
    btc_full['source'] = btc_full['has_real_data'].map({True: 'real', False: 'annual_avg'})
    
    return btc_full[['date', 'close', 'source']]

btc_full = generate_btc_daily()
print("\nBTC 完整日线数据:")
print(f"  日期范围: {btc_full['date'].min()} 至 {btc_full['date'].max()}")
print(f"  数据条数: {len(btc_full)}")
print(f"  真实数据: {len(btc_full[btc_full['source']=='real'])} 条")
print(f"  年度平均: {len(btc_full[btc_full['source']=='annual_avg'])} 条")
print()
print(btc_full.head(10))
print("...")
print(btc_full.tail(10))

# 保存 BTC CSV
btc_full.to_csv(r'E:\agent_dev\multi-agent\tmp\btc_daily.csv', index=False)
print("\nBTC CSV 已保存至: E:\\agent_dev\\multi-agent\\tmp\\btc_daily.csv")
