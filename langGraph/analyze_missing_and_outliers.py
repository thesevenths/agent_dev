import pandas as pd
import numpy as np

# 读取数据
file_path = 'E:\\agent_dev\\langGraph\\student_habits_performance.csv'
print(f"正在读取文件: {file_path}")
df = pd.read_csv(file_path, encoding='utf-8')

# 显示基本信息
echo_info = f"数据形状: {df.shape}\n"
echo_info += "列名:\n" + "\n".join(df.columns.tolist()) + "\n"
echo_info += "\n前5行数据:\n" + str(df.head()) + "\n"
print(echo_info)

# 统计每列的缺失值数量
missing_values = df.isnull().sum()
print("\n每列缺失值数量:")
print(missing_values)

# 识别数值型字段中的异常值
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\n数值型字段: {numeric_columns}")

outliers_summary = {}
for col in numeric_columns:
    # 使用四分位距法 (IQR) 检测异常值
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 找出异常值
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    outliers_summary[col] = {
        'count': len(outliers),
        'values': outliers.tolist(),
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }
    
    print(f"\n{col} 异常值检测:")
    print(f"范围: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"异常值数量: {len(outliers)}")
    if len(outliers) > 0:
        print(f"异常值示例 (最多5个): {outliers.head().tolist()}")

# 输出总体总结
print("\n\n--- 分析总结 ---")
print("缺失值总览:")
for col, miss_count in missing_values.items():
    if miss_count > 0:
        print(f"{col}: {miss_count} 个缺失值")

print("\n异常值总览 (仅数值型字段)")
for col, info in outliers_summary.items():
    if info['count'] > 0:
        print(f"{col}: 共 {info['count']} 个异常值，检测区间 [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]")
