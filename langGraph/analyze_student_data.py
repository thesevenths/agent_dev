import pandas as pd

# 读取CSV文件
file_path = 'E:\agent_dev\langGraph\student_habits_performance.csv'
df = pd.read_csv(file_path, encoding='utf-8')

# 显示数据前几行以确认加载正确
print("数据前5行：\n", df.head())

# 分离数值型和分类变量
numeric_columns = df.select_dtypes(include=['number']).columns
print(f"\n数值型字段: {list(numeric_columns)}")

categorical_columns = df.select_dtypes(include=['object']).columns
print(f"分类变量字段: {list(categorical_columns)}")

# 对数值型字段使用describe()
if len(numeric_columns) > 0:
    numeric_summary = df[numeric_columns].describe()
    print("\n数值型字段统计摘要：\n", numeric_summary)
else:
    print("\n未找到数值型字段。")

# 对分类变量统计频次分布
if len(categorical_columns) > 0:
    for col in categorical_columns:
        print(f"\n【{col}】频次分布：")
        freq_dist = df[col].value_counts()
        print(freq_dist)
else:
    print("\n未找到分类变量字段。")
