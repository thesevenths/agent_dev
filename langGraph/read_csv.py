import pandas as pd

# 读取CSV文件
file_path = 'E:\\agent_dev\\langGraph\\student_habits_performance.csv'
df = pd.read_csv(file_path, encoding='utf-8')

# 打印前5行数据
print(df.head())

# 打印列名
print(df.columns.tolist())