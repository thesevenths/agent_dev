import pandas as pd

# 读取CSV文件
file_path = 'E:\\agent_dev\\langGraph\\student_habits_performance.csv'
df = pd.read_csv(file_path)

# 输出数据维度
print("数据维度 (行数, 列数):", df.shape)

# 查看数据类型和非空值数量
df.info()

# 检查重复行数量
duplicated_rows = df.duplicated().sum()
print(f"重复行数量: {duplicated_rows}")
