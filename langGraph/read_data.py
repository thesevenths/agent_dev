# 使用pandas读取CSV文件
import pandas as pd

# 读取数据文件
file_path = 'E:\agent_dev\langGraph\student_habits_performance.csv'
df = pd.read_csv(file_path, encoding='utf-8')

# 显示数据前5行，确认加载成功
print("数据加载成功！前5行数据如下：")
print(df.head())

# 输出数据集基本信息
print("\n数据集基本信息：")
df.info()

# 输出数值型字段的统计描述
print("\n数值型字段统计描述：")
print(df.describe())
