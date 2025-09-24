# 检查文件路径是否存在并读取CSV数据
import os
import pandas as pd

# 定义文件路径
file_path = 'E:\\agent_dev\\langGraph\\student_habits_performance.csv'

# 检查文件是否存在
if os.path.exists(file_path):
    print(f"文件存在: {file_path}")
    try:
        # 使用utf-8编码读取CSV文件
        df = pd.read_csv(file_path, encoding='utf-8')
        print("CSV文件读取成功。")
        # 打印前5行数据
        print(df.head())
    except Exception as e:
        print(f"读取CSV时发生错误: {e}")
else:
    print(f"文件不存在，请检查路径是否正确: {file_path}")
