import pandas as pd

# 读取CSV文件，使用utf-8编码
file_path = 'E:\\agent_dev\\langGraph\\student_habits_performance.csv'
try:
    df = pd.read_csv(file_path, encoding='utf-8')
    print("数据成功加载！")
    print(f"数据形状: {df.shape}")
    print("前5行数据:")
    print(df.head())
except Exception as e:
    print(f"加载数据时出错: {e}")