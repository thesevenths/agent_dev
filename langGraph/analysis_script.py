# -*- coding: utf-8 -*-
import pandas as pd

# 读取CSV文件
file_path = 'E:\\agent_dev\\langGraph\\student_habits_performance.csv'
data = pd.read_csv(file_path, encoding='utf-8')

# 对数值型字段进行描述性统计
descriptive_stats = data.describe()
print("\n描述性统计表：\n", descriptive_stats)