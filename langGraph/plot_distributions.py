# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
file_path = 'E:\\agent_dev\\langGraph\\student_habits_performance.csv'
df = pd.read_csv(file_path, encoding='utf-8')

# 确认数值型变量
numeric_columns = ['math score', 'reading score', 'writing score']

# 创建保存图像的目录
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

# 绘制每个数值变量的分布直方图
for col in numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[col], kde=True, bins=20, color='skyblue', edgecolor='black')
    plt.title(f'{col} 分布直方图')
    plt.xlabel(col)
    plt.ylabel('频数')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plot_path = os.path.join(output_dir, f'{col.replace(" ", "_")}_distribution.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"已生成 {col} 的分布图: {plot_path}")
    plt.close()

print("所有分布直方图已生成完毕。")