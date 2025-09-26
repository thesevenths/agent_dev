# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 确保输出目录存在
os.makedirs('plots', exist_ok=True)

# 读取数据
file_path = 'E:/agent_dev/langGraph/student_habits_performance.csv'
df = pd.read_csv(file_path, encoding='utf-8')

# 检查列名
print("当前列名：", df.columns.tolist())

# 假设我们根据 'test preparation course'（测试准备课程）作为分类变量进行分组分析
# 计算各组成绩均值
category_col = 'test preparation course'
score_cols = ['math score', 'reading score', 'writing score']

grouped_means = df.groupby(category_col)[score_cols].mean()
print("\n按 \"{}\" 分组的各科成绩均值：".format(category_col))
print(grouped_means)

# 绘制柱状图
grouped_means.plot(kind='bar', figsize=(10, 6))
plt.title('按测试准备课程完成情况分组的各科平均成绩')
plt.xlabel('是否完成测试准备课程')
plt.ylabel('平均成绩')
plt.xticks(rotation=0)
plt.legend(title='科目')
plt.tight_layout()
plt.savefig('plots/scores_by_test_prep.png')
plt.close()

print("\n柱状图已保存至 plots/scores_by_test_prep.png")