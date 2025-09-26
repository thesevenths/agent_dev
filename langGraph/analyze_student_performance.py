import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
file_path = 'E:\\agent_dev\\langGraph\\student_habits_performance.csv'
data = pd.read_csv(file_path, encoding='utf-8')

# 检查必要的列是否存在
required_columns = ['gender', 'family_background', 'final_grade']
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"缺少必要字段: {col}")

# 绘制按性别分组的成绩箱线图
plt.figure(figsize=(10, 6))
sns.boxplot(x='gender', y='final_grade', data=data)
plt.title('按性别分组的最终成绩箱线图')
plt.xlabel('性别')
plt.ylabel('最终成绩')
plt.savefig('gender_performance_boxplot.png')
plt.close()
print("已生成按性别分组的箱线图：gender_performance_boxplot.png")

# 绘制按家庭背景分组的成绩箱线图
plt.figure(figsize=(12, 7))
sns.boxplot(x='family_background', y='final_grade', data=data)
plt.title('按家庭背景分组的最终成绩箱线图')
plt.xlabel('家庭背景')
plt.ylabel('最终成绩')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('family_background_performance_boxplot.png')
plt.close()
print("已生成按家庭背景分组的箱线图：family_background_performance_boxplot.png")

# 输出基本统计信息
print("\n各群体成绩统计摘要：")
print(data.groupby('gender')['final_grade'].describe())
print("\n")
print(data.groupby('family_background')['final_grade'].describe())