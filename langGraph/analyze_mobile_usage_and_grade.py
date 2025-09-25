import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据文件
file_path = 'E:\\agent_dev\\langGraph\\student_habits_performance.csv'
data = pd.read_csv(file_path, encoding='utf-8')

# 选取指定变量
selected_columns = ['mobile_usage_hours', 'final_grade']
data_selected = data[selected_columns]

# 计算相关系数矩阵
correlation_matrix = data_selected.corr()
print("相关系数矩阵：")
print(correlation_matrix)

# 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True, fmt='.2f')
plt.title('手机使用时长与最终成绩的相关系数热力图')
plt.savefig('mobile_usage_vs_final_grade_heatmap.png')
plt.close()
print("热力图已保存为 mobile_usage_vs_final_grade_heatmap.png")