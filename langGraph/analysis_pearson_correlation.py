import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
file_path = 'E:\\agent_dev\\langGraph\\student_habits_performance.csv'
df = pd.read_csv(file_path)

# 计算皮尔逊相关系数矩阵
numeric_columns = ['math score', 'reading score', 'writing score']
correlation_matrix = df[numeric_columns].corr(method='pearson')

# 打印相关系数矩阵
print("皮尔逊相关系数矩阵：")
print(correlation_matrix)

# 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True,
            xticklabels=numeric_columns, yticklabels=numeric_columns)
plt.title('数值型变量之间的皮尔逊相关系数热力图')
plt.tight_layout()

# 保存图像
output_plot_path = 'plots/pearson_correlation_heatmap.png'
plt.savefig(output_plot_path)
plt.close()

print(f"热力图已保存至 {output_plot_path}")