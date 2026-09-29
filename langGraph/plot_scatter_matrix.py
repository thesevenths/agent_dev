import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
file_path = 'E:\\agent_dev\\langGraph\\student_habits_performance.csv'
df = pd.read_csv(file_path, encoding='utf-8')

# 提取三科成绩
scores_df = df[['math score', 'reading score', 'writing score']]

# 绘制成对关系图（散点图矩阵）并添加趋势线
pair_plot = sns.pairplot(scores_df, kind='scatter', plot_kws={'alpha':0.6}, diag_kind='hist')
pair_plot.fig.suptitle('数学、阅读、写作成绩两两关系散点图矩阵', y=1.02, fontsize=16)

# 保存图像
pair_plot.savefig('subject_scores_scatter_matrix.png', dpi=300, bbox_inches='tight')
print("散点图矩阵已保存为 subject_scores_scatter_matrix.png")