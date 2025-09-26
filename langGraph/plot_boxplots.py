import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
file_path = 'E:\\agent_dev\\langGraph\\student_habits_performance.csv'
df = pd.read_csv(file_path, encoding='utf-8')

# 确保 plots 目录存在
if not os.path.exists('plots'):
    os.makedirs('plots')

# 绘制数学、阅读、写作成绩的箱线图
plt.figure(figsize=(10, 6))
df[['math score', 'reading score', 'writing score']].boxplot()
plt.title('学生成绩箱线图（数学、阅读、写作）')
plt.ylabel('分数')
plt.xlabel('科目')
plt.grid(False)

# 保存图像
boxplot_path = 'plots/score_boxplots.png'
plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"箱线图已保存至 {boxplot_path}")

# 识别离群点
outliers = {}
for col in ['math score', 'reading score', 'writing score']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outliers[col] = col_outliers.index.tolist()
    print(f"{col} 的离群点索引: {outliers[col]}")
    if len(outliers[col]) > 0:
        print(f"{col} 的离群点数据:\n{col_outliers[['gender', 'race/ethnicity', 'parental level of education', col]]}\n")

# 将离群点信息保存为文件以便后续引用
outliers_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in outliers.items()]))
outliers_file = 'analysis_results/outliers_summary.csv'
if not os.path.exists('analysis_results'):
    os.makedirs('analysis_results')
outliers_df.to_csv(outliers_file, index_label='index')
print(f"离群点汇总已保存至 {outliers_file}")