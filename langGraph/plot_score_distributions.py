import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
file_path = 'E:\\agent_dev\\langGraph\\student_habits_performance.csv'
df = pd.read_csv(file_path, encoding='utf-8')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形和子图
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# 绘制数学成绩直方图
axs[0].hist(df['math score'], bins=20, color='skyblue', edgecolor='black')
axs[0].set_title('数学成绩分布')
axs[0].set_xlabel('成绩')
axs[0].set_ylabel('频数')
axs[0].grid(True)

# 绘制阅读成绩直方图
axs[1].hist(df['reading score'], bins=20, color='lightgreen', edgecolor='black')
axs[1].set_title('阅读成绩分布')
axs[1].set_xlabel('成绩')
axs[1].set_ylabel('频数')
axs[1].grid(True)

# 绘制作文成绩直方图
axs[2].hist(df['writing score'], bins=20, color='salmon', edgecolor='black')
axs[2].set_title('作文成绩分布')
axs[2].set_xlabel('成绩')
axs[2].set_ylabel('频数')
axs[2].grid(True)

# 调整子图间距
plt.tight_layout()

# 保存图像
plt.savefig('score_distributions.png')
plt.close()

print("三科成绩直方图已生成并保存为 score_distributions.png")