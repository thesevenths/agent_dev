import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
file_path = 'E:\\agent_dev\\langGraph\\student_habits_performance.csv'
df = pd.read_csv(file_path, encoding='utf-8')

# 按是否完成考试准备课程分组，计算各科目平均分
grouped_means = df.groupby('test preparation course')[['math score', 'reading score', 'writing score']].mean()
print("按是否完成考试准备课程分组的各科目平均分：")
print(grouped_means)

# 绘制柱状图
ax = grouped_means.plot(kind='bar', figsize=(10, 6))
plt.title('完成与未完成考试准备课程学生的成绩对比')
plt.xlabel('考试准备课程')
plt.ylabel('平均分')
plt.xticks(rotation=0)
plt.legend(['数学', '阅读', '写作'])
plt.grid(axis='y')

# 在柱子上方显示数值
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', padding=3)

plt.tight_layout()
plt.savefig('test_prep_impact.png')
plt.close()

print("柱状图已保存为 test_prep_impact.png")