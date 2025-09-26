import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体
font_path = 'C:\\Windows\\Fonts\\simsun.ttc'
cn_font = fm.FontProperties(fname=font_path)

# 读取数据
file_path = 'E:\\agent_dev\\langGraph\\student_habits_performance.csv'
df = pd.read_csv(file_path, encoding='utf-8')

# 按性别分组计算三科平均成绩
gender_grouped = df.groupby('gender')[['math score', 'reading score', 'writing score']].mean()
print('按性别分组的平均成绩：')
print(gender_grouped)

# 绘制箱形图展示男女学生在三科上的成绩分布差异
fig, ax = plt.subplots(figsize=(10, 6))

data_m = df[df['gender'] == 'male'][['math score', 'reading score', 'writing score']].values
data_f = df[df['gender'] == 'female'][['math score', 'reading score', 'writing score']].values

bp_m = ax.boxplot(data_m, positions=[1, 2, 3], widths=0.6, patch_artist=True, boxprops=dict(facecolor='blue', alpha=0.5), medianprops=dict(color='black'))
bp_f = ax.boxplot(data_f, positions=[1.5, 2.5, 3.5], widths=0.6, patch_artist=True, boxprops=dict(facecolor='pink', alpha=0.5), medianprops=dict(color='black'))

ax.set_xticks([1.25, 2.25, 3.25])
ax.set_xticklabels(['Math Score', 'Reading Score', 'Writing Score'], fontproperties=cn_font)
ax.set_ylabel('Score', fontproperties=cn_font)
ax.set_title('Gender Comparison of Subject Scores Distribution', fontproperties=cn_font)
ax.legend([bp_m['boxes'][0], bp_f['boxes'][0]], ['Male', 'Female'], prop=cn_font)

plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('gender_comparison_boxplot.png')
plt.close()

print('箱形图已保存为 gender_comparison_boxplot.png')