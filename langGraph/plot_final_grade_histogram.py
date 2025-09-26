import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文字体路径
font_path = 'SimSun.ttf'

# 读取数据文件
file_path = 'E:\\agent_dev\\langGraph\\student_habits_performance.csv'
data = pd.read_csv(file_path, encoding='utf-8')

# 查看列名，确认成绩列的名称
print("数据列名：", data.columns.tolist())

# 假设最终成绩列为 'final_grade' 或类似名称，尝试匹配常见命名
grade_columns = [col for col in data.columns if 'grade' in col.lower() or 'score' in col.lower()]

if not grade_columns:
    raise ValueError("未找到包含 'grade' 或 'score' 的列，请检查 CSV 文件。")

# 使用第一个匹配的成绩列
final_grade_col = grade_columns[0]
print(f"使用列 '{final_grade_col}' 作为最终成绩进行分析。")

# 绘制直方图
plt.figure(figsize=(10, 6))
sns.histplot(data[final_grade_col], kde=True, bins=20, color='skyblue', edgecolor='black')
plt.title(f'{final_grade_col} 分布直方图', fontsize=16, fontfamily='SimSun')
plt.xlabel(final_grade_col, fontsize=12, fontfamily='SimSun')
plt.ylabel('频数', fontsize=12, fontfamily='SimSun')

# 保存图像
output_image_path = 'final_grade_distribution_histogram.png'
plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
print(f"图像已保存至: {output_image_path}")

# 显示图表
plt.show()

# 检查文件是否生成
if os.path.exists(output_image_path):
    print("文件生成成功。")
else:
    print("文件生成失败。")