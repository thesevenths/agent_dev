import pandas as pd

# 读取数据
file_path = 'E:\\agent_dev\\langGraph\\student_habits_performance.csv'
df = pd.read_csv(file_path)

# 确认列名（处理可能的空格或大小写问题）
column_mapping = {
    'gender': 'gender',
    'race/ethnicity': 'race_ethnicity',
    'parental level of education': 'parent_education',
    'lunch': 'lunch',
    'test preparation course': 'test_prep_course',
    'math score': 'math_score',
    'reading score': 'reading_score',
    'writing score': 'writing_score'
}
df.rename(columns=column_mapping, inplace=True)

# 定义分类变量和成绩变量
categorical_vars = ['gender', 'race_ethnicity', 'parent_education', 'lunch', 'test_prep_course']
score_vars = ['math_score', 'reading_score', 'writing_score']

# 存储所有分组结果
all_group_results = {}

# 对每个分类变量与每项成绩进行分组统计平均值
for cat_var in categorical_vars:
    group_stats = {}
    print(f"\\n=== 按 {cat_var} 分组的平均成绩 ===")
    for score_var in score_vars:
        grouped_mean = df.groupby(cat_var)[score_var].mean().round(2)
        group_stats[score_var] = grouped_mean
        print(f"{score_var}:\n{grouped_mean}\n")
    all_group_results[cat_var] = group_stats

# 将结果保存为字典形式，便于后续引用
print("\\n所有分组统计完成。")

# 导出分组结果到CSV文件以便验证
group_results_df = pd.DataFrame()
for cat_var, stats in all_group_results.items():
    for score_var, data in stats.items():
        temp_df = data.reset_index()
        temp_df.columns = [cat_var, f'{score_var}_mean']
        temp_df['metric'] = score_var
        group_results_df = pd.concat([group_results_df, temp_df], ignore_index=True)

group_results_df.to_csv('grouped_means.csv', index=False)
print("分组均值已导出至 'grouped_means.csv'")