import pandas as pd

# 读取CSV文件
df = pd.read_csv('E:\\agent_dev\\langGraph\\student_habits_performance.csv', encoding='utf-8')

# 计算每列的缺失值数量和占比
missing_info = df.isnull().sum()  # 缺失值数量
total_rows = len(df)
missing_percentage = (missing_info / total_rows) * 100  # 缺失值占比

# 合并缺失值信息
missing_df = pd.DataFrame({
    'Missing_Count': missing_info,
    'Missing_Percentage': missing_percentage
})

# 检查重复行
duplicated_rows = df.duplicated().sum()
print(f"重复行数量: {duplicated_rows}")

# 输出缺失值统计
print("\n各列缺失值统计:")
print(missing_df)

# 判断是否需要去重
if duplicated_rows > 0:
    print("\n存在重复行，建议进行去重处理。")
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    print(f"去重后数据形状: {df_cleaned.shape}")
else:
    print("\n无重复行，无需去重处理。")
    df_cleaned = df

# 保存清洗后的数据（可选）
df_cleaned.to_csv('cleaned_student_data.csv', index=False, encoding='utf-8-sig')
print("\n清洗后的数据已保存为 'cleaned_student_data.csv'")