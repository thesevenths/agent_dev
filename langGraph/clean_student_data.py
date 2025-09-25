import pandas as pd

# 读取CSV文件
file_path = 'E:\\agent_dev\\langGraph\\student_habits_performance.csv'
print("正在读取数据...")
df = pd.read_csv(file_path, encoding='utf-8')

# 显示原始数据基本信息
print("\n原始数据形状:", df.shape)
print("\n前5行数据:")
print(df.head())

# 检查缺失值
print("\n缺失值统计:")
print(df.isnull().sum())

# 处理缺失值：数值型列用均值填充，分类变量用众数填充
numeric_cols = df.select_dtypes(include=['number']).columns
category_cols = df.select_dtypes(include=['object']).columns

for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mean(), inplace=True)
        print(f"{col} 中的缺失值已使用均值填充。")

for col in category_cols:
    if df[col].isnull().sum() > 0:
        mode_value = df[col].mode()[0]
        df[col].fillna(mode_value, inplace=True)
        print(f"{col} 中的缺失值已使用众数 '{mode_value}' 填充。")

# 去除重复行
initial_rows = df.shape[0]
df.drop_duplicates(inplace=True)
dropped_rows = initial_rows - df.shape[0]
print(f"\n共删除了 {dropped_rows} 行重复数据。")

# 修正数据类型：将分类变量转换为category类型
for col in category_cols:
    df[col] = df[col].astype('category')
print("\n分类变量已转换为 'category' 类型。")

# 显示处理后的数据信息
print("\n处理后的数据形状:", df.shape)
print("\n数据类型:")
print(df.dtypes)

# 保存清洗后的数据
output_path = 'cleaned_student_habits_performance.csv'
df.to_csv(output_path, index=False, encoding='utf-8')
print(f"\n清洗后的数据已保存至 '{output_path}'。")