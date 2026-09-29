import pandas as pd

# 读取CSV文件
file_path = 'E:\\agent_dev\\langGraph\\student_habits_performance.csv'
df = pd.read_csv(file_path)

# 检查缺失值
def check_missing_values():
    print("缺失值统计：")
    print(df.isnull().sum())

# 查看各列数据类型
def check_dtypes():
    print("\n数据类型：")
    print(df.dtypes)

# 识别异常值或不一致的数据条目
def check_anomalies():
    print("\n数值型数据的基本统计信息：")
    print(df.describe())
    
    # 检查数学、阅读、写作成绩是否在合理范围内（0-100）
    math_outliers = df[(df['math score'] < 0) | (df['math score'] > 100)]
    reading_outliers = df[(df['reading score'] < 0) | (df['reading score'] > 100)]
    writing_outliers = df[(df['writing score'] < 0) | (df['writing score'] > 100)]
    
    print(f"\n数学成绩异常值数量: {len(math_outliers)}")
    print(f"阅读成绩异常值数量: {len(reading_outliers)}")
    print(f"写作成绩异常值数量: {len(writing_outliers)}")

# 执行检查函数
check_missing_values()
check_dtypes()
check_anomalies()