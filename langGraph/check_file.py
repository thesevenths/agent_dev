import os

# 定义文件路径
file_path = r'E:\agent_dev\langGraph\student_habits_performance.csv'

# 检查文件是否存在
if not os.path.exists(file_path):
    print(f"文件不存在：{file_path}")
else:
    print(f"文件存在：{file_path}")

    # 尝试以UTF-8编码打开文件，读取前几行进行验证
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i in range(5):  # 读取前5行
                line = f.readline()
                if not line:
                    break
                print(f"第{i+1}行: {line.strip()}")
        print("\n文件成功以UTF-8编码读取，格式正常。")
    except UnicodeDecodeError as e:
        print(f"\n文件读取失败，编码错误：{e}")
    except Exception as e:
        print(f"\n发生未知错误：{e}")