import os

# 定义文件路径
file_path = r'E:\agent_dev\langGraph\student_habits_performance.csv'

# 检查文件是否存在
if not os.path.exists(file_path):
    print(f"文件不存在：{file_path}")
else:
    print(f"文件存在：{file_path}")

    # 尝试以UTF-8编码打开文件，检查是否可读
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(1024)  # 读取前1024个字符进行验证
            print("文件成功以UTF-8编码读取前1024字符。")
            print("文件开头内容预览：\n", content[:200])
    except UnicodeDecodeError as e:
        print(f"UTF-8解码错误：{e}")
    except Exception as e:
        print(f"读取文件时发生其他错误：{e}")