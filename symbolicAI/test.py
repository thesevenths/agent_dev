import os

# # 设置通义千问的 API 地址
# os.environ["OPENAI_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"


from symai import Symbol

# 语义模式
s1 = Symbol("大象是一种动物", semantic=True)
s2 = Symbol("elephants are animals", semantic=True)

print(s1 == s2)  # 语义上等价
