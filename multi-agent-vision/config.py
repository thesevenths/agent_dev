import os
from dotenv import load_dotenv
load_dotenv()
import psycopg2

# Load from environment variables to avoid committing to GitHub
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")  # Default for local testing, override with env
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
PG_CONN_STR = os.getenv("PG_CONN_STR")  # e.g., "postgresql://user:password@host:port/dbname"
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")  # For LangSmith tracing

print("Current working directory:", os.getcwd())
print(DASHSCOPE_API_KEY)
print(TAVILY_API_KEY)
print(PG_CONN_STR)
print(LANGSMITH_API_KEY)



# # 获取连接字符串
# pg_conn_str = os.getenv("PG_CONN_STR")
#
# if not pg_conn_str:
#     print("❌ 错误：PG_CONN_STR 环境变量未设置！")
#     exit(1)
#
# print(f"🔌 尝试连接数据库: {pg_conn_str}")
#
# try:
#     # 建立连接
#     conn = psycopg2.connect(pg_conn_str)
#     cursor = conn.cursor()
#
#     # 执行简单查询
#     cursor.execute("SELECT version();")
#     db_version = cursor.fetchone()
#     print("✅ 成功连接到 PostgreSQL！")
#     print(f"📦 数据库版本: {db_version[0]}")
#
#     # 可选：列出所有表（仅限当前 schema）
#     cursor.execute("""
#         SELECT table_name
#         FROM information_schema.tables
#         WHERE table_schema = 'public'
#         LIMIT 5;
#     """)
#     tables = cursor.fetchall()
#     if tables:
#         print("🗃️  public schema 中的部分表:")
#         for table in tables:
#             print(f"  - {table[0]}")
#     else:
#         print("📭 public schema 中没有表。")
#
#     # 关闭连接
#     cursor.close()
#     conn.close()
#     print("🔒 数据库连接已关闭。")
#
# except Exception as e:
#     print(f"❌ 连接失败: {e}")


# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(
#     model="qwen-plus",
#     temperature=0.0,
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     api_key=DASHSCOPE_API_KEY  # 替换为你的实际 key
# )
#
# try:
#     response = llm.invoke("你好")
#     print(response.content)
# except Exception as e:
#     print("Error:", e)
