from dotenv import load_dotenv
import os
from pathlib import Path
# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# 模型配置
EMBEDDING_MODEL = "text-embedding-v3"  # DashScope embedding 模型
RERANK_MODEL = "rerank-v1"              # DashScope rerank 模型
LLM_MODEL = "qwen-max"                 # DashScope LLM 模型

# Load from environment variables to avoid committing to GitHub
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")   
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
PG_CONN_STR = os.getenv("PG_CONN_STR")   
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")   
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
print(f"config: LLAMA_CLOUD_API_KEY: {LLAMA_CLOUD_API_KEY}")

TOP_N = int(os.getenv("TOP_N", 5))  # Default to 5 if not set

QQ_EMAIL = os.getenv("QQ_EMAIL")
QQ_APP_PASSWORD = os.getenv("QQ_APP_PASSWORD")

CRYPTO_SENTIMENT_KEY = os.getenv("CRYPTO_SENTIMENT_KEY")