
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()

# Load from environment variables to avoid committing to GitHub
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")   
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
PG_CONN_STR = os.getenv("PG_CONN_STR")   
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")   

# === 在线大模型（DashScope / 通义千问）配置 —— 保留备用，已不再硬编码在 agent.py ===
# 切回在线模型时，把 .env 里的 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 改成下面这些值即可：
#   DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
#   DASHSCOPE_MODEL    = "qwen-plus"
# （API_KEY 仍走环境变量 DASHSCOPE_API_KEY，下方保留读取，未删除）

# === 本地 / 在线大模型统一配置（OpenAI 兼容接口）===
# 只改下面 4 个环境变量即可在「本地 vLLM」与「在线 DashScope」之间切换，无需动代码
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus")
# 是否校验 SSL 证书；本地自签名/内网证书请设为 false（等价于 curl -k）
LLM_VERIFY_SSL = os.getenv("LLM_VERIFY_SSL", "true").strip().lower() in ("1", "true", "yes", "y", "on")

TOP_N = int(os.getenv("TOP_N", 5))  # Default to 5 if not set

QQ_EMAIL = os.getenv("QQ_EMAIL")
QQ_APP_PASSWORD = os.getenv("QQ_APP_PASSWORD")

CRYPTO_SENTIMENT_KEY = os.getenv("CRYPTO_SENTIMENT_KEY")