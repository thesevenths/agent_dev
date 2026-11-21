
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


# === 自建端点必须绕过系统代理 ===
# 症状：模型「连不上」（SSL: UNEXPECTED_EOF_WHILE_READING / 502 / Connection error），
# 但 curl 与 openssl CLI 同一时刻都能连通。根因在 Windows 注册表里的系统代理（例如
# Clash 的 127.0.0.1:7890）：requests / httpx / urllib 在 Windows 上会自动采纳它，而代理的
# bypass 名单是**按名字**匹配的（名单里有 192.168.* 却匹配不到 vllm.xxx.com 这种域名），
# 于是到内网自建推理服务的请求被塞进代理，代理连不上目标就把连接掐了。
# 只有当端点解析到私网地址时才 bypass：公网端点（如 DashScope）在该网络里恰恰**需要**代理，
# 无差别 bypass 会把在线模型这条路打死。解析失败时保持原行为（宁可不动，不做反效果修复）。
def _is_private_host(host: str) -> bool:
    import ipaddress
    import socket
    if not host:
        return False
    try:
        addrs = {info[4][0] for info in socket.getaddrinfo(host, None)}
    except OSError:
        return False
    try:
        return all(ipaddress.ip_address(a).is_private for a in addrs) if addrs else False
    except ValueError:
        return False


def _bypass_proxy_for(*urls: str) -> None:
    hosts = []
    for u in urls:
        h = u.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0].strip().lower()
        if h and _is_private_host(h) and h not in hosts:
            hosts.append(h)
    if not hosts:
        return
    for name in ("NO_PROXY", "no_proxy"):
        cur = [x.strip() for x in os.environ.get(name, "").split(",") if x.strip()]
        merged = cur + [h for h in hosts if h not in cur]
        if merged != cur:
            os.environ[name] = ",".join(merged)


_bypass_proxy_for(LLM_BASE_URL, os.getenv("AGENT_MEMORY_EMBED_BASE_URL", ""))

TOP_N = int(os.getenv("TOP_N", 5))  # Default to 5 if not set

QQ_EMAIL = os.getenv("QQ_EMAIL")
QQ_APP_PASSWORD = os.getenv("QQ_APP_PASSWORD")

CRYPTO_SENTIMENT_KEY = os.getenv("CRYPTO_SENTIMENT_KEY")