"""临时诊断：用 .env 里的真实 key，验证「plain HTTP + Host 头」这条通道能否跑 chat。

只打印状态与片段，不打印密钥。
"""
import json
import os
import urllib.request

from dotenv import load_dotenv

load_dotenv(".env", override=False)

KEY = os.getenv("LLM_API_KEY", "")
BASE = os.getenv("LLM_BASE_URL", "")
MODEL = os.getenv("LLM_MODEL", "")
print("env:", "KEY_LEN", len(KEY), "BASE", BASE, "MODEL", MODEL)

URL = "http://192.168.50.45/v1/chat/completions"
HEADERS = {
    "Host": "vllm.softsafe-tech.com",
    "Authorization": f"Bearer {KEY}",
    "Content-Type": "application/json",
}
BODY = json.dumps({
    "model": MODEL,
    "messages": [{"role": "user", "content": "只回复两个字：通了"}],
    "max_tokens": 64, "temperature": 0,
}).encode("utf-8")


def call(url, headers):
    req = urllib.request.Request(url, data=BODY, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            data = json.loads(r.read().decode("utf-8"))
        txt = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        return f"HTTP {r.status} reply={txt!r} usage={data.get('usage')}"
    except Exception as e:
        body = ""
        try:
            body = e.read()[:200].decode("utf-8", "replace")
        except Exception:
            pass
        return f"FAIL {type(e).__name__} {str(e)[:120]} {body}"


print("http+Host  :", call(URL, HEADERS))

HEADERS2 = dict(HEADERS)
HEADERS2.pop("Host")
print("http noHost:", call(URL, HEADERS2))
