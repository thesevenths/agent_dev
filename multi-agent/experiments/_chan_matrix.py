"""临时诊断：把「目标(IP/域名) × 路径(models/chat)」的矩阵跑一遍，找出可用的 http 通道。"""
import json
import os
import urllib.request
import urllib.error

import httpx
from dotenv import load_dotenv

load_dotenv(".env", override=False)
KEY = os.getenv("LLM_API_KEY", "")
MODEL = os.getenv("LLM_MODEL", "")
AUTH = {"Authorization": "Bearer " + KEY}
BODY = json.dumps({"model": MODEL, "messages": [
    {"role": "user", "content": "只回复两个字：通了"}], "max_tokens": 32, "temperature": 0}).encode()


def via(target, path, host=None):
    url = f"http://{target}/v1/{path}"
    h = dict(AUTH)
    if host:
        h["Host"] = host
    try:
        if path == "models":
            req = urllib.request.Request(url, headers=h)
        else:
            req = urllib.request.Request(url, data=BODY, headers={**h, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return f"HTTP {r.status} {r.read()[:110]!r}"
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code} {e.read()[:90]!r}"
    except Exception as e:
        return f"FAIL {type(e).__name__} {str(e)[:80]}"


print("IP   +Host models :", via("192.168.50.45", "models", "vllm.softsafe-tech.com"))
print("IP   +Host chat   :", via("192.168.50.45", "chat/completions", "vllm.softsafe-tech.com"))
print("DOM        models :", via("vllm.softsafe-tech.com", "models"))
print("DOM        chat   :", via("vllm.softsafe-tech.com", "chat/completions"))
print("HTTPS DOM  chat   :", end=" ")
try:
    c = httpx.Client(base_url="https://vllm.softsafe-tech.com/v1", headers=AUTH, timeout=60)
    r = c.post("/chat/completions", json={"model": MODEL, "messages": [
        {"role": "user", "content": "只回复两个字：通了"}], "max_tokens": 32, "temperature": 0})
    print(r.status_code, r.text[:110])
except Exception as e:
    print("FAIL", type(e).__name__, str(e)[:90])
