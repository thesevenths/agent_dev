"""临时诊断：curl（Schannel）走 https 时，vLLM 上游此刻到底是活着还是挂了。

密钥写进临时 curl 配置文件传递，不出现在进程命令行，也不打印。
"""
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env", override=False)
KEY = os.getenv("LLM_API_KEY", "")
MODEL = os.getenv("LLM_MODEL", "")


def cfg(data=None):
    lines = [f'header = "Authorization: Bearer {KEY}"',
             'header = "Content-Type: application/json"']
    if data is not None:
        Path("_curl_body.json").write_text(data, encoding="utf-8")
        lines.append('data = "@_curl_body.json"')
    f = Path("_curl_cfg")
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return f


def run(url, data=None, timeout=25):
    c = cfg(data)
    p = subprocess.run(["curl.exe", "-sS", "-m", str(timeout), "-o", "-",
                        "-w", "|CODE=%{http_code}", "-K", str(c), url],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return ((p.stdout or "")[:200].replace("\n", " "), (p.stderr or "")[:150])


for u in ("https://vllm.softsafe-tech.com/v1/models", "http://192.168.50.45/v1/models"):
    out, err = run(u)
    print(f"{u}\n   -> {out}\n   err={err}")

body = ('{"model":"%s","messages":[{"role":"user","content":"reply OK"}],'
        '"max_tokens":8,"temperature":0}') % MODEL
out, err = run("https://vllm.softsafe-tech.com/v1/chat/completions", body, timeout=90)
print("https chat ->", out, "| err=", err)
