"""临时网络诊断：找一条 Python 可用的通道连到自托管 vLLM。

结论会写进 stdout（由调用方重定向到文件），确认不可用后即可删除本文件。
"""
import socket
import urllib.request
import urllib.error

HOST = "vllm.softsafe-tech.com"
IP = "192.168.50.45"


def get(url, headers=None, timeout=8):
    try:
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return f"OK {r.status} {r.read()[:120]!r}"
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code} {e.read()[:100]!r}"
    except Exception as e:
        return f"FAIL {type(e).__name__} {str(e)[:90]}"


print("== 端口连通性（plain HTTP）==")
for port in (80, 8000, 8080, 8100, 3000, 5000, 8081, 9000, 18000, 11434, 443):
    try:
        s = socket.create_connection((IP, port), timeout=4)
        s.close()
        print(f"  TCP OK   :{port}")
    except Exception as e:
        print(f"  TCP FAIL :{port} {type(e).__name__}")

print("== 用 Host 头命中 vhost ==")
for port in (80, 8080, 8100):
    print(f"  :{port} /v1/models  ->", get(f"http://{IP}:{port}/v1/models",
                                           {"Host": HOST}))
    print(f"  :{port} /v1/chat    ->", get(f"http://{IP}:{port}/v1/chat/completions",
                                           {"Host": HOST}))
