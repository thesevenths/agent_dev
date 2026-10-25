"""临时诊断：同一时刻并排测「raw socket / urllib / httpx」三条路，定位是哪一层被劫持。

前面 raw socket 全部成功、httpx/urllib 全部失败 —— 高度怀疑 Windows 注册表里的系统代理
（127.0.0.1:7890）被 urllib/httpx 自动采纳，而 bypass 列表按名字匹配，域名不在名单里。
"""
import os
import socket
import ssl

import httpx
import urllib.request

HOST = "vllm.softsafe-tech.com"
IP = "192.168.50.45"
URLS = [f"https://{HOST}/v1/models", f"http://{HOST}/v1/models",
        f"http://{IP}/v1/models"]

print("proxies seen by urllib.getproxies():", urllib.request.getproxies())
print("httpx env proxies:", {k: v for k, v in os.environ.items() if "proxy" in k.lower()})

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
for url in URLS:
    try:
        with socket.create_connection((IP, 443), timeout=10) as s:
            with ctx.wrap_socket(s, server_hostname=HOST) as t:
                t.sendall(b"GET /v1/models HTTP/1.1\r\nHost: " + HOST.encode() +
                          b"\r\nConnection: close\r\n\r\n")
                first = t.recv(40)
        print(f"raw     {url:45s} -> {first[:22]!r}")
    except Exception as e:
        print(f"raw     {url:45s} -> FAIL {type(e).__name__} {str(e)[:60]}")

for url in URLS:
    for label, fn in (("urllib", lambda u: urllib.request.urlopen(u, timeout=15).status),
                      ("httpx", lambda u: httpx.get(u, timeout=15).status_code),
                      ("httpx-notrust", lambda u: httpx.get(u, timeout=15, trust_env=False).status_code)):
        try:
            print(f"{label:12s} {url:45s} -> {fn(url)}")
        except Exception as e:
            print(f"{label:12s} {url:45s} -> FAIL {type(e).__name__} {str(e)[:70]}")
