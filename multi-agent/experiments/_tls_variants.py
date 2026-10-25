"""临时诊断：Python ssl 与该 nginx 的握手到底哪一步被掐。

openssl CLI（同一份 OpenSSL 3.0.16）能连上并协商 TLSv1.2 / ECDHE-RSA-AES128-GCM-SHA256，
而 Python 的 ssl 一律 UNEXPECTED_EOF。这里逐个对齐参数，找出决定性差异。
"""
import os
import socket
import ssl

HOST = "vllm.softsafe-tech.com"
IP = "192.168.50.45"
PATH = "/v1/models"


def attempt(name, *, cipher=None, versions=None, alpn=None, options_off=None,
            groups=None, send_http=True):
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        if cipher:
            ctx.set_ciphers(cipher)
        if versions:
            ctx.minimum_version = ctx.maximum_version = versions
        if alpn:
            ctx.set_alpn_protocols(alpn)
        if options_off:
            for o in options_off:
                ctx.options &= ~o
        if groups:
            ctx.set_ecdh_curve(groups)
    except Exception as e:
        print(f"{name:38s} SETUP-FAIL {type(e).__name__}: {e}")
        return
    try:
        with socket.create_connection((IP, 443), timeout=12) as sock:
            with ctx.wrap_socket(sock, server_hostname=HOST) as tls:
                if not send_http:
                    print(f"{name:38s} HANDSHAKE-OK {tls.version()}")
                    return
                tls.sendall(f"GET {PATH} HTTP/1.1\r\nHost: {HOST}\r\n"
                            f"Connection: close\r\n\r\n".encode())
                head = tls.recv(80)
        print(f"{name:38s} OK {tls.version()} {head[:24]!r}")
    except Exception as e:
        print(f"{name:38s} FAIL {type(e).__name__} {str(e)[:70]}")


attempt("bare handshake (no cipher tweak)", send_http=False)
attempt("exact cipher AES128-GCM + 1.2", cipher="ECDHE-RSA-AES128-GCM-SHA256",
        versions=ssl.TLSVersion.TLSv1_2)
attempt("seclevel0 all ciphers", cipher="ALL@SECLEVEL=0")
attempt("no middlebox compat", options_off=[ssl.OP_ENABLE_MIDDLEBOX_COMPAT])
attempt("with ALPN h2+http1.1", alpn=["h2", "http/1.1"])
attempt("group secp384r1", groups="secp384r1")
attempt("group prime256v1", groups="prime256v1")
print("env SSL_CERT_FILE =", os.environ.get("SSL_CERT_FILE"),
      "| OPENSSL_CONF =", os.environ.get("OPENSSL_CONF"))
