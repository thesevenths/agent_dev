import ssl
import urllib.request

URL = "https://vllm.softsafe-tech.com/v1/models"


def t(name, setup):
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        setup(ctx)
        with urllib.request.urlopen(URL, timeout=12, context=ctx) as r:
            print("OK  ", name, r.status)
    except Exception as e:
        print("FAIL", name, type(e).__name__, str(e)[:110])


def v12_only(c):
    c.maximum_version = ssl.TLSVersion.TLSv1_2
    c.minimum_version = ssl.TLSVersion.TLSv1_2


def v13_only(c):
    c.minimum_version = ssl.TLSVersion.TLSv1_3


t("default", lambda c: None)
t("tls1.2 only", v12_only)
t("tls1.3 only", v13_only)
t("x25519 only", lambda c: c.set_ecdh_curve("X25519"))
t("x25519+tls1.2", lambda c: (c.set_ecdh_curve("X25519"), v12_only(c)))
t("seclevel2", lambda c: c.set_ciphers("DEFAULT@SECLEVEL=2"))
t("alpn http/1.1", lambda c: c.set_alpn_protocols(["http/1.1"]))
t("no compress ext", lambda c: c.set_ciphers("AES256-GCM-SHA384"))
