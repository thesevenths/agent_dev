"""探针 v5：验证根因 = 取码器把『用户 prompt』当成了代码。

只从 assistant 文本取码（_extract_block 对围栏块本就能正确抽出函数），
跑 manual + harness grade。若 /0 → 7/7、/10 → 5/5，则证明：
  - 模型能力没问题
  - 修复方向是 extract_code_from_state 跳过 user 消息 + 收紧 bare-def fallback
"""
from __future__ import annotations
import sys, json, ssl
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import urllib.request  # noqa: E402

from adaptive_runtime.humaneval import (  # noqa: E402
    load_tasks, extract_code_from_state, functional_correctness, _extract_block,
)


def read_env(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


def call_vllm(base_url, api_key, model, verify_ssl, sys_text, human_text, temp=0.0):
    url = base_url.rstrip("/") + "/chat/completions"
    body = {"model": model, "temperature": temp,
            "messages": [{"role": "system", "content": sys_text},
                         {"role": "user", "content": human_text}]}
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    ctx = None if verify_ssl else ssl.create_default_context()
    if not verify_ssl:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
        return json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"] or ""


def manual(entry_point, test, code):
    if not code:
        return "EMPTY"
    ns = {}
    try:
        exec(code, ns)
    except Exception as e:
        return f"EXEC FAIL {type(e).__name__}: {e}"
    fn = ns.get(entry_point)
    if not callable(fn):
        return f"NO FN {entry_point}"
    tns = dict(ns)
    try:
        exec(test, tns)
        tns["check"](fn)
        return "PASS"
    except Exception as e:
        return f"CHECK FAIL {type(e).__name__}: {e}"


def main():
    env = read_env(ROOT / ".env")
    base = env["LLM_BASE_URL"]; key = env["LLM_API_KEY"]; model = env["LLM_MODEL"]
    verify = env.get("LLM_VERIFY_SSL", "true").strip().lower() in ("1", "true", "yes", "y", "on")
    tasks = load_tasks(n=12)
    sys_p = ("You are an expert Python programmer. Complete the function. "
             "Respond with ONLY a single ```python code block containing the full "
             "function. No explanations, no tests, no extra text.")
    for tid in ("HumanEval/0", "HumanEval/10"):
        t = next(x for x in tasks if x["task_id"] == tid)
        text = call_vllm(base, key, model, verify, sys_p, t["query"], temp=0.0)
        # 只从 assistant 文本取码（正确地抽围栏块）
        code = _extract_block(text)
        print(f"\n===== {tid} =====")
        print("extracted code head:", (code or "")[:60].replace("\n", " "))
        print("harness grade (assistant-only):", functional_correctness(t["entry_point"], t["test"], code))
        print("manual check:", manual(t["entry_point"], t["test"], code))


if __name__ == "__main__":
    main()
