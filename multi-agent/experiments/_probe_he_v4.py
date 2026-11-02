"""一次性探针 v4（langchain-free）：决定性诊断 HumanEval 0 分根因。

直接用 requests 调 vLLM（凭据从 .env 解析，不打印 key），复用 harness 的纯 Python 评分。
判读：
  - RAW 里就是正确函数但 harness 给 0 → 取码/评分 bug（修 harness 即可解锁）
  - RAW 本身就是错逻辑 → 模型能力问题，修 agent 出码无法解锁
"""
from __future__ import annotations
import sys, os, re, json, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import urllib.request  # noqa: E402
import ssl  # noqa: E402

from adaptive_runtime.humaneval import (  # noqa: E402
    load_tasks, extract_code_from_state, functional_correctness,
)


def read_env(path: Path) -> dict:
    out: dict = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


def loose_extract(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"def\s+\w+\s*\(", text)
    if not m:
        return None
    return text[m.start():].strip()


def manual(entry_point: str, test: str, code: str) -> str:
    if not code:
        return "CODE EMPTY"
    ns: dict = {}
    try:
        exec(code, ns)
    except Exception as e:  # noqa: BLE001
        return f"EXEC FAIL: {type(e).__name__}: {e}\n{traceback.format_exc()}"
    fn = ns.get(entry_point)
    if not callable(fn):
        return f"NO FN '{entry_point}' (keys: {[k for k in ns if not k.startswith('__')][:12]})"
    tns = dict(ns)
    try:
        exec(test, tns)
        tns["check"](fn)
        return "MANUAL check() PASSED"
    except Exception as e:  # noqa: BLE001
        return f"CHECK FAIL: {type(e).__name__}: {e}\n{traceback.format_exc()}"


def call_vllm(base_url: str, api_key: str, model: str, verify_ssl: bool,
              sys_text: str, human_text: str, temp: float = 0.0) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "temperature": temp,
        "messages": [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": human_text},
        ],
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    ctx = None if verify_ssl else ssl.create_default_context()
    if not verify_ssl:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"] or ""


def main() -> None:
    env = read_env(ROOT / ".env")
    base_url = env.get("LLM_BASE_URL", "")
    api_key = env.get("LLM_API_KEY", "")
    model = env.get("LLM_MODEL", "")
    verify = env.get("LLM_VERIFY_SSL", "true").strip().lower() in ("1", "true", "yes", "y", "on")
    print(f"[cfg] model={model} base={base_url} verify_ssl={verify} key_present={bool(api_key)}")

    tasks = load_tasks(n=12)
    sys_p = ("You are an expert Python programmer. Complete the function. "
             "Respond with ONLY a single ```python code block containing the full "
             "function. No explanations, no tests, no extra text.")
    for tid in ("HumanEval/0", "HumanEval/10"):
        t = next(x for x in tasks if x["task_id"] == tid)
        query = t["query"]
        try:
            text = call_vllm(base_url, api_key, model, verify, sys_p, query, temp=0.0)
        except Exception as e:  # noqa: BLE001
            print(f"\n===== {tid} =====\nLLM ERROR: {type(e).__name__}: {e}")
            continue
        print(f"\n{'='*70}\n===== {tid} (ep={t['entry_point']}) =====")
        print("--- RAW OUTPUT (first 2500 chars) ---")
        print(text[:2500])
        hc = extract_code_from_state(
            {"messages": [{"role": "user", "content": query},
                          {"role": "assistant", "content": text}]}, ROOT)
        lc = loose_extract(text)
        print("--- HARNESS-EXTRACTED CODE ---")
        print(hc)
        print("--- LOOSE-EXTRACTED CODE ---")
        print(lc)
        print(f"--- HARNESS grade: {functional_correctness(t['entry_point'], t['test'], hc)}")
        print(f"--- LOOSE   grade: {functional_correctness(t['entry_point'], t['test'], lc)}")
        print("--- MANUAL (harness code) ---")
        print(manual(t["entry_point"], t["test"], hc))
        print("--- MANUAL (loose code) ---")
        print(manual(t["entry_point"], t["test"], lc))


if __name__ == "__main__":
    main()
