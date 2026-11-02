"""探针 v6：用真实消息顺序（user prompt 在前 + assistant 代码在后）验证修复后的
extract_code_from_state 不再把 prompt 当代码。模拟 agent 真实 final_state["messages"]。"""
from __future__ import annotations
import sys, json, ssl
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import urllib.request  # noqa: E402
from adaptive_runtime.humaneval import (  # noqa: E402
    load_tasks, extract_code_from_state, functional_correctness,
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


def main():
    env = read_env(ROOT / ".env")
    base = env["LLM_BASE_URL"]; key = env["LLM_API_KEY"]; model = env["LLM_MODEL"]
    verify = env.get("LLM_VERIFY_SSL", "true").strip().lower() in ("1", "true", "yes", "y", "on")
    tasks = load_tasks(n=12)
    sys_p = ("You are an expert Python programmer. Complete the function. "
             "Respond with ONLY a single ```python code block containing the full "
             "function. No explanations, no tests, no extra text.")
    for tid in ("HumanEval/0", "HumanEval/10", "HumanEval/1"):
        t = next(x for x in tasks if x["task_id"] == tid)
        text = call_vllm(base, key, model, verify, sys_p, t["query"], temp=0.0)
        # 模拟 agent 真实 final_state：user 在前（含 def 的 prompt），assistant 在后
        state = {"messages": [
            {"role": "user", "content": t["query"]},
            {"role": "assistant", "content": text},
        ]}
        code = extract_code_from_state(state, ROOT)
        head = (code or "").splitlines()[0] if code else "<None>"
        grade = functional_correctness(t["entry_point"], t["test"], code)
        print(f"{tid}: first_code_line={head!r}  grade={grade}")


if __name__ == "__main__":
    main()
