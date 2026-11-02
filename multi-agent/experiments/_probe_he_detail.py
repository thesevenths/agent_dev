"""一次性探针 v2：确认 HumanEval/0 直连出码到底是『逻辑错』还是『取码/评分 bug』。

不改动任何现有代码，只做测量。
"""
from __future__ import annotations
import sys, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "adaptive_runtime"))

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage  # noqa: E402
from llm import coder_llm  # noqa: E402
from adaptive_runtime.humaneval import load_tasks, extract_code_from_state  # noqa: E402
import os as _os
print("LLM_MODEL =", _os.environ.get("LLM_MODEL"), "| BASE =", _os.environ.get("LLM_BASE_URL"))


def run_manual(entry_point: str, test: str, code: str):
    ns: dict = {}
    try:
        exec(code, ns)
    except Exception as e:  # noqa: BLE001
        return f"EXEC ERROR: {type(e).__name__}: {e}"
    fn = ns.get(entry_point)
    if not callable(fn):
        return f"NO FUNCTION {entry_point} (got keys: {[k for k in ns if not k.startswith('__')][:10]})"
    test_ns = dict(ns)
    try:
        exec(test, test_ns)
        test_ns.get("check")(fn)
        return "MANUAL check() PASSED"
    except Exception as e:  # noqa: BLE001
        return f"MANUAL check() FAILED: {type(e).__name__}: {e}"


def main() -> None:
    tasks = load_tasks(n=1)
    t = tasks[0]
    print(f"\n=== {t['task_id']} ===")
    print("--- test source ---")
    print(t["test"])
    sys_prompt = SystemMessage(
        "You are a Python code completion engine. Given a function signature with "
        "docstring, return ONLY the completed Python function as a single fenced "
        "code block. Do not explain, do not write tests.")
    try:
        resp = coder_llm.invoke([sys_prompt, HumanMessage(content=t["query"])])
        text = getattr(resp, "content", None) or ""
    except Exception as e:  # noqa: BLE001
        print("LLM ERROR:", e); return
    code = extract_code_from_state(
        {"messages": [HumanMessage(content=t["query"]), AIMessage(content=text)]}, ROOT)
    print("--- extracted code (full) ---")
    print(code)
    print("--- manual run ---")
    print(run_manual(t["entry_point"], t["test"], code))


if __name__ == "__main__":
    main()
