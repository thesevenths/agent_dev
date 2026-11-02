"""一次性探针：验证『直连模型单轮出码』能否让 HumanEval 出非零分。

不改动任何现有代码，只做测量。用来决定 HumanEval 的正向对照能不能走通。
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "adaptive_runtime"))

from langchain_core.messages import SystemMessage, HumanMessage  # noqa: E402

from llm import coder_llm  # 直连 vLLM 的代码 LLM（temperature=0.3）  # noqa: E402
from adaptive_runtime.humaneval import (  # noqa: E402
    load_tasks, extract_code_from_state, functional_correctness,
)


def main() -> None:
    tasks = load_tasks(n=4)
    print(f"[probe] 载入 {len(tasks)} 题")
    sys_prompt = SystemMessage(
        "You are a Python code completion engine. Given a function signature with "
        "docstring, return ONLY the completed Python function as a single fenced "
        "code block. Do not explain, do not write tests.")
    for t in tasks[:4]:
        query = t["query"]
        print(f"\n=== {t['task_id']} ===")
        for ti, temp in enumerate((0.3, 0.0)):
            try:
                llm = coder_llm.with_config({"temperature": temp}) \
                    if hasattr(coder_llm, "with_config") else coder_llm
                resp = llm.invoke([sys_prompt, HumanMessage(content=query)])
                text = getattr(resp, "content", None) or ""
            except Exception as e:  # noqa: BLE001
                print(f"  temp={temp} ERROR: {type(e).__name__}: {e}")
                continue
            from langchain_core.messages import AIMessage
            code = extract_code_from_state({"messages": [HumanMessage(query),
                                                         AIMessage(content=text)]}, ROOT)
            passed, total = functional_correctness(t["entry_point"], t["test"], code)
            q = 100.0 * passed / total if total else 0.0
            print(f"  temp={temp} extracted={code is not None} "
                  f"passed={passed}/{total} Q={q:.1f} code_len={len(code or '')}")
            if code:
                print("    code head:", (code or "")[:80].replace("\n", " "))


if __name__ == "__main__":
    main()
