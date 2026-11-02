"""一次性探针 v3：换提示词格式，确认 Qwen-27B 能否真正完成 HumanEval 函数体。

不改动任何现有代码，只做测量 + 对比两种 prompt 策略。
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "adaptive_runtime"))

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage  # noqa: E402
from llm import coder_llm  # noqa: E402
from adaptive_runtime.humaneval import load_tasks, extract_code_from_state, functional_correctness  # noqa: E402
import json as _json


def grade(entry_point, test, code):
    passed, total = functional_correctness(entry_point, test, code)
    return passed, total


def call(sys_text, human_text):
    try:
        resp = coder_llm.invoke([SystemMessage(content=sys_text), HumanMessage(content=human_text)])
        return getattr(resp, "content", None) or ""
    except Exception as e:  # noqa: BLE001
        return f"<<ERROR {e}>>"


def main():
    # 载入原始 HumanEval prompt（不含 _INSTRUCTION）
    p = ROOT / "experiments" / "humaneval" / "HumanEval.jsonl"
    rows = [ _json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip() ]
    rows.sort(key=lambda r: str(r.get("task_id","")))
    for tid in ("HumanEval/0", "HumanEval/10"):
        r = next(x for x in rows if x["task_id"] == tid)
        prompt = str(r["prompt"]).strip()   # 纯函数签名+docstring
        test = str(r["test"]); ep = str(r["entry_point"])
        print(f"\n=== {tid} (ep={ep}) ===")
        # 策略 A：指令放 system，human 只留纯函数 prompt
        sysA = ("You are an expert Python programmer. Complete the function below. "
                "Reply with ONLY a single ```python fenced code block containing the full "
                "function implementation (signature + docstring + body). No explanations, "
                "no tests, no extra text.")
        textA = call(sysA, prompt)
        codeA = extract_code_from_state({"messages":[HumanMessage(prompt), AIMessage(textA)]}, ROOT)
        pa, ta = grade(ep, test, codeA)
        print(f"  [A system=instr] extracted={codeA is not None} passed={pa}/{ta} "
              f"len={len(codeA or '')} | head: {(codeA or '')[:70]!r}")
        # 策略 B：human = HumanEval 原 prompt + 清晰指令（无 _INSTRUCTION 复述）
        sysB = "You are an expert Python programmer."
        humanB = (prompt + "\n\nImplement the function above. Output ONLY a single Python code "
                          "block with the full implementation. Do not write explanations or tests.")
        textB = call(sysB, humanB)
        codeB = extract_code_from_state({"messages":[HumanMessage(humanB), AIMessage(textB)]}, ROOT)
        pb, tb = grade(ep, test, codeB)
        print(f"  [B clear instr] extracted={codeB is not None} passed={pb}/{tb} "
              f"len={len(codeB or '')} | head: {(codeB or '')[:70]!r}")


if __name__ == "__main__":
    main()
