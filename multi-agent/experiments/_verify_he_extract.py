"""验证 create_file 取码修复：模拟真实 agent 的 final_state 结构，确认能抽到代码并满分。

不依赖 langgraph / requests，只用仓库自带的 humaneval 适配层 + 真实 HumanEval 数据集。
"""
import sys, json
sys.path.insert(0, ".")

from adaptive_runtime.humaneval import (
    load_tasks, extract_code_from_state, functional_correctness,
)

# 取真实数据集全部任务（按 task_id 索引）
tasks = {t["task_id"]: t for t in load_tasks()}

CASES = {
    "HumanEval/0": dict(
        entry="has_close_elements",
        file_contents=(
            "from typing import List\n\n\n"
            "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
            "    for i in range(len(numbers)):\n"
            "        for j in range(i + 1, len(numbers)):\n"
            "            if abs(numbers[i] - numbers[j]) < threshold:\n"
            "                return True\n"
            "    return False\n"
        ),
    ),
    "HumanEval/104": dict(
        entry="unique_digits",
        file_contents=(
            "def unique_digits(x):\n"
            "    result = []\n"
            "    for num in x:\n"
            "        s = str(num)\n"
            "        if all(int(d) % 2 == 1 for d in s):\n"
            "            result.append(num)\n"
            "    return sorted(result)\n"
        ),
    ),
}


def make_final_state(file_contents: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": "implement has_close_elements"},
            {"role": "assistant", "content": ""},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "name": "create_file",
                        "args": {"file_name": "x.py", "file_contents": file_contents},
                    }
                ],
            },
        ]
    }


def main() -> None:
    all_ok = True
    for tid, spec in CASES.items():
        t = tasks[tid]
        entry = spec["entry"]
        fs = make_final_state(spec["file_contents"])
        code = extract_code_from_state(fs, root=".", entry_point=entry)
        extracted = code is not None
        # 诊断：手动 exec 看是否真能定义函数
        diag_fn = None
        if code:
            ns = {}
            try:
                exec(code, ns)
                diag_fn = ns.get(entry)
            except Exception as e:
                diag_fn = f"EXEC-ERR:{e}"
        passed, total = functional_correctness(entry, t["test"], code)
        q = round(100.0 * passed / total, 1) if total else 0.0
        status = "OK " if (extracted and passed == total) else "FAIL"
        if not (extracted and passed == total):
            all_ok = False
        print(f"[{status}] {tid}  entry={entry}")
        print(f"        extracted={extracted}  passed={passed}/{total}  Q={q}  diag_fn={diag_fn!r}")
        if not extracted:
            print(f"        !! code was None")
    print("\n==> %s" % ("ALL GREEN ✅" if all_ok else "STILL BROKEN ❌"))


if __name__ == "__main__":
    main()
