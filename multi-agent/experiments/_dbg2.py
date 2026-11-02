import sys
sys.path.insert(0, ".")
from adaptive_runtime.humaneval import functional_correctness, load_tasks, _count_asserts

tasks = {t["task_id"]: t for t in load_tasks()}
t = tasks["HumanEval/0"]
test = t["test"]
entry = t["entry_point"]
print("entry_point =", entry)
print("test total asserts via _count:", _count_asserts(test))

# 真实 agent 写出的形态（带 typing 注解）
code_annotated = (
    "from typing import List\n"
    "\n"
    "\n"
    "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
    "    for i in range(len(numbers)):\n"
    "        for j in range(i + 1, len(numbers)):\n"
    "            if abs(numbers[i] - numbers[j]) < threshold:\n"
    "                return True\n"
    "    return False\n"
)
# 无注解形态（直连探针验证过的）
code_plain = (
    "def has_close_elements(numbers, threshold):\n"
    "    for i in range(len(numbers)):\n"
    "        for j in range(i + 1, len(numbers)):\n"
    "            if abs(numbers[i] - numbers[j]) < threshold:\n"
    "                return True\n"
    "    return False\n"
)

for label, code in [("ANNOTATED", code_annotated), ("PLAIN", code_plain)]:
    ns = {}
    try:
        exec(code, ns)
        fn = ns.get(entry)
        print(f"[{label}] exec OK, fn callable={callable(fn)}")
    except Exception as e:
        print(f"[{label}] EXEC-ERR: {type(e).__name__}: {e}")
        fn = None
    if fn:
        p, tot = functional_correctness(entry, test, code)
        print(f"[{label}] functional_correctness -> passed={p}/{tot} Q={round(100*p/tot,1) if tot else 0}")
    else:
        print(f"[{label}] fn is None -> Q will be 0")
