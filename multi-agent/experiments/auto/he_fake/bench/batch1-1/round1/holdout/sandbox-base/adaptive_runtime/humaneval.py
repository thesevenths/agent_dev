"""HumanEval 适配器：把 OpenAI HumanEval 接进现有 RRSI 引擎，不需要 Docker。

为什么是这一层
--------------
RRSI 闭环唯一的评分货币是 ``Q``（``probe.run_one`` → ``runner_once`` → ``evaluator.score_trace``）。
HumanEval 不是交付物任务，它的 Q 必须是「单测是否通过」，而不是交付物的 A/E/C。
本模块提供三件事，全部新增、不改动现有评分链路：

1. ``load_tasks``        ：把 HumanEval 数据集映射成 ``{task_id, query, benchmark, entry_point, test}``，
                          其中 ``query`` 已嵌入「只输出 Python 代码」的指令，交给现有 agent 跑。
2. ``extract_code_from_state`` / ``functional_correctness`` ：从 agent 产出里抽取代码，用 HumanEval 自带的
                          ``test``（一组 ``assert``）跑出 **断言通过比例** 作为连续 Q（0–100）。
                          用比例而不是二值：二值 Q 的 null sd 接近 50，gate 几乎永远 REJECT；
                          比例 Q 方差更小，真实改进才有可能过门。
3. ``grade_run``         ：``runner_once`` 在 ``--benchmark humaneval`` 时调用，返回可直接写 ``score.json`` 的结果。

数据来源（按优先级）
--------------------
``load_tasks(data_path)`` 依次尝试：① 显式 ``--data``；② ``experiments/humaneval/HumanEval.jsonl``；
③ 已安装的 ``human_eval`` 包自带数据。任一命中即可。数据集是公开单文件（164 题 ≈ 80KB），
可用 ``experiments/fetch_humaneval.py`` 一次性下载（需联网）。

注意：本模块**不** import ``probe`` / ``benchmark`` / ``runner_once``，避免任何循环依赖；
它只被 ``runner_once`` 单向调用。
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from .settings import PATHS

#: HumanEval 数据集落盘目录（由 ``fetch_humaneval.py`` 写入）。
TASKS_DIR = PATHS.experiments / "humaneval"

#: 拼到每个 HumanEval prompt 后的指令：把「交付物 agent」引导成「代码生成 agent」。
_INSTRUCTION = (
    "Complete the function above and return ONLY a single Python code block "
    "containing the full function definition (signature + docstring + body). "
    "Do not write explanations and do not write the test harness."
)


# ---------------------------------------------------------------- 数据加载

def load_tasks(data_path: str | Path | None = None, n: int | None = None,
               seed: int = 0) -> list[dict]:
    """把 HumanEval 数据集映射成闭环任务。

    ``n`` 取前 N 个（按任务 id 排序，确定性、可复现），不随机打乱 —— 跨轮可比比随机更稳。
    ``seed`` 保留为接口兼容位，当前未用于洗牌。
    """
    p = Path(data_path) if data_path else (TASKS_DIR / "HumanEval.jsonl")
    if not p.exists():
        raise SystemExit(
            f"HumanEval 数据集不存在：{p}\n"
            f"  → 先跑 `python experiments/fetch_humaneval.py`（需联网），"
            f"或用 --data 指定本地 HumanEval.jsonl")
    rows: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        raise SystemExit(f"HumanEval 数据集为空：{p}")

    # HumanEval 的 task_id 形如 HumanEval/0 .. /163，按 id 排序后取前 N 即确定性子集。
    rows.sort(key=lambda r: str(r.get("task_id", "")))
    if n is not None and 0 < n < len(rows):
        rows = rows[:n]

    tasks: list[dict] = []
    for r in rows:
        prompt = str(r.get("prompt") or "").strip()
        if not prompt:
            continue
        query = prompt + "\n\n" + _INSTRUCTION
        tasks.append({
            "task_id": str(r["task_id"]),
            "query": query,
            "benchmark": "humaneval",
            "entry_point": str(r.get("entry_point") or ""),
            "test": str(r.get("test") or ""),
            "family": "coding",
            "source": "HumanEval",
        })
    if not tasks:
        raise SystemExit("没有可用的 HumanEval 任务（prompt 全部为空？）")
    return tasks


def fetch(dest: str | Path | None = None) -> Path:
    """一次性下载 HumanEval 数据集并解压到 ``experiments/humaneval/HumanEval.jsonl``（需联网）。

    openai/human-eval 仓库只提供压缩版 ``data/HumanEval.jsonl.gz``，
    未压缩的 ``HumanEval.jsonl`` 直接访问会 404，所以这里统一走 ``.gz`` 并解压。
    """
    import gzip
    import urllib.request

    out = Path(dest) if dest else (TASKS_DIR / "HumanEval.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    url = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
    print(f"[fetch] 下载 {url}")
    print(f"[fetch] 写入 {out}")
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (已知公开 URL)
        raw = resp.read()
    # 兼容两种返回：gzip 字节（正常）或明文（罕见回退）
    try:
        text = gzip.decompress(raw).decode("utf-8")
    except (OSError, gzip.BadGzipFile):
        text = raw.decode("utf-8", "replace")
    out.write_text(text, encoding="utf-8")
    n = len([l for l in text.splitlines() if l.strip()])
    print(f"[fetch] 完成：{n} 题")
    return out


# ---------------------------------------------------------------- 代码抽取

_FENCE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.S | re.I)


def _extract_block(text: str) -> str | None:
    """从一段文本里抽取 Python 代码：优先取最长的 ```python 围栏块；没有围栏但整体像函数则整段用。"""
    if not text:
        return None
    blocks = [b.strip() for b in _FENCE.findall(text) if b.strip()]
    if blocks:
        blocks.sort(key=lambda b: (0 if "def " in b else 1, -len(b)))
        return blocks[0]
    if re.search(r"^\s*def\s+\w+\s*\(", text, re.M):
        return text.strip()
    return None


def extract_code_from_state(final_state: Any, root: str | Path) -> str | None:
    """从 agent 产出抽取代码：先扫最终消息，再退到 ``documents/*.py``。"""
    candidates: list[str] = []
    try:
        msgs = (final_state or {}).get("messages") or []
    except Exception:
        msgs = []
    for m in msgs:
        content = getattr(m, "content", None)
        if content is None and isinstance(m, dict):
            content = m.get("content")
        if isinstance(content, str):
            candidates.append(content)
    # 退路：agent 可能用 create_file 把代码写进了 documents/
    try:
        for py in sorted(Path(root).glob("documents/*.py")):
            candidates.append(py.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        pass
    for text in candidates:
        code = _extract_block(text)
        if code:
            return code
    return None


# ---------------------------------------------------------------- 功能性判分

def _count_asserts(test_src: str) -> int:
    try:
        tree = ast.parse(test_src)
    except Exception:
        return 0
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == "check":
            return sum(1 for s in n.body if isinstance(s, ast.Assert))
    return 0


def _assert_fraction(test_src: str, ns: dict, entry_point: str) -> tuple[int, int] | None:
    """逐条执行 ``check`` 里的 assert，返回 (通过数, 总数)。

    只处理 HumanEval 最常见形态：``def check(candidate):  <setup 语句>; <assert ...>``。
    assert 表达式在 ``check`` 的局部命名空间里求值（已绑定 ``candidate`` 为解题函数）。
    嵌套在循环/分支里的 assert 不在直接子节点上，此时回退到二值判定（见 ``functional_correctness``）。
    """
    try:
        tree = ast.parse(test_src)
    except Exception:
        return None
    check = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "check"), None)
    if check is None:
        return None
    setup = [s for s in check.body if not isinstance(s, ast.Assert)]
    asserts = [s for s in check.body if isinstance(s, ast.Assert)]
    if not asserts:
        return None

    local: dict[str, Any] = dict(ns)
    local["candidate"] = ns.get(entry_point)
    builtins = {"__builtins__": __builtins__}  # type: ignore[dict-item]
    try:
        for s in setup:
            exec(compile(ast.Module([s], []), "<he_setup>", "exec"), builtins, local)  # noqa: S102
    except Exception:
        return None
    local["candidate"] = ns.get(entry_point)
    passed = 0
    for a in asserts:
        try:
            val = eval(compile(ast.Expression(a.test), "<he_assert>", "eval"), builtins, local)  # noqa: S307
            if bool(val):
                passed += 1
        except Exception:
            pass
    return (passed, len(asserts))


def functional_correctness(entry_point: str, test: str, code: str | None
                           ) -> tuple[int, int]:
    """返回 (通过的断言数, 断言总数)。code 为空或不可执行 → (0, 总数)。

    连续 Q = 100 * passed / total；用比例而非二值是为了压低 null sd，让 gate 能检出真实改进。
    """
    total = max(1, _count_asserts(test))
    if not code:
        return (0, total)
    ns: dict[str, Any] = {}
    try:
        exec(code, ns)  # noqa: S102  解题代码来自本仓库自己的模型输出，非外部不可信输入
    except Exception:
        return (0, total)
    fn = ns.get(entry_point)
    if not callable(fn):
        return (0, total)
    frac = _assert_fraction(test, ns, entry_point)
    if frac is not None:
        return frac
    # 回退二值：直接跑 ``check(candidate)``，抛错即 0 分。
    test_ns = dict(ns)
    try:
        exec(test, test_ns)  # noqa: S102
        test_ns.get("check")(fn)
        return (total, total)
    except Exception:
        return (0, total)


# ---------------------------------------------------------------- 单 run 评分（给 runner_once 用）

def grade_run(entry_point: str, test: str, final_state: Any, root: str | Path) -> dict:
    """在 ``runner_once`` 里调用：抽取代码 → 跑单测 → 产出结构化结果。"""
    code = extract_code_from_state(final_state, root)
    passed, total = functional_correctness(entry_point, test, code)
    q = 100.0 * passed / total if total else 0.0
    a = passed / total if total else 0.0
    return {
        "Q": q,
        "A": a,
        "passed": passed,
        "total": total,
        "all_pass": bool(total) and passed == total,
        "extracted": code is not None,
        "code_len": len(code or ""),
    }
