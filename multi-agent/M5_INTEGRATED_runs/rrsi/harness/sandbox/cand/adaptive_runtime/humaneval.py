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
    # 兜底：仅当整段文本**以** def/class/@ 开头（绝对行首，无前导 import/自然语言）
    # 才视为裸代码块。不能用 re.M 的「文本中某行含 def」匹配 —— HumanEval 的
    # prompt 本身以函数签名开头、其后跟自然语言指令，会被误判为代码 → SyntaxError → 0 分。
    stripped = text.strip()
    if re.match(r"(def|class|@)", stripped):
        return stripped
    return None


def _msg_role(m: Any) -> str:
    """从 langchain 消息对象或 dict 两种形态里提取 role 字符串（小写）。

    langchain BaseMessage 用 ``.type``（"human"/"ai"/"system"/"tool"）；
    走序列化 dict 时用 ``role`` 字段（"user"/"assistant"/...）。
    无法识别时返回空串，由调用方按「非生成侧」处理。
    """
    if isinstance(m, dict):
        r = m.get("role")
    else:
        r = getattr(m, "type", None) or getattr(m, "role", None)
    return (r or "").lower()


def _msg_content(m: Any) -> str | None:
    """从 langchain 消息对象或 dict 提取文本内容。"""
    if isinstance(m, dict):
        c = m.get("content")
    else:
        c = getattr(m, "content", None)
    return c if isinstance(c, str) else None


def _tool_calls_of(m: Any) -> list:
    """从 langchain 消息对象或 dict 提取 tool_calls 列表（每个含 name/args）。

    langchain 的 ``AIMessage.tool_calls`` 在运行时是 ``ToolCall``（dict 子类）；
    走序列化 dict 时是 ``{"name":..., "args":...}``。两种形态都兼容。
    """
    if isinstance(m, dict):
        tcs = m.get("tool_calls")
    else:
        tcs = getattr(m, "tool_calls", None)
    if not tcs:
        return []
    out: list = []
    for tc in tcs:
        if isinstance(tc, dict):
            out.append(tc)
        else:
            args = getattr(tc, "args", None) or {}
            out.append({"name": getattr(tc, "name", ""),
                        "args": dict(args) if isinstance(args, dict) else {}})
    return out


def _defines(code: str, entry_point: str) -> bool:
    """判断 code 里是否定义了名为 entry_point 的函数。"""
    return bool(re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", code or ""))


def extract_code_from_state(final_state: Any, root: str | Path,
                            entry_point: str | None = None) -> str | None:
    """从 agent 产出抽取解题代码。

    覆盖三类真实来源（根因修复）：
    1. ``create_file`` 工具调用里的 ``file_contents`` —— 真实 agent 把 HumanEval 解答
       写进这个参数（如 ``_sandbox/tmp/<entry>.py``），但**不落持久盘**，故必须直接读
       tool_call 参数。这是之前所有 HumanEval 0 分的真正根因：旧实现既没扫
       tool_calls，也没 glob ``_sandbox/tmp/*.py``（磁盘文件本身也不持久化）。
    2. 磁盘上的 ``.py`` 文件（``_sandbox/tmp/*.py``、``documents/*.py``）—— 兜底，
       按 entry_point 命中优先（注意：文件通常不持久化，故第 1 类才是主路径）。
    3. assistant 消息里的围栏代码块 —— 直连/简单路径。

    选择优先级（命中 ``entry_point`` 的定义永远最优先，避免取到测试/脚手架文件）：
    含 entry_point 的 create_file 内容 > 含 entry_point 的磁盘文件 > 含 entry_point
    的围栏块 > 任意含 ``def`` 的 create_file/文件 > 任意围栏块 > 首个非空候选。
    """
    text_cands: list[str] = []     # assistant 消息文本（跑 _extract_block）
    tool_cands: list[str] = []     # create_file 的 file_contents（整段用）
    file_cands: list[str] = []     # 磁盘 .py（整段用）

    try:
        msgs = (final_state or {}).get("messages") or []
    except Exception:
        msgs = []

    for m in msgs:
        role = _msg_role(m)
        if role in ("ai", "assistant"):
            c = _msg_content(m)
            if c:
                text_cands.append(c)
        # create_file 的 file_contents 是解答的权威来源，与消息 role 无关都要扫
        for tc in _tool_calls_of(m):
            if str(tc.get("name", "")).lower() == "create_file":
                fc = (tc.get("args") or {}).get("file_contents")
                if isinstance(fc, str) and fc.strip():
                    tool_cands.append(fc)

    # 磁盘 .py 兜底（仅限定两个真实落点，避免误扫 sandbox 里的 scaffold 代码）
    for pat in ("documents/*.py", "_sandbox/tmp/*.py"):
        try:
            for py in sorted(Path(root).glob(pat)):
                file_cands.append(py.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            pass

    # ---- 命中 entry_point 的优先 ----
    if entry_point:
        for c in reversed(tool_cands + file_cands):
            if _defines(c, entry_point):
                return c
        for c in text_cands:
            blk = _extract_block(c)
            if blk and _defines(blk, entry_point):
                return blk

    # ---- 无 entry_point 或都未命中：退而求其次 ----
    for c in tool_cands + file_cands:
        if re.search(r"\bdef\s+\w+\s*\(", c or ""):
            return c
    for c in text_cands:
        blk = _extract_block(c)
        if blk:
            return blk
    # 最后的兜底：首个非空候选
    for c in tool_cands + file_cands + text_cands:
        if c and c.strip():
            return c
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
    code = extract_code_from_state(final_state, root, entry_point)
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
