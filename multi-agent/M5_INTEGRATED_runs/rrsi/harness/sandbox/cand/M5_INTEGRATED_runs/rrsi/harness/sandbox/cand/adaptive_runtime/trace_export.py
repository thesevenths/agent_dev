"""Trace 旁路导出：把既有 runlog 文本解析成结构化事件，不改任何现有埋点。

为什么是「解析」而不是「重写埋点」
----------------------------------
runlog.py 已经在写 ``[step-failure] kind= agent=`` / ``[critic+summary] ... PASS``
/ ``[tool][Agent] ✔ name (1.2s, 300 chars)`` 这些结构化程度足够的事件。
再插一套埋点等于双源，且要改生产文件 —— 违反本方案的第一个不变量。

已覆盖的真实格式（2026-10 抽样核对）
------------------------------------
- ``[supervisor] created execution plan (goal='...'):`` + ``N.    [pending] 标题: 描述 → agent``
- ``[critic+summary] step 1/5 CrawlerAgent: PASS via FUSED judge+summary ...``
- ``[critic] step 1/3 rejected → retry 1/2 with feedback: ...``
- ``[step-failure] step 3 attempt #1 kind=recursion agent=CodeAgent`` + 下一行 ``  WHY: ...``
- ``[tool][CodeAgent] ▶ create_file({...})`` / ``[tool][CodeAgent] ✔ create_file (0.01s, 778 chars) -> ...``
- ``[supervisor] re-planning step 4/5:`` / ``EARLY FINISH at step 4`` / ``FINISH``
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .io_util import read_text, safe_relpath, sha256_text, write_json

# ---------------------------------------------------------------- 正则

_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3} INFO (.*)$")
_PLAN_HEADER = re.compile(r"\[supervisor\] created execution plan \(goal='(.*)'\):")
_PLAN_ITEM = re.compile(r"^\s*(\d+)\.\s+\[(\w+)\]\s+(.+?)\s*→\s*([A-Za-z_][\w]*)\s*$")
_CRITIC_PASS = re.compile(r"\[critic\+summary\] step (\d+)/(\d+) ([A-Za-z_][\w]*): PASS")
_CRITIC_REJECT = re.compile(
    r"\[critic\] step (\d+)/(\d+) ([A-Za-z_][\w]*): rejected → retry (\d+)/(\d+)")
_STEP_FAIL = re.compile(
    r"\[step-failure\] step (\d+) attempt #(\d+) kind=(\w+) agent=([A-Za-z_][\w]*)")
_TOOL_START = re.compile(r"\[tool\]\[([A-Za-z_?][\w]*)\] ▶ ([\w.]+)\(")
_TOOL_END = re.compile(
    r"\[tool\]\[([A-Za-z_?][\w]*)\] ([✔✖]) ([\w.]+) \(([\d.]+)s, (\d+) chars\)")
_REPLAN = re.compile(r"\[supervisor\] re-planning step (\d+)/(\d+)")
_EARLY = re.compile(r"\[supervisor\] EARLY FINISH at step (\d+)")
_FINISH = re.compile(r"\[supervisor\] FINISH")
_RUN_START = re.compile(r"=== run log started: (.*?) ===")
_RECALL = re.compile(r"for query '(.*?)':", re.S)

#: 可交付产物后缀（与 agents._PATH_EXTS 的语义一致，这里是独立副本，
#: 因为冻结清单已经把 agents.py 锁定，不能 import 它的私有常量）
DELIVERABLE_EXTS = {".md", ".docx", ".doc", ".pdf", ".html", ".htm", ".xlsx", ".xls",
                    ".csv", ".pptx", ".png", ".jpg", ".jpeg", ".svg", ".zip", ".txt"}


# ---------------------------------------------------------------- 数据

@dataclass
class StepState:
    step: int
    passes: int = 0
    rejects: int = 0
    failures: int = 0
    last: str = "unknown"          # pass | fail | reject | unknown
    fail_kind: str = ""
    agent: str = ""

    @property
    def achieved(self) -> bool:
        return self.last == "pass"


@dataclass
class Trace:
    run_id: str
    log_path: str
    started_at: str = ""
    goal: str = ""
    query: str = ""
    plan: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    steps: dict[int, StepState] = field(default_factory=dict)
    total_steps: int = 0
    artifacts: list[dict] = field(default_factory=list)
    result: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    counters: dict = field(default_factory=dict)
    unknown_events: int = 0

    # ---- 派生指标（evaluator 用） ----
    @property
    def passed_steps(self) -> int:
        return sum(1 for s in self.steps.values() if s.achieved)

    @property
    def contract_rate(self) -> float:
        total = self.total_steps or len(self.steps)
        if total <= 0:
            return 0.0
        return max(0.0, min(1.0, self.passed_steps / total))

    @property
    def truncated(self) -> bool:
        return self.result.get("truncated", False)

    @property
    def salvaged(self) -> bool:
        return self.result.get("salvaged", False)

    def deliverables(self) -> list[dict]:
        return [a for a in self.artifacts if a.get("deliverable")]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["steps"] = {str(k): asdict(v) for k, v in self.steps.items()}
        d["derived"] = {
            "passed_steps": self.passed_steps,
            "contract_rate": round(self.contract_rate, 4),
            "deliverables": len(self.deliverables()),
            "truncated": self.truncated,
            "salvaged": self.salvaged,
        }
        return d


# ---------------------------------------------------------------- 解析

def _blocks(text: str) -> list[tuple[str, str]]:
    """把日志切成 (时间戳, 首行消息 + 续行) 的块。"""
    out: list[tuple[str, str]] = []
    for raw in text.splitlines():
        m = _TS.match(raw)
        if m:
            out.append((m.group(1), m.group(2)))
        elif out:
            ts, msg = out[-1]
            out[-1] = (ts, msg + "\n" + raw)
    return out


def _derive_run_id(path: Path) -> str:
    stem = path.stem
    if "_" in stem:
        parts = stem.split("_")
        if len(parts) >= 3:
            return f"{parts[-2]}_{parts[-1]}"
    return stem


def export_from_log(log_path: str | Path, root: str | Path | None = None,
                    run_id: str | None = None, window_max: float = 7200.0) -> Trace:
    path = Path(log_path)
    text = read_text(path)
    root = Path(root) if root else Path(__file__).resolve().parent.parent
    tr = Trace(run_id=run_id or _derive_run_id(path), log_path=str(path))

    blocks = _blocks(text)
    if blocks:
        tr.started_at = blocks[0][0]

    plan_mode = False
    current_fail: dict | None = None
    counters: dict[str, int] = {}

    for ts, msg in blocks:
        head = msg.split("\n", 1)[0]

        if _RUN_START.search(head):
            counters["run_start"] = counters.get("run_start", 0) + 1
            continue

        m = _PLAN_HEADER.search(head)
        if m:
            tr.goal = m.group(1)
            plan_mode = True
            tr.plan = []
            # 计划条目是以「无时间戳的续行」形式跟在 header 后面的，必须在本块内解析
            for line in msg.split("\n")[1:]:
                mi = _PLAN_ITEM.match(line)
                if mi:
                    tr.plan.append({
                        "step_no": int(mi.group(1)),
                        "status": mi.group(2),
                        "title": mi.group(3).strip(),
                        "agent": mi.group(4),
                    })
            continue

        if plan_mode:
            mi = _PLAN_ITEM.match(msg)
            if mi:
                tr.plan.append({
                    "step_no": int(mi.group(1)),
                    "status": mi.group(2),
                    "title": mi.group(3).strip(),
                    "agent": mi.group(4),
                })
                continue
            plan_mode = False

        m = _CRITIC_PASS.search(head)
        if m:
            step, total, agent = int(m.group(1)), int(m.group(2)), m.group(3)
            st = tr.steps.setdefault(step, StepState(step))
            st.passes += 1
            st.last = "pass"
            st.agent = st.agent or agent
            tr.total_steps = max(tr.total_steps, total, step)
            tr.events.append({"t": ts, "type": "gate", "gate": "fused_critic",
                              "step_no": step, "agent": agent, "pass": True})
            counters["gate_pass"] = counters.get("gate_pass", 0) + 1
            continue

        m = _CRITIC_REJECT.search(head)
        if m:
            step, total, agent = int(m.group(1)), int(m.group(2)), m.group(3)
            st = tr.steps.setdefault(step, StepState(step))
            st.rejects += 1
            st.last = "reject"
            st.agent = st.agent or agent
            tr.total_steps = max(tr.total_steps, total, step)
            tr.events.append({"t": ts, "type": "gate", "gate": "fused_critic",
                              "step_no": step, "agent": agent, "pass": False,
                              "retry": int(m.group(4)), "reason_head": head[:200]})
            counters["gate_reject"] = counters.get("gate_reject", 0) + 1
            continue

        m = _STEP_FAIL.search(head)
        if m:
            step, attempt, kind, agent = (int(m.group(1)), int(m.group(2)),
                                          m.group(3), m.group(4))
            why = ""
            if "\n" in msg:
                for line in msg.split("\n")[1:]:
                    if line.strip().upper().startswith("WHY:"):
                        why = line.split(":", 1)[1].strip()
                        break
            st = tr.steps.setdefault(step, StepState(step))
            st.failures += 1
            st.last = "fail"
            st.fail_kind = kind
            st.agent = st.agent or agent
            tr.events.append({"t": ts, "type": "step_failure", "step_no": step,
                              "attempt": attempt, "kind": kind, "agent": agent,
                              "why": why})
            counters[f"fail_{kind}"] = counters.get(f"fail_{kind}", 0) + 1
            current_fail = {"kind": kind, "why": why}
            continue

        m = _TOOL_END.search(head)
        if m:
            agent, ok, name, dur, chars = (m.group(1), m.group(2), m.group(3),
                                           float(m.group(4)), int(m.group(5)))
            tr.events.append({"t": ts, "type": "tool", "agent": agent, "name": name,
                              "ok": ok == "✔", "ms": int(dur * 1000), "chars": chars})
            counters["tool"] = counters.get("tool", 0) + 1
            continue

        m = _TOOL_START.search(head)
        if m:
            counters["tool_start"] = counters.get("tool_start", 0) + 1
            continue

        m = _REPLAN.search(head)
        if m:
            step, total = int(m.group(1)), int(m.group(2))
            tr.total_steps = max(tr.total_steps, total, step)
            tr.events.append({"t": ts, "type": "replan", "step_no": step})
            counters["replan"] = counters.get("replan", 0) + 1
            continue

        m = _EARLY.search(head)
        if m:
            tr.events.append({"t": ts, "type": "early_finish", "step_no": int(m.group(1))})
            tr.result["early_finish"] = int(m.group(1))
            counters["early_finish"] = counters.get("early_finish", 0) + 1
            continue

        if _FINISH.search(head):
            tr.result["finalized"] = True
            counters["finish"] = counters.get("finish", 0) + 1
            continue

        m = _RECALL.search(msg)
        if m and not tr.query:
            tr.query = m.group(1).strip()
            continue

    # ---- 兜底：plan 里出现但从未上报的步骤也算进分母 ----
    if tr.plan:
        tr.total_steps = max(tr.total_steps, max(p["step_no"] for p in tr.plan))
        for p in tr.plan:
            tr.steps.setdefault(p["step_no"], StepState(p["step_no"], agent=p.get("agent", "")))

    tr.result.setdefault("finalized", bool(counters.get("finish")))
    tr.result["truncated"] = any(s.fail_kind == "truncated" for s in tr.steps.values())
    tr.result["salvaged"] = any("已按完成处理" in (e.get("why") or "")
                                for e in tr.events if e.get("type") == "step_failure")
    tr.counters = counters
    # 已识别块 = 产生事件的块 + 计划行 + 仅有计数的块
    recognized = (len(tr.events) + len(tr.plan)
                  + counters.get("run_start", 0) + counters.get("tool_start", 0)
                  + counters.get("finish", 0))
    tr.unknown_events = max(0, len(blocks) - recognized)

    # ---- 产物扫描（窗口 = [run 开始, min(日志最后写入, 开始+window_max)]）----
    # 历史日志的 mtime 可能被整体拷动过（本仓库 2026-10-08 一次拷贝把 95 个日志和
    # tmp/ 产物的 mtime 全刷成了同一时刻）。此时按 mtime 归属产物是不可能的，
    # 必须显式标记 artifacts_unavailable，而不是把别人的产物算到这次 run 头上。
    since = _parse_ts(tr.started_at)
    try:
        log_mtime = path.stat().st_mtime
    except OSError:
        log_mtime = since
    capped = log_mtime > since + window_max
    if capped:
        tr.artifacts = []
        tr.result["artifacts_unavailable"] = True
    else:
        tr.artifacts = scan_artifacts(root, since, min(log_mtime + 2.0, since + window_max))
    tr.result["artifacts_window_capped"] = bool(capped)
    return tr


def _parse_ts(ts: str) -> float:
    try:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").timestamp()
    except (ValueError, TypeError):
        return 0.0


def scan_artifacts(root: str | Path, since_ts: float,
                   until_ts: float | None = None, limit: int = 800) -> list[dict]:
    """扫描本次 run 期间落盘的文件（documents/ tmp/ 根目录）。

    用 mtime 而不是解析 tool 入参：tool 入参预览会被截断（``...[+474 chars]``），
    解析它必然漏文件；mtime 是确定性的。
    必须同时给上界（日志最后写入时刻），否则回放旧 run 会把之后所有产物都算进去。
    """
    root = Path(root)
    candidates: list[Path] = []
    for sub in ("documents", "tmp"):
        d = root / sub
        if d.is_dir():
            try:
                candidates.extend([p for p in d.rglob("*") if p.is_file()])
            except OSError:
                pass
    try:
        candidates.extend([p for p in root.glob("*.*") if p.is_file()])
    except OSError:
        pass

    out: list[dict] = []
    for p in candidates[:limit]:
        try:
            st = p.stat()
        except OSError:
            continue
        if st.st_mtime + 0.5 < since_ts:
            continue
        if until_ts is not None and st.st_mtime > until_ts:
            continue
        ext = p.suffix.lower()
        out.append({
            "path": safe_relpath(p, root),
            "bytes": st.st_size,
            "ext": ext,
            "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            "deliverable": ext in DELIVERABLE_EXTS and st.st_size > 0,
        })
    out.sort(key=lambda a: a["bytes"], reverse=True)
    return out


# ---------------------------------------------------------------- 写出

def write_trace(trace: Trace, out_dir: str | Path, name: str = "trace.json") -> Path:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    return write_json(d / name, trace.to_dict())


def trace_fingerprint(trace: Trace) -> str:
    """轨迹指纹：用于去重与复现（不含时间戳）。"""
    parts = [trace.goal]
    for p in trace.plan:
        parts.append(f"{p['step_no']}|{p['agent']}|{p['title'][:40]}")
    for e in trace.events:
        if e.get("type") in ("step_failure", "gate"):
            parts.append(f"{e.get('type')}|{e.get('step_no')}|{e.get('kind')}|{e.get('pass')}")
    return sha256_text("\n".join(parts))[:16]


def load_trace(path: str | Path) -> dict:
    from .io_util import read_json
    return read_json(path, default={}) or {}


def iter_logs(log_dir: str | Path) -> Iterable[Path]:
    p = Path(log_dir)
    if not p.is_dir():
        return []
    return sorted(p.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
