"""Task Harness 组装：Base + 本轮检索到的经验（≤3 条）。

版本模型（V3 §4）
----------------
- **Base Harness**：git tag ``base/vN`` + 冻结的 env 快照，只经人工 promote 变化；
- **Task Harness**：Base + 检索到的经验，仅存活于单轮，落 ``runs/<task>/<round>/task_harness.json``；
- **Incumbent**：本轮内被接受的候选，轮末丢弃；
- **Candidate**：worktree/沙箱上的一次具体改动。

这样既避免「无限累积漂移」，又让改进能真正落地（人工 promote 门）。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable

from .io_util import now_iso, write_json
from .settings import CFG, Settings


@dataclass
class TaskHarness:
    base_tag: str
    failure_mode: str
    env_overrides: dict[str, str] = field(default_factory=dict)
    lessons: list[dict] = field(default_factory=list)
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict:
        return asdict(self)


def assemble(base_tag: str, failure_mode: str, experiences: Iterable[object],
             env_overrides: dict[str, str] | None = None,
             cfg: Settings | None = None, top: int | None = None) -> TaskHarness:
    """从经验库里挑 ≤3 条组装。低置信经验必须标注 unverified，并保留负面证据数。"""
    cfg = cfg or CFG
    top = top or cfg.exp_retrieve_top
    lessons: list[dict] = []
    for e in list(experiences)[:top]:
        conf = float(getattr(e, "confidence", 0.0))
        lessons.append({
            "experience_id": getattr(e, "experience_id", ""),
            "claim": getattr(e, "claim", ""),
            "component": getattr(e, "component", ""),
            "confidence": conf,
            "verified": conf >= cfg.exp_min_confidence,
            "negative_tasks": list(getattr(e, "negative_tasks", []) or []),
            "text": e.lesson(verified=conf >= cfg.exp_min_confidence)
                    if hasattr(e, "lesson") else str(getattr(e, "claim", "")),
        })
    return TaskHarness(base_tag=base_tag, failure_mode=failure_mode,
                       env_overrides=dict(env_overrides or {}), lessons=lessons)


def write(th: TaskHarness, out_dir: str | Path) -> Path:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    return write_json(d / "task_harness.json", th.to_dict())


def lessons_block(th: TaskHarness) -> str:
    """注入提示的文本块。未验证经验必须显式标注，防止把猜测当规则。"""
    if not th.lessons:
        return ""
    lines = ["[历史经验（来自跨任务验证，注意置信度）]"]
    for l in th.lessons:
        tag = "已验证" if l.get("verified") else "待验证"
        neg = f"，在 {len(l.get('negative_tasks', []))} 个任务上无效" if l.get("negative_tasks") else ""
        lines.append(f"- ({tag} {l.get('confidence', 0):.2f}{neg}) {l.get('claim', '')}")
    return "\n".join(lines)
