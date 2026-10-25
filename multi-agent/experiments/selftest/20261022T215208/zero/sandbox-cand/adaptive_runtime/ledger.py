"""假设台账：失败假设不可原样重提。

规则
----
- ``fingerprint`` 相同 → 必须引用原 hyp_id 并说明差异，否则 DUPLICATE_HYPOTHESIS；
- ``inconclusive``（样本不足/判分不稳）与 ``rejected`` 分开记，不能互相覆盖；
- 连续 3 次 ``inconclusive`` → 提升为 ``rejected``（该假设在本 harness 下不可判定）。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

from .io_util import append_jsonl, new_id, now_iso, read_jsonl
from .settings import PATHS

STATUSES = ("proposed", "running", "accepted", "rejected", "inconclusive")
_INCONCLUSIVE_LIMIT = 3


def fingerprint(target: str, old: Any, new: Any, taxonomy: str) -> str:
    return f"{target}|{old}->{new}|{taxonomy}"


@dataclass
class Hypothesis:
    hyp_id: str
    fingerprint: str
    claim: str
    target: str
    taxonomy: str = ""
    status: str = "proposed"
    evidence: dict = field(default_factory=dict)
    tasks: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=now_iso)
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class Ledger:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else PATHS.ledger / "hypotheses.jsonl"
        self._rows: list[dict] | None = None

    # ---- IO ----
    @property
    def rows(self) -> list[dict]:
        if self._rows is None:
            self._rows = read_jsonl(self.path)
        return self._rows

    def reload(self) -> None:
        self._rows = None

    def _append(self, row: dict) -> None:
        append_jsonl(self.path, [row])
        if self._rows is not None:
            self._rows.append(row)

    # ---- 查询 ----
    def latest(self, fp: str) -> dict | None:
        for row in reversed(self.rows):
            if row.get("fingerprint") == fp:
                return row
        return None

    def is_duplicate(self, fp: str) -> tuple[bool, dict | None]:
        """同一 fingerprint 只要曾经提出过（proposed/running）或已有结论，就不许原样重提。

        只有 ``inconclusive`` 允许重跑 —— 它的语义是「样本不足/判分不稳」，不是「已证伪」。
        """
        row = self.latest(fp)
        if row and row.get("status") in ("proposed", "running", "rejected", "accepted"):
            return True, row
        return False, row

    def inconclusive_streak(self, fp: str) -> int:
        streak = 0
        for row in reversed(self.rows):
            if row.get("fingerprint") != fp:
                continue
            if row.get("status") == "inconclusive":
                streak += 1
            else:
                break
        return streak

    # ---- 写入 ----
    def register(self, hyp: Hypothesis) -> Hypothesis:
        dup, prev = self.is_duplicate(hyp.fingerprint)
        if dup:
            raise ValueError(f"DUPLICATE_HYPOTHESIS: {hyp.fingerprint} 已存在（{prev.get('status')}）")
        self._append(hyp.to_dict())
        return hyp

    def create(self, target: str, old: Any, new: Any, taxonomy: str, claim: str,
               tasks: list[str] | None = None) -> Hypothesis:
        hyp = Hypothesis(hyp_id=new_id("hyp"), fingerprint=fingerprint(target, old, new, taxonomy),
                         claim=claim, target=target, taxonomy=taxonomy,
                         tasks=list(tasks or []))
        return self.register(hyp)

    def update(self, fp: str, status: str, evidence: dict | None = None,
               notes: str = "", tasks: list[str] | None = None) -> dict:
        if status not in STATUSES:
            raise ValueError(f"未知状态 {status}")
        if status == "rejected" and self.inconclusive_streak(fp) >= _INCONCLUSIVE_LIMIT:
            notes = (notes + " | 由连续 3 次 inconclusive 提升为 rejected").strip(" |")
        row = {"hyp_id": new_id("hyp"), "fingerprint": fp, "status": status,
               "evidence": evidence or {}, "notes": notes,
               "tasks": list(tasks or []), "created_at": now_iso()}
        self._append(row)
        return row

    # ---- 给 proposer 的上下文 ----
    def rejected_context(self, limit: int = 12) -> str:
        rows = [r for r in self.rows if r.get("status") in ("rejected", "inconclusive")]
        if not rows:
            return ""
        lines = []
        for r in rows[-limit:]:
            ev = r.get("evidence") or {}
            lines.append(f"- {r.get('fingerprint')} → {r.get('status')}"
                         f"（n={ev.get('n', '?')}, dQ={ev.get('dQ', '?')}, reason={ev.get('reason', '?')}）")
        return "\n".join(lines)

    def stats(self) -> dict:
        out: dict[str, int] = {}
        for r in self.rows:
            out[r.get("status", "?")] = out.get(r.get("status", "?"), 0) + 1
        return out
