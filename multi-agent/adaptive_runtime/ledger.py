"""假设台账：失败假设不可原样重提。

规则
----
- ``fingerprint`` 相同 → 必须引用原 hyp_id 并说明差异，否则 DUPLICATE_HYPOTHESIS；
- ``inconclusive``（样本不足/判分不稳）与 ``rejected`` 分开记，不能互相覆盖；
- 同一假设拿不出结论到第 3 次就**不再花机时**（见 ``exhausted``），但这不等于证伪：
  状态仍是 ``inconclusive``，不写 ``rejected``。模块早期版本写的「提升为 rejected」
  是个错的设计：基建故障连轮三轮就会把无辜的旋钮永久烧掉。
- 测完了但与噪声不可区分（``reason=within_noise``）也是一条 ``inconclusive``，而它
  **不该重提**：再跑一轮只会拿到同一个平局。区分这两类是 ``exhausted`` 的唯一职责。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

from .io_util import append_jsonl, new_id, now_iso, read_jsonl
from .settings import PATHS

STATUSES = ("proposed", "running", "accepted", "rejected", "inconclusive")
_INCONCLUSIVE_LIMIT = 3
#: 「测完了，但和噪声不可区分」的 reason（gate.Reason.WITHIN_NOISE）。
#: 不在此处 import gate（闸门不依赖台账，保持单向依赖），所以取字面值。
DECISIVE_WASH = "within_noise"


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

    def exhausted(self, fp: str) -> tuple[bool, dict | None]:
        """这个假设是否「再测也不会有新东西」—— proposer 据此不再为它花一轮机时。

        为什么需要单独一条判据（而不只靠 ``is_duplicate``）：``is_duplicate`` 故意放行
        ``inconclusive``，因为它的语义是「还没得出结论」。但 ``inconclusive`` 里混着两种
        完全不同的东西：

        1. **测完了，是个平局**（``reason=within_noise``）：再来一轮 78 个 run 只会得到
           同一个答案。这是 rrsi2 修完台账投毒后暴露的真瓶颈 —— 把噪声误判从
           ``rejected`` 降级成 ``inconclusive`` 之后，若没人记得「这个平局已经测过」，
           闭环就每轮重提同一个旋钮、每轮烧掉同样多的墙钟，永不前进。
        2. **没测成**（缺 run / 止损 / 判分不稳）：这些值得重测；但同一个假设连试 3 次
           还是拿不出结论，就不该继续占用机时。

        两者都只影响「要不要再花这一轮」，**不写 ``rejected``**：状态保持诚实，
        日后基线或其入参变了（fingerprint 随之改变）仍可重新评估。
        """
        rows = [r for r in self.rows if r.get("fingerprint") == fp]
        if not rows:
            return False, None
        last = rows[-1]
        if last.get("status") != "inconclusive":
            return False, last
        reason = str((last.get("evidence") or {}).get("reason") or "")
        if reason == DECISIVE_WASH:
            return True, last
        if self.inconclusive_streak(fp) >= _INCONCLUSIVE_LIMIT:
            return True, last
        return False, last

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
            notes = (notes + " | 注：连续 inconclusive 不会自动变成 rejected（那会把基建故障"
                          "当成证伪），只看 exhausted").strip(" |")
        elif status == "inconclusive" and self.inconclusive_streak(fp) + 1 >= _INCONCLUSIVE_LIMIT:
            # 把「机时预算已用尽」写进注里：它不改变状态，但 proposer 会从这一行读出
            # 「别再提了」，报告里也能看见这个假设是被放弃而不是被证伪。
            notes = (notes + f" | 第 {self.inconclusive_streak(fp) + 1} 次拿不出结论，"
                          "propose 不再为它花机时（见 Ledger.exhausted）").strip(" |")
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
