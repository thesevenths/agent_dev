"""M5-00 -- autonomous loop recovery: reconcile and resume without re-running.

A crashed loop must resume from persisted evidence:
  * the loop state row + append-only hash-chained events (autonomous_loop_state);
  * the per-round evidence JSON written by the runner (objective / plan /
    approval / campaign / report / experience ids);
  * the real ResearchDirector journal (state machine) for the current round.

Recovery NEVER creates a campaign, never grants an approval, never fabricates
Experience, never re-finalizes a round and never re-consumes a round budget. It
produces a RECONCILIATION REPORT then moves the loop to a legal ACTIVE state.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .autonomous_loop_state import (
    LoopEventStore, LoopRecord, S_INTERRUPTED, S_RECOVERING, S_ROUND_COMPLETED,
    TERMINAL,
)


@dataclass
class RoundReconciliation:
    round_no: int
    has_objective: bool = False
    has_plan: bool = False
    has_approval: bool = False
    has_campaign: bool = False
    has_report: bool = False
    has_experience: bool = False
    director_state: str = ""
    complete: bool = False
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"round_no": self.round_no, "has_objective": self.has_objective,
                "has_plan": self.has_plan, "has_approval": self.has_approval,
                "has_campaign": self.has_campaign, "has_report": self.has_report,
                "has_experience": self.has_experience,
                "director_state": self.director_state,
                "complete": self.complete, "reasons": list(self.reasons)}


@dataclass
class LoopRecoveryReport:
    loop_id: str
    previous_state: str
    resolved_state: str
    rounds: list[RoundReconciliation] = field(default_factory=list)
    integrity_ok: bool = False
    corruption: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"loop_id": self.loop_id,
                "previous_state": self.previous_state,
                "resolved_state": self.resolved_state,
                "rounds": [r.to_dict() for r in self.rounds],
                "integrity_ok": self.integrity_ok,
                "corruption": list(self.corruption),
                "next_actions": list(self.next_actions),
                "applied": self.applied}


def load_round_records(rounds_dir: str | Path) -> list[dict[str, Any]]:
    """Read the runner's per-round evidence JSON (read-only)."""
    p = Path(rounds_dir)
    if not p.is_dir():
        return []
    out = []
    for f in sorted(p.glob("round_*.json")):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001 - a corrupt file is reported, never dropped
            out.append({"file": f.name, "corrupt": True})
    return out


def reconcile(*, store: LoopEventStore, record: LoopRecord,
              rounds_dir: str | Path) -> LoopRecoveryReport:
    """Read-only reconciliation of one loop against its persisted round facts."""
    integrity = store.verify_integrity(record.loop_id)
    round_records = load_round_records(rounds_dir)
    rounds: list[RoundReconciliation] = []
    corruption: list[str] = []
    for rr in round_records:
        if rr.get("corrupt"):
            corruption.append(f"corrupt round file: {rr.get('file')}")
            continue
        n = int(rr.get("round_no", 0))
        rec = RoundReconciliation(round_no=n)
        rec.has_objective = bool(rr.get("objective_id"))
        rec.has_plan = bool(rr.get("plan_id"))
        rec.has_approval = bool(rr.get("approval_id"))
        rec.has_campaign = bool(rr.get("campaign_id"))
        rec.has_report = bool(rr.get("report_id"))
        rec.has_experience = bool(rr.get("experience_id"))
        rec.director_state = str(rr.get("director_state") or "")
        rec.complete = all((rec.has_objective, rec.has_plan, rec.has_approval,
                            rec.has_campaign, rec.has_report, rec.has_experience))
        if not rec.complete:
            rec.reasons = [k for k, v in (
                ("objective", rec.has_objective), ("plan", rec.has_plan),
                ("approval", rec.has_approval), ("campaign", rec.has_campaign),
                ("report", rec.has_report), ("experience", rec.has_experience))
                if not v]
        rounds.append(rec)

    previous = record.state
    if previous in TERMINAL:
        resolved = previous
        actions = ["terminal loop; no recovery action"]
    elif integrity and all(r.complete for r in rounds) and rounds:
        resolved = S_ROUND_COMPLETED
        actions = ["all persisted rounds complete; loop may finalize or start "
                   "the next round"]
    else:
        resolved = S_RECOVERING
        actions = (["repair event-chain integrity before resuming"]
                   if not integrity else []) + [
            f"round {r.round_no}: complete missing " + ", ".join(r.reasons)
            for r in rounds if not r.complete]

    return LoopRecoveryReport(
        loop_id=record.loop_id, previous_state=previous,
        resolved_state=resolved, rounds=rounds, integrity_ok=integrity,
        corruption=corruption, next_actions=actions, applied=False)


def apply_recovery(*, store: LoopEventStore, record: LoopRecord,
                   report: LoopRecoveryReport,
                   resolved_state: str | None = None) -> LoopRecoveryReport:
    """Apply a reconciled legal state (idempotent; never re-runs a round)."""
    if report.previous_state in TERMINAL:
        return report
    target = resolved_state or report.resolved_state
    if record.state != target:
        store.append(loop_id=record.loop_id, round_no=record.round_no,
                     event="recovery_applied",
                     payload={"from": record.state, "to": target,
                              "integrity_ok": report.integrity_ok})
        record.state = target
    store.save(record)
    report.resolved_state = target
    report.applied = True
    return report


__all__ = [
    "RoundReconciliation", "LoopRecoveryReport", "load_round_records",
    "reconcile", "apply_recovery",
]
