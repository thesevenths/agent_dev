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
from .loop_reality_reconciler import (
    AWAITING_APPROVAL, CAMPAIGN_IN_FLIGHT, CORRUPT, EXPERIENCE_PENDING,
    RECOVERY_REQUIRED, ROUND_COMPLETED, reconcile_round,
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
    classification: str = RECOVERY_REQUIRED
    unresolved: list[str] = field(default_factory=list)
    corruption: list[str] = field(default_factory=list)
    gate_ok: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"round_no": self.round_no, "has_objective": self.has_objective,
                "has_plan": self.has_plan, "has_approval": self.has_approval,
                "has_campaign": self.has_campaign, "has_report": self.has_report,
                "has_experience": self.has_experience,
                "director_state": self.director_state,
                "complete": self.complete, "reasons": list(self.reasons),
                "classification": self.classification,
                "unresolved": list(self.unresolved),
                "corruption": list(self.corruption),
                "gate_ok": self.gate_ok}


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
    classification: str = RECOVERY_REQUIRED
    realities: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"loop_id": self.loop_id,
                "previous_state": self.previous_state,
                "resolved_state": self.resolved_state,
                "rounds": [r.to_dict() for r in self.rounds],
                "integrity_ok": self.integrity_ok,
                "corruption": list(self.corruption),
                "next_actions": list(self.next_actions),
                "applied": self.applied,
                "classification": self.classification,
                "realities": list(self.realities)}


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
              rounds_dir: str | Path, service: Any | None = None,
              state_dir: str | Path | None = None,
              budget_ledger: Any | None = None,
              config_sha: str = "") -> LoopRecoveryReport:
    """Read-only reconciliation; with ``service``, resolve REAL store records.

    Without a service this keeps the legacy round-JSON-only contract (used by
    engine-free unit tests). The loop itself MUST pass the real service, so a
    non-empty id can never be mistaken for a resolved artifact.
    """
    integrity = store.verify_integrity(record.loop_id)
    round_records = load_round_records(rounds_dir)
    rounds: list[RoundReconciliation] = []
    realities: list[dict[str, Any]] = []
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
        rec.classification = ROUND_COMPLETED if rec.complete else RECOVERY_REQUIRED
        if not rec.complete:
            rec.reasons = [k for k, v in (
                ("objective", rec.has_objective), ("plan", rec.has_plan),
                ("approval", rec.has_approval), ("campaign", rec.has_campaign),
                ("report", rec.has_report), ("experience", rec.has_experience))
                if not v]
        if service is not None:
            reality = reconcile_round(
                service, rr, rounds_dir=rounds_dir, state_dir=state_dir,
                budget_ledger=budget_ledger, loop_id=record.loop_id,
                config_sha=config_sha)
            realities.append(reality.to_dict())
            rec.classification = reality.classification
            rec.unresolved = list(reality.unresolved)
            rec.corruption = list(reality.corruption)
            rec.gate_ok = reality.gate_ok
            rec.director_state = reality.director_state or rec.director_state
            rec.complete = bool(rec.complete
                                and reality.classification == ROUND_COMPLETED)
            if not rec.complete:
                rec.reasons.extend(reality.unresolved
                                   or [reality.classification])
            corruption.extend(f"round {n}: {c}" for c in reality.corruption)
        rounds.append(rec)

    if not integrity:
        corruption.append("loop event chain integrity failed")
    severity = {
        CORRUPT: 5, RECOVERY_REQUIRED: 4, CAMPAIGN_IN_FLIGHT: 3,
        EXPERIENCE_PENDING: 2, AWAITING_APPROVAL: 1, ROUND_COMPLETED: 0,
    }
    if corruption:
        classification = CORRUPT
    elif rounds:
        classification = max((r.classification for r in rounds),
                             key=lambda c: severity.get(c, 4))
    else:
        classification = RECOVERY_REQUIRED

    previous = record.state
    if previous in TERMINAL:
        resolved = previous
        actions = ["terminal loop; no recovery action"]
    elif classification == ROUND_COMPLETED and integrity:
        resolved = S_ROUND_COMPLETED
        actions = ["all REAL round artifacts resolved; loop may finalize or "
                   "start the next round"]
    else:
        resolved = S_RECOVERING
        actions = [f"recovery classification={classification}"]
        if not integrity:
            actions.append("repair event-chain integrity before resuming")
        actions.extend(f"round {r.round_no}: " + ", ".join(r.reasons)
                       for r in rounds if not r.complete)

    return LoopRecoveryReport(
        loop_id=record.loop_id, previous_state=previous,
        resolved_state=resolved, rounds=rounds, integrity_ok=integrity,
        corruption=corruption, next_actions=actions, applied=False,
        classification=classification, realities=realities)


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
    "reconcile", "apply_recovery", "ROUND_COMPLETED", "AWAITING_APPROVAL",
    "CAMPAIGN_IN_FLIGHT", "EXPERIENCE_PENDING", "RECOVERY_REQUIRED", "CORRUPT",
]
