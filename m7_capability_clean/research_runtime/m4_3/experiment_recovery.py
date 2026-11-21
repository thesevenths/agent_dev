"""M4.3-02 -- experiment recovery: reconcile and resume without re-running.

A crashed / killed experiment must resume from persisted evidence:

  * the controller's state row + append-only event log (``experiment_state``);
  * the ExperimentRegistry's blocks / runs / finalized flags;
  * the real campaign ids already bound to runs.

Recovery NEVER creates a campaign, never grants an approval, never re-finalizes
a run and never re-consumes budget. It produces a RECONCILIATION REPORT first
(what is complete, what is missing, what would be the next legal action) and
only then moves the controller to a legal ACTIVE state via the state machine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..campaign.schemas import CAMPAIGN_TERMINAL_STATUSES
from .experiment_controller import ExperimentController
from .experiment_state import (
    ACTIVE_STATES, S_BLOCK_COMPLETED, S_CONFIRMATORY_AWAITING_APPROVAL,
    S_CONFIRMATORY_EXECUTING, S_SELECTING_OBJECTIVES, TERMINAL_STATES,
)
from .experiment_registry import ARMS
from .registry_store import RegistryStore


@dataclass
class BlockReconciliation:
    block_id: str
    arms_registered: list[str] = field(default_factory=list)
    arms_finalized: list[str] = field(default_factory=list)
    missing_arms: list[str] = field(default_factory=list)
    unfinalized_arms: list[str] = field(default_factory=list)
    campaign_ids: dict[str, str] = field(default_factory=dict)
    duplicate_campaigns: list[str] = field(default_factory=list)
    complete: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"block_id": self.block_id,
                "arms_registered": list(self.arms_registered),
                "arms_finalized": list(self.arms_finalized),
                "missing_arms": list(self.missing_arms),
                "unfinalized_arms": list(self.unfinalized_arms),
                "campaign_ids": dict(self.campaign_ids),
                "duplicate_campaigns": list(self.duplicate_campaigns),
                "complete": self.complete}


@dataclass
class RecoveryReport:
    experiment_id: str
    previous_state: str
    resolved_state: str
    blocks: list[BlockReconciliation] = field(default_factory=list)
    registry_blocks: int = 0
    registry_runs: int = 0
    duplicate_campaign_ids: list[str] = field(default_factory=list)
    corruption: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"experiment_id": self.experiment_id,
                "previous_state": self.previous_state,
                "resolved_state": self.resolved_state,
                "blocks": [b.to_dict() for b in self.blocks],
                "registry_blocks": self.registry_blocks,
                "registry_runs": self.registry_runs,
                "duplicate_campaign_ids": list(self.duplicate_campaign_ids),
                "corruption": list(self.corruption),
                "next_actions": list(self.next_actions),
                "applied": self.applied}


def inspect_registry(db: Any) -> tuple[list[dict[str, Any]],
                                       list[dict[str, Any]]]:
    """Read blocks/runs from the M4.3 tables (read-only)."""
    store = RegistryStore(db=db)
    return store.load_blocks(), store.load_runs()


def reconcile(controller: ExperimentController, *,
              db: Any | None = None) -> RecoveryReport:
    """Read-only reconciliation of ONE experiment against its registry."""
    db = db if db is not None else controller.store.db
    blocks_raw, runs_raw = inspect_registry(db)
    runs_by_block: dict[str, list[dict[str, Any]]] = {}
    for run in runs_raw:
        runs_by_block.setdefault(str(run.get("block_id", "")), []).append(run)

    report = RecoveryReport(
        experiment_id=controller.record.experiment_id,
        previous_state=controller.state,
        resolved_state=controller.state,
        registry_blocks=len(blocks_raw), registry_runs=len(runs_raw))

    campaign_owner: dict[str, list[str]] = {}
    for run in runs_raw:
        cid = str(run.get("campaign_id") or "")
        if cid:
            campaign_owner.setdefault(cid, []).append(str(run.get("run_id")))

    seen_dupes: set[str] = set()
    for block in blocks_raw:
        bid = str(block.get("block_id", ""))
        arms = {str(r.get("arm")): r for r in runs_by_block.get(bid, [])}
        rec = BlockReconciliation(block_id=bid)
        rec.arms_registered = sorted(arms)
        rec.arms_finalized = sorted(a for a, r in arms.items()
                                    if r.get("finalized"))
        rec.missing_arms = [a for a in ARMS if a not in arms]
        rec.unfinalized_arms = [a for a in arms if not arms[a].get("finalized")]
        rec.campaign_ids = {a: str(r.get("campaign_id") or "")
                            for a, r in arms.items()}
        for cid in rec.campaign_ids.values():
            owners = campaign_owner.get(cid, [])
            if cid and len(set(owners)) > 1:
                rec.duplicate_campaigns.append(cid)
                seen_dupes.add(cid)
        rec.complete = (not rec.missing_arms and not rec.unfinalized_arms)
        report.blocks.append(rec)

    report.duplicate_campaign_ids = sorted(seen_dupes)

    # ------------------------------------------------------ resolved state
    state = controller.state
    complete_blocks = [b for b in report.blocks if b.complete]
    blocking = [b for b in report.blocks if not b.complete]

    if state in TERMINAL_STATES:
        report.resolved_state = state
        report.next_actions.append(
            "experiment is terminal; nothing to resume")
        return report

    if state in (S_CONFIRMATORY_EXECUTING, S_CONFIRMATORY_AWAITING_APPROVAL):
        report.resolved_state = state
        report.next_actions.append(
            "confirmatory phase: resume from the frozen protocol; do not "
            "create any new campaign for a block that already has one")
        return report

    if blocking:
        report.resolved_state = S_SELECTING_OBJECTIVES
        for b in blocking:
            if b.unfinalized_arms:
                report.next_actions.append(
                    f"block {b.block_id}: finalize arm(s) "
                    f"{','.join(b.unfinalized_arms)} from the persisted "
                    "campaign record (no new campaign)")
            if b.missing_arms:
                report.next_actions.append(
                    f"block {b.block_id}: run arm(s) "
                    f"{','.join(b.missing_arms)}")
    elif complete_blocks:
        report.resolved_state = S_BLOCK_COMPLETED
        report.next_actions.append(
            "all blocks complete; continue with pilot analysis")
    else:
        report.resolved_state = S_SELECTING_OBJECTIVES
        report.next_actions.append("no blocks yet; start the first block")

    if report.duplicate_campaign_ids:
        report.corruption.append(
            "duplicate campaign ids across runs: "
            + ", ".join(report.duplicate_campaign_ids))

    if report.resolved_state not in ACTIVE_STATES:
        report.resolved_state = S_SELECTING_OBJECTIVES
    return report


def recover_experiment(controller: ExperimentController, *,
                       db: Any | None = None,
                       apply: bool = True) -> RecoveryReport:
    """Reconcile and (optionally) move the controller to the legal state."""
    report = reconcile(controller, db=db)
    if not apply or controller.state in TERMINAL_STATES:
        return report
    if report.resolved_state == controller.state and \
            report.previous_state == report.resolved_state:
        report.applied = True
        return report
    controller.recover(resolved_state=report.resolved_state,
                       detail={"recovery": report.to_dict()})
    report.applied = True
    return report


def assert_campaigns_not_duplicated(db: Any) -> None:
    """Fail closed if two runs share one campaign id (cross-run reuse)."""
    _, runs = inspect_registry(db)
    seen: dict[str, set[str]] = {}
    for run in runs:
        cid = str(run.get("campaign_id") or "")
        if cid:
            seen.setdefault(cid, set()).add(str(run.get("run_id")))
    dupes = {cid: sorted(r) for cid, r in seen.items() if len(r) > 1}
    if dupes:
        raise ValueError(f"campaign reuse across runs detected: {dupes}")


def campaign_is_terminal(record: Any) -> bool:
    status = getattr(record, "status", "") if record is not None else ""
    return status in CAMPAIGN_TERMINAL_STATUSES


__all__ = [
    "BlockReconciliation", "RecoveryReport", "inspect_registry", "reconcile",
    "recover_experiment", "assert_campaigns_not_duplicated",
    "campaign_is_terminal",
]
