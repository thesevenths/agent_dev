"""M4.3-02 -- Experiment Controller: ONE coordinator for stages and budget.

The controller owns:
  * the experiment identity and its durable state (``experiment_state``);
  * the per-block lifecycle (select -> review -> approve -> execute -> record);
  * the budget guard (campaigns / wall-clock) that stops a run honestly when
    the real environment cannot afford more;
  * the human-review and external-approval bookkeeping.

It does NOT own the science: objectives come from a real ``ObjectiveDiscovery``
or an operator, hypotheses/plans/approvals/campaigns come from the EXISTING
``ResearchService`` through ``ResearchDirector``, and the gate verdict comes
from the read-only gate adapter. No engine call is made here and no approval is
granted here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .experiment_state import (
    E_ANALYZE_PILOT, E_APPROVAL_CONFIRMED, E_BEGIN_BLOCK, E_COMPLETE,
    E_COMPLETE_BLOCK, E_CONFIRMATORY_APPROVAL_REQUEST,
    E_CONFIRMATORY_COMPLETED, E_CONFIRMATORY_START, E_EVALUATING, E_EXECUTING,
    E_FAIL, E_FREEZE_CONFIRMATORY, E_FREEZE_SNAPSHOTS, E_INTERRUPT,
    E_PILOT_COMPLETED, E_PREFLIGHT, E_PREPARE_CONFIRMATORY, E_RECORDING,
    E_RECOVER, E_REPORT, E_REQUEST_APPROVAL, E_REQUEST_REVIEW,
    E_RESOLVE_RECOVERY, E_REVIEW_COMPLETE, E_REVIEW_DEFERRED, E_STOP,
    IllegalTransition, ExperimentEventStore, ExperimentRecord,
    ExperimentStateMachine, S_AWAITING_APPROVAL, S_AWAITING_REVIEW,
    S_BLOCK_COMPLETED, S_COMPLETED, S_CONFIRMATORY_AWAITING_APPROVAL,
    S_CONFIRMATORY_COMPLETED, S_CONFIRMATORY_EXECUTING, S_CONFIRMATORY_FROZEN,
    S_CONFIRMATORY_PREPARING, S_CREATED, S_EVALUATING, S_EXECUTING, S_FAILED,
    S_INTERRUPTED, S_PILOT_ANALYZING, S_PILOT_COMPLETED, S_PREFLIGHTING,
    S_RECORDING, S_RECOVERING, S_REPORTING, S_SELECTING_OBJECTIVES,
    S_SNAPSHOTS_FROZEN, S_STOPPED, TERMINAL_STATES,
)

STAGE_PILOT = "pilot"
STAGE_CONFIRMATORY = "confirmatory"
STAGES: tuple[str, ...] = (STAGE_PILOT, STAGE_CONFIRMATORY)


class BudgetExhausted(RuntimeError):
    """The real environment cannot afford another campaign (stop honestly)."""


@dataclass
class BudgetGuard:
    """Fixed campaign / wall-clock budget for ONE experiment.

    The guard is intentionally conservative: it refuses to START work it cannot
    pay for, and ``stop_reason`` records WHY an experiment halted, so the report
    can say "budget exhausted" instead of silently truncating the sample.
    """
    max_campaigns: int = 0            # 0 = unlimited (pilot may be unlimited)
    max_wall_time_s: float = 0.0      # 0 = unlimited
    campaigns_used: int = 0
    seconds_used: float = 0.0
    stopped_reason: str = ""

    def can_afford_campaign(self) -> bool:
        if self.max_campaigns and self.campaigns_used >= self.max_campaigns:
            return False
        if self.max_wall_time_s and self.seconds_used >= self.max_wall_time_s:
            return False
        return True

    def reserve_campaign(self) -> None:
        if not self.can_afford_campaign():
            self.stopped_reason = (
                "budget exhausted: "
                f"campaigns_used={self.campaigns_used}/"
                f"{self.max_campaigns or 'unlimited'}, "
                f"seconds_used={round(self.seconds_used, 1)}/"
                f"{self.max_wall_time_s or 'unlimited'}")
            raise BudgetExhausted(self.stopped_reason)
        self.campaigns_used += 1

    def charge_seconds(self, seconds: float) -> None:
        self.seconds_used += max(0.0, float(seconds))

    def to_dict(self) -> dict[str, Any]:
        return {"max_campaigns": self.max_campaigns,
                "max_wall_time_s": self.max_wall_time_s,
                "campaigns_used": self.campaigns_used,
                "seconds_used": round(self.seconds_used, 3),
                "campaigns_remaining": (
                    max(0, self.max_campaigns - self.campaigns_used)
                    if self.max_campaigns else None),
                "stopped_reason": self.stopped_reason}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BudgetGuard":
        return cls(max_campaigns=int(d.get("max_campaigns") or 0),
                   max_wall_time_s=float(d.get("max_wall_time_s") or 0.0),
                   campaigns_used=int(d.get("campaigns_used") or 0),
                   seconds_used=float(d.get("seconds_used") or 0.0),
                   stopped_reason=str(d.get("stopped_reason") or ""))


@dataclass
class BlockProgress:
    """Per-block progress as seen by the controller (not the science record)."""
    block_id: str
    state: str = S_SELECTING_OBJECTIVES
    review_status: str = "pending"     # pending | complete | deferred
    review_note: str = ""
    approval_ids: dict[str, str] = field(default_factory=dict)   # arm -> id
    campaign_ids: dict[str, str] = field(default_factory=dict)
    outcomes: dict[str, str] = field(default_factory=dict)
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"block_id": self.block_id, "state": self.state,
                "review_status": self.review_status,
                "review_note": self.review_note,
                "approval_ids": dict(self.approval_ids),
                "campaign_ids": dict(self.campaign_ids),
                "outcomes": dict(self.outcomes),
                "seconds": round(self.seconds, 3)}


class ExperimentController:
    """State / budget / approval coordinator for ONE experiment.

    Construct with ``ExperimentController.create(...)`` for a NEW experiment or
    ``ExperimentController.load(experiment_id, ...)`` to resume an existing one.
    """

    def __init__(self, *, record: ExperimentRecord, store: ExperimentEventStore,
                 budget: BudgetGuard | None = None,
                 approval_verifier: Callable[[str, str], bool] | None = None):
        self.record = record
        self.store = store
        self.sm = ExperimentStateMachine()
        self.blocks: dict[str, BlockProgress] = {}
        self._approval_verifier = approval_verifier
        cfg_blocks = record.config.get("blocks") or {}
        for bid, payload in cfg_blocks.items():
            if isinstance(payload, dict):
                self.blocks[bid] = BlockProgress(
                    block_id=bid, state=payload.get("state",
                                                    S_SELECTING_OBJECTIVES),
                    review_status=payload.get("review_status", "pending"),
                    review_note=payload.get("review_note", ""),
                    approval_ids=dict(payload.get("approval_ids") or {}),
                    campaign_ids=dict(payload.get("campaign_ids") or {}),
                    outcomes=dict(payload.get("outcomes") or {}),
                    seconds=float(payload.get("seconds") or 0.0))
        # The persisted budget is authoritative when resuming; a freshly
        # supplied guard wins only for a NEW experiment (create()).
        if budget is not None:
            self.budget = budget
        elif record.config.get("budget"):
            self.budget = BudgetGuard.from_dict(record.config["budget"])
        else:
            self.budget = BudgetGuard()

    # -------------------------------------------------------------- factory
    @classmethod
    def create(cls, *, experiment_id: str, stage: str = STAGE_PILOT,
               db: Any = None, db_path: str | Path | None = None,
               config: dict[str, Any] | None = None,
               budget: BudgetGuard | None = None,
               approval_verifier: Callable[[str, str], bool] | None = None
               ) -> "ExperimentController":
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}; expected {STAGES}")
        store = ExperimentEventStore(db=db, db_path=None if db is not None
                                     else str(db_path) if db_path else None)
        cfg = dict(config or {})
        if budget is not None:
            cfg["budget"] = budget.to_dict()
        rec = store.create(experiment_id=experiment_id, stage=stage,
                           config=cfg)
        return cls(record=rec, store=store, budget=budget,
                   approval_verifier=approval_verifier)

    @classmethod
    def load(cls, experiment_id: str, *, db: Any = None,
             db_path: str | Path | None = None,
             approval_verifier: Callable[[str, str], bool] | None = None
             ) -> "ExperimentController":
        store = ExperimentEventStore(db=db, db_path=None if db is not None
                                     else str(db_path) if db_path else None)
        rec = store.load(experiment_id)
        if rec is None:
            raise ValueError(f"unknown experiment {experiment_id!r}")
        return cls(record=rec, store=store,
                   approval_verifier=approval_verifier)

    # -------------------------------------------------------------- state
    @property
    def state(self) -> str:
        return self.record.state

    @property
    def is_terminal(self) -> bool:
        return self.sm.is_terminal(self.record.state)

    def history(self) -> list[dict[str, Any]]:
        return self.store.history(self.record.experiment_id)

    def _persist_config(self) -> None:
        """Keep block/budget progress next to the identity row (same db)."""
        self.record.config["blocks"] = {
            bid: b.to_dict() for bid, b in self.blocks.items()}
        self.record.config["budget"] = self.budget.to_dict()
        with self.store.db.session() as conn:
            conn.execute(
                "UPDATE m4_3_experiments SET config_json=? WHERE experiment_id=?",
                (json.dumps(self.record.config, sort_keys=True, default=str),
                 self.record.experiment_id))

    def _transition(self, event: str, payload: dict[str, Any] | None = None):
        from_state = self.record.state
        to_state = self.sm.require(event, from_state)
        self.store.transition(experiment_id=self.record.experiment_id,
                             event=event, from_state=from_state,
                             to_state=to_state, payload=payload or {})
        self.record.state = to_state
        self._persist_config()
        return to_state

    # ------------------------------------------------------- pilot lifecycle
    def preflight(self, *, detail: dict[str, Any] | None = None) -> str:
        return self._transition(E_PREFLIGHT, detail)

    def freeze_snapshots(self, snapshot_ids: list[str] | tuple[str, ...],
                         detail: dict[str, Any] | None = None) -> str:
        ids = list(snapshot_ids)
        if not ids:
            raise ValueError("snapshot set cannot be empty")
        self.record.config["snapshot_ids"] = ids
        return self._transition(E_FREEZE_SNAPSHOTS,
                                {**(detail or {}), "snapshot_ids": ids})

    def begin_block(self, block_id: str, *,
                    detail: dict[str, Any] | None = None) -> BlockProgress:
        if block_id in self.blocks:
            raise ValueError(f"block {block_id!r} already started")
        self.blocks[block_id] = BlockProgress(block_id=block_id)
        self._transition(E_BEGIN_BLOCK, {"block_id": block_id, **(detail or {})})
        return self.blocks[block_id]

    def request_review(self, block_id: str,
                       detail: dict[str, Any] | None = None) -> str:
        return self._transition(E_REQUEST_REVIEW,
                                {"block_id": block_id, **(detail or {})})

    def complete_review(self, block_id: str, *,
                        reviewers: list[str], review_ids: list[str],
                        detail: dict[str, Any] | None = None) -> str:
        """Advance only with COMPLETE human blind review (>= 2 reviewers)."""
        if len(set(reviewers)) < 2:
            raise ValueError(
                "a completed review requires at least two DISTINCT reviewers")
        block = self.blocks[block_id]
        block.review_status = "complete"
        return self._transition(E_REVIEW_COMPLETE, {
            "block_id": block_id, "reviewers": sorted(set(reviewers)),
            "review_ids": sorted(set(review_ids)), **(detail or {})})

    def defer_review(self, block_id: str, *, reason: str) -> str:
        """EXPLICIT, recorded deferral (no human reviewers available yet).

        This is not a silent skip: the deferral is an event with a reason, and
        downstream analysis must treat the objective-quality metric as
        INCOMPLETE for this block.
        """
        if not reason.strip():
            raise ValueError("a review deferral must record a reason")
        block = self.blocks[block_id]
        block.review_status = "deferred"
        block.review_note = reason
        return self._transition(E_REVIEW_DEFERRED,
                                {"block_id": block_id, "reason": reason})

    def request_approval(self, block_id: str, *,
                         detail: dict[str, Any] | None = None) -> str:
        return self._transition(E_REQUEST_APPROVAL,
                                {"block_id": block_id, **(detail or {})})

    def confirm_approval(self, block_id: str, *, arm: str, approval_id: str,
                         plan_id: str = "",
                         detail: dict[str, Any] | None = None) -> str:
        """Record ONE observed external approval; never grant one.

        Recording an approval does NOT start execution: the experiment only
        enters EXECUTING once EVERY arm of the block has a real approval (see
        ``begin_execution``). When an ``approval_verifier`` is configured the
        approval must be verifiable in the trusted store, otherwise the
        record is refused.
        """
        if not approval_id:
            raise ValueError("confirm_approval requires a real approval_id")
        if self._approval_verifier is not None and \
                not self._approval_verifier(approval_id, plan_id):
            raise IllegalTransition(E_APPROVAL_CONFIRMED, self.record.state,
                                    to_state=S_EXECUTING)
        if self.state != S_AWAITING_APPROVAL:
            raise IllegalTransition(E_APPROVAL_CONFIRMED, self.state,
                                    to_state=S_EXECUTING)
        block = self.blocks[block_id]
        block.approval_ids[arm] = approval_id
        self.store.transition(
            experiment_id=self.record.experiment_id,
            event="approval_observed", from_state=self.state,
            to_state=self.state,
            payload={"block_id": block_id, "arm": arm,
                     "approval_id": approval_id, "plan_id": plan_id,
                     **(detail or {})})
        self._persist_config()
        return self.state

    def begin_execution(self, block_id: str, *, arms: list[str] | None = None,
                        detail: dict[str, Any] | None = None) -> str:
        """AWAITING_APPROVAL -> EXECUTING, only when every EXECUTING arm is
        approved. An arm that was retained as INVALID does not need one."""
        block = self.blocks[block_id]
        wanted = list(arms) if arms else ["A", "B", "C"]
        missing = [a for a in wanted if not block.approval_ids.get(a)]
        if missing:
            raise IllegalTransition(E_EXECUTING, self.state,
                                    to_state=S_EXECUTING)
        return self._transition(E_EXECUTING, {
            "block_id": block_id,
            "approval_ids": dict(block.approval_ids), **(detail or {})})

    # Backwards-compatible alias: explicit execution entry point.
    def mark_executing(self, block_id: str, *,
                       detail: dict[str, Any] | None = None) -> str:
        return self.begin_execution(block_id, detail=detail)

    def mark_evaluating(self, block_id: str, *, campaign_id: str = "",
                        seconds: float = 0.0,
                        detail: dict[str, Any] | None = None) -> str:
        """EXECUTING -> EVALUATING. ``detail.arm`` (optional) tags a campaign."""
        block = self.blocks[block_id]
        arm = str((detail or {}).get("arm") or "")
        if campaign_id and arm:
            block.campaign_ids[arm] = campaign_id
        block.seconds += max(0.0, seconds)
        self.budget.charge_seconds(seconds)
        return self._transition(E_EVALUATING, {
            "block_id": block_id, "campaign_id": campaign_id,
            "seconds": seconds, **(detail or {})})

    def mark_recording(self, block_id: str, *, outcomes: dict[str, str],
                       detail: dict[str, Any] | None = None) -> str:
        block = self.blocks[block_id]
        block.outcomes.update(dict(outcomes))
        return self._transition(E_RECORDING, {
            "block_id": block_id, "outcomes": dict(outcomes), **(detail or {})})

    def complete_block(self, block_id: str,
                       detail: dict[str, Any] | None = None) -> str:
        block = self.blocks[block_id]
        block.state = S_BLOCK_COMPLETED
        return self._transition(E_COMPLETE_BLOCK,
                                {"block_id": block_id, **(detail or {})})

    # ------------------------------------------------------- pilot closing
    def analyze_pilot(self, detail: dict[str, Any] | None = None) -> str:
        return self._transition(E_ANALYZE_PILOT, detail)

    def pilot_completed(self, detail: dict[str, Any] | None = None) -> str:
        return self._transition(E_PILOT_COMPLETED, detail)

    # -------------------------------------------------- confirmatory phase
    def prepare_confirmatory(self, *, protocol_draft: dict[str, Any],
                             detail: dict[str, Any] | None = None) -> str:
        self.record.config["confirmatory_draft"] = dict(protocol_draft)
        return self._transition(E_PREPARE_CONFIRMATORY,
                                {**(detail or {}),
                                 "protocol_draft": dict(protocol_draft)})

    def freeze_confirmatory(self, *, protocol_sha256: str,
                            detail: dict[str, Any] | None = None) -> str:
        if len(protocol_sha256) != 64:
            raise ValueError("a frozen protocol requires a 64-char SHA-256")
        self.record.config["confirmatory_protocol_sha256"] = protocol_sha256
        return self._transition(E_FREEZE_CONFIRMATORY,
                                {"protocol_sha256": protocol_sha256,
                                 **(detail or {})})

    def request_confirmatory_approval(self,
                                     detail: dict[str, Any] | None = None) -> str:
        return self._transition(E_CONFIRMATORY_APPROVAL_REQUEST, detail)

    def start_confirmatory(self, *, approval_reference: str,
                           detail: dict[str, Any] | None = None) -> str:
        """The ONLY entry into confirmatory execution.

        Requires a frozen protocol AND an explicit external approval reference;
        there is no default, no auto-start and no bypass.
        """
        if not self.record.config.get("confirmatory_protocol_sha256"):
            raise IllegalTransition(E_CONFIRMATORY_START, self.record.state,
                                    to_state=S_CONFIRMATORY_EXECUTING)
        if not approval_reference.strip():
            raise ValueError(
                "confirmatory execution requires an explicit human approval "
                "reference; it is never started automatically")
        self.record.config["confirmatory_approval_reference"] = approval_reference
        return self._transition(E_CONFIRMATORY_START, {
            "approval_reference": approval_reference, **(detail or {})})

    def confirmatory_completed(self,
                               detail: dict[str, Any] | None = None) -> str:
        return self._transition(E_CONFIRMATORY_COMPLETED, detail)

    # ------------------------------------------------------------- closing
    def reporting(self, detail: dict[str, Any] | None = None) -> str:
        return self._transition(E_REPORT, detail)

    def completed(self, detail: dict[str, Any] | None = None) -> str:
        return self._transition(E_COMPLETE, detail)

    # --------------------------------------------------------------- halts
    def fail(self, error: str) -> str:
        return self._transition(E_FAIL, {"error": error})

    def interrupt(self, reason: str = "") -> str:
        return self._transition(E_INTERRUPT, {"reason": reason})

    def stop(self, reason: str = "") -> str:
        return self._transition(E_STOP, {"reason": reason})

    # ------------------------------------------------------------- recovery
    def recover(self, *, resolved_state: str,
                detail: dict[str, Any] | None = None) -> str:
        """Enter RECOVERING then resolve to the reconciled ACTIVE state."""
        if self.state in TERMINAL_STATES:
            raise IllegalTransition(E_RECOVER, self.state)
        to_recovering = self.sm.require(E_RECOVER, self.state)
        self.store.transition(experiment_id=self.record.experiment_id,
                             event=E_RECOVER, from_state=self.state,
                             to_state=to_recovering,
                             payload=dict(detail or {}))
        self.record.state = to_recovering
        resolved = self.sm.require(E_RESOLVE_RECOVERY, S_RECOVERING,
                                   recovery_target=resolved_state)
        self.store.transition(experiment_id=self.record.experiment_id,
                             event=E_RESOLVE_RECOVERY, from_state=S_RECOVERING,
                             to_state=resolved,
                             payload=dict(detail or {}))
        self.record.state = resolved
        self._persist_config()
        return resolved

    # ---------------------------------------------------------------- views
    def status(self) -> dict[str, Any]:
        return {
            "experiment_id": self.record.experiment_id,
            "stage": self.record.stage,
            "state": self.state,
            "is_terminal": self.is_terminal,
            "snapshot_ids": list(self.record.config.get("snapshot_ids") or []),
            "blocks": {bid: b.to_dict() for bid, b in self.blocks.items()},
            "budget": self.budget.to_dict(),
            "confirmatory_protocol_sha256":
                self.record.config.get("confirmatory_protocol_sha256", ""),
            "n_events": len(self.history()),
        }


__all__ = [
    "STAGE_PILOT", "STAGE_CONFIRMATORY", "STAGES", "BudgetExhausted",
    "BudgetGuard", "BlockProgress", "ExperimentController",
]
