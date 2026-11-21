"""M4.3-02 -- unified experiment state machine + append-only event log.

ONE state vocabulary for the whole A/B/C value-validation runtime. Every
transition is validated against a fixed table (an illegal transition is
REFUSED, never silently coerced) and every accepted transition is appended to a
durable, per-experiment event log, so a restarted process can resume exactly
where it stopped without re-creating a Campaign or re-consuming budget.

States (spec M4.3-02 section 3)
-------------------------------
    CREATED -> PREFLIGHTING -> SNAPSHOTS_FROZEN -> SELECTING_OBJECTIVES
      -> AWAITING_REVIEW -> AWAITING_APPROVAL -> EXECUTING -> EVALUATING
      -> RECORDING -> BLOCK_COMPLETED  (loop per block)
      -> PILOT_ANALYZING -> PILOT_COMPLETED -> CONFIRMATORY_PREPARING
      -> CONFIRMATORY_FROZEN -> CONFIRMATORY_AWAITING_APPROVAL
      -> CONFIRMATORY_EXECUTING -> CONFIRMATORY_COMPLETED -> REPORTING
      -> COMPLETED
    any active state -> FAILED | INTERRUPTED | STOPPED
    any non-terminal state -> RECOVERING -> (resolved target)

This module is PURE orchestration bookkeeping: it never runs an engine, never
grants an approval and never computes a metric.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..research_store.db import ResearchDatabase, _dump, _load, _now

# --------------------------------------------------------------------- states
S_CREATED = "CREATED"
S_PREFLIGHTING = "PREFLIGHTING"
S_SNAPSHOTS_FROZEN = "SNAPSHOTS_FROZEN"
S_SELECTING_OBJECTIVES = "SELECTING_OBJECTIVES"
S_AWAITING_REVIEW = "AWAITING_REVIEW"
S_AWAITING_APPROVAL = "AWAITING_APPROVAL"
S_EXECUTING = "EXECUTING"
S_EVALUATING = "EVALUATING"
S_RECORDING = "RECORDING"
S_BLOCK_COMPLETED = "BLOCK_COMPLETED"
S_PILOT_ANALYZING = "PILOT_ANALYZING"
S_PILOT_COMPLETED = "PILOT_COMPLETED"
S_CONFIRMATORY_PREPARING = "CONFIRMATORY_PREPARING"
S_CONFIRMATORY_FROZEN = "CONFIRMATORY_FROZEN"
S_CONFIRMATORY_AWAITING_APPROVAL = "CONFIRMATORY_AWAITING_APPROVAL"
S_CONFIRMATORY_EXECUTING = "CONFIRMATORY_EXECUTING"
S_CONFIRMATORY_COMPLETED = "CONFIRMATORY_COMPLETED"
S_REPORTING = "REPORTING"
S_COMPLETED = "COMPLETED"
S_FAILED = "FAILED"
S_INTERRUPTED = "INTERRUPTED"
S_RECOVERING = "RECOVERING"
S_STOPPED = "STOPPED"

STATES: tuple[str, ...] = (
    S_CREATED, S_PREFLIGHTING, S_SNAPSHOTS_FROZEN, S_SELECTING_OBJECTIVES,
    S_AWAITING_REVIEW, S_AWAITING_APPROVAL, S_EXECUTING, S_EVALUATING,
    S_RECORDING, S_BLOCK_COMPLETED, S_PILOT_ANALYZING, S_PILOT_COMPLETED,
    S_CONFIRMATORY_PREPARING, S_CONFIRMATORY_FROZEN,
    S_CONFIRMATORY_AWAITING_APPROVAL, S_CONFIRMATORY_EXECUTING,
    S_CONFIRMATORY_COMPLETED, S_REPORTING, S_COMPLETED, S_FAILED,
    S_INTERRUPTED, S_RECOVERING, S_STOPPED,
)

# Terminal states cannot transition anywhere (a finished experiment is final).
TERMINAL_STATES: frozenset[str] = frozenset({S_COMPLETED, S_FAILED, S_STOPPED})
# Interruptions are resumable: they may only enter RECOVERING.
RESUMABLE_STATES: frozenset[str] = frozenset({S_INTERRUPTED})
# "Active" = a live experiment that can still make progress.
ACTIVE_STATES: frozenset[str] = frozenset(
    set(STATES) - TERMINAL_STATES - RESUMABLE_STATES)

# --------------------------------------------------------------------- events
E_CREATE = "create"
E_PREFLIGHT = "preflight"
E_FREEZE_SNAPSHOTS = "freeze_snapshots"
E_BEGIN_BLOCK = "begin_block"
E_REQUEST_REVIEW = "request_review"
E_REVIEW_COMPLETE = "review_complete"
E_REVIEW_DEFERRED = "review_deferred"
E_REQUEST_APPROVAL = "request_approval"
E_APPROVAL_CONFIRMED = "approval_confirmed"
E_EXECUTING = "executing"
E_EVALUATING = "evaluating"
E_RECORDING = "recording"
E_COMPLETE_BLOCK = "complete_block"
E_ANALYZE_PILOT = "analyze_pilot"
E_PILOT_COMPLETED = "pilot_completed"
E_PREPARE_CONFIRMATORY = "prepare_confirmatory"
E_FREEZE_CONFIRMATORY = "freeze_confirmatory"
E_CONFIRMATORY_APPROVAL_REQUEST = "confirmatory_approval_request"
E_CONFIRMATORY_START = "confirmatory_start"
E_CONFIRMATORY_COMPLETED = "confirmatory_completed"
E_REPORT = "report"
E_COMPLETE = "complete"
E_FAIL = "fail"
E_INTERRUPT = "interrupt"
E_STOP = "stop"
E_RECOVER = "recover"
E_RESOLVE_RECOVERY = "resolve_recovery"

# (event, from_state) -> to_state. ``"*"`` means "any ACTIVE state".
_TRANSITIONS: dict[str, dict[str, str]] = {
    E_CREATE: {"": S_CREATED},
    E_PREFLIGHT: {S_CREATED: S_PREFLIGHTING},
    E_FREEZE_SNAPSHOTS: {S_PREFLIGHTING: S_SNAPSHOTS_FROZEN},
    # A new block may start from the frozen set or after a completed block.
    E_BEGIN_BLOCK: {S_SNAPSHOTS_FROZEN: S_SELECTING_OBJECTIVES,
                    S_BLOCK_COMPLETED: S_SELECTING_OBJECTIVES},
    E_REQUEST_REVIEW: {S_SELECTING_OBJECTIVES: S_AWAITING_REVIEW},
    E_REVIEW_COMPLETE: {S_AWAITING_REVIEW: S_AWAITING_APPROVAL},
    # A deferral is EXPLICIT (recorded reason); the experiment may proceed to
    # approval while the human blind review remains pending. It never counts
    # as a completed review downstream.
    E_REVIEW_DEFERRED: {S_AWAITING_REVIEW: S_AWAITING_APPROVAL},
    E_REQUEST_APPROVAL: {S_AWAITING_REVIEW: S_AWAITING_APPROVAL},
    E_APPROVAL_CONFIRMED: {S_AWAITING_APPROVAL: S_EXECUTING},
    E_EXECUTING: {S_AWAITING_APPROVAL: S_EXECUTING},
    E_EVALUATING: {S_EXECUTING: S_EVALUATING},
    E_RECORDING: {S_EVALUATING: S_RECORDING},
    E_COMPLETE_BLOCK: {S_RECORDING: S_BLOCK_COMPLETED},
    E_ANALYZE_PILOT: {S_BLOCK_COMPLETED: S_PILOT_ANALYZING},
    E_PILOT_COMPLETED: {S_PILOT_ANALYZING: S_PILOT_COMPLETED},
    E_PREPARE_CONFIRMATORY: {S_PILOT_COMPLETED: S_CONFIRMATORY_PREPARING},
    E_FREEZE_CONFIRMATORY: {S_CONFIRMATORY_PREPARING: S_CONFIRMATORY_FROZEN},
    E_CONFIRMATORY_APPROVAL_REQUEST: {
        S_CONFIRMATORY_FROZEN: S_CONFIRMATORY_AWAITING_APPROVAL},
    E_CONFIRMATORY_START: {
        S_CONFIRMATORY_AWAITING_APPROVAL: S_CONFIRMATORY_EXECUTING},
    E_CONFIRMATORY_COMPLETED: {
        S_CONFIRMATORY_EXECUTING: S_CONFIRMATORY_COMPLETED},
    E_REPORT: {S_CONFIRMATORY_COMPLETED: S_REPORTING,
               S_PILOT_COMPLETED: S_REPORTING},
    E_COMPLETE: {S_REPORTING: S_COMPLETED},
    E_FAIL: {"*": S_FAILED},
    E_INTERRUPT: {"*": S_INTERRUPTED},
    E_STOP: {"*": S_STOPPED},
    E_RECOVER: {"*": S_RECOVERING},
}


class IllegalTransition(ValueError):
    """An event is not legal from the current state (fail-closed)."""

    def __init__(self, event: str, from_state: str,
                 to_state: str | None = None):
        target = f" -> {to_state}" if to_state else ""
        super().__init__(
            f"illegal experiment transition: event {event!r} is not allowed "
            f"from state {from_state!r}{target}")
        self.event, self.from_state, self.to_state = event, from_state, to_state


class ExperimentStateMachine:
    """Validates transitions. No I/O, no side effects."""

    def can(self, event: str, from_state: str) -> bool:
        table = _TRANSITIONS.get(event)
        if table is None:
            return False
        if from_state in table:
            return True
        # Wildcard events (fail/interrupt/stop/recover) apply to ACTIVE states
        # AND to a RESUMABLE (interrupted) experiment -- only ``recover`` is
        # actually allowed from there (enforced in ``target``).
        return "*" in table and from_state in (ACTIVE_STATES | RESUMABLE_STATES)

    def is_terminal(self, state: str) -> bool:
        return state in TERMINAL_STATES

    def target(self, event: str, from_state: str,
               recovery_target: str | None = None) -> str:
        if event == E_RESOLVE_RECOVERY:
            if from_state != S_RECOVERING:
                raise IllegalTransition(event, from_state, recovery_target)
            if recovery_target not in ACTIVE_STATES:
                raise IllegalTransition(event, from_state, recovery_target)
            return recovery_target
        table = _TRANSITIONS.get(event)
        if table is None or not self.can(event, from_state):
            raise IllegalTransition(event, from_state)
        to_state = table.get(from_state) or table["*"]
        if from_state in TERMINAL_STATES:
            raise IllegalTransition(event, from_state, to_state)
        if from_state in RESUMABLE_STATES and event != E_RECOVER:
            raise IllegalTransition(event, from_state, to_state)
        return to_state

    def require(self, event: str, from_state: str,
                recovery_target: str | None = None) -> str:
        return self.target(event, from_state, recovery_target=recovery_target)


# --------------------------------------------------------------- persistence
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS m4_3_experiments (
    experiment_id    TEXT PRIMARY KEY,
    stage            TEXT NOT NULL,
    state            TEXT NOT NULL,
    config_json      TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS m4_3_experiment_events (
    seq              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id    TEXT NOT NULL,
    event            TEXT NOT NULL,
    from_state       TEXT NOT NULL,
    to_state         TEXT NOT NULL,
    payload_json     TEXT NOT NULL,
    created_at       TEXT NOT NULL
);
"""


@dataclass
class ExperimentRecord:
    """Durable identity + current state of ONE experiment."""
    experiment_id: str
    stage: str
    state: str
    config: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"experiment_id": self.experiment_id, "stage": self.stage,
                "state": self.state, "config": dict(self.config),
                "created_at": self.created_at, "updated_at": self.updated_at}


class ExperimentEventStore:
    """Append-only event log + current-state row in the ARR SQLite database.

    Reuses ``ResearchDatabase`` (no second database layer). Event rows are
    never updated or deleted; the current state is a derived convenience row
    that is always written in the SAME transaction as the event.
    """

    def __init__(self, db: ResearchDatabase | None = None, *,
                 db_path: str | None = None):
        if db is not None:
            self.db = db
            with self.db.session() as conn:
                for statement in SCHEMA_SQL.split(";"):
                    if statement.strip():
                        conn.execute(statement)
        else:
            if not db_path:
                raise ValueError("db_path is required when db is not injected")
            self.db = ResearchDatabase(db_path, schema_sql=SCHEMA_SQL)

    # ------------------------------------------------------------- create
    def create(self, *, experiment_id: str, stage: str,
               config: dict[str, Any]) -> ExperimentRecord:
        now = _now()
        with self.db.session() as conn:
            exists = conn.execute(
                "SELECT 1 FROM m4_3_experiments WHERE experiment_id=?",
                (experiment_id,)).fetchone()
            if exists is not None:
                raise ValueError(f"experiment {experiment_id!r} already exists")
            conn.execute(
                """INSERT INTO m4_3_experiments
                   (experiment_id, stage, state, config_json, created_at,
                    updated_at) VALUES (?,?,?,?,?,?)""",
                (experiment_id, stage, S_CREATED, _dump(config), now, now))
            self._append(conn, experiment_id=experiment_id, event=E_CREATE,
                         from_state="", to_state=S_CREATED,
                         payload={"stage": stage})
        return ExperimentRecord(experiment_id=experiment_id, stage=stage,
                                state=S_CREATED, config=dict(config),
                                created_at=now, updated_at=now)

    def _append(self, conn, *, experiment_id: str, event: str,
                from_state: str, to_state: str,
                payload: dict[str, Any]) -> None:
        conn.execute(
            """INSERT INTO m4_3_experiment_events
               (experiment_id, event, from_state, to_state, payload_json,
                created_at) VALUES (?,?,?,?,?,?)""",
            (experiment_id, event, from_state, to_state, _dump(payload),
             _now()))

    # -------------------------------------------------------------- read
    def load(self, experiment_id: str) -> ExperimentRecord | None:
        rows = self.db.rows(
            "SELECT * FROM m4_3_experiments WHERE experiment_id=?",
            (experiment_id,))
        if not rows:
            return None
        r = rows[0]
        return ExperimentRecord(
            experiment_id=r["experiment_id"], stage=r["stage"],
            state=r["state"], config=_load(r["config_json"], {}),
            created_at=r["created_at"], updated_at=r["updated_at"])

    def history(self, experiment_id: str) -> list[dict[str, Any]]:
        rows = self.db.rows(
            """SELECT * FROM m4_3_experiment_events
               WHERE experiment_id=? ORDER BY seq""", (experiment_id,))
        return [{"seq": r["seq"], "event": r["event"],
                 "from_state": r["from_state"], "to_state": r["to_state"],
                 "payload": _load(r["payload_json"], {}),
                 "created_at": r["created_at"]} for r in rows]

    def list_experiments(self) -> list[ExperimentRecord]:
        rows = self.db.rows("SELECT * FROM m4_3_experiments ORDER BY created_at")
        return [ExperimentRecord(
            experiment_id=r["experiment_id"], stage=r["stage"],
            state=r["state"], config=_load(r["config_json"], {}),
            created_at=r["created_at"], updated_at=r["updated_at"]) for r in rows]

    # --------------------------------------------------------- transition
    def transition(self, *, experiment_id: str, event: str,
                   from_state: str, to_state: str,
                   payload: dict[str, Any] | None = None) -> None:
        """Persist ONE transition. The caller has already validated it."""
        with self.db.session() as conn:
            row = conn.execute(
                "SELECT state FROM m4_3_experiments WHERE experiment_id=?",
                (experiment_id,)).fetchone()
            if row is None:
                raise ValueError(f"unknown experiment {experiment_id!r}")
            if row["state"] != from_state:
                raise IllegalTransition(event, row["state"], to_state)
            conn.execute(
                "UPDATE m4_3_experiments SET state=?, updated_at=? "
                "WHERE experiment_id=?",
                (to_state, _now(), experiment_id))
            self._append(conn, experiment_id=experiment_id, event=event,
                         from_state=from_state, to_state=to_state,
                         payload=dict(payload or {}))

    def export(self, experiment_id: str) -> dict[str, Any]:
        rec = self.load(experiment_id)
        return {"experiment": rec.to_dict() if rec else None,
                "events": self.history(experiment_id)}


__all__ = [
    "STATES", "ACTIVE_STATES", "TERMINAL_STATES", "RESUMABLE_STATES",
    "S_CREATED", "S_PREFLIGHTING", "S_SNAPSHOTS_FROZEN",
    "S_SELECTING_OBJECTIVES", "S_AWAITING_REVIEW", "S_AWAITING_APPROVAL",
    "S_EXECUTING", "S_EVALUATING", "S_RECORDING", "S_BLOCK_COMPLETED",
    "S_PILOT_ANALYZING", "S_PILOT_COMPLETED", "S_CONFIRMATORY_PREPARING",
    "S_CONFIRMATORY_FROZEN", "S_CONFIRMATORY_AWAITING_APPROVAL",
    "S_CONFIRMATORY_EXECUTING", "S_CONFIRMATORY_COMPLETED", "S_REPORTING",
    "S_COMPLETED", "S_FAILED", "S_INTERRUPTED", "S_RECOVERING", "S_STOPPED",
    "E_CREATE", "E_PREFLIGHT", "E_FREEZE_SNAPSHOTS", "E_BEGIN_BLOCK",
    "E_REQUEST_REVIEW", "E_REVIEW_COMPLETE", "E_REVIEW_DEFERRED",
    "E_REQUEST_APPROVAL", "E_APPROVAL_CONFIRMED", "E_EXECUTING",
    "E_EVALUATING", "E_RECORDING", "E_COMPLETE_BLOCK", "E_ANALYZE_PILOT",
    "E_PILOT_COMPLETED", "E_PREPARE_CONFIRMATORY", "E_FREEZE_CONFIRMATORY",
    "E_CONFIRMATORY_APPROVAL_REQUEST", "E_CONFIRMATORY_START",
    "E_CONFIRMATORY_COMPLETED", "E_REPORT", "E_COMPLETE", "E_FAIL",
    "E_INTERRUPT", "E_STOP", "E_RECOVER", "E_RESOLVE_RECOVERY",
    "IllegalTransition", "ExperimentStateMachine", "ExperimentRecord",
    "ExperimentEventStore", "SCHEMA_SQL",
]
