"""M5-00 -- single-domain Autonomous Research Loop state + durable event log.

THIN orchestration bookkeeping ONLY. This module never runs an engine, never
grants an approval, never fabricates Experience/Evidence and never computes a
verdict. It mirrors the proven M4.3 pattern (fixed transition table + append-only
hash-chained events) for ONE autonomous loop made of ROUNDS.

Round lifecycle
---------------
    CREATED -> PREFLIGHTED -> (per round) DISCOVERING -> OBJECTIVE_READY
      -> HYPOTHESIS_READY -> PLAN_READY -> AWAITING_APPROVAL -> EXECUTING
      -> EVALUATING -> EXPERIENCE_RECORDED -> ROUND_COMPLETED
      -> (next round, feeding prior Experience back into Discovery)
      -> LOOP_COMPLETED
    any active state -> FAILED | INTERRUPTED | STOPPED
    active/interrupted -> RECOVERING -> resolved legal active state
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .research_store.db import ResearchDatabase, _dump, _load, _now

S_CREATED = "CREATED"
S_PREFLIGHTED = "PREFLIGHTED"
S_DISCOVERING = "DISCOVERING"
S_OBJECTIVE_READY = "OBJECTIVE_READY"
S_HYPOTHESIS_READY = "HYPOTHESIS_READY"
S_PLAN_READY = "PLAN_READY"
S_AWAITING_APPROVAL = "AWAITING_APPROVAL"
S_EXECUTING = "EXECUTING"
S_EVALUATING = "EVALUATING"
S_EXPERIENCE_RECORDED = "EXPERIENCE_RECORDED"
S_ROUND_COMPLETED = "ROUND_COMPLETED"
S_LOOP_COMPLETED = "LOOP_COMPLETED"
S_FAILED = "FAILED"
S_INTERRUPTED = "INTERRUPTED"
S_STOPPED = "STOPPED"
S_RECOVERING = "RECOVERING"

STATES: tuple[str, ...] = (
    S_CREATED, S_PREFLIGHTED, S_DISCOVERING, S_OBJECTIVE_READY,
    S_HYPOTHESIS_READY, S_PLAN_READY, S_AWAITING_APPROVAL, S_EXECUTING,
    S_EVALUATING, S_EXPERIENCE_RECORDED, S_ROUND_COMPLETED,
    S_LOOP_COMPLETED, S_FAILED, S_INTERRUPTED, S_STOPPED, S_RECOVERING,
)
TERMINAL = frozenset({S_LOOP_COMPLETED, S_FAILED, S_STOPPED})
RESUMABLE = frozenset({S_INTERRUPTED})
ACTIVE = frozenset(set(STATES) - TERMINAL - RESUMABLE)

E_CREATE = "create"
E_PREFLIGHT = "preflight"
E_BEGIN_ROUND = "begin_round"
E_OBJECTIVE = "objective_proposed"
E_HYPOTHESIS = "hypotheses_ready"
E_PLAN = "plan_created"
E_APPROVAL_REQUEST = "approval_requested"
E_APPROVAL_CONFIRMED = "approval_observed"
E_EXECUTE = "executing"
E_EVALUATE = "evaluating"
E_EXPERIENCE = "experience_recorded"
E_COMPLETE_ROUND = "complete_round"
E_COMPLETE_LOOP = "complete_loop"
E_FAIL = "fail"
E_INTERRUPT = "interrupt"
E_STOP = "stop"
E_RECOVER = "recover"
E_RESOLVE = "resolve_recovery"

_TRANSITIONS: dict[str, dict[str, str]] = {
    E_CREATE: {"": S_CREATED},
    E_PREFLIGHT: {S_CREATED: S_PREFLIGHTED},
    E_BEGIN_ROUND: {S_PREFLIGHTED: S_DISCOVERING,
                    S_ROUND_COMPLETED: S_DISCOVERING},
    E_OBJECTIVE: {S_DISCOVERING: S_OBJECTIVE_READY},
    E_HYPOTHESIS: {S_OBJECTIVE_READY: S_HYPOTHESIS_READY},
    E_PLAN: {S_HYPOTHESIS_READY: S_PLAN_READY},
    E_APPROVAL_REQUEST: {S_PLAN_READY: S_AWAITING_APPROVAL},
    E_APPROVAL_CONFIRMED: {S_AWAITING_APPROVAL: S_AWAITING_APPROVAL},
    E_EXECUTE: {S_AWAITING_APPROVAL: S_EXECUTING},
    E_EVALUATE: {S_EXECUTING: S_EVALUATING},
    E_EXPERIENCE: {S_EVALUATING: S_EXPERIENCE_RECORDED},
    E_COMPLETE_ROUND: {S_EXPERIENCE_RECORDED: S_ROUND_COMPLETED},
    E_COMPLETE_LOOP: {S_ROUND_COMPLETED: S_LOOP_COMPLETED},
    E_RECOVER: {s: S_RECOVERING for s in ACTIVE | RESUMABLE},
    E_RESOLVE: {S_RECOVERING: S_ROUND_COMPLETED},
}
_WILDCARD = frozenset({E_FAIL, E_INTERRUPT, E_STOP})


class IllegalTransition(RuntimeError):
    def __init__(self, event: str, state: str, to_state: str = ""):
        super().__init__(
            f"illegal transition event={event!r} from state={state!r}"
            + (f" (wanted {to_state!r})" if to_state else ""))
        self.event, self.state, self.to_state = event, state, to_state


def can_transition(state: str, event: str) -> bool:
    if event in _WILDCARD:
        return state in ACTIVE or state in RESUMABLE
    return state in _TRANSITIONS.get(event, {})


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS m5_loop_state (
    loop_id     TEXT PRIMARY KEY,
    state       TEXT NOT NULL,
    round_no    INTEGER NOT NULL DEFAULT 0,
    record_json TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS m5_loop_events (
    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
    loop_id      TEXT NOT NULL,
    round_no     INTEGER NOT NULL DEFAULT 0,
    event        TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_sha  TEXT NOT NULL,
    prev_sha     TEXT NOT NULL,
    chain_sha    TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
"""


@dataclass
class LoopRecord:
    loop_id: str
    domain: str = "scinet"
    state: str = S_CREATED
    round_no: int = 0
    max_rounds: int = 2
    config: dict[str, Any] = field(default_factory=dict)
    config_sha: str = ""
    rounds: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"loop_id": self.loop_id, "domain": self.domain,
                "state": self.state, "round_no": self.round_no,
                "max_rounds": self.max_rounds, "config": dict(self.config),
                "config_sha": self.config_sha,
                "rounds": list(self.rounds)}


class LoopEventStore:
    """Append-only, hash-chained event log for one autonomous loop."""

    def __init__(self, *, db: Any):
        self.db = db
        with self.db.session() as conn:
            for stmt in SCHEMA_SQL.split(";"):
                if stmt.strip():
                    conn.execute(stmt)

    def save(self, record: LoopRecord) -> None:
        with self.db.session() as conn:
            conn.execute(
                """INSERT INTO m5_loop_state
                   (loop_id, state, round_no, record_json, updated_at)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(loop_id) DO UPDATE SET
                     state=excluded.state, round_no=excluded.round_no,
                     record_json=excluded.record_json,
                     updated_at=excluded.updated_at""",
                (record.loop_id, record.state, record.round_no,
                 _dump(record.to_dict()), _now()))

    def load(self, loop_id: str) -> LoopRecord | None:
        row = self.db.rows(
            "SELECT record_json FROM m5_loop_state WHERE loop_id=?",
            (loop_id,))
        if not row:
            return None
        d = _load(row[0]["record_json"], {})
        return LoopRecord(
            loop_id=d.get("loop_id", loop_id), domain=d.get("domain", "scinet"),
            state=d.get("state", S_CREATED), round_no=int(d.get("round_no", 0)),
            max_rounds=int(d.get("max_rounds", 2)),
            config=d.get("config") or {},
            config_sha=str(d.get("config_sha") or ""),
            rounds=d.get("rounds") or [])

    def append(self, *, loop_id: str, round_no: int, event: str,
               payload: dict[str, Any]) -> str:
        payload_json = _dump(payload)
        payload_sha = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        with self.db.session() as conn:
            last = conn.execute(
                """SELECT chain_sha FROM m5_loop_events WHERE loop_id=?
                   ORDER BY seq DESC LIMIT 1""", (loop_id,)).fetchone()
            prev = last["chain_sha"] if last else "0" * 64
            chain = hashlib.sha256(
                (prev + payload_sha + event + str(round_no)).encode("utf-8")
            ).hexdigest()
            conn.execute(
                """INSERT INTO m5_loop_events
                   (loop_id, round_no, event, payload_json, payload_sha,
                    prev_sha, chain_sha, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (loop_id, round_no, event, payload_json, payload_sha,
                 prev, chain, _now()))
        return chain

    def history(self, loop_id: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.rows(
            "SELECT * FROM m5_loop_events WHERE loop_id=? ORDER BY seq",
            (loop_id,))]

    def verify_integrity(self, loop_id: str) -> bool:
        prev = "0" * 64
        for row in self.history(loop_id):
            if row["prev_sha"] != prev:
                return False
            # Recompute the payload hash from the ACTUAL stored JSON: trusting
            # the stored payload_sha alone would let a tampered payload pass.
            actual_payload_sha = hashlib.sha256(
                str(row["payload_json"]).encode("utf-8")).hexdigest()
            if actual_payload_sha != row["payload_sha"]:
                return False
            expected = hashlib.sha256(
                (row["prev_sha"] + actual_payload_sha + row["event"]
                 + str(row["round_no"])).encode("utf-8")).hexdigest()
            if expected != row["chain_sha"]:
                return False
            prev = row["chain_sha"]
        return True


__all__ = [
    "S_CREATED", "S_PREFLIGHTED", "S_DISCOVERING", "S_OBJECTIVE_READY",
    "S_HYPOTHESIS_READY", "S_PLAN_READY", "S_AWAITING_APPROVAL", "S_EXECUTING",
    "S_EVALUATING", "S_EXPERIENCE_RECORDED", "S_ROUND_COMPLETED",
    "S_LOOP_COMPLETED", "S_FAILED", "S_INTERRUPTED", "S_STOPPED",
    "S_RECOVERING", "STATES", "TERMINAL", "RESUMABLE", "ACTIVE",
    "IllegalTransition", "can_transition", "LoopRecord", "LoopEventStore",
    "SCHEMA_SQL",
]
