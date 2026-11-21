"""M5 Integrated -- durable GLOBAL runtime state, checkpoint and journal.

One SQLite database coordinates the whole run: global + per-domain budget,
round/domain/rrsi state, config+manifest SHA binding, an APPEND-ONLY event
journal (hash-chained), and recovery classification. It does NOT reimplement
the per-loop ledger (LoopBudgetLedger) or the per-domain budget
(DomainBudgetManager) -- it mirrors/reserves through them and records the
global accounting so a process restart cannot regain spent budget.

Semantics: at-most-once campaign execution + resume-in-place. No exactly-once
claim. Fail-closed on SHA drift or unverifiable state.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from .research_store.db import ResearchDatabase, ResearchStoreConflict

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS rt_run (
    run_id TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    config_sha TEXT NOT NULL,
    dataset_shas TEXT NOT NULL,
    max_rounds INTEGER NOT NULL,
    global_budget INTEGER NOT NULL,
    campaigns_consumed INTEGER NOT NULL DEFAULT 0,
    started_epoch REAL NOT NULL,
    executor_id TEXT NOT NULL DEFAULT '',
    terminal_reason TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rt_round (
    run_id TEXT NOT NULL,
    round_no INTEGER NOT NULL,
    domain_id TEXT NOT NULL,
    state TEXT NOT NULL,
    objective_id TEXT NOT NULL DEFAULT '',
    campaign_id TEXT NOT NULL DEFAULT '',
    report_id TEXT NOT NULL DEFAULT '',
    experience_id TEXT NOT NULL DEFAULT '',
    payload_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, round_no, domain_id)
);
CREATE TABLE IF NOT EXISTS rt_budget (
    run_id TEXT NOT NULL,
    domain_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    state TEXT NOT NULL,
    config_sha TEXT NOT NULL,
    outcome TEXT NOT NULL DEFAULT '',
    requested_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, domain_id, campaign_id)
);
-- P0-A: a reservation exists BEFORE any campaign id is known. Its identity is
-- deterministic (run_id|round|domain|objective), independent of the engine.
-- state: reserved -> bound (campaign id known) -> consumed | released.
CREATE TABLE IF NOT EXISTS rt_reservation (
    run_id TEXT NOT NULL,
    reservation_id TEXT NOT NULL,
    domain_id TEXT NOT NULL DEFAULT '',
    round_no INTEGER NOT NULL DEFAULT 0,
    campaign_id TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL,
    outcome TEXT NOT NULL DEFAULT '',
    config_sha TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, reservation_id)
);
CREATE TABLE IF NOT EXISTS rt_journal (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    round_no INTEGER NOT NULL DEFAULT 0,
    event TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_sha TEXT NOT NULL,
    prev_sha TEXT NOT NULL,
    chain_sha TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rt_checkpoint (
    run_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    state_json TEXT NOT NULL,
    state_sha TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, checkpoint_id)
);
-- ARR v1.0 WP-01 durable per-campaign execution claim and state machine.
-- States CREATED CLAIMED RUNNING COMPLETED FAILED RECOVERY_REQUIRED UNKNOWN.
-- Ownership via executor_id makes the claim AT-MOST-ONCE not exactly-once.
CREATE TABLE IF NOT EXISTS rt_campaign_exec (
    campaign_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    state TEXT NOT NULL,
    executor_id TEXT NOT NULL DEFAULT '',
    config_sha TEXT NOT NULL DEFAULT '',
    harness_sha TEXT NOT NULL DEFAULT '',
    result_json TEXT NOT NULL DEFAULT '',
    result_sha TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

S_INIT = "INIT"
S_RUNNING = "RUNNING"
S_ROUND2 = "ROUND2_CROSS_DOMAIN"
S_RRSI = "RRSI_EVOLVING"
S_COMPLETED = "COMPLETED"
S_STOPPED = "STOPPED"
S_EXHAUSTED = "BUDGET_EXHAUSTED"
S_FAILED = "FAILED"
S_RECOVERING = "RECOVERING"

# recovery classification (ordered by severity)
C_COMPLETE = "ROUND_COMPLETED"
C_AWAITING = "AWAITING_APPROVAL"
C_INFLIGHT = "CAMPAIGN_IN_FLIGHT"
C_EXPERIENCE = "EXPERIENCE_PENDING"
C_RECOVERY = "RECOVERY_REQUIRED"
C_CORRUPT = "CORRUPT"


def _now() -> str:
    import datetime
    return datetime.datetime.now().isoformat(timespec="seconds")


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, ensure_ascii=False, default=str
    ).encode("utf-8")).hexdigest()


def config_sha(config: Any) -> str:
    d = config.__dict__ if hasattr(config, "__dict__") else dict(config)
    return _sha(d)


class RuntimeStateError(RuntimeError):
    pass


class RuntimeState:
    def __init__(self, *, db: ResearchDatabase):
        self.db = db
        with self.db.session() as conn:
            for stmt in SCHEMA_SQL.split(";"):
                if stmt.strip():
                    conn.execute(stmt)
            # Idempotent forward migration for DBs created before P0-B added
            # executor/terminal bookkeeping. ALTER ... ADD COLUMN once.
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(rt_run)")}
            if "executor_id" not in cols:
                conn.execute(
                    "ALTER TABLE rt_run ADD COLUMN executor_id TEXT NOT NULL DEFAULT ''")
            if "terminal_reason" not in cols:
                conn.execute(
                    "ALTER TABLE rt_run ADD COLUMN terminal_reason TEXT NOT NULL DEFAULT ''")

    # ------------------------------------------- P0-B: executor / terminal guard
    TERMINAL_STATES = frozenset({"COMPLETED", "STOPPED", "BUDGET_EXHAUSTED",
                                 "FAILED"})

    def claim_executor(self, run_id: str, executor_id: str) -> bool:
        """Atomically claim the sole execution right for a run.

        Two processes racing with different executor ids: exactly one wins.
        A terminal run can NEVER be claimed. The same executor re-claiming is
        idempotent (resume after its own restart).
        """
        if not executor_id:
            raise RuntimeStateError("executor_id required")
        with self.db.session() as conn:
            row = conn.execute(
                "SELECT state, executor_id FROM rt_run WHERE run_id=?",
                (run_id,)).fetchone()
            if row is None:
                raise RuntimeStateError(f"unknown run {run_id!r}")
            if row["state"] in self.TERMINAL_STATES:
                return False
            if row["executor_id"] and row["executor_id"] != executor_id:
                return False
            conn.execute(
                "UPDATE rt_run SET executor_id=?,updated_at=? WHERE run_id=?",
                (executor_id, _now(), run_id))
        return True

    def release_executor(self, run_id: str, executor_id: str) -> bool:
        with self.db.session() as conn:
            row = conn.execute(
                "SELECT executor_id FROM rt_run WHERE run_id=?",
                (run_id,)).fetchone()
            if row is None or row["executor_id"] != executor_id:
                return False
            conn.execute(
                "UPDATE rt_run SET executor_id='',updated_at=? WHERE run_id=?",
                (_now(), run_id))
        return True

    def mark_terminal(self, run_id: str, state: str, reason: str) -> None:
        if state not in self.TERMINAL_STATES:
            raise RuntimeStateError(f"{state!r} is not a terminal state")
        with self.db.session() as conn:
            conn.execute(
                "UPDATE rt_run SET state=?,terminal_reason=?,executor_id='',"
                "updated_at=? WHERE run_id=?", (state, reason, _now(), run_id))

    # ---------------------------------------------------------------- run init
    def initialize(self, run_id: str, *, config: Any,
                   dataset_shas: dict[str, str], max_rounds: int,
                   global_budget: int) -> dict[str, Any]:
        csha = config_sha(config)
        now = _now()
        with self.db.session() as conn:
            row = conn.execute(
                "SELECT * FROM rt_run WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                conn.execute(
                    """INSERT INTO rt_run
                       (run_id,state,config_sha,dataset_shas,max_rounds,
                        global_budget,campaigns_consumed,started_epoch,updated_at)
                       VALUES (?,?,?,?,?,?,0,?,?)""",
                    (run_id, S_INIT, csha,
                     json.dumps(dataset_shas, sort_keys=True),
                     int(max_rounds), int(global_budget), time.time(), now))
            else:
                if row["config_sha"] != csha:
                    raise RuntimeStateError(
                        f"run {run_id!r} config SHA drift: stored="
                        f"{row['config_sha']}, computed={csha}")
                if row["dataset_shas"] != json.dumps(dataset_shas, sort_keys=True):
                    raise RuntimeStateError(
                        f"run {run_id!r} dataset manifest SHA drift; refusing")
        self.append(run_id, 0, "run_initialized",
                    {"config_sha": csha, "dataset_shas": dataset_shas,
                     "max_rounds": max_rounds, "global_budget": global_budget})
        return self.run(run_id)

    def run(self, run_id: str) -> dict[str, Any]:
        with self.db.session() as conn:
            row = conn.execute(
                "SELECT * FROM rt_run WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row else {}

    def set_state(self, run_id: str, state: str, *, round_no: int = 0,
                  event: str = "", payload: dict[str, Any] | None = None) -> dict[str, Any]:
        with self.db.session() as conn:
            row = conn.execute(
                "SELECT state FROM rt_run WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise RuntimeStateError(f"unknown run {run_id!r}")
            conn.execute("UPDATE rt_run SET state=?,updated_at=? WHERE run_id=?",
                         (state, _now(), run_id))
        if event:
            self.append(run_id, round_no, event, payload or {"to_state": state})
        return self.run(run_id)

    # ------------------------------------------------------------- append journal
    def append(self, run_id: str, round_no: int, event: str,
               payload: dict[str, Any]) -> dict[str, Any]:
        """Hash-chained append-only event. Never update/delete a prior row."""
        now = _now()
        with self.db.session() as conn:
            last = conn.execute(
                "SELECT chain_sha FROM rt_journal WHERE run_id=? "
                "ORDER BY seq DESC LIMIT 1", (run_id,)).fetchone()
            prev = last["chain_sha"] if last else "0" * 64
            psha = _sha(payload)
            chain = hashlib.sha256(
                (prev + psha + event + str(round_no)).encode("utf-8")
            ).hexdigest()
            cur = conn.execute(
                """INSERT INTO rt_journal
                   (run_id,round_no,event,payload_json,payload_sha,prev_sha,
                    chain_sha,created_at) VALUES (?,?,?,?,?,?,?,?)""",
                (run_id, int(round_no), event,
                 json.dumps(payload, sort_keys=True, ensure_ascii=False,
                            default=str), psha, prev, chain, now))
            return {"seq": cur.lastrowid, "chain_sha": chain}

    def journal(self, run_id: str) -> list[dict[str, Any]]:
        with self.db.session() as conn:
            rows = conn.execute(
                "SELECT * FROM rt_journal WHERE run_id=? ORDER BY seq",
                (run_id,)).fetchall()
        return [dict(r) for r in rows]

    def verify_journal(self, run_id: str) -> dict[str, Any]:
        rows = self.journal(run_id)
        prev = "0" * 64
        for i, r in enumerate(rows):
            psha = _sha(json.loads(r["payload_json"]))
            chain = hashlib.sha256(
                (prev + psha + r["event"] + str(r["round_no"])).encode("utf-8")
            ).hexdigest()
            if r["prev_sha"] != prev or r["chain_sha"] != chain or r["payload_sha"] != psha:
                return {"ok": False, "broken_at": i, "seq": r["seq"]}
            prev = r["chain_sha"]
        return {"ok": True, "n_events": len(rows), "head_sha": prev}

    # -------------------------------------------------------------- global budget
    def reserve_global(self, run_id: str, campaign_id: str, *,
                       config_sha: str) -> dict[str, Any]:
        now = _now()
        with self.db.session() as conn:
            run = conn.execute(
                "SELECT * FROM rt_run WHERE run_id=?", (run_id,)).fetchone()
            if run is None:
                raise RuntimeStateError(f"unknown run {run_id!r}")
            if run["config_sha"] != config_sha:
                raise RuntimeStateError(f"run {run_id!r} config SHA drift on reserve")
            existing = conn.execute(
                "SELECT * FROM rt_budget WHERE run_id=? AND campaign_id=?",
                (run_id, campaign_id)).fetchone()
            if existing is not None:
                return dict(existing)  # idempotent: never charge twice
            limit = int(run["global_budget"])
            used = (int(run["campaigns_consumed"])
                    + conn.execute(
                        "SELECT COUNT(*) c FROM rt_budget "
                        "WHERE run_id=? AND state='reserved'",
                        (run_id,)).fetchone()["c"])
            if limit > 0 and used >= limit:
                raise RuntimeBudgetExhausted(
                    f"run {run_id!r} global budget exhausted: "
                    f"limit={limit}, used={used}")
            conn.execute(
                """INSERT INTO rt_budget
                   (run_id,domain_id,campaign_id,state,config_sha,outcome,
                    requested_at,updated_at) VALUES (?,?,?,?,?,?,?,?)""",
                (run_id, "", campaign_id, "reserved", config_sha, "", now, now))
            return {"campaign_id": campaign_id, "state": "reserved"}

    def consume_global(self, run_id: str, campaign_id: str, *,
                       outcome: str) -> dict[str, Any]:
        now = _now()
        with self.db.session() as conn:
            row = conn.execute(
                "SELECT * FROM rt_budget WHERE run_id=? AND campaign_id=?",
                (run_id, campaign_id)).fetchone()
            if row is None:
                raise RuntimeStateError(
                    f"campaign {campaign_id!r} has no global reservation")
            if row["state"] == "consumed":
                return dict(row)  # idempotent
            conn.execute(
                "UPDATE rt_budget SET state='consumed',outcome=?,updated_at=? "
                "WHERE run_id=? AND campaign_id=?",
                (outcome, now, run_id, campaign_id))
            conn.execute(
                "UPDATE rt_run SET campaigns_consumed=campaigns_consumed+1,"
                "updated_at=? WHERE run_id=?", (now, run_id))
        # NOTE: leave the write transaction before opening a new read session;
        # nested BEGIN IMMEDIATE on the same connection deadlocks.
        return self.run(run_id)

    def global_budget_snapshot(self, run_id: str) -> dict[str, Any]:
        run = self.run(run_id)
        with self.db.session() as conn:
            reserved = conn.execute(
                "SELECT COUNT(*) c FROM rt_budget WHERE run_id=? AND state='reserved'",
                (run_id,)).fetchone()["c"]
            # P0-A: an active execution reservation also spends budget, even
            # before its campaign_id exists. Never let a restart regain it.
            active_res = conn.execute(
                "SELECT COUNT(*) c FROM rt_reservation WHERE run_id=? "
                "AND state IN ('reserved','bound')", (run_id,)).fetchone()["c"]
        limit = int(run.get("global_budget", 0))
        consumed = int(run.get("campaigns_consumed", 0))
        reserved_total = reserved + active_res
        return {"limit": limit, "reserved": reserved_total, "consumed": consumed,
                "active_reservations": active_res,
                "remaining": (limit - reserved_total - consumed) if limit > 0 else -1}

    # ------------------------------------------- P0-A: pre-execution reservations
    @staticmethod
    def reservation_id(run_id: str, round_no: int, domain_id: str,
                       objective_id: str) -> str:
        """Deterministic identity available BEFORE the campaign exists."""
        return hashlib.sha256(
            f"{run_id}|{round_no}|{domain_id}|{objective_id}".encode("utf-8")
        ).hexdigest()

    def reserve_execution(self, run_id: str, reservation_id: str, *,
                          config_sha: str, domain_id: str = "",
                          round_no: int = 0) -> dict[str, Any]:
        """Atomically reserve ONE campaign slot BEFORE execution starts.

        A second domain cannot start when only one slot remains: the budget
        check and insert happen in the SAME BEGIN IMMEDIATE transaction.
        Idempotent by reservation_id; config SHA drift is rejected.
        """
        now = _now()
        with self.db.session() as conn:
            run = conn.execute(
                "SELECT * FROM rt_run WHERE run_id=?", (run_id,)).fetchone()
            if run is None:
                raise RuntimeStateError(f"unknown run {run_id!r}")
            if run["config_sha"] != config_sha:
                raise RuntimeStateError(
                    f"run {run_id!r} config SHA drift on reserve_execution")
            existing = conn.execute(
                "SELECT * FROM rt_reservation WHERE run_id=? AND reservation_id=?",
                (run_id, reservation_id)).fetchone()
            if existing is not None:
                # Resume-in-place: an identical reservation is not a new spend.
                return dict(existing)
            limit = int(run["global_budget"])
            consumed = int(run["campaigns_consumed"])
            active = conn.execute(
                "SELECT COUNT(*) c FROM rt_reservation WHERE run_id=? "
                "AND state IN ('reserved','bound')", (run_id,)).fetchone()["c"]
            legacy = conn.execute(
                "SELECT COUNT(*) c FROM rt_budget WHERE run_id=? AND state='reserved'",
                (run_id,)).fetchone()["c"]
            if limit > 0 and (consumed + active + legacy) >= limit:
                raise RuntimeBudgetExhausted(
                    f"run {run_id!r} global budget exhausted before execution: "
                    f"limit={limit}, consumed={consumed}, active={active}")
            conn.execute(
                """INSERT INTO rt_reservation
                   (run_id,reservation_id,domain_id,round_no,campaign_id,state,
                    outcome,config_sha,requested_at,updated_at)
                   VALUES (?,?,?,?,?,'reserved','',?,?,?)""",
                (run_id, reservation_id, domain_id, int(round_no), "",
                 config_sha, now, now))
        self.append(run_id, round_no, "budget_reserved",
                    {"reservation_id": reservation_id, "domain_id": domain_id,
                     "config_sha": config_sha})
        return {"reservation_id": reservation_id, "state": "reserved"}

    def bind_reservation(self, run_id: str, reservation_id: str, *,
                         campaign_id: str) -> dict[str, Any]:
        """Bind a real campaign id to an existing reservation (once)."""
        if not campaign_id:
            raise RuntimeStateError("cannot bind an empty campaign_id")
        with self.db.session() as conn:
            row = conn.execute(
                "SELECT * FROM rt_reservation WHERE run_id=? AND reservation_id=?",
                (run_id, reservation_id)).fetchone()
            if row is None:
                raise RuntimeStateError(
                    f"reservation {reservation_id!r} not found for run {run_id!r}")
            if row["state"] == "released":
                raise RuntimeStateError(
                    f"reservation {reservation_id!r} was released; cannot bind")
            if row["campaign_id"] and row["campaign_id"] != campaign_id:
                raise RuntimeStateError(
                    f"reservation {reservation_id!r} already bound to "
                    f"{row['campaign_id']!r}, not {campaign_id!r}")
            if row["state"] == "reserved":
                conn.execute(
                    "UPDATE rt_reservation SET state='bound',campaign_id=?,"
                    "updated_at=? WHERE run_id=? AND reservation_id=?",
                    (campaign_id, _now(), run_id, reservation_id))
        return {"reservation_id": reservation_id, "campaign_id": campaign_id,
                "state": "bound"}

    def consume_reservation(self, run_id: str, reservation_id: str, *,
                            outcome: str) -> dict[str, Any]:
        """Settle a reservation on success/failure/interruption (idempotent)."""
        now = _now()
        with self.db.session() as conn:
            row = conn.execute(
                "SELECT * FROM rt_reservation WHERE run_id=? AND reservation_id=?",
                (run_id, reservation_id)).fetchone()
            if row is None:
                raise RuntimeStateError(
                    f"reservation {reservation_id!r} not found for run {run_id!r}")
            if row["state"] == "consumed":
                return dict(row)  # idempotent
            conn.execute(
                "UPDATE rt_reservation SET state='consumed',outcome=?,updated_at=? "
                "WHERE run_id=? AND reservation_id=?",
                (outcome, now, run_id, reservation_id))
            conn.execute(
                "UPDATE rt_run SET campaigns_consumed=campaigns_consumed+1,"
                "updated_at=? WHERE run_id=?", (now, run_id))
        return self.run(run_id)

    def release_reservation(self, run_id: str, reservation_id: str, *,
                            reason: str) -> dict[str, Any]:
        """Release a reservation only when NO campaign was ever started."""
        with self.db.session() as conn:
            row = conn.execute(
                "SELECT * FROM rt_reservation WHERE run_id=? AND reservation_id=?",
                (run_id, reservation_id)).fetchone()
            if row is None:
                raise RuntimeStateError(f"reservation {reservation_id!r} not found")
            if row["state"] in ("bound", "consumed") or row["campaign_id"]:
                raise RuntimeStateError(
                    f"reservation {reservation_id!r} reached campaign; "
                    "must be consumed, not released")
            conn.execute(
                "UPDATE rt_reservation SET state='released',outcome=?,updated_at=? "
                "WHERE run_id=? AND reservation_id=?",
                (f"released:{reason}", _now(), run_id, reservation_id))
        return {"reservation_id": reservation_id, "state": "released"}

    def reservations(self, run_id: str) -> list[dict[str, Any]]:
        with self.db.session() as conn:
            rows = conn.execute(
                "SELECT * FROM rt_reservation WHERE run_id=? "
                "ORDER BY round_no, domain_id", (run_id,)).fetchall()
        return [dict(r) for r in rows]

    def orphan_reservations(self, run_id: str) -> list[dict[str, Any]]:
        """Reservations still active after a crash: budget must not be regained."""
        return [r for r in self.reservations(run_id)
                if r["state"] in ("reserved", "bound")]

    # ------------------------------------------------------------------- rounds
    # P0-C: only the FINAL state means a research round is truly done. Earlier
    # states describe in-flight stages and must never be read as completion.
    S_CAMPAIGN_DONE = "CAMPAIGN_COMPLETED"
    S_REPORT_PENDING = "REPORT_PENDING"
    S_EXPERIENCE_PENDING = "EXPERIENCE_PENDING"
    S_ROUND_DONE = "ROUND_COMPLETED"

    def put_round(self, run_id: str, round_no: int, domain_id: str,
                  state: str, **fields: Any) -> dict[str, Any]:
        """Upsert round state (idempotent for identical payload).

        P0-C: a ROUND_COMPLETED write is REJECTED unless a real campaign id,
        report id AND experience id are all present. "No evidence, no done."
        """
        if state == self.S_ROUND_DONE:
            missing = [k for k in ("campaign_id", "report_id", "experience_id")
                       if not str(fields.get(k, "")).strip()]
            if missing:
                raise ResearchStoreConflict(
                    f"round {run_id}/{round_no}/{domain_id} cannot be "
                    f"ROUND_COMPLETED without {missing}")
        payload = {k: v for k, v in fields.items() if k != "payload"}
        blob = json.dumps(fields.get("payload") or {}, sort_keys=True,
                          ensure_ascii=False, default=str)
        now = _now()
        with self.db.session() as conn:
            old = conn.execute(
                "SELECT * FROM rt_round WHERE run_id=? AND round_no=? "
                "AND domain_id=?", (run_id, round_no, domain_id)).fetchone()
            if old is not None:
                # Evidence ids are IMMUTABLE: check conflicts BEFORE any
                # idempotency short-circuit, otherwise a changed report_id with
                # an otherwise identical payload would be silently ignored.
                for col in ("objective_id", "campaign_id", "report_id",
                            "experience_id"):
                    if col not in payload:
                        continue
                    new_val = str(payload[col])
                    old_val = str(old[col])
                    if old_val and new_val and old_val != new_val:
                        raise ResearchStoreConflict(
                            f"round {run_id}/{round_no}/{domain_id} {col} changed "
                            f"from {old_val!r} to {new_val!r}")
                    # empty -> real id is a legitimate stage upgrade; never
                    # allow real id -> empty (that would erase evidence).
                    if old_val and not new_val:
                        raise ResearchStoreConflict(
                            f"round {run_id}/{round_no}/{domain_id} cannot "
                            f"erase {col}={old_val!r}")
                if (old["state"] == state and old["objective_id"] ==
                        str(payload.get("objective_id", old["objective_id"]))
                        and old["payload_json"] == blob):
                    return dict(old)
            conn.execute(
                """INSERT INTO rt_round
                   (run_id,round_no,domain_id,state,objective_id,campaign_id,
                    report_id,experience_id,payload_json,updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(run_id,round_no,domain_id) DO UPDATE SET
                     state=excluded.state, payload_json=excluded.payload_json,
                     campaign_id=CASE WHEN excluded.campaign_id<>''
                       THEN excluded.campaign_id ELSE rt_round.campaign_id END,
                     report_id=CASE WHEN excluded.report_id<>''
                       THEN excluded.report_id ELSE rt_round.report_id END,
                     experience_id=CASE WHEN excluded.experience_id<>''
                       THEN excluded.experience_id ELSE rt_round.experience_id END,
                     updated_at=excluded.updated_at""",
                (run_id, round_no, domain_id, state,
                 str(payload.get("objective_id", "")),
                 str(payload.get("campaign_id", "")),
                 str(payload.get("report_id", "")),
                 str(payload.get("experience_id", "")), blob, now))
        return {"run_id": run_id, "round_no": round_no,
                "domain_id": domain_id, "state": state, **payload}

    def rounds(self, run_id: str) -> list[dict[str, Any]]:
        with self.db.session() as conn:
            rows = conn.execute(
                "SELECT * FROM rt_round WHERE run_id=? "
                "ORDER BY round_no, domain_id", (run_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d.pop("payload_json") or "{}")
            out.append(d)
        return out

    # --------------------------------------------------------------- checkpoint
    def checkpoint(self, run_id: str, checkpoint_id: str) -> dict[str, Any]:
        """Durable snapshot of current global state; immutable per id."""
        state = {"run": self.run(run_id), "rounds": self.rounds(run_id),
                 "budget": self.global_budget_snapshot(run_id),
                 "journal": self.verify_journal(run_id)}
        sha = _sha(state)
        blob = json.dumps(state, sort_keys=True, ensure_ascii=False, default=str)
        now = _now()
        with self.db.session() as conn:
            old = conn.execute(
                "SELECT * FROM rt_checkpoint WHERE run_id=? AND checkpoint_id=?",
                (run_id, checkpoint_id)).fetchone()
            if old is not None:
                if old["state_sha"] != sha:
                    raise ResearchStoreConflict(
                        f"checkpoint {checkpoint_id!r} already exists with "
                        "different content; refusing overwrite")
                return dict(old)
            conn.execute(
                """INSERT INTO rt_checkpoint
                   (run_id,checkpoint_id,state_json,state_sha,created_at)
                   VALUES (?,?,?,?,?)""",
                (run_id, checkpoint_id, blob, sha, now))
        # The rt_checkpoint table is itself the durable, append-only record.
        # Do NOT also append a journal event here: that would mutate the very
        # journal whose hash is part of `state`, breaking idempotent reuse.
        return {"checkpoint_id": checkpoint_id, "state_sha": sha}

    def checkpoint_load(self, run_id: str, checkpoint_id: str) -> dict[str, Any]:
        with self.db.session() as conn:
            row = conn.execute(
                "SELECT * FROM rt_checkpoint WHERE run_id=? AND checkpoint_id=?",
                (run_id, checkpoint_id)).fetchone()
        if row is None:
            return {}
        d = dict(row)
        d["state"] = json.loads(d["state_json"])
        return d

    # ----------------------------------------------------------------- recovery
    def classify(self, run_id: str, *,
                 reality_verifier: Any = None) -> dict[str, Any]:
        """Read-only recovery classification from REAL persisted state.

        P0-C: a round's own ``state`` field is NEVER sufficient. A round is
        ROUND_COMPLETED only when it has a real campaign_id + report_id +
        experience_id AND (when supplied) ``reality_verifier`` resolves all
        three against the REAL stores with matching SHA/binding. Any unresolved
        id, orphan reservation, or unverifiable evidence -> fail-closed.
        """
        integ = self.verify_journal(run_id)
        run = self.run(run_id)
        if not run:
            return {"classification": C_RECOVERY, "reason": "run_not_found"}
        if not integ["ok"]:
            return {"classification": C_CORRUPT, "journal": integ}
        rounds = self.rounds(run_id)
        budget = self.global_budget_snapshot(run_id)
        reasons: list[str] = []

        # An unsettled reservation means a campaign may be in flight after a
        # crash. Budget must not be regained and the round cannot be complete.
        orphans = self.orphan_reservations(run_id)
        if orphans:
            reasons.append(
                "orphan_reservations:" + ",".join(
                    o["reservation_id"][:12] for o in orphans))

        for r in rounds:
            state = r["state"]
            if state == self.S_ROUND_DONE:
                missing = [k for k in ("campaign_id", "report_id",
                                       "experience_id")
                           if not str(r.get(k, "")).strip()]
                if missing:
                    reasons.append(
                        f"round{r['round_no']}/{r['domain_id']}:missing:{missing}")
                elif reality_verifier is not None:
                    try:
                        v = reality_verifier(r)
                        if not v.get("ok"):
                            reasons.append(
                                f"round{r['round_no']}/{r['domain_id']}:"
                                f"unresolved:" + ",".join(
                                    v.get("reasons", ["reality_check_failed"])))
                    except Exception as exc:  # noqa: BLE001 - fail closed
                        reasons.append(
                            f"round{r['round_no']}/{r['domain_id']}:"
                            f"verifier_error:{type(exc).__name__}: {exc}")
            elif state in (self.S_CAMPAIGN_DONE, self.S_REPORT_PENDING):
                reasons.append(
                    f"round{r['round_no']}/{r['domain_id']}:{state}")
            elif state == self.S_EXPERIENCE_PENDING:
                reasons.append(
                    f"round{r['round_no']}/{r['domain_id']}:EXPERIENCE_PENDING")
            elif state in ("RESERVED", "DISPATCHED"):
                reasons.append(f"round{r['round_no']}/{r['domain_id']}:in_flight")
            elif state == "FAILED":
                reasons.append(f"round{r['round_no']}/{r['domain_id']}:failed")
            elif state == "BUDGET_BLOCKED":
                reasons.append(
                    f"round{r['round_no']}/{r['domain_id']}:budget_blocked")
            elif state == "COMPLETED" or state not in (
                    self.S_ROUND_DONE, self.S_CAMPAIGN_DONE,
                    self.S_REPORT_PENDING, self.S_EXPERIENCE_PENDING,
                    "RESERVED", "DISPATCHED", "FAILED", "BUDGET_BLOCKED"):
                # Legacy COMPLETED marker (or unknown state): treat with the
                # SAME strict evidence rules rather than trusting the flag --
                # this is the exact source-review defect. Missing ids or an
                # unresolved real-store check make the round NOT complete.
                missing = [k for k in ("campaign_id", "report_id",
                                       "experience_id")
                           if not str(r.get(k, "")).strip()]
                if missing:
                    reasons.append(
                        f"round{r['round_no']}/{r['domain_id']}:"
                        f"legacy_completed_missing:{missing}")
                elif reality_verifier is not None and not reality_verifier(r).get("ok"):
                    reasons.append(
                        f"round{r['round_no']}/{r['domain_id']}:legacy_unresolved")

        if reasons:
            # Most-specific classification from the first blocking reason.
            first = reasons[0]
            if "EXPERIENCE_PENDING" in first:
                cls = C_EXPERIENCE
            elif "in_flight" in first or "orphan_reservations" in first:
                cls = C_INFLIGHT
            else:
                cls = C_RECOVERY
            return {"classification": cls, "reasons": reasons,
                    "journal": integ, "budget": budget,
                    "orphan_reservations": [o["reservation_id"] for o in orphans]}
        return {"classification": C_COMPLETE, "journal": integ,
                "budget": budget, "n_rounds": len(rounds)}

    # ----------------------------------------- ARR v1.0 WP-01: campaign claims
    CAMPAIGN_STATES = frozenset({
        "CREATED", "CLAIMED", "RUNNING", "COMPLETED", "FAILED",
        "RECOVERY_REQUIRED", "UNKNOWN"})
    CAMPAIGN_TERMINAL = frozenset({"COMPLETED", "FAILED"})

    def claim_campaign(self, *, campaign_id: str, run_id: str,
                       executor_id: str, config_sha: str,
                       harness_sha: str) -> dict[str, Any]:
        """Atomically claim ONE campaign for execution (at-most-once).

        Semantics:
          * first call      -> state=CLAIMED, ok=True;
          * same executor re-claim while CLAIMED/RUNNING -> idempotent ok=True;
          * COMPLETED/FAILED -> ok=False + cached result (NEVER re-execute);
          * different executor while CLAIMED/RUNNING -> ok=False (concurrent);
          * RECOVERY_REQUIRED/UNKNOWN -> ok=False (unsafe to auto-resume);
        """
        if not campaign_id or not executor_id:
            raise RuntimeStateError("campaign_id and executor_id required")
        if not harness_sha:
            raise RuntimeStateError("harness_sha required for campaign claim")
        with self.db.session() as conn:  # BEGIN IMMEDIATE -> atomic claim
            row = conn.execute(
                "SELECT * FROM rt_campaign_exec WHERE campaign_id=?",
                (campaign_id,)).fetchone()
            if row is None:
                conn.execute(
                    """INSERT INTO rt_campaign_exec
                       (campaign_id,run_id,state,executor_id,config_sha,
                        harness_sha,result_json,result_sha,reason,
                        created_at,updated_at)
                       VALUES (?,?, 'CLAIMED', ?,?,?, '', '', '', ?, ?)""",
                    (campaign_id, run_id, executor_id, config_sha,
                     harness_sha, _now(), _now()))
                return {"ok": True, "state": "CLAIMED", "first_claim": True}
            state = row["state"]
            if state in self.CAMPAIGN_TERMINAL:
                return {"ok": False, "state": state, "already_terminal": True,
                        "result": _load_json(row["result_json"], {}),
                        "result_sha": row["result_sha"]}
            if state in ("RECOVERY_REQUIRED", "UNKNOWN"):
                return {"ok": False, "state": state,
                        "reason": "unsafe_recovery_requires_human_or_verifier"}
            if row["executor_id"] != executor_id:
                return {"ok": False, "state": state,
                        "reason": "claimed_by_another_executor"}
            # same executor, CLAIMED/RUNNING: resume-in-place, never duplicate
            conn.execute(
                "UPDATE rt_campaign_exec SET state='CLAIMED',updated_at=? "
                "WHERE campaign_id=?", (_now(), campaign_id))
            return {"ok": True, "state": "CLAIMED", "first_claim": False,
                    "resume_in_place": True}

    def campaign_state(self, campaign_id: str) -> dict[str, Any] | None:
        row = self.db.rows(
            "SELECT * FROM rt_campaign_exec WHERE campaign_id=?",
            (campaign_id,))
        if not row:
            return None
        r = row[0]
        return {"campaign_id": r["campaign_id"], "state": r["state"],
                "executor_id": r["executor_id"],
                "config_sha": r["config_sha"], "harness_sha": r["harness_sha"],
                "result": _load_json(r["result_json"], {}),
                "result_sha": r["result_sha"], "reason": r["reason"],
                "created_at": r["created_at"], "updated_at": r["updated_at"]}

    def set_campaign_state(self, campaign_id: str, state: str, *,
                           executor_id: str = "", result: Any = None,
                           result_sha: str = "", reason: str = "") -> dict[str, Any]:
        """Validate a campaign state transition and persist it + optional result."""
        if state not in self.CAMPAIGN_STATES:
            raise RuntimeStateError(f"unknown campaign state {state!r}")
        with self.db.session() as conn:
            row = conn.execute(
                "SELECT state,executor_id FROM rt_campaign_exec WHERE campaign_id=?",
                (campaign_id,)).fetchone()
            if row is None:
                raise RuntimeStateError(f"unknown campaign {campaign_id!r}")
            current = row["state"]
            allowed = {
                "CREATED": {"CLAIMED"},
                "CLAIMED": {"RUNNING", "RECOVERY_REQUIRED", "FAILED"},
                "RUNNING": {"COMPLETED", "FAILED", "RECOVERY_REQUIRED"},
                "RECOVERY_REQUIRED": {"RUNNING", "UNKNOWN", "FAILED"},
                "UNKNOWN": {"FAILED"},
                "COMPLETED": set(), "FAILED": set(),
            }
            if state not in allowed.get(current, set()):
                raise RuntimeStateError(
                    f"illegal campaign transition {current}->{state}")
            if executor_id and row["executor_id"] and row["executor_id"] != executor_id:
                raise RuntimeStateError("campaign owned by another executor")
            payload = json.dumps(result, sort_keys=True, ensure_ascii=False,
                                 default=str) if result is not None else ""
            conn.execute(
                """UPDATE rt_campaign_exec SET state=?,result_json=?,
                   result_sha=?,reason=?,updated_at=? WHERE campaign_id=?""",
                (state, payload, result_sha, reason, _now(), campaign_id))
        return {"ok": True, "state": state,
                "result_sha": result_sha or self.campaign_state(campaign_id)["result_sha"]}

    def recover_interrupted_campaigns(self, run_id: str,
                                      *, executor_id: str) -> list[dict[str, Any]]:
        """Mark CLAIMED/RUNNING campaigns as RECOVERY_REQUIRED after a restart.

        NEVER auto-resumes a backend that may have physically executed: it
        records uncertainty so a verifier/human decides (exactly-once honesty).
        """
        with self.db.session() as conn:
            rows = conn.execute(
                """SELECT campaign_id,state FROM rt_campaign_exec
                   WHERE run_id=? AND state IN ('CLAIMED','RUNNING')""",
                (run_id,)).fetchall()
            out = []
            for r in rows:
                conn.execute(
                    """UPDATE rt_campaign_exec SET state='RECOVERY_REQUIRED',
                       reason=?,updated_at=? WHERE campaign_id=?""",
                    (f"interrupted_from_{r['state']}", _now(), r["campaign_id"]))
                out.append({"campaign_id": r["campaign_id"],
                            "from_state": r["state"],
                            "to_state": "RECOVERY_REQUIRED"})
        return out


def _load_json(raw: str, fallback: Any) -> Any:
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return fallback


class RuntimeBudgetExhausted(RuntimeError):
    pass


__all__ = [
    "RuntimeState", "RuntimeStateError", "RuntimeBudgetExhausted",
    "config_sha", "SCHEMA_SQL", "S_INIT", "S_RUNNING", "S_ROUND2", "S_RRSI",
    "S_COMPLETED", "S_STOPPED", "S_EXHAUSTED", "S_FAILED", "S_RECOVERING",
    "C_COMPLETE", "C_AWAITING", "C_INFLIGHT", "C_EXPERIENCE", "C_RECOVERY",
    "C_CORRUPT",
]
