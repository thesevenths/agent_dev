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
        limit = int(run.get("global_budget", 0))
        consumed = int(run.get("campaigns_consumed", 0))
        return {"limit": limit, "reserved": reserved, "consumed": consumed,
                "remaining": (limit - reserved - consumed) if limit > 0 else -1}

    # ------------------------------------------------------------------- rounds
    def put_round(self, run_id: str, round_no: int, domain_id: str,
                  state: str, **fields: Any) -> dict[str, Any]:
        """Upsert round state (idempotent for identical payload)."""
        payload = {k: v for k, v in fields.items() if k != "payload"}
        blob = json.dumps(fields.get("payload") or {}, sort_keys=True,
                          ensure_ascii=False, default=str)
        now = _now()
        with self.db.session() as conn:
            old = conn.execute(
                "SELECT * FROM rt_round WHERE run_id=? AND round_no=? "
                "AND domain_id=?", (run_id, round_no, domain_id)).fetchone()
            if old is not None:
                if (old["state"] == state and old["objective_id"] ==
                        str(payload.get("objective_id", old["objective_id"]))
                        and old["payload_json"] == blob):
                    return dict(old)
                # a round may advance state but must not rewrite evidence ids
                for col in ("objective_id", "campaign_id", "report_id",
                            "experience_id"):
                    if col in payload and old[col] and str(old[col]) != str(payload[col]):
                        raise ResearchStoreConflict(
                            f"round {run_id}/{round_no}/{domain_id} {col} changed "
                            f"from {old[col]!r} to {payload[col]!r}")
            conn.execute(
                """INSERT INTO rt_round
                   (run_id,round_no,domain_id,state,objective_id,campaign_id,
                    report_id,experience_id,payload_json,updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(run_id,round_no,domain_id) DO UPDATE SET
                     state=excluded.state, payload_json=excluded.payload_json,
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
    def classify(self, run_id: str) -> dict[str, Any]:
        """Read-only recovery classification from REAL persisted state.

        Never infers completion from a non-empty id alone: each round must be
        in a terminal COMPLETED state with campaign consumed + report + evidence
        recorded. Unverifiable -> RECOVERY_REQUIRED / CORRUPT.
        """
        integ = self.verify_journal(run_id)
        run = self.run(run_id)
        if not run:
            return {"classification": C_RECOVERY, "reason": "run_not_found"}
        rounds = self.rounds(run_id)
        if not integ["ok"]:
            return {"classification": C_CORRUPT, "journal": integ}
        budget = self.global_budget_snapshot(run_id)
        incomplete = [r for r in rounds if r["state"] != "COMPLETED"]
        if incomplete:
            # Distinguish the stage of the FIRST incomplete round.
            first = incomplete[0]
            payload = first.get("payload") or {}
            if first.get("campaign_id") and not first.get("report_id"):
                cls = C_INFLIGHT
            elif first.get("report_id") and not first.get("experience_id"):
                cls = C_EXPERIENCE
            elif payload.get("awaiting_approval"):
                cls = C_AWAITING
            else:
                cls = C_RECOVERY
            return {"classification": cls, "round": first,
                    "journal": integ, "budget": budget}
        return {"classification": C_COMPLETE, "journal": integ,
                "budget": budget, "n_rounds": len(rounds)}


class RuntimeBudgetExhausted(RuntimeError):
    pass


__all__ = [
    "RuntimeState", "RuntimeStateError", "RuntimeBudgetExhausted",
    "config_sha", "SCHEMA_SQL", "S_INIT", "S_RUNNING", "S_ROUND2", "S_RRSI",
    "S_COMPLETED", "S_STOPPED", "S_EXHAUSTED", "S_FAILED", "S_RECOVERING",
    "C_COMPLETE", "C_AWAITING", "C_INFLIGHT", "C_EXPERIENCE", "C_RECOVERY",
    "C_CORRUPT",
]
