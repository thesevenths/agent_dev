"""M5-00.1 / P0-01 -- durable, idempotent loop budget ledger.

The M5-00 loop previously counted campaigns in process memory and measured
wall-clock from ``time.time()`` at construction. Both reset on restart, and the
counter was incremented only AFTER a campaign finished. That makes a crash
between start and completion look like unspent budget -- exactly the window in
which an at-most-once runtime must not lose its accounting.

This module adds a SMALL ledger to the same ``ResearchDatabase``:

  * ``loop_budget_state``: one row per loop (limits, persisted start time,
    aggregate reserved/consumed, config SHA);
  * ``loop_budget_entries``: one row per campaign, unique per loop, so the same
    campaign can never be charged twice.

Semantics
---------
  ``reserved``  campaigns started/pending but not finalised
  ``consumed``  campaigns finalised (completed, failed, interrupted all spend)
  ``remaining`` = limit - reserved - consumed

Reservation is written BEFORE the campaign is executed. On restart the ledger
is read from SQLite; consumed budget can never be reclaimed by a new process.
"""
from __future__ import annotations

import time
from typing import Any

from .research_store.db import ResearchDatabase, ResearchStoreConflict, content_sha256

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS loop_budget_state (
    loop_id          TEXT PRIMARY KEY,
    config_sha       TEXT NOT NULL,
    max_campaigns    INTEGER NOT NULL,
    max_wall_time_s  REAL NOT NULL,
    started_epoch    REAL NOT NULL,
    reserved         INTEGER NOT NULL DEFAULT 0,
    consumed         INTEGER NOT NULL DEFAULT 0,
    status           TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS loop_budget_entries (
    loop_id       TEXT NOT NULL,
    campaign_id   TEXT NOT NULL,
    config_sha    TEXT NOT NULL,
    state         TEXT NOT NULL,
    outcome       TEXT NOT NULL DEFAULT '',
    requested_at  TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (loop_id, campaign_id)
);
"""

ACTIVE = "active"
EXHAUSTED = "exhausted"
STOPPED = "stopped"


class LoopBudgetExhausted(RuntimeError):
    pass


def config_sha256(config: Any) -> str:
    """Canonical SHA over a LoopConfig (dict or dataclass-like object)."""
    payload = config if isinstance(config, dict) else dict(config.__dict__)
    return content_sha256(payload)


class LoopBudgetLedger:
    """Fail-closed ledger; all mutations happen in one SQLite transaction."""

    def __init__(self, *, db: ResearchDatabase):
        self.db = db
        with self.db.session() as conn:
            for stmt in SCHEMA_SQL.split(";"):
                if stmt.strip():
                    conn.execute(stmt)

    @staticmethod
    def _now() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%S")

    def initialize(self, loop_id: str, *, config_sha: str,
                   max_campaigns: int, max_wall_time_s: float,
                   started_epoch: float | None = None) -> dict[str, Any]:
        now = self._now()
        epoch = float(started_epoch if started_epoch is not None else time.time())
        with self.db.session() as conn:
            row = conn.execute(
                "SELECT * FROM loop_budget_state WHERE loop_id=?",
                (loop_id,)).fetchone()
            if row is None:
                conn.execute(
                    """INSERT INTO loop_budget_state
                       (loop_id, config_sha, max_campaigns, max_wall_time_s,
                        started_epoch, reserved, consumed, status, updated_at)
                       VALUES (?,?,?,?,?,0,0,?,?)""",
                    (loop_id, config_sha, int(max_campaigns),
                     float(max_wall_time_s), epoch, ACTIVE, now))
            else:
                self._assert_config(row, config_sha, max_campaigns,
                                    max_wall_time_s, loop_id)
        return self.snapshot(loop_id)

    def snapshot(self, loop_id: str) -> dict[str, Any]:
        with self.db.session() as conn:
            row = conn.execute(
                "SELECT * FROM loop_budget_state WHERE loop_id=?",
                (loop_id,)).fetchone()
        if row is None:
            raise ResearchStoreConflict(
                f"loop {loop_id!r} has no budget ledger; refusing to guess limits")
        limit = int(row["max_campaigns"])
        reserved = int(row["reserved"])
        consumed = int(row["consumed"])
        remaining = None if limit <= 0 else limit - reserved - consumed
        return {
            "loop_id": loop_id,
            "config_sha": row["config_sha"],
            "budget_limit": limit or None,
            "reserved": reserved,
            "consumed": consumed,
            "remaining": remaining,
            "max_wall_time_s": float(row["max_wall_time_s"]),
            "started_epoch": float(row["started_epoch"]),
            "elapsed_s": round(time.time() - float(row["started_epoch"]), 3),
            "status": row["status"],
        }

    @staticmethod
    def _assert_config(row: Any, config_sha: str, max_campaigns: int,
                       max_wall_time_s: float, loop_id: str) -> None:
        checks = (
            ("config_sha", row["config_sha"], config_sha),
            ("max_campaigns", int(row["max_campaigns"]), int(max_campaigns)),
            ("max_wall_time_s", float(row["max_wall_time_s"]),
             float(max_wall_time_s)),
        )
        for name, stored, supplied in checks:
            if stored != supplied:
                raise ResearchStoreConflict(
                    f"loop {loop_id!r} budget {name} changed after creation: "
                    f"stored={stored!r}, supplied={supplied!r}")

    def reserve(self, loop_id: str, campaign_id: str, *,
                config_sha: str) -> dict[str, Any]:
        """Persist a campaign reservation BEFORE any execution starts."""
        now = self._now()
        with self.db.session() as conn:
            state = conn.execute(
                "SELECT * FROM loop_budget_state WHERE loop_id=?",
                (loop_id,)).fetchone()
            if state is None:
                raise ResearchStoreConflict(
                    f"loop {loop_id!r} has no budget ledger")
            self._assert_config(state, config_sha,
                                int(state["max_campaigns"]),
                                float(state["max_wall_time_s"]), loop_id)
            existing = conn.execute(
                """SELECT * FROM loop_budget_entries
                   WHERE loop_id=? AND campaign_id=?""",
                (loop_id, campaign_id)).fetchone()
            if existing is not None:
                return dict(existing)   # idempotent: never charge twice
            limit = int(state["max_campaigns"])
            used = int(state["reserved"]) + int(state["consumed"])
            if limit > 0 and used >= limit:
                raise LoopBudgetExhausted(
                    f"loop {loop_id!r} budget exhausted: "
                    f"limit={limit}, reserved={state['reserved']}, "
                    f"consumed={state['consumed']}")
            conn.execute(
                """INSERT INTO loop_budget_entries
                   (loop_id, campaign_id, config_sha, state, outcome,
                    requested_at, updated_at)
                   VALUES (?,?,?,'reserved','',?,?)""",
                (loop_id, campaign_id, config_sha, now, now))
            conn.execute(
                "UPDATE loop_budget_state SET reserved=reserved+1, "
                "updated_at=? WHERE loop_id=?", (now, loop_id))
            return {"loop_id": loop_id, "campaign_id": campaign_id,
                    "state": "reserved", "config_sha": config_sha,
                    "requested_at": now, "updated_at": now}

    def consume(self, loop_id: str, campaign_id: str, *, outcome: str,
                config_sha: str) -> dict[str, Any]:
        """Finalise a reservation; failed/interrupted also spend the budget."""
        now = self._now()
        with self.db.session() as conn:
            state = conn.execute(
                "SELECT * FROM loop_budget_state WHERE loop_id=?",
                (loop_id,)).fetchone()
            if state is None:
                raise ResearchStoreConflict(
                    f"loop {loop_id!r} has no budget ledger")
            self._assert_config(state, config_sha,
                                int(state["max_campaigns"]),
                                float(state["max_wall_time_s"]), loop_id)
            entry = conn.execute(
                """SELECT * FROM loop_budget_entries
                   WHERE loop_id=? AND campaign_id=?""",
                (loop_id, campaign_id)).fetchone()
            if entry is None:
                raise ResearchStoreConflict(
                    f"campaign {campaign_id!r} was never reserved in loop "
                    f"{loop_id!r}; refusing to retro-charge")
            if entry["state"] == "consumed":
                return dict(entry)   # idempotent finalisation
            conn.execute(
                """UPDATE loop_budget_entries
                   SET state='consumed', outcome=?, updated_at=?
                   WHERE loop_id=? AND campaign_id=?""",
                (str(outcome), now, loop_id, campaign_id))
            conn.execute(
                """UPDATE loop_budget_state
                   SET reserved=reserved-1, consumed=consumed+1, updated_at=?
                   WHERE loop_id=?""", (now, loop_id))
            return {"loop_id": loop_id, "campaign_id": campaign_id,
                    "state": "consumed", "outcome": str(outcome),
                    "config_sha": config_sha, "requested_at": entry["requested_at"],
                    "updated_at": now}

    def release(self, loop_id: str, campaign_id: str, *, reason: str,
                config_sha: str) -> dict[str, Any]:
        """Cancel a reservation that never became a campaign (no budget spent)."""
        now = self._now()
        with self.db.session() as conn:
            entry = conn.execute(
                """SELECT * FROM loop_budget_entries
                   WHERE loop_id=? AND campaign_id=?""",
                (loop_id, campaign_id)).fetchone()
            if entry is None:
                return {"state": "absent", "campaign_id": campaign_id}
            if entry["state"] != "reserved":
                return dict(entry)
            conn.execute(
                """UPDATE loop_budget_entries
                   SET state='released', outcome=?, updated_at=?
                   WHERE loop_id=? AND campaign_id=?""",
                (f"released:{reason}", now, loop_id, campaign_id))
            conn.execute(
                "UPDATE loop_budget_state SET reserved=reserved-1, "
                "updated_at=? WHERE loop_id=?", (now, loop_id))
            return {"loop_id": loop_id, "campaign_id": campaign_id,
                    "state": "released", "outcome": f"released:{reason}"}

    def stop(self, loop_id: str, reason: str) -> dict[str, Any]:
        with self.db.session() as conn:
            conn.execute(
                "UPDATE loop_budget_state SET status=?, updated_at=? "
                "WHERE loop_id=?", (STOPPED, self._now(), loop_id))
        return self.snapshot(loop_id)


__all__ = [
    "LoopBudgetLedger", "LoopBudgetExhausted", "config_sha256",
    "SCHEMA_SQL", "ACTIVE", "EXHAUSTED", "STOPPED",
]
