"""M5-01 -- independent, durable budgets per domain.

One domain exhausting its budget must never stop another domain. Like the
M5-00.1 ledger, reservations are persisted BEFORE a campaign runs and finalised
for completed/failed/interrupted outcomes; a campaign id can be charged once.
"""
from __future__ import annotations

import time
from typing import Any

from .research_store.db import ResearchDatabase, ResearchStoreConflict

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS domain_budget_state (
    run_id         TEXT NOT NULL,
    domain_id      TEXT NOT NULL,
    max_campaigns  INTEGER NOT NULL,
    reserved       INTEGER NOT NULL DEFAULT 0,
    consumed       INTEGER NOT NULL DEFAULT 0,
    status         TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    PRIMARY KEY (run_id, domain_id)
);
CREATE TABLE IF NOT EXISTS domain_budget_entries (
    run_id       TEXT NOT NULL,
    domain_id    TEXT NOT NULL,
    campaign_id  TEXT NOT NULL,
    state        TEXT NOT NULL,
    outcome      TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (run_id, domain_id, campaign_id)
);
"""


class DomainBudgetExhausted(RuntimeError):
    pass


class DomainBudgetManager:
    def __init__(self, *, db: ResearchDatabase):
        self.db = db
        with self.db.session() as conn:
            for stmt in SCHEMA_SQL.split(";"):
                if stmt.strip():
                    conn.execute(stmt)

    @staticmethod
    def _now() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%S")

    def initialize(self, run_id: str, domain_id: str, *,
                   max_campaigns: int) -> dict[str, Any]:
        with self.db.session() as conn:
            row = conn.execute(
                """SELECT * FROM domain_budget_state
                   WHERE run_id=? AND domain_id=?""",
                (run_id, domain_id)).fetchone()
            if row is None:
                conn.execute(
                    """INSERT INTO domain_budget_state
                       (run_id, domain_id, max_campaigns, reserved, consumed,
                        status, updated_at) VALUES (?,?,?,0,0,'active',?)""",
                    (run_id, domain_id, int(max_campaigns), self._now()))
            elif int(row["max_campaigns"]) != int(max_campaigns):
                raise ResearchStoreConflict(
                    f"domain budget {domain_id} changed after creation")
        return self.snapshot(run_id, domain_id)

    def snapshot(self, run_id: str, domain_id: str) -> dict[str, Any]:
        with self.db.session() as conn:
            row = conn.execute(
                """SELECT * FROM domain_budget_state
                   WHERE run_id=? AND domain_id=?""",
                (run_id, domain_id)).fetchone()
        if row is None:
            raise ResearchStoreConflict(
                f"no budget for run={run_id!r}, domain={domain_id!r}")
        limit = int(row["max_campaigns"])
        reserved = int(row["reserved"])
        consumed = int(row["consumed"])
        return {"run_id": run_id, "domain_id": domain_id,
                "budget_limit": limit or None, "reserved": reserved,
                "consumed": consumed,
                "remaining": None if limit <= 0 else limit - reserved - consumed,
                "status": row["status"]}

    def reserve(self, run_id: str, domain_id: str, campaign_id: str) -> dict[str, Any]:
        now = self._now()
        with self.db.session() as conn:
            existing = conn.execute(
                """SELECT * FROM domain_budget_entries
                   WHERE run_id=? AND domain_id=? AND campaign_id=?""",
                (run_id, domain_id, campaign_id)).fetchone()
            if existing is not None:
                return dict(existing)
            state = conn.execute(
                """SELECT * FROM domain_budget_state
                   WHERE run_id=? AND domain_id=?""",
                (run_id, domain_id)).fetchone()
            if state is None:
                raise ResearchStoreConflict(
                    f"no budget for domain {domain_id!r}")
            limit = int(state["max_campaigns"])
            used = int(state["reserved"]) + int(state["consumed"])
            if limit > 0 and used >= limit:
                raise DomainBudgetExhausted(
                    f"domain {domain_id!r} budget exhausted: {used}/{limit}")
            conn.execute(
                """INSERT INTO domain_budget_entries
                   (run_id, domain_id, campaign_id, state, outcome,
                    created_at, updated_at) VALUES (?,?,?,'reserved','',?,?)""",
                (run_id, domain_id, campaign_id, now, now))
            conn.execute(
                """UPDATE domain_budget_state SET reserved=reserved+1,
                   updated_at=? WHERE run_id=? AND domain_id=?""",
                (now, run_id, domain_id))
            return {"state": "reserved", "campaign_id": campaign_id}

    def release(self, run_id: str, domain_id: str, campaign_id: str, *,
                reason: str) -> dict[str, Any]:
        now = self._now()
        with self.db.session() as conn:
            entry = conn.execute(
                """SELECT * FROM domain_budget_entries
                   WHERE run_id=? AND domain_id=? AND campaign_id=?""",
                (run_id, domain_id, campaign_id)).fetchone()
            if entry is None or entry["state"] != "reserved":
                return dict(entry) if entry else {"state": "absent"}
            conn.execute(
                """UPDATE domain_budget_entries SET state='released',
                   outcome=?, updated_at=?
                   WHERE run_id=? AND domain_id=? AND campaign_id=?""",
                (f"released:{reason}", now, run_id, domain_id, campaign_id))
            conn.execute(
                """UPDATE domain_budget_state SET reserved=reserved-1,
                   updated_at=? WHERE run_id=? AND domain_id=?""",
                (now, run_id, domain_id))
            return {"state": "released", "campaign_id": campaign_id}

    def consume(self, run_id: str, domain_id: str, campaign_id: str, *,
                outcome: str) -> dict[str, Any]:
        now = self._now()
        with self.db.session() as conn:
            entry = conn.execute(
                """SELECT * FROM domain_budget_entries
                   WHERE run_id=? AND domain_id=? AND campaign_id=?""",
                (run_id, domain_id, campaign_id)).fetchone()
            if entry is None:
                raise ResearchStoreConflict(
                    f"campaign {campaign_id!r} was never reserved in domain "
                    f"{domain_id!r}")
            if entry["state"] == "consumed":
                return dict(entry)
            conn.execute(
                """UPDATE domain_budget_entries
                   SET state='consumed', outcome=?, updated_at=?
                   WHERE run_id=? AND domain_id=? AND campaign_id=?""",
                (str(outcome), now, run_id, domain_id, campaign_id))
            conn.execute(
                """UPDATE domain_budget_state
                   SET reserved=reserved-1, consumed=consumed+1, updated_at=?
                   WHERE run_id=? AND domain_id=?""",
                (now, run_id, domain_id))
            return {"state": "consumed", "campaign_id": campaign_id,
                    "outcome": str(outcome)}


__all__ = ["DomainBudgetManager", "DomainBudgetExhausted", "SCHEMA_SQL"]
