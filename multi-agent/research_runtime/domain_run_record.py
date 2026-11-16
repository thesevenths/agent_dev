"""M5-01 -- append-only cross-domain run records.

Each record ties the GLOBAL run id to one domain, objective, approval, campaign,
report and the real source artifacts. Records are immutable: a second write of
the same content is a no-op; a conflicting write raises ResearchStoreConflict.
"""
from __future__ import annotations

import time
from typing import Any

from .research_store.db import (
    ResearchDatabase, ResearchStoreConflict, content_sha256,
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS domain_run_records (
    run_id        TEXT NOT NULL,
    objective_id  TEXT NOT NULL,
    domain_id     TEXT NOT NULL,
    campaign_id   TEXT NOT NULL,
    approval_id   TEXT NOT NULL,
    plan_id       TEXT NOT NULL,
    session_id    TEXT NOT NULL,
    report_id     TEXT NOT NULL,
    payload_sha   TEXT NOT NULL,
    status        TEXT NOT NULL,
    payload_json  TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (run_id, objective_id)
);
"""


class DomainRunRecordStore:
    def __init__(self, *, db: ResearchDatabase):
        self.db = db
        with self.db.session() as conn:
            for stmt in SCHEMA_SQL.split(";"):
                if stmt.strip():
                    conn.execute(stmt)

    def write(self, *, run_id: str, objective_id: str, domain_id: str,
              campaign_id: str, approval_id: str, plan_id: str,
              session_id: str, report_id: str, status: str,
              sources: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "run_id": run_id, "objective_id": objective_id,
            "domain_id": domain_id, "campaign_id": campaign_id,
            "approval_id": approval_id, "plan_id": plan_id,
            "session_id": session_id, "report_id": report_id,
            "status": status, "sources": sources or {},
        }
        digest = content_sha256(payload)
        import json
        blob = json.dumps(payload, indent=2, sort_keys=True,
                          ensure_ascii=False, default=str)
        with self.db.session() as conn:
            existing = conn.execute(
                """SELECT payload_sha FROM domain_run_records
                   WHERE run_id=? AND objective_id=?""",
                (run_id, objective_id)).fetchone()
            if existing is not None:
                if existing["payload_sha"] != digest:
                    raise ResearchStoreConflict(
                        f"domain run record for {objective_id!r} already "
                        "exists with different content")
                return {"status": "unchanged", "payload_sha": digest}
            conn.execute(
                """INSERT INTO domain_run_records
                   (run_id, objective_id, domain_id, campaign_id, approval_id,
                    plan_id, session_id, report_id, payload_sha, status,
                    payload_json, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, objective_id, domain_id, campaign_id, approval_id,
                 plan_id, session_id, report_id, digest, status, blob,
                 time.strftime("%Y-%m-%dT%H:%M:%S")))
        return {"status": "created", "payload_sha": digest}

    def get(self, run_id: str, objective_id: str) -> dict[str, Any] | None:
        import json
        with self.db.session() as conn:
            row = conn.execute(
                """SELECT payload_json FROM domain_run_records
                   WHERE run_id=? AND objective_id=?""",
                (run_id, objective_id)).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def all(self, run_id: str | None = None) -> list[dict[str, Any]]:
        import json
        sql = "SELECT payload_json FROM domain_run_records"
        params: tuple[Any, ...] = ()
        if run_id:
            sql += " WHERE run_id=?"
            params = (run_id,)
        sql += " ORDER BY created_at, objective_id"
        with self.db.session() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [json.loads(r["payload_json"]) for r in rows]


__all__ = ["DomainRunRecordStore", "SCHEMA_SQL"]
