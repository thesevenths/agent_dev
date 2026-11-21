"""M5-02 -- durable, append-only evolution experiment/version registry.

Every candidate, including REJECTED / INCONCLUSIVE / failed ones, is retained.
A record binds a layer, a candidate SHA, a parent SHA and the real experiment
evidence. Cross-layer write conflicts are rejected.
"""
from __future__ import annotations

import json
import time
from typing import Any

from ..research_store.db import ResearchDatabase, ResearchStoreConflict, content_sha256

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS rrsi_evolution_records (
    experiment_id  TEXT PRIMARY KEY,
    layer          TEXT NOT NULL,
    candidate_id   TEXT NOT NULL,
    candidate_sha  TEXT NOT NULL,
    parent_sha     TEXT NOT NULL DEFAULT '',
    status         TEXT NOT NULL,
    gate_passed    INTEGER NOT NULL DEFAULT 0,
    evidence_json  TEXT NOT NULL,
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rrsi_layer ON rrsi_evolution_records(layer);
"""


class EvolutionRegistry:
    def __init__(self, *, db: ResearchDatabase):
        self.db = db
        with self.db.session() as conn:
            for stmt in SCHEMA_SQL.split(";"):
                if stmt.strip():
                    conn.execute(stmt)

    def record(self, *, experiment_id: str, layer: str, candidate_id: str,
               candidate_sha: str, parent_sha: str = "", status: str,
               gate_passed: bool = False, evidence: dict[str, Any] | None = None
               ) -> dict[str, Any]:
        payload = {"experiment_id": experiment_id, "layer": layer,
                   "candidate_id": candidate_id,
                   "candidate_sha": candidate_sha, "parent_sha": parent_sha,
                   "status": status, "gate_passed": bool(gate_passed),
                   "evidence": evidence or {}}
        digest = content_sha256(payload)
        blob = json.dumps(payload, indent=2, sort_keys=True,
                          ensure_ascii=False, default=str)
        with self.db.session() as conn:
            old = conn.execute(
                "SELECT * FROM rrsi_evolution_records WHERE experiment_id=?",
                (experiment_id,)).fetchone()
            if old is not None:
                if old["layer"] != layer or old["candidate_sha"] != candidate_sha:
                    raise ResearchStoreConflict(
                        f"experiment {experiment_id!r} already used by another "
                        "layer/candidate; refusing cross-layer reuse")
                if old["evidence_json"] != blob:
                    raise ResearchStoreConflict(
                        f"evolution experiment {experiment_id!r} is immutable")
                return {"status": "unchanged", "experiment_id": experiment_id}
            conn.execute(
                """INSERT INTO rrsi_evolution_records
                   (experiment_id, layer, candidate_id, candidate_sha,
                    parent_sha, status, gate_passed, evidence_json, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (experiment_id, layer, candidate_id, candidate_sha, parent_sha,
                 status, int(bool(gate_passed)), blob,
                 time.strftime("%Y-%m-%dT%H:%M:%S")))
        # Return the VERSION sha supplied by the coordinator, not the SHA of
        # the bookkeeping row; parent inheritance must chain candidate versions.
        return {"status": "created", "experiment_id": experiment_id,
                "candidate_sha": candidate_sha}

    def get(self, experiment_id: str) -> dict[str, Any] | None:
        with self.db.session() as conn:
            row = conn.execute(
                "SELECT evidence_json FROM rrsi_evolution_records "
                "WHERE experiment_id=?", (experiment_id,)).fetchone()
        return json.loads(row["evidence_json"]) if row else None

    def all(self, layer: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT evidence_json FROM rrsi_evolution_records"
        params: tuple[Any, ...] = ()
        if layer:
            sql += " WHERE layer=?"
            params = (layer,)
        sql += " ORDER BY created_at, experiment_id"
        with self.db.session() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [json.loads(r["evidence_json"]) for r in rows]

    def parent_for(self, layer: str) -> str:
        """SHA of the most recent PROMOTED version in this layer, or ''."""
        for rec in reversed(self.all(layer)):
            if rec.get("status") == "PROMOTED":
                return str(rec.get("candidate_sha") or "")
        return ""


__all__ = ["EvolutionRegistry", "SCHEMA_SQL"]
