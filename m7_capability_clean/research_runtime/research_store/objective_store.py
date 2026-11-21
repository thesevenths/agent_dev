"""M4.0-01 -- Objective persistence in the EXISTING research.sqlite.

Reuses ``ResearchDatabase`` (WAL + ``BEGIN IMMEDIATE`` + RLock),
``content_sha256`` / ``canonical_bytes`` and the existing ``ResearchObjective``
contract from ``research_runtime/schemas.py``. NO new database, NO ORM.

The 10 required columns are present verbatim. ``provenance_json`` is one
additional column: it records WHY an objective exists (explicit historical
record vs M4.0 reconstruction) together with the source path + SHA, so a
backfilled objective can never be mistaken for a native historical one.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..schemas import ResearchObjective
from .db import ResearchDatabase, ResearchStoreConflict, content_sha256
from .schemas import StoredObjective

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS objectives (
    objective_id          TEXT PRIMARY KEY,
    domain                TEXT NOT NULL,
    description           TEXT NOT NULL,
    constraints_json      TEXT NOT NULL,
    success_criteria_json TEXT NOT NULL,
    parent_objective_id   TEXT,
    status                TEXT NOT NULL,
    origin_experience_id  TEXT,
    content_sha256        TEXT NOT NULL,
    created_at            TEXT NOT NULL,
    provenance_json       TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_objectives_domain ON objectives(domain, status);
CREATE INDEX IF NOT EXISTS idx_objectives_parent ON objectives(parent_objective_id);
"""

REQUIRED_COLUMNS = (
    "objective_id", "domain", "description", "constraints_json",
    "success_criteria_json", "parent_objective_id", "status",
    "origin_experience_id", "content_sha256", "created_at",
)


class ObjectiveStore:
    """Idempotent, fail-closed objective persistence.

    A second save of the SAME content is a no-op (idempotent migration/backfill).
    A save of the SAME id with DIFFERENT content raises ``ResearchStoreConflict``
    rather than silently overwriting research state.
    """

    def __init__(self, db: ResearchDatabase):
        self.db = db

    @staticmethod
    def _canonical_payload(record: StoredObjective) -> dict[str, Any]:
        # created_at is metadata, not identity: backfill must be able to
        # re-derive the same content hash on a later run.
        return {
            "objective_id": record.objective_id,
            "domain": record.domain,
            "description": record.description,
            "constraints": record.constraints,
            "success_criteria": record.success_criteria,
            "parent_objective_id": record.parent_objective_id,
            "status": record.status,
            "origin_experience_id": record.origin_experience_id,
        }

    def save(self, record: StoredObjective) -> dict[str, Any]:
        digest = content_sha256(self._canonical_payload(record))
        if record.content_sha256 and record.content_sha256 != digest:
            raise ResearchStoreConflict(
                f"objective {record.objective_id!r} content_sha256 mismatch: "
                f"declared {record.content_sha256} != computed {digest}")
        with self.db.session() as conn:
            existing = conn.execute(
                "SELECT content_sha256 FROM objectives WHERE objective_id=?",
                (record.objective_id,)).fetchone()
            if existing is not None:
                if existing["content_sha256"] != digest:
                    raise ResearchStoreConflict(
                        f"objective {record.objective_id!r} already exists with "
                        "different content; refusing to overwrite history")
                return {"status": "unchanged", "objective_id": record.objective_id}
            conn.execute(
                """INSERT INTO objectives
                   (objective_id, domain, description, constraints_json,
                    success_criteria_json, parent_objective_id, status,
                    origin_experience_id, content_sha256, created_at,
                    provenance_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (record.objective_id, record.domain, record.description,
                 record._dump_json(record.constraints),
                 record._dump_json(record.success_criteria),
                 record.parent_objective_id, record.status,
                 record.origin_experience_id, digest, record.created_at,
                 record._dump_json(record.provenance)),
            )
        return {"status": "created", "objective_id": record.objective_id}

    def get(self, objective_id: str) -> StoredObjective | None:
        rows = self.db.rows("SELECT * FROM objectives WHERE objective_id=?",
                            (objective_id,))
        if not rows:
            return None
        return StoredObjective.from_row(rows[0])

    def all(self, *, domain: str | None = None,
            status: str | None = None) -> list[StoredObjective]:
        sql = "SELECT * FROM objectives WHERE 1=1"
        params: list[Any] = []
        if domain:
            sql += " AND domain=?"
            params.append(domain)
        if status:
            sql += " AND status=?"
            params.append(status)
        sql += " ORDER BY objective_id"
        return [StoredObjective.from_row(r) for r in self.db.rows(sql, tuple(params))]

    @staticmethod
    def migrate(db_path: str | Path) -> dict[str, Any]:
        """Explicit idempotent migration entry point (safe to call repeatedly)."""
        ResearchDatabase(db_path, schema_sql=SCHEMA_SQL)
        return schema_structure(db_path)


def schema_structure(db_path: str | Path) -> dict[str, Any]:
    """STRUCTURE ONLY (no row counts) -- comparable across migrations."""
    db = ResearchDatabase(db_path, schema_sql=SCHEMA_SQL)
    rows = db.rows("PRAGMA table_info(objectives)")
    cols = [{"name": r["name"], "type": r["type"], "notnull": r["notnull"],
             "pk": r["pk"], "default": r["dflt_value"]} for r in rows]
    idx = [r["name"] for r in db.rows("PRAGMA index_list(objectives)")]
    return {
        "table": "objectives",
        "columns": cols,
        "column_names": [c["name"] for c in cols],
        "required_columns_present": all(
            c in {x["name"] for x in cols} for c in REQUIRED_COLUMNS),
        "indexes": sorted(idx),
    }


def database_snapshot(db_path: str | Path) -> dict[str, Any]:
    """Full DB snapshot: schema for EVERY table + row counts (M4.0-03).

    ``structure`` is directly comparable before/after a repeated migration;
    ``statistics`` is intentionally separate so data growth cannot masquerade
    as a schema change.
    """
    db = ResearchDatabase(db_path, schema_sql=SCHEMA_SQL)
    tables = [r["name"] for r in db.rows(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    schema: dict[str, Any] = {}
    stats: dict[str, Any] = {}
    for t in tables:
        cols = [{"name": r["name"], "type": r["type"],
                 "notnull": r["notnull"], "pk": r["pk"]}
                for r in db.rows(f'PRAGMA table_info("{t}")')]
        schema[t] = cols
        stats[t] = db.rows(f'SELECT COUNT(*) AS n FROM "{t}"')[0]["n"]
    indexes = [r["name"] for r in db.rows(
        "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name")]
    return {
        "schema": schema,
        "statistics": stats,
        "indexes": sorted(indexes),
        "table_count": len(tables),
        "structure_signature": {
            "tables": tables,
            "columns": schema,
            "indexes": sorted(indexes),
        },
    }


def describe_schema(db_path: str | Path) -> dict[str, Any]:
    """Portable JSON snapshot of the objectives table structure + row count."""
    out = schema_structure(db_path)
    db = ResearchDatabase(db_path, schema_sql=SCHEMA_SQL)
    out["row_count"] = db.rows("SELECT COUNT(*) AS n FROM objectives")[0]["n"]
    return out


def from_research_objective(obj: ResearchObjective, *, status: str = "proposed",
                            parent_objective_id: str | None = None,
                            origin_experience_id: str | None = None,
                            provenance: dict[str, Any] | None = None
                            ) -> StoredObjective:
    """Adapt the EXISTING ResearchObjective schema to the persisted record."""
    payload = {
        "objective_id": obj.objective_id, "domain": obj.domain,
        "description": obj.description, "constraints": dict(obj.constraints),
        "success_criteria": dict(obj.success_criteria),
        "parent_objective_id": parent_objective_id, "status": status,
        "origin_experience_id": origin_experience_id,
    }
    return StoredObjective(
        **payload,
        content_sha256=content_sha256(payload),
        created_at=obj.created_at,
        provenance=provenance or {},
    )
