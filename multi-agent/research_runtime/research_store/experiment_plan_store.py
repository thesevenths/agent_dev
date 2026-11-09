"""A1: content-addressed, versioned ExperimentPlan storage.

The approval is bound to ``plan_sha256``. Editing a plan (even a single
guardrail value) creates a NEW immutable version and moves the ``plan_current``
pointer; the previously granted approval then points at a hash that is no longer
current and cannot authorise execution. History is never silently overwritten.
"""
from __future__ import annotations

import sqlite3
from typing import Any

from ..schemas import ExperimentPlan
from .db import ResearchDatabase, ResearchStoreError, _dump, _load, _now, content_sha256
from .schemas import StoredExperimentPlan

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS plan_versions (
    plan_id          TEXT NOT NULL,
    plan_sha256      TEXT NOT NULL,
    version          INTEGER NOT NULL,
    objective_id     TEXT NOT NULL,
    hypothesis_ids   TEXT NOT NULL,
    record_json      TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    PRIMARY KEY (plan_id, plan_sha256)
);
CREATE TABLE IF NOT EXISTS plan_current (
    plan_id          TEXT PRIMARY KEY,
    plan_sha256      TEXT NOT NULL,
    version          INTEGER NOT NULL
);
"""


def canonical_plan(plan: ExperimentPlan | dict[str, Any]) -> dict[str, Any]:
    model = (plan if isinstance(plan, ExperimentPlan)
             else ExperimentPlan.model_validate(plan))
    data = model.model_dump(mode="json")
    # A timestamp is not part of the scientific design being approved.
    data.pop("created_at", None)
    return data


def plan_sha256(plan: ExperimentPlan | dict[str, Any]) -> str:
    return content_sha256(canonical_plan(plan))


class ExperimentPlanStore:
    def __init__(self, db: ResearchDatabase):
        self.db = db

    def save(self, plan: ExperimentPlan | dict[str, Any]) -> StoredExperimentPlan:
        payload = canonical_plan(plan)
        digest = content_sha256(payload)
        plan_id = str(payload["plan_id"])
        hypothesis_ids = [str(c.get("hypothesis_id"))
                          for c in payload.get("candidates", [])]
        try:
            with self.db.session() as conn:
                current = conn.execute(
                    "SELECT * FROM plan_current WHERE plan_id=?", (plan_id,)
                ).fetchone()
                if current is not None and current["plan_sha256"] == digest:
                    # Stay INSIDE this transaction: opening a second connection
                    # here would deadlock against our own BEGIN IMMEDIATE lock.
                    same = conn.execute(
                        """SELECT * FROM plan_versions
                           WHERE plan_id=? AND plan_sha256=?""",
                        (plan_id, digest),
                    ).fetchone()
                    return self._row(same)
                row = conn.execute(
                    """SELECT COALESCE(MAX(version), 0) AS v FROM plan_versions
                       WHERE plan_id=?""", (plan_id,),
                ).fetchone()
                version = int(row["v"]) + 1
                conn.execute(
                    """INSERT INTO plan_versions
                       (plan_id, plan_sha256, version, objective_id,
                        hypothesis_ids, record_json, created_at)
                       VALUES (?,?,?,?,?,?,?)""",
                    (plan_id, digest, version, str(payload["objective_id"]),
                     _dump(hypothesis_ids), _dump(payload), _now()),
                )
                conn.execute(
                    """INSERT INTO plan_current (plan_id, plan_sha256, version)
                       VALUES (?,?,?)
                       ON CONFLICT(plan_id) DO UPDATE SET
                           plan_sha256=excluded.plan_sha256,
                           version=excluded.version""",
                    (plan_id, digest, version),
                )
        except sqlite3.Error as exc:
            raise ResearchStoreError(
                f"cannot save plan {plan_id!r}: {exc}") from exc
        return StoredExperimentPlan(
            plan_id=plan_id, objective_id=str(payload["objective_id"]),
            plan=payload, plan_sha256=digest, version=version,
            hypothesis_ids=hypothesis_ids,
        )

    def get(self, plan_id: str) -> StoredExperimentPlan | None:
        rows = self.db.rows(
            """SELECT v.* FROM plan_versions v
               JOIN plan_current c ON c.plan_id=v.plan_id
                    AND c.plan_sha256=v.plan_sha256
               WHERE v.plan_id=?""", (plan_id,))
        return self._row(rows[0]) if rows else None

    def get_version(self, plan_id: str,
                    sha256: str) -> StoredExperimentPlan | None:
        rows = self.db.rows(
            """SELECT * FROM plan_versions WHERE plan_id=? AND plan_sha256=?""",
            (plan_id, sha256))
        return self._row(rows[0]) if rows else None

    def versions(self, plan_id: str) -> list[StoredExperimentPlan]:
        rows = self.db.rows(
            "SELECT * FROM plan_versions WHERE plan_id=? ORDER BY version",
            (plan_id,))
        return [self._row(row) for row in rows]

    def current_sha256(self, plan_id: str) -> str | None:
        rows = self.db.rows(
            "SELECT plan_sha256 FROM plan_current WHERE plan_id=?", (plan_id,))
        return str(rows[0]["plan_sha256"]) if rows else None

    @staticmethod
    def _row(row) -> StoredExperimentPlan:
        return StoredExperimentPlan(
            plan_id=row["plan_id"], objective_id=row["objective_id"],
            plan=_load(row["record_json"], {}), plan_sha256=row["plan_sha256"],
            version=int(row["version"]),
            hypothesis_ids=_load(row["hypothesis_ids"], []),
            created_at=row["created_at"],
        )
