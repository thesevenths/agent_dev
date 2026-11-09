"""Immutable model lineage: checkpoint + config hash + dataset + parent.

Without this, a successful candidate is just a file on disk and you cannot
answer "trained from which version, on which data, with which config?".
Promotion is explicit and recorded with a reason; the evaluator/decision layer
is separate from the model-producing layer.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .schemas import ModelVersion
from .settings import ensure_data_dir

_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_versions (
    version_id          TEXT PRIMARY KEY,
    experiment_id       TEXT NOT NULL,
    job_id              TEXT NOT NULL,
    engine              TEXT,
    checkpoint_path     TEXT,
    checkpoint_sha256   TEXT,
    config_path         TEXT,
    config_sha256       TEXT,
    code_version        TEXT,
    runtime_code_version TEXT,
    code_dirty          INTEGER,
    dataset_version     TEXT,
    parent_version_id   TEXT,
    metrics             TEXT,
    promoted            INTEGER NOT NULL DEFAULT 0,
    promotion_reason    TEXT,
    created_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mv_experiment ON model_versions(experiment_id);
"""


class ModelRegistry:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else ensure_data_dir() / "models.sqlite"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._session() as conn:
            conn.executescript(_SCHEMA)
            self._migrate(conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Idempotent additive migration for registries created before A08."""
        existing = {row["name"] for row in conn.execute(
            "PRAGMA table_info(model_versions)"
        )}
        for column, ddl in (("runtime_code_version", "TEXT"),
                            ("code_dirty", "INTEGER")):
            if column not in existing:
                conn.execute(
                    f"ALTER TABLE model_versions ADD COLUMN {column} {ddl}"
                )

    @contextmanager
    def _session(self):
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _to_row(mv: ModelVersion) -> tuple:
        return (
            mv.version_id, mv.experiment_id, mv.job_id, mv.engine,
            mv.checkpoint_path, mv.checkpoint_sha256, mv.config_path,
            mv.config_sha256, mv.code_version,
            mv.runtime_code_version,
            None if mv.code_dirty is None else int(mv.code_dirty),
            mv.dataset_version,
            mv.parent_version_id, json.dumps(mv.metrics, sort_keys=True),
            int(mv.promoted), mv.promotion_reason, mv.created_at,
        )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> ModelVersion:
        data = dict(row)
        data["metrics"] = json.loads(data.get("metrics") or "{}")
        data["promoted"] = bool(data.get("promoted"))
        if "code_dirty" in data and data["code_dirty"] is not None:
            data["code_dirty"] = bool(data["code_dirty"])
        return ModelVersion(**data)

    def register(self, mv: ModelVersion) -> ModelVersion:
        with self._session() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO model_versions
                   (version_id, experiment_id, job_id, engine,
                   checkpoint_path, checkpoint_sha256, config_path,
                   config_sha256, code_version, runtime_code_version,
                   code_dirty, dataset_version, parent_version_id, metrics,
                   promoted, promotion_reason, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                self._to_row(mv),
            )
        return mv

    def get(self, version_id: str) -> ModelVersion | None:
        with self._session() as conn:
            row = conn.execute(
                "SELECT * FROM model_versions WHERE version_id=?", (version_id,)
            ).fetchone()
        return self._from_row(row) if row else None

    def exists(self, version_id: str) -> bool:
        return self.get(version_id) is not None

    def promote(self, version_id: str, reason: str) -> ModelVersion | None:
        mv = self.get(version_id)
        if mv is None:
            return None
        mv.promoted = True
        mv.promotion_reason = reason
        return self.register(mv)

    def list_versions(self, *, experiment_id: str | None = None,
                      promoted_only: bool = False) -> list[ModelVersion]:
        sql = "SELECT * FROM model_versions"
        clauses, params = [], []
        if experiment_id:
            clauses.append("experiment_id=?")
            params.append(experiment_id)
        if promoted_only:
            clauses.append("promoted=1")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC"
        with self._session() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._from_row(r) for r in rows]

    def children_of(self, version_id: str) -> list[ModelVersion]:
        with self._session() as conn:
            rows = conn.execute(
                "SELECT * FROM model_versions WHERE parent_version_id=?",
                (version_id,),
            ).fetchall()
        return [self._from_row(r) for r in rows]
