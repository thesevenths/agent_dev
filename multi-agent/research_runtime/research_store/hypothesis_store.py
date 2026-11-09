"""A1: persistent hypotheses + the REAL deterministic Critic verdict.

``HypothesisManager.review()`` only sets ``status="approved"`` on an in-memory
object. That is a useful in-process gate but NOT durable evidence. Here the
verdict is stored, bound to the exact hypothesis content hash, so a later edit
cannot inherit an old review.
"""
from __future__ import annotations

import sqlite3
import uuid
from typing import Any

from ..schemas import Hypothesis
from .db import ResearchDatabase, ResearchStoreError, _dump, _load, _now, content_sha256
from .schemas import CriticReviewRecord, StoredHypothesis

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS hypothesis_versions (
    hypothesis_id       TEXT NOT NULL,
    content_sha256      TEXT NOT NULL,
    version             INTEGER NOT NULL,
    objective_id        TEXT NOT NULL,
    source_experiment_id TEXT NOT NULL,
    status              TEXT NOT NULL,
    record_json         TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    PRIMARY KEY (hypothesis_id, content_sha256)
);
CREATE TABLE IF NOT EXISTS critic_reviews (
    seq                 INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id           TEXT NOT NULL UNIQUE,
    hypothesis_id       TEXT NOT NULL,
    hypothesis_sha256   TEXT NOT NULL,
    reviewer            TEXT NOT NULL,
    passed              INTEGER NOT NULL,
    errors_json         TEXT NOT NULL,
    review_version      TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    UNIQUE (hypothesis_id, hypothesis_sha256, reviewer)
);
CREATE INDEX IF NOT EXISTS idx_reviews_hypothesis
    ON critic_reviews(hypothesis_id, seq);
"""

# Volatile fields are deliberately excluded from the content hash: a proposal
# re-emitted one second later is the SAME scientific claim, and a Critic verdict
# must follow the claim, not the timestamp or its workflow status.
_VOLATILE = ("created_at", "status")


def canonical_hypothesis(hypothesis: Hypothesis | dict[str, Any]) -> dict[str, Any]:
    model = (hypothesis if isinstance(hypothesis, Hypothesis)
             else Hypothesis.model_validate(hypothesis))
    data = model.model_dump(mode="json")
    for key in _VOLATILE:
        data.pop(key, None)
    return data


def hypothesis_sha256(hypothesis: Hypothesis | dict[str, Any]) -> str:
    return content_sha256(canonical_hypothesis(hypothesis))


class HypothesisStore:
    def __init__(self, db: ResearchDatabase):
        self.db = db

    def save(self, hypothesis: Hypothesis | dict[str, Any]) -> StoredHypothesis:
        payload = canonical_hypothesis(hypothesis)
        digest = content_sha256(payload)
        hid = str(payload["hypothesis_id"])
        try:
            with self.db.session() as conn:
                existing = conn.execute(
                    """SELECT * FROM hypothesis_versions
                       WHERE hypothesis_id=? AND content_sha256=?""",
                    (hid, digest),
                ).fetchone()
                if existing is not None:
                    return self._row(existing)
                row = conn.execute(
                    """SELECT COALESCE(MAX(version), 0) AS v FROM hypothesis_versions
                       WHERE hypothesis_id=?""", (hid,),
                ).fetchone()
                version = int(row["v"]) + 1
                conn.execute(
                    """INSERT INTO hypothesis_versions
                       (hypothesis_id, content_sha256, version, objective_id,
                        source_experiment_id, status, record_json, created_at)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (hid, digest, version, str(payload["objective_id"]),
                     str(payload.get("source_experiment_id") or ""),
                     "proposed", _dump(payload), _now()),
                )
        except sqlite3.Error as exc:
            raise ResearchStoreError(
                f"cannot save hypothesis {hid!r}: {exc}") from exc
        return StoredHypothesis(
            hypothesis_id=hid, objective_id=str(payload["objective_id"]),
            hypothesis=payload, content_sha256=digest, version=version,
            source_experiment_id=str(payload.get("source_experiment_id") or ""),
            status="proposed",
        )

    def get(self, hypothesis_id: str) -> StoredHypothesis | None:
        rows = self.db.rows(
            """SELECT * FROM hypothesis_versions WHERE hypothesis_id=?
               ORDER BY version DESC LIMIT 1""", (hypothesis_id,))
        return self._row(rows[0]) if rows else None

    def versions(self, hypothesis_id: str) -> list[StoredHypothesis]:
        rows = self.db.rows(
            """SELECT * FROM hypothesis_versions WHERE hypothesis_id=?
               ORDER BY version""", (hypothesis_id,))
        return [self._row(row) for row in rows]

    def save_review(self, hypothesis_id: str, *, passed: bool,
                    errors: list[str] | None = None,
                    reviewer: str = "deterministic_critic",
                    review_version: str = "p1-02") -> CriticReviewRecord:
        stored = self.get(hypothesis_id)
        if stored is None:
            raise ResearchStoreError(
                f"cannot review unknown hypothesis {hypothesis_id!r}")
        digest = stored.content_sha256
        with self.db.session() as conn:
            existing = conn.execute(
                """SELECT * FROM critic_reviews WHERE hypothesis_id=?
                   AND hypothesis_sha256=? AND reviewer=?""",
                (hypothesis_id, digest, reviewer),
            ).fetchone()
            if existing is not None:
                return self._review_row(existing)
            review_id = "rev_" + uuid.uuid4().hex[:16]
            conn.execute(
                """INSERT INTO critic_reviews
                   (review_id, hypothesis_id, hypothesis_sha256, reviewer,
                    passed, errors_json, review_version, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (review_id, hypothesis_id, digest, reviewer, int(bool(passed)),
                 _dump(list(errors or [])), review_version, _now()),
            )
        return CriticReviewRecord(
            review_id=review_id, hypothesis_id=hypothesis_id,
            hypothesis_sha256=digest, reviewer=reviewer, passed=bool(passed),
            errors=list(errors or []), review_version=review_version,
        )

    def latest_review(self, hypothesis_id: str) -> CriticReviewRecord | None:
        rows = self.db.rows(
            """SELECT * FROM critic_reviews WHERE hypothesis_id=?
               ORDER BY seq DESC LIMIT 1""", (hypothesis_id,))
        return self._review_row(rows[0]) if rows else None

    def passed_review(self, hypothesis_id: str,
                      hypothesis_sha256: str) -> CriticReviewRecord | None:
        rows = self.db.rows(
            """SELECT * FROM critic_reviews WHERE hypothesis_id=?
               AND hypothesis_sha256=? AND passed=1
               ORDER BY seq DESC LIMIT 1""",
            (hypothesis_id, hypothesis_sha256))
        return self._review_row(rows[0]) if rows else None

    # ------------------------------------------------------------- internals
    @staticmethod
    def _row(row) -> StoredHypothesis:
        return StoredHypothesis(
            hypothesis_id=row["hypothesis_id"],
            objective_id=row["objective_id"],
            hypothesis=_load(row["record_json"], {}),
            content_sha256=row["content_sha256"],
            version=int(row["version"]),
            source_experiment_id=row["source_experiment_id"] or "",
            status=row["status"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _review_row(row) -> CriticReviewRecord:
        return CriticReviewRecord(
            review_id=row["review_id"], hypothesis_id=row["hypothesis_id"],
            hypothesis_sha256=row["hypothesis_sha256"], reviewer=row["reviewer"],
            passed=bool(row["passed"]), errors=_load(row["errors_json"], []),
            review_version=row["review_version"], created_at=row["created_at"],
        )
