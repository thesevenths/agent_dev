"""M4.3-02 -- persistent HUMAN review queue (a real submission entry point).

The blind-review classes in ``blind_review`` are correct but in-memory: there
was no way for a real person to receive a blinded objective, score it, and have
that submission persisted append-only. This module is that entry point.

Guarantees
----------
  * a queued item is addressed by an OPAQUE ``review_id`` (no objective_id, no
    arm, no ranker signal anywhere in the reviewer-visible payload);
  * the ``review_id -> (objective_id, arm, block, snapshot, ranker)`` mapping is
    sealed in a SEPARATE table and is unreadable until ``unblind()``;
  * submissions are append-only and unique per (review_id, reviewer_id,
    review_kind): a reviewer cannot silently overwrite an earlier score;
  * completeness requires >= ``required_human_reviews`` DISTINCT HUMAN
    reviewers -- an ``automated`` score is stored separately and NEVER counts;
  * disagreements are preserved: every reviewer's own scores and reason stay in
    their own row (no averaging, no discard);
  * ``unblind()`` refuses until every queued item is complete.

The queue reuses ``RegistryStore.save_review`` for the review rows, so the
append-only + hash-chained audit trail is the same one the registry uses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..research_store.db import ResearchDatabase, _dump, _load, _now
from .blind_review import (
    FORBIDDEN_BLIND_FIELDS, REVIEW_KIND_AUTOMATED, REVIEW_KIND_HUMAN_BLIND,
    REVIEW_KINDS, RUBRIC_DIMENSIONS, RUBRIC_SCALE, BlindedObjective,
    BlindingViolation, QualityReview, blind,
)
from .registry_store import RegistryConflict, RegistryStore

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS m4_3_review_queue (
    review_id        TEXT PRIMARY KEY,
    experiment_id    TEXT NOT NULL,
    blinded_json     TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    unblinded_at     TEXT
);
CREATE TABLE IF NOT EXISTS m4_3_review_seals (
    review_id        TEXT PRIMARY KEY,
    experiment_id    TEXT NOT NULL,
    seal_json        TEXT NOT NULL,
    sealed_at        TEXT NOT NULL
);
"""


class ReviewQueueViolation(ValueError):
    """A submission breaks the review protocol (fail-closed)."""


def _validate_scores(scores: dict[str, Any]) -> dict[str, int]:
    clean: dict[str, int] = {}
    for dim in RUBRIC_DIMENSIONS:
        value = scores.get(dim)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ReviewQueueViolation(
                f"rubric dimension {dim!r} is missing or non-numeric")
        if not (RUBRIC_SCALE[0] <= value <= RUBRIC_SCALE[1]):
            raise ReviewQueueViolation(
                f"rubric dimension {dim!r} out of scale {RUBRIC_SCALE}: {value}")
        clean[dim] = int(value)
    return clean


@dataclass
class QueueItem:
    review_id: str
    experiment_id: str
    blinded: dict[str, Any]
    created_at: str
    unblinded_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"review_id": self.review_id,
                "experiment_id": self.experiment_id,
                "blinded": dict(self.blinded),
                "created_at": self.created_at,
                "unblinded_at": self.unblinded_at}


class ReviewQueue:
    """Durable review queue backed by the ARR SQLite database."""

    def __init__(self, *, db: Any = None, db_path: str | None = None,
                 required_human_reviews: int = 2):
        if required_human_reviews < 2:
            raise ValueError("a Pilot review requires >= 2 human reviewers")
        self.required_human_reviews = required_human_reviews
        if db is not None:
            self.db = db
            with self.db.session() as conn:
                for statement in SCHEMA_SQL.split(";"):
                    if statement.strip():
                        conn.execute(statement)
        else:
            if not db_path:
                raise ValueError("db_path is required when db is not injected")
            self.db = ResearchDatabase(db_path, schema_sql=SCHEMA_SQL)
        self._registry = RegistryStore(db=self.db)

    # ------------------------------------------------------------- enqueue
    def enqueue(self, *, experiment_id: str, objective: dict[str, Any],
                arm: str, block_id: str, snapshot_id: str,
                review_id: str | None = None) -> QueueItem:
        """Seal one objective into the queue and return its blinded payload."""
        if not experiment_id:
            raise ValueError("experiment_id is required")
        # The INPUT may carry arm / ranker / block metadata (it is used to
        # build the SEALED mapping); none of it may reach the reviewer, so it
        # is stripped before blinding. ``blind`` would otherwise fail closed.
        clean = {k: v for k, v in objective.items()
                 if k not in FORBIDDEN_BLIND_FIELDS}
        blinded = blind(clean)
        rid = review_id or blinded.review_id
        payload = blinded.to_dict()
        leaked = FORBIDDEN_BLIND_FIELDS.intersection(payload)
        if leaked:
            raise BlindingViolation(f"blinded payload leaks {sorted(leaked)}")
        seal = {
            "review_id": rid, "experiment_id": experiment_id,
            "objective_id": str(objective.get("objective_id") or ""),
            "arm": arm, "block_id": block_id, "snapshot_id": snapshot_id,
            "generation_source": str(objective.get("generation_source") or ""),
            "rank_score": objective.get("rank_score"),
            "ranker_version": str(objective.get("ranker_version") or ""),
            "sealed_at": _now(),
        }
        with self.db.session() as conn:
            if conn.execute("SELECT 1 FROM m4_3_review_queue WHERE review_id=?",
                            (rid,)).fetchone() is not None:
                raise RegistryConflict(f"review {rid!r} is already queued")
            conn.execute(
                """INSERT INTO m4_3_review_queue
                   (review_id, experiment_id, blinded_json, created_at,
                    unblinded_at) VALUES (?,?,?,?,NULL)""",
                (rid, experiment_id, _dump(payload), _now()))
            conn.execute(
                """INSERT INTO m4_3_review_seals
                   (review_id, experiment_id, seal_json, sealed_at)
                   VALUES (?,?,?,?)""",
                (rid, experiment_id, _dump(seal), _now()))
        return QueueItem(review_id=rid, experiment_id=experiment_id,
                         blinded=payload, created_at=_now())

    # ---------------------------------------------------------------- read
    def _item(self, review_id: str) -> QueueItem:
        rows = self.db.rows(
            "SELECT * FROM m4_3_review_queue WHERE review_id=?", (review_id,))
        if not rows:
            raise ReviewQueueViolation(f"unknown review {review_id!r}")
        r = rows[0]
        return QueueItem(review_id=r["review_id"],
                         experiment_id=r["experiment_id"],
                         blinded=_load(r["blinded_json"], {}),
                         created_at=r["created_at"],
                         unblinded_at=r["unblinded_at"] or "")

    def show(self, review_id: str) -> dict[str, Any]:
        """The reviewer-visible payload ONLY (never the sealed mapping)."""
        item = self._item(review_id)
        payload = dict(item.blinded)
        leaked = FORBIDDEN_BLIND_FIELDS.intersection(payload)
        if leaked:
            raise BlindingViolation(f"queued payload leaks {sorted(leaked)}")
        return payload

    def reviews(self, review_id: str,
                *, kind: str | None = None) -> list[dict[str, Any]]:
        rows = self.db.rows(
            """SELECT record_json FROM m4_3_reviews WHERE review_id=?
               ORDER BY seq""", (review_id,))
        out = [_load(r["record_json"], {}) for r in rows]
        if kind is not None:
            out = [r for r in out if r.get("review_kind") == kind]
        return out

    def human_reviews(self, review_id: str) -> list[dict[str, Any]]:
        return self.reviews(review_id, kind=REVIEW_KIND_HUMAN_BLIND)

    # -------------------------------------------------------------- submit
    def submit(self, *, review_id: str, reviewer_id: str,
               scores: dict[str, Any], reason: str,
               review_kind: str = REVIEW_KIND_HUMAN_BLIND) -> dict[str, Any]:
        """Persist ONE review. Append-only: the same reviewer cannot redo it."""
        if review_kind not in REVIEW_KINDS:
            raise ReviewQueueViolation(f"unknown review_kind {review_kind!r}")
        if not reviewer_id or not reviewer_id.strip():
            raise ReviewQueueViolation("reviewer_id is required")
        if not reason or not reason.strip():
            raise ReviewQueueViolation("a review must record a reason")
        item = self._item(review_id)
        if item.unblinded_at:
            raise ReviewQueueViolation(
                "the mapping is already unblinded; no new review may be added")
        if review_kind == REVIEW_KIND_HUMAN_BLIND:
            # An automated scorer may never impersonate a human reviewer id.
            if reviewer_id.lower().startswith(("auto", "scorer", "llm")):
                raise ReviewQueueViolation(
                    "an automated scorer cannot submit a human_blind review")
        review = QualityReview(
            review_id=review_id, reviewer_id=reviewer_id.strip(),
            scores=_validate_scores(scores), reason=reason.strip(),
            reviewed_at=_now(), review_kind=review_kind)
        try:
            self._registry.save_review(review.to_dict())
        except RegistryConflict as exc:
            raise ReviewQueueViolation(
                f"reviewer {reviewer_id!r} already submitted review "
                f"{review_id!r}; the original result cannot be overwritten"
            ) from exc
        return review.to_dict()

    # -------------------------------------------------------- completeness
    def completeness(self, review_id: str) -> dict[str, Any]:
        human = self.human_reviews(review_id)
        automated = self.reviews(review_id, kind=REVIEW_KIND_AUTOMATED)
        reviewers = sorted({r["reviewer_id"] for r in human})
        return {
            "review_id": review_id,
            "required_human_reviews": self.required_human_reviews,
            "n_human_reviews": len(human),
            "n_distinct_human_reviewers": len(reviewers),
            "reviewers": reviewers,
            "n_automated_scores": len(automated),
            "complete": len(reviewers) >= self.required_human_reviews,
            "scores": [{r["reviewer_id"]: r["scores"],
                        "total": r.get("total"), "reason": r["reason"]}
                       for r in human],
        }

    def queue_status(self, experiment_id: str) -> dict[str, Any]:
        rows = self.db.rows(
            """SELECT review_id, unblinded_at FROM m4_3_review_queue
               WHERE experiment_id=? ORDER BY created_at""", (experiment_id,))
        items = []
        for r in rows:
            c = self.completeness(r["review_id"])
            items.append({"review_id": r["review_id"],
                          "unblinded_at": r["unblinded_at"] or "",
                          "n_human_reviews": c["n_human_reviews"],
                          "complete": c["complete"]})
        pending = [i for i in items if not i["complete"]]
        return {"experiment_id": experiment_id,
                "n_queued": len(items),
                "n_complete": len(items) - len(pending),
                "n_pending": len(pending),
                "pending_review_ids": [i["review_id"] for i in pending],
                "items": items}

    def list_pending(self, experiment_id: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM m4_3_review_queue"
        params: tuple[Any, ...] = ()
        if experiment_id:
            sql += " WHERE experiment_id=?"
            params = (experiment_id,)
        rows = self.db.rows(sql + " ORDER BY created_at", params)
        out = []
        for r in rows:
            c = self.completeness(r["review_id"])
            if not c["complete"]:
                out.append({"review_id": r["review_id"],
                            "experiment_id": r["experiment_id"],
                            "n_human_reviews": c["n_human_reviews"],
                            "required": c["required_human_reviews"]})
        return out

    # ------------------------------------------------------------- unblind
    def unblind(self, experiment_id: str) -> dict[str, Any]:
        """Release the sealed mapping, ONLY after every item is complete."""
        rows = self.db.rows(
            """SELECT review_id, unblinded_at FROM m4_3_review_queue
               WHERE experiment_id=? ORDER BY created_at""", (experiment_id,))
        if not rows:
            raise ReviewQueueViolation(
                f"no queued review for experiment {experiment_id!r}")
        pending = [r["review_id"] for r in rows
                   if not self.completeness(r["review_id"])["complete"]]
        if pending:
            raise ReviewQueueViolation(
                "cannot unblind: "
                f"{len(pending)} review(s) still missing independent human "
                f"submissions: {pending}")
        seals = {}
        with self.db.session() as conn:
            for r in rows:
                seal = conn.execute(
                    "SELECT seal_json FROM m4_3_review_seals WHERE review_id=?",
                    (r["review_id"],)).fetchone()
                seals[r["review_id"]] = _load(seal["seal_json"], {})
                conn.execute(
                    "UPDATE m4_3_review_queue SET unblinded_at=? "
                    "WHERE review_id=? AND unblinded_at IS NULL",
                    (_now(), r["review_id"]))
        return {"experiment_id": experiment_id,
                "n_unblinded": len(seals), "mapping": seals}

    def arm_mapping(self, review_id: str) -> dict[str, Any]:
        """Sealed until unblinding: the arm MUST stay hidden from reviewers."""
        item = self._item(review_id)
        if not item.unblinded_at:
            raise ReviewQueueViolation(
                "the arm mapping is sealed until every review is complete and "
                "unblind() has been called")
        rows = self.db.rows(
            "SELECT seal_json FROM m4_3_review_seals WHERE review_id=?",
            (review_id,))
        return _load(rows[0]["seal_json"], {}) if rows else {}

    def export(self, experiment_id: str) -> dict[str, Any]:
        rows = self.db.rows(
            "SELECT * FROM m4_3_review_queue WHERE experiment_id=? "
            "ORDER BY created_at", (experiment_id,))
        items, seals = [], {}
        for r in rows:
            items.append({"review_id": r["review_id"],
                          "blinded": _load(r["blinded_json"], {}),
                          "created_at": r["created_at"],
                          "unblinded_at": r["unblinded_at"] or "",
                          "reviews": self.reviews(r["review_id"])})
            if r["unblinded_at"]:
                seal_rows = self.db.rows(
                    "SELECT seal_json FROM m4_3_review_seals WHERE review_id=?",
                    (r["review_id"],))
                seals[r["review_id"]] = _load(seal_rows[0]["seal_json"], {}) \
                    if seal_rows else {}
        return {"experiment_id": experiment_id,
                "required_human_reviews": self.required_human_reviews,
                "items": items,
                "mapping": seals if seals else "SEALED",
                "status": self.queue_status(experiment_id)}


__all__ = [
    "ReviewQueueViolation", "QueueItem", "ReviewQueue", "SCHEMA_SQL",
]
