"""M4.3-01.1 P0-03 -- durable, append-only ExperimentRegistry storage.

Reuses ARR's EXISTING SQLite plumbing (``research_store.db.ResearchDatabase``
and ``content_sha256``) instead of inventing a second database layer. Four
additive tables live in the SAME research SQLite file when a ``ResearchDatabase``
is injected, or in a dedicated file when a path is given:

    m4_3_blocks   -- block_id, snapshot_id, seed, protocol/snapshot SHA, status
    m4_3_runs     -- full RunRecord JSON, finalized flag, all trusted IDs
    m4_3_reviews  -- append-only independent review rows (never overwritten)
    m4_3_events   -- append-only, hash-CHAINED audit events

Every state change appends an event. The chain hash links each event to the
previous one, so an independent export can be verified for integrity and any
tampering / truncation is detectable. A FINALIZED run is immutable: updating it
raises ``RegistryConflict`` instead of silently overwriting evidence.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..research_store.db import ResearchDatabase, content_sha256, _dump, _load, _now

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS m4_3_blocks (
    block_id         TEXT PRIMARY KEY,
    snapshot_id      TEXT NOT NULL,
    seed             INTEGER NOT NULL,
    protocol_sha256  TEXT NOT NULL,
    snapshot_sha256  TEXT NOT NULL,
    status           TEXT NOT NULL,
    record_json      TEXT NOT NULL,
    created_at       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS m4_3_runs (
    run_id           TEXT PRIMARY KEY,
    block_id         TEXT NOT NULL,
    arm              TEXT NOT NULL,
    objective_id     TEXT NOT NULL,
    plan_id          TEXT NOT NULL DEFAULT '',
    approval_id      TEXT NOT NULL DEFAULT '',
    campaign_id      TEXT NOT NULL DEFAULT '',
    report_id        TEXT NOT NULL DEFAULT '',
    experience_id    TEXT NOT NULL DEFAULT '',
    protocol_sha256  TEXT NOT NULL DEFAULT '',
    snapshot_sha256  TEXT NOT NULL DEFAULT '',
    review_id        TEXT NOT NULL DEFAULT '',
    outcome          TEXT NOT NULL,
    finalized        INTEGER NOT NULL DEFAULT 0,
    record_json      TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    finalized_at     TEXT
);
CREATE TABLE IF NOT EXISTS m4_3_reviews (
    seq              INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id        TEXT NOT NULL,
    reviewer_id      TEXT NOT NULL,
    review_kind      TEXT NOT NULL,
    record_json      TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    UNIQUE (review_id, reviewer_id, review_kind)
);
CREATE TABLE IF NOT EXISTS m4_3_events (
    seq              INTEGER PRIMARY KEY AUTOINCREMENT,
    scope            TEXT NOT NULL,
    event_type       TEXT NOT NULL,
    payload_json     TEXT NOT NULL,
    payload_sha256   TEXT NOT NULL,
    prev_sha256      TEXT NOT NULL,
    chain_sha256     TEXT NOT NULL,
    created_at       TEXT NOT NULL
);
"""

_GENESIS = "0" * 64


class RegistryConflict(RuntimeError):
    """Immutable record would be overwritten / duplicated."""


def snapshot_sha256(snapshot: Any) -> str:
    """Stable SHA of a research-opportunity Snapshot (any dict/obj)."""
    if snapshot is None:
        return content_sha256({"snapshot": None})
    if isinstance(snapshot, dict):
        return content_sha256(snapshot)
    if hasattr(snapshot, "to_dict"):
        return content_sha256(snapshot.to_dict())
    if hasattr(snapshot, "model_dump"):
        return content_sha256(snapshot.model_dump(mode="json"))
    return content_sha256(str(snapshot))


class RegistryStore:
    """Durable, append-only backend for ExperimentRegistry."""

    def __init__(self, db: ResearchDatabase | None = None,
                 *, db_path: str | Path | None = None):
        if db is not None:
            self.db = db
            # Additive schema on the SHARED ARR database (idempotent).
            with self.db.session() as conn:
                for statement in SCHEMA_SQL.split(";"):
                    if statement.strip():
                        conn.execute(statement)
        else:
            if db_path is None:
                raise ValueError("db_path is required when db is not injected")
            self.db = ResearchDatabase(db_path, schema_sql=SCHEMA_SQL)
        self.db_path = Path(self.db.db_path)

    # ------------------------------------------------------------- internals
    def _append_event(self, conn: sqlite3.Connection, *, scope: str,
                      event_type: str, payload: dict[str, Any]) -> str:
        prev = conn.execute(
            "SELECT chain_sha256 FROM m4_3_events ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        prev_sha = prev["chain_sha256"] if prev else _GENESIS
        payload_sha = content_sha256(payload)
        chain_sha = content_sha256({
            "prev_sha256": prev_sha, "scope": scope,
            "event_type": event_type, "payload_sha256": payload_sha,
        })
        conn.execute(
            """INSERT INTO m4_3_events
               (scope, event_type, payload_json, payload_sha256, prev_sha256,
                chain_sha256, created_at) VALUES (?,?,?,?,?,?,?)""",
            (scope, event_type, _dump(payload), payload_sha, prev_sha,
             chain_sha, _now()))
        return chain_sha

    # ------------------------------------------------------------- blocks
    def save_block(self, block: Any, *, protocol_sha256: str,
                   snapshot_sha256: str) -> None:
        payload = asdict(block) if hasattr(block, "__dataclass_fields__") \
            else dict(block)
        with self.db.session() as conn:
            exists = conn.execute(
                "SELECT 1 FROM m4_3_blocks WHERE block_id=?",
                (payload["block_id"],)).fetchone()
            if exists is not None:
                raise RegistryConflict(
                    f"block {payload['block_id']!r} already exists")
            conn.execute(
                """INSERT INTO m4_3_blocks
                   (block_id, snapshot_id, seed, protocol_sha256,
                    snapshot_sha256, status, record_json, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (payload["block_id"], payload["snapshot_id"],
                 int(payload["seed"]), protocol_sha256, snapshot_sha256,
                 "open", _dump(payload), _now()))
            self._append_event(
                conn, scope=f"block:{payload['block_id']}",
                event_type="block_created",
                payload={"block_id": payload["block_id"],
                         "snapshot_id": payload["snapshot_id"],
                         "seed": int(payload["seed"]),
                         "protocol_sha256": protocol_sha256,
                         "snapshot_sha256": snapshot_sha256})

    def load_blocks(self) -> list[dict[str, Any]]:
        rows = self.db.rows(
            "SELECT * FROM m4_3_blocks ORDER BY rowid")
        return [_load(r["record_json"], {}) | {
            "protocol_sha256": r["protocol_sha256"],
            "snapshot_sha256": r["snapshot_sha256"]} for r in rows]

    # --------------------------------------------------------------- runs
    def save_run(self, rec: Any, *, protocol_sha256: str = "",
                 snapshot_sha256: str = "") -> None:
        payload = rec.to_dict() if hasattr(rec, "to_dict") else dict(rec)
        rid = payload["run_id"]
        with self.db.session() as conn:
            if conn.execute("SELECT 1 FROM m4_3_runs WHERE run_id=?",
                            (rid,)).fetchone() is not None:
                raise RegistryConflict(f"run {rid!r} already exists")
            conn.execute(
                """INSERT INTO m4_3_runs
                   (run_id, block_id, arm, objective_id, plan_id, approval_id,
                    campaign_id, report_id, experience_id, protocol_sha256,
                    snapshot_sha256, review_id, outcome, finalized, record_json,
                    created_at, finalized_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (rid, payload["block_id"], payload["arm"],
                 payload.get("objective_id", ""), payload.get("plan_id", ""),
                 payload.get("approval_id", ""), payload.get("campaign_id", ""),
                 payload.get("report_id", ""), payload.get("experience_id", ""),
                 protocol_sha256, snapshot_sha256, payload.get("review_id", ""),
                 payload.get("outcome", "interrupted"), 0, _dump(payload),
                 _now(), None))
            self._append_event(
                conn, scope=f"run:{rid}", event_type="run_registered",
                payload={"run_id": rid, "block_id": payload["block_id"],
                         "arm": payload["arm"],
                         "objective_id": payload.get("objective_id", ""),
                         "protocol_sha256": protocol_sha256,
                         "snapshot_sha256": snapshot_sha256})

    def update_run(self, rec: Any, *, event_type: str = "run_updated",
                   changes: dict[str, Any] | None = None) -> None:
        """Persist a NON-finalized run. Finalized runs are immutable."""
        payload = rec.to_dict() if hasattr(rec, "to_dict") else dict(rec)
        rid = payload["run_id"]
        with self.db.session() as conn:
            row = conn.execute(
                "SELECT finalized FROM m4_3_runs WHERE run_id=?",
                (rid,)).fetchone()
            if row is None:
                raise RegistryConflict(f"unknown run {rid!r}")
            if row["finalized"]:
                raise RegistryConflict(
                    f"run {rid!r} is finalized; completed runs cannot be "
                    "overwritten (append-only audit)")
            conn.execute(
                """UPDATE m4_3_runs SET record_json=?, outcome=?,
                   plan_id=?, approval_id=?, campaign_id=?, report_id=?,
                   experience_id=?, review_id=? WHERE run_id=? AND finalized=0""",
                (_dump(payload), payload.get("outcome", "interrupted"),
                 payload.get("plan_id", ""), payload.get("approval_id", ""),
                 payload.get("campaign_id", ""), payload.get("report_id", ""),
                 payload.get("experience_id", ""), payload.get("review_id", ""),
                 rid))
            self._append_event(
                conn, scope=f"run:{rid}", event_type=event_type,
                payload={"run_id": rid,
                         "changes": dict(changes or {}),
                         "outcome": payload.get("outcome", "interrupted")})

    def finalize_run(self, rec: Any, *, changes: dict[str, Any] | None = None
                     ) -> None:
        """Mark a run FINAL exactly once. A second attempt is a conflict."""
        payload = rec.to_dict() if hasattr(rec, "to_dict") else dict(rec)
        rid = payload["run_id"]
        with self.db.session() as conn:
            row = conn.execute(
                "SELECT finalized FROM m4_3_runs WHERE run_id=?",
                (rid,)).fetchone()
            if row is None:
                raise RegistryConflict(f"unknown run {rid!r}")
            if row["finalized"]:
                raise RegistryConflict(
                    f"run {rid!r} already finalized; no double execution / "
                    "no duplicate campaign / no budget re-consumption")
            now = _now()
            conn.execute(
                """UPDATE m4_3_runs SET finalized=1, finalized_at=?,
                   record_json=?, outcome=?, plan_id=?, approval_id=?,
                   campaign_id=?, report_id=?, experience_id=?, review_id=?
                   WHERE run_id=? AND finalized=0""",
                (now, _dump(payload), payload.get("outcome", "interrupted"),
                 payload.get("plan_id", ""), payload.get("approval_id", ""),
                 payload.get("campaign_id", ""), payload.get("report_id", ""),
                 payload.get("experience_id", ""), payload.get("review_id", ""),
                 rid))
            self._append_event(
                conn, scope=f"run:{rid}", event_type="run_finalized",
                payload={"run_id": rid,
                         "outcome": payload.get("outcome", "interrupted"),
                         "campaign_id": payload.get("campaign_id", ""),
                         "changes": dict(changes or {})})

    def load_runs(self) -> list[dict[str, Any]]:
        rows = self.db.rows("SELECT * FROM m4_3_runs ORDER BY rowid")
        out = []
        for r in rows:
            d = _load(r["record_json"], {})
            d["finalized"] = bool(r["finalized"])
            d["protocol_sha256"] = r["protocol_sha256"]
            d["snapshot_sha256"] = r["snapshot_sha256"]
            out.append(d)
        return out

    # ------------------------------------------------------------- reviews
    def save_review(self, review: dict[str, Any]) -> None:
        rid = review.get("review_id", "")
        with self.db.session() as conn:
            try:
                conn.execute(
                    """INSERT INTO m4_3_reviews
                       (review_id, reviewer_id, review_kind, record_json,
                        created_at) VALUES (?,?,?,?,?)""",
                    (rid, review.get("reviewer_id", ""),
                     review.get("review_kind", "human_blind"),
                     _dump(review), _now()))
            except sqlite3.IntegrityError as exc:
                raise RegistryConflict(
                    f"review ({rid}, {review.get('reviewer_id')}, "
                    f"{review.get('review_kind')}) already recorded") from exc
            self._append_event(
                conn, scope=f"review:{rid}", event_type="review_recorded",
                payload={"review_id": rid,
                         "reviewer_id": review.get("reviewer_id", ""),
                         "review_kind": review.get("review_kind", "")})

    def load_reviews(self) -> list[dict[str, Any]]:
        rows = self.db.rows("SELECT * FROM m4_3_reviews ORDER BY seq")
        return [_load(r["record_json"], {}) for r in rows]

    # ------------------------------------------------------------- events
    def append_event(self, *, scope: str, event_type: str,
                     payload: dict[str, Any]) -> str:
        with self.db.session() as conn:
            return self._append_event(
                conn, scope=scope, event_type=event_type, payload=payload)

    def load_events(self) -> list[dict[str, Any]]:
        rows = self.db.rows("SELECT * FROM m4_3_events ORDER BY seq")
        return [dict(r) for r in rows]

    # ---------------------------------------------- export / integrity
    def export(self) -> dict[str, Any]:
        """Independent, self-contained export incl. the full event chain."""
        bundle = {
            "schema": "m4_3_registry_export_v1",
            "db_path": str(self.db_path),
            "blocks": self.load_blocks(),
            "runs": self.load_runs(),
            "reviews": self.load_reviews(),
            "events": self.load_events(),
        }
        bundle["export_sha256"] = content_sha256({
            k: bundle[k] for k in ("blocks", "runs", "reviews", "events")})
        return bundle

    def verify_integrity(self) -> bool:
        """Re-derive the whole event hash chain; reject tampering/truncation."""
        events = self.load_events()
        prev = _GENESIS
        for ev in events:
            payload = _load(ev["payload_json"], {})
            expected_payload = content_sha256(payload)
            expected_chain = content_sha256({
                "prev_sha256": prev, "scope": ev["scope"],
                "event_type": ev["event_type"],
                "payload_sha256": expected_payload})
            if (ev["prev_sha256"] != prev
                    or ev["payload_sha256"] != expected_payload
                    or ev["chain_sha256"] != expected_chain):
                return False
            prev = ev["chain_sha256"]
        return True

    @staticmethod
    def verify_export(bundle: dict[str, Any]) -> bool:
        """Verify an EXPORTED bundle without touching the database."""
        events = bundle.get("events", [])
        prev = _GENESIS
        for ev in events:
            payload = json.loads(ev["payload_json"]) if isinstance(
                ev["payload_json"], str) else ev["payload_json"]
            expected_payload = content_sha256(payload)
            expected_chain = content_sha256({
                "prev_sha256": prev, "scope": ev["scope"],
                "event_type": ev["event_type"],
                "payload_sha256": expected_payload})
            if (ev["prev_sha256"] != prev
                    or ev["payload_sha256"] != expected_payload
                    or ev["chain_sha256"] != expected_chain):
                return False
            prev = ev["chain_sha256"]
        expected_export = content_sha256({
            k: bundle[k] for k in ("blocks", "runs", "reviews", "events")})
        return bundle.get("export_sha256") == expected_export


__all__ = [
    "SCHEMA_SQL", "RegistryConflict", "RegistryStore", "snapshot_sha256",
]
