"""A1: human approval records, bound to a resource HASH.

Trust boundary
--------------
``request()`` is reachable from an Agent tool and can ONLY create a ``pending``
record. ``grant()`` is deliberately NOT exposed as a tool: it lives behind the
out-of-band console (``approval_cli.py``) that a human runs in a terminal. An LLM
therefore cannot mint its own "human" approval through the tool surface, and a
``pending`` record can never satisfy the campaign gate.

``source`` distinguishes a human decision from an automated system policy. The
campaign gate (``research_service``) decides which sources are authoritative;
by default ONLY ``human`` can authorise execution.
"""
from __future__ import annotations

import sqlite3
import uuid
from typing import Any

from .db import (
    ResearchDatabase,
    ResearchStoreConflict,
    ResearchStoreError,
    _dump,
    _load,
    _now,
)
from .schemas import APPROVAL_SCOPES, ApprovalRecord, ApprovalSource

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS approvals (
    approval_id       TEXT PRIMARY KEY,
    resource_type     TEXT NOT NULL,
    resource_id       TEXT NOT NULL,
    resource_sha256   TEXT NOT NULL,
    status            TEXT NOT NULL,
    source            TEXT NOT NULL,
    approved_by       TEXT,
    approved_at       TEXT,
    requested_by      TEXT,
    requested_at      TEXT NOT NULL,
    scope_json        TEXT NOT NULL,
    notes             TEXT,
    created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_approvals_resource
    ON approvals(resource_type, resource_id, status);
CREATE TABLE IF NOT EXISTS approval_events (
    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
    approval_id  TEXT NOT NULL,
    event_type   TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
"""


class ApprovalStore:
    def __init__(self, db: ResearchDatabase):
        self.db = db

    def request(self, *, resource_type: str, resource_id: str,
                resource_sha256: str, requested_by: str = "research_agent",
                scope: list[str] | None = None,
                notes: str = "") -> ApprovalRecord:
        scopes = list(scope or ["run_campaign", "resume_campaign"])
        unknown = [item for item in scopes if item not in APPROVAL_SCOPES]
        if unknown:
            raise ResearchStoreError(f"unknown approval scopes: {unknown}")
        record = ApprovalRecord(
            approval_id="approval_" + uuid.uuid4().hex[:12],
            resource_type=resource_type, resource_id=resource_id,
            resource_sha256=resource_sha256, status="pending", source="human",
            requested_by=requested_by, requested_at=_now(), scope=scopes,
            notes=notes,
        )
        self._insert(record, "approval_requested")
        return record

    def grant(self, approval_id: str, *, approved_by: str,
              source: ApprovalSource = "human",
              current_resource_sha256: str | None = None,
              notes: str = "") -> ApprovalRecord:
        """Out-of-band, human-side operation. NEVER call from an Agent tool."""
        if not approved_by or not approved_by.strip():
            raise ResearchStoreError("approved_by is required for a human grant")
        if source not in ("human", "system_policy"):
            raise ResearchStoreError(f"invalid approval source {source!r}")
        with self.db.session() as conn:
            row = conn.execute(
                "SELECT * FROM approvals WHERE approval_id=?", (approval_id,)
            ).fetchone()
            if row is None:
                raise ResearchStoreError(f"unknown approval {approval_id!r}")
            if row["status"] != "pending":
                raise ResearchStoreConflict(
                    f"approval {approval_id!r} is already {row['status']!r}; "
                    "an approval decision is immutable")
            if (current_resource_sha256 is not None
                    and current_resource_sha256 != row["resource_sha256"]):
                raise ResearchStoreConflict(
                    "resource content changed since the request; re-request "
                    "approval for the current version")
            now = _now()
            cur = conn.execute(
                """UPDATE approvals SET status='approved', source=?,
                       approved_by=?, approved_at=?, notes=?
                   WHERE approval_id=? AND status='pending'""",
                (source, approved_by.strip(), now, notes, approval_id),
            )
            if cur.rowcount != 1:
                raise ResearchStoreConflict("concurrent approval decision")
            self._event_row(conn, approval_id, "approval_granted", {
                "approved_by": approved_by.strip(), "source": source,
            })
        return self.get(approval_id)  # type: ignore[return-value]

    def reject(self, approval_id: str, *, decided_by: str,
               notes: str = "") -> ApprovalRecord:
        return self._decide(approval_id, "rejected", decided_by, notes)

    def revoke(self, approval_id: str, *, decided_by: str,
               notes: str = "") -> ApprovalRecord:
        """Invalidate a granted approval (e.g. scope abuse discovered)."""
        return self._decide(approval_id, "revoked", decided_by, notes)

    def _decide(self, approval_id: str, status: str, decided_by: str,
                notes: str) -> ApprovalRecord:
        with self.db.session() as conn:
            cur = conn.execute(
                """UPDATE approvals SET status=?, approved_by=?, approved_at=?,
                       notes=? WHERE approval_id=? AND status IN
                       ('pending','approved')""",
                (status, decided_by, _now(), notes, approval_id),
            )
            if cur.rowcount != 1:
                raise ResearchStoreConflict(
                    f"cannot {status} approval {approval_id!r}")
            self._event_row(conn, approval_id, f"approval_{status}",
                            {"decided_by": decided_by, "notes": notes})
        return self.get(approval_id)  # type: ignore[return-value]

    def get(self, approval_id: str) -> ApprovalRecord | None:
        rows = self.db.rows(
            "SELECT * FROM approvals WHERE approval_id=?", (approval_id,))
        return self._row(rows[0]) if rows else None

    def for_resource(self, resource_id: str,
                     resource_type: str = "experiment_plan") -> list[ApprovalRecord]:
        rows = self.db.rows(
            """SELECT * FROM approvals WHERE resource_type=? AND resource_id=?
               ORDER BY created_at""", (resource_type, resource_id))
        return [self._row(row) for row in rows]

    def events(self, approval_id: str) -> list[dict[str, Any]]:
        rows = self.db.rows(
            """SELECT * FROM approval_events WHERE approval_id=? ORDER BY seq""",
            (approval_id,))
        return [{"event_type": row["event_type"],
                 "payload": _load(row["payload_json"], {}),
                 "created_at": row["created_at"]} for row in rows]

    # ------------------------------------------------------------- internals
    def _insert(self, record: ApprovalRecord, event_type: str) -> None:
        try:
            with self.db.session() as conn:
                conn.execute(
                    """INSERT INTO approvals
                       (approval_id, resource_type, resource_id, resource_sha256,
                        status, source, approved_by, approved_at, requested_by,
                        requested_at, scope_json, notes, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (record.approval_id, record.resource_type, record.resource_id,
                     record.resource_sha256, record.status, record.source,
                     record.approved_by, record.approved_at, record.requested_by,
                     record.requested_at, _dump(record.scope), record.notes,
                     record.created_at),
                )
                self._event_row(conn, record.approval_id, event_type,
                                {"resource_id": record.resource_id,
                                 "resource_sha256": record.resource_sha256})
        except sqlite3.Error as exc:
            raise ResearchStoreError(f"cannot create approval: {exc}") from exc

    @staticmethod
    def _event_row(conn: sqlite3.Connection, approval_id: str,
                   event_type: str, payload: dict[str, Any]) -> None:
        conn.execute(
            """INSERT INTO approval_events
               (approval_id, event_type, payload_json, created_at)
               VALUES (?,?,?,?)""",
            (approval_id, event_type, _dump(payload), _now()),
        )

    @staticmethod
    def _row(row) -> ApprovalRecord:
        return ApprovalRecord(
            approval_id=row["approval_id"], resource_type=row["resource_type"],
            resource_id=row["resource_id"],
            resource_sha256=row["resource_sha256"], status=row["status"],
            source=row["source"], approved_by=row["approved_by"] or "",
            approved_at=row["approved_at"], requested_by=row["requested_by"] or "",
            requested_at=row["requested_at"], scope=_load(row["scope_json"], []),
            notes=row["notes"] or "", created_at=row["created_at"],
        )
