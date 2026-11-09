"""A1: research sessions, idempotent campaign bindings, reports and experience.

A ``ResearchSession`` is the ONLY place that joins an Objective to its
Hypotheses / Critic reviews / Plans / Approvals / Campaigns / ComparisonReports
/ final ResearchReport. The campaign binding table is the idempotency key that
stops two concurrent ``run_research_campaign`` tool calls from launching the
same experiment twice; it is also how the two-rounds-per-objective cap is
enforced without trusting Agent memory.
"""
from __future__ import annotations

import sqlite3
import uuid
from typing import Any

from ..schemas import ResearchObjective
from .db import ResearchDatabase, ResearchStoreError, _dump, _load, _now, content_sha256
from .schemas import CampaignBinding, ResearchReport, ResearchSession, safe_id

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS research_sessions (
    session_id   TEXT PRIMARY KEY,
    objective_id TEXT NOT NULL,
    status       TEXT NOT NULL,
    record_json  TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS research_session_events (
    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    event_type   TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS campaign_bindings (
    campaign_id     TEXT PRIMARY KEY,
    plan_id         TEXT NOT NULL,
    approval_id     TEXT NOT NULL,
    plan_sha256     TEXT NOT NULL,
    objective_id    TEXT NOT NULL,
    session_id      TEXT,
    created_at      TEXT NOT NULL,
    UNIQUE (plan_id, approval_id, plan_sha256)
);
CREATE TABLE IF NOT EXISTS research_reports (
    report_id       TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    report_sha256   TEXT NOT NULL,
    record_json     TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS research_experience (
    experience_id   TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    objective_id    TEXT NOT NULL,
    kind            TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_session_events
    ON research_session_events(session_id, seq);
CREATE INDEX IF NOT EXISTS idx_bindings_objective
    ON campaign_bindings(objective_id);
"""


class ResearchSessionStore:
    def __init__(self, db: ResearchDatabase):
        self.db = db

    # -------------------------------------------------------------- session
    def create_session(self, objective: ResearchObjective | dict[str, Any], *,
                       session_id: str | None = None,
                       max_campaign_rounds: int = 2) -> ResearchSession:
        model = (objective if isinstance(objective, ResearchObjective)
                 else ResearchObjective.model_validate(objective))
        sid = session_id or "session_" + uuid.uuid4().hex[:12]
        safe_id(sid, "session_id")
        session = ResearchSession(
            session_id=sid, objective_id=model.objective_id,
            objective=model.model_dump(mode="json"),
            max_campaign_rounds=max_campaign_rounds,
        )
        try:
            with self.db.session() as conn:
                conn.execute(
                    """INSERT INTO research_sessions
                       (session_id, objective_id, status, record_json,
                        created_at, updated_at)
                       VALUES (?,?,?,?,?,?)""",
                    (sid, session.objective_id, session.status,
                     _dump(session.model_dump(mode="json")),
                     session.created_at, session.updated_at),
                )
                self._event(conn, sid, "session_created",
                            {"objective_id": session.objective_id})
        except sqlite3.IntegrityError as exc:
            raise ResearchStoreError(
                f"session {sid!r} already exists") from exc
        except sqlite3.Error as exc:
            raise ResearchStoreError(f"cannot create session: {exc}") from exc
        return session

    def get_session(self, session_id: str) -> ResearchSession | None:
        rows = self.db.rows(
            "SELECT record_json FROM research_sessions WHERE session_id=?",
            (session_id,))
        if not rows:
            return None
        return ResearchSession.model_validate(_load(rows[0]["record_json"], {}))

    def session_for_objective(self, objective_id: str) -> ResearchSession | None:
        rows = self.db.rows(
            """SELECT record_json FROM research_sessions WHERE objective_id=?
               ORDER BY created_at DESC LIMIT 1""", (objective_id,))
        if not rows:
            return None
        return ResearchSession.model_validate(_load(rows[0]["record_json"], {}))

    def save(self, session: ResearchSession, *, event_type: str | None = None,
             payload: dict[str, Any] | None = None) -> ResearchSession:
        session.updated_at = _now()
        with self.db.session() as conn:
            cur = conn.execute(
                """UPDATE research_sessions SET status=?, record_json=?,
                   updated_at=? WHERE session_id=?""",
                (session.status, _dump(session.model_dump(mode="json")),
                 session.updated_at, session.session_id),
            )
            if cur.rowcount != 1:
                raise ResearchStoreError(
                    f"session {session.session_id!r} not found")
            if event_type:
                self._event(conn, session.session_id, event_type, payload or {})
        return session

    def events(self, session_id: str) -> list[dict[str, Any]]:
        rows = self.db.rows(
            """SELECT * FROM research_session_events WHERE session_id=?
               ORDER BY seq""", (session_id,))
        return [{"event_type": row["event_type"],
                 "payload": _load(row["payload_json"], {}),
                 "created_at": row["created_at"]} for row in rows]

    # ------------------------------------------------------------- bindings
    def bind_campaign(self, binding: CampaignBinding) -> tuple[CampaignBinding, bool]:
        """Atomically return the existing binding or create a new one.

        Returns ``(binding, created_new)``. The unique constraint plus
        ``BEGIN IMMEDIATE`` makes concurrent duplicate submissions safe.
        """
        safe_id(binding.campaign_id, "campaign_id")
        with self.db.session() as conn:
            existing = conn.execute(
                """SELECT * FROM campaign_bindings
                   WHERE plan_id=? AND approval_id=? AND plan_sha256=?""",
                (binding.plan_id, binding.approval_id, binding.plan_sha256),
            ).fetchone()
            if existing is not None:
                return self._binding_row(existing), False
            try:
                conn.execute(
                    """INSERT INTO campaign_bindings
                       (campaign_id, plan_id, approval_id, plan_sha256,
                        objective_id, session_id, created_at)
                       VALUES (?,?,?,?,?,?,?)""",
                    (binding.campaign_id, binding.plan_id, binding.approval_id,
                     binding.plan_sha256, binding.objective_id,
                     binding.session_id, binding.created_at),
                )
            except sqlite3.IntegrityError:
                rows = conn.execute(
                    """SELECT * FROM campaign_bindings
                       WHERE plan_id=? AND approval_id=? AND plan_sha256=?""",
                    (binding.plan_id, binding.approval_id, binding.plan_sha256),
                ).fetchall()
                if rows:
                    return self._binding_row(rows[0]), False
                raise ResearchStoreError(
                    "campaign_id collision; refusing to reuse an existing campaign")
            return binding, True

    def campaign_for(self, plan_id: str, approval_id: str,
                     plan_sha256: str) -> CampaignBinding | None:
        rows = self.db.rows(
            """SELECT * FROM campaign_bindings
               WHERE plan_id=? AND approval_id=? AND plan_sha256=?""",
            (plan_id, approval_id, plan_sha256))
        return self._binding_row(rows[0]) if rows else None

    def campaigns_for_objective(self, objective_id: str) -> list[CampaignBinding]:
        rows = self.db.rows(
            """SELECT * FROM campaign_bindings WHERE objective_id=?
               ORDER BY created_at""", (objective_id,))
        return [self._binding_row(row) for row in rows]

    # -------------------------------------------------------------- reports
    def save_report(self, report: ResearchReport) -> ResearchReport:
        payload = report.model_dump(mode="json")
        payload.pop("report_sha256", None)
        digest = content_sha256(payload)
        final = report.model_copy(update={"report_sha256": digest})
        with self.db.session() as conn:
            existing = conn.execute(
                "SELECT report_sha256 FROM research_reports WHERE report_id=?",
                (final.report_id,),
            ).fetchone()
            if existing is not None:
                if existing["report_sha256"] != digest:
                    raise ResearchStoreError(
                        f"report {final.report_id!r} already persisted with "
                        "different content; reports are immutable")
                return final
            conn.execute(
                """INSERT INTO research_reports
                   (report_id, session_id, report_sha256, record_json, created_at)
                   VALUES (?,?,?,?,?)""",
                (final.report_id, final.session_id, digest,
                 _dump(final.model_dump(mode="json")), final.created_at),
            )
        return final

    def get_report(self, report_id: str) -> ResearchReport | None:
        rows = self.db.rows(
            "SELECT record_json FROM research_reports WHERE report_id=?",
            (report_id,))
        if not rows:
            return None
        return ResearchReport.model_validate(_load(rows[0]["record_json"], {}))

    def reports_for_session(self, session_id: str) -> list[ResearchReport]:
        rows = self.db.rows(
            """SELECT record_json FROM research_reports WHERE session_id=?
               ORDER BY created_at""", (session_id,))
        return [ResearchReport.model_validate(_load(r["record_json"], {}))
                for r in rows]

    # ---------------------------------------------------------- experience
    def record_experience(self, *, session_id: str, objective_id: str,
                          kind: str, payload: dict[str, Any]) -> str:
        """Append-only structured experience. Negative results are evidence too."""
        ref = "exp_" + uuid.uuid4().hex[:12]
        with self.db.session() as conn:
            conn.execute(
                """INSERT INTO research_experience
                   (experience_id, session_id, objective_id, kind,
                    payload_json, created_at) VALUES (?,?,?,?,?,?)""",
                (ref, session_id, objective_id, kind, _dump(payload), _now()),
            )
        return ref

    def experience(self, session_id: str) -> list[dict[str, Any]]:
        rows = self.db.rows(
            """SELECT * FROM research_experience WHERE session_id=?
               ORDER BY created_at""", (session_id,))
        return [{"experience_id": r["experience_id"], "kind": r["kind"],
                 "payload": _load(r["payload_json"], {}),
                 "created_at": r["created_at"]} for r in rows]

    # ------------------------------------------------------------- internals
    @staticmethod
    def _event(conn: sqlite3.Connection, session_id: str,
              event_type: str, payload: dict[str, Any]) -> None:
        conn.execute(
            """INSERT INTO research_session_events
               (session_id, event_type, payload_json, created_at)
               VALUES (?,?,?,?)""",
            (session_id, event_type, _dump(payload), _now()),
        )

    @staticmethod
    def _binding_row(row) -> CampaignBinding:
        return CampaignBinding(
            campaign_id=row["campaign_id"], plan_id=row["plan_id"],
            approval_id=row["approval_id"], plan_sha256=row["plan_sha256"],
            objective_id=row["objective_id"], session_id=row["session_id"],
            created_at=row["created_at"],
        )
