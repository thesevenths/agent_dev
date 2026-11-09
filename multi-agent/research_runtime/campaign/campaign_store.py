"""P1-04: SQLite-backed campaign state, events and recovery queries.

Why SQLite rather than a JSON file
----------------------------------
Every state transition must be durable BEFORE the corresponding engine call,
and the state write plus its audit event must be ONE transaction. If the DB
write fails, the runner must not submit another job: otherwise a crash would
leave an unrecorded training run that resume() can never find.

The Event table is append-only. ``record_json`` is a cache of the current
state; the trials table is authoritative for trial rows, so a partially
written campaign can still be reconstructed.
"""
from __future__ import annotations

import datetime
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .exceptions import CampaignStoreError
from .schemas import CampaignRecord, TrialRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS campaigns (
    campaign_id     TEXT PRIMARY KEY,
    plan_id         TEXT NOT NULL,
    objective_id    TEXT NOT NULL,
    status          TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    record_json     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS campaign_trials (
    trial_id          TEXT PRIMARY KEY,
    campaign_id       TEXT NOT NULL,
    arm_type          TEXT NOT NULL,
    arm_id            TEXT NOT NULL,
    seed              INTEGER NOT NULL,
    status            TEXT NOT NULL,
    position          INTEGER NOT NULL,
    train_job_id      TEXT,
    evaluation_job_id TEXT,
    record_json       TEXT NOT NULL,
    FOREIGN KEY(campaign_id) REFERENCES campaigns(campaign_id)
);
CREATE INDEX IF NOT EXISTS idx_campaign_trial_status
    ON campaign_trials(campaign_id, status);
CREATE TABLE IF NOT EXISTS campaign_events (
    event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id  TEXT NOT NULL,
    trial_id     TEXT,
    event_type   TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_campaign_events
    ON campaign_events(campaign_id, event_id);
"""


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


class CampaignStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._session() as conn:
                conn.executescript(_SCHEMA)
        except sqlite3.Error as exc:
            raise CampaignStoreError(f"cannot initialise campaign DB: {exc}") from exc

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @contextmanager
    def _session(self):
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _event_summary(event_type: str, trial_id: str | None) -> str:
        target = f" trial={trial_id}" if trial_id else ""
        return f"{_now()} | {event_type}{target}"

    # -------------------------------------------------------------- writes
    def create_campaign(self, record: CampaignRecord, *,
                        event_type: str = "campaign_created",
                        payload: dict[str, Any] | None = None) -> CampaignRecord:
        try:
            with self._session() as conn:
                conn.execute(
                    """INSERT INTO campaigns
                       (campaign_id, plan_id, objective_id, status,
                        dataset_version, created_at, updated_at, record_json)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (record.campaign_id, record.plan_id, record.objective_id,
                     record.status, record.dataset_version, record.created_at,
                     record.updated_at, record.model_dump_json()),
                )
                for position, trial in enumerate(record.trials):
                    self._write_trial_row(conn, record.campaign_id, trial, position)
                self._write_event_row(conn, record.campaign_id, None,
                                      event_type, payload or {})
        except sqlite3.IntegrityError as exc:
            raise CampaignStoreError(
                f"campaign {record.campaign_id!r} already exists"
            ) from exc
        except sqlite3.Error as exc:
            raise CampaignStoreError(f"create_campaign failed: {exc}") from exc
        return record

    @staticmethod
    def _write_trial_row(conn: sqlite3.Connection, campaign_id: str,
                         trial: TrialRecord, position: int) -> None:
        conn.execute(
            """INSERT OR REPLACE INTO campaign_trials
               (trial_id, campaign_id, arm_type, arm_id, seed, status,
                position, train_job_id, evaluation_job_id, record_json)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (trial.trial_id, campaign_id, trial.arm_type, trial.arm_id,
             trial.seed, trial.status, position, trial.train_job_id,
             trial.evaluation_job_id, trial.model_dump_json()),
        )

    @staticmethod
    def _write_event_row(conn: sqlite3.Connection, campaign_id: str,
                         trial_id: str | None, event_type: str,
                         payload: dict[str, Any]) -> None:
        conn.execute(
            """INSERT INTO campaign_events
               (campaign_id, trial_id, event_type, payload_json, created_at)
               VALUES (?,?,?,?,?)""",
            (campaign_id, trial_id, event_type,
             json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str),
             _now()),
        )

    def save_campaign(self, record: CampaignRecord, *,
                      event_type: str | None = None,
                      payload: dict[str, Any] | None = None,
                      trial_id: str | None = None) -> CampaignRecord:
        """Campaign status + audit event in ONE atomic transaction."""
        record.updated_at = _now()
        if event_type:
            record.event_log.append(self._event_summary(event_type, trial_id))
        try:
            with self._session() as conn:
                cur = conn.execute(
                    """UPDATE campaigns SET status=?, updated_at=?, record_json=?
                       WHERE campaign_id=?""",
                    (record.status, record.updated_at, record.model_dump_json(),
                     record.campaign_id),
                )
                if cur.rowcount != 1:
                    raise CampaignStoreError(
                        f"campaign {record.campaign_id!r} not found")
                if event_type:
                    self._write_event_row(conn, record.campaign_id, trial_id,
                                          event_type, payload or {})
        except CampaignStoreError:
            raise
        except sqlite3.Error as exc:
            raise CampaignStoreError(f"save_campaign failed: {exc}") from exc
        return record

    def save_trial(self, record: CampaignRecord, trial: TrialRecord, *,
                   event_type: str, payload: dict[str, Any] | None = None
                   ) -> TrialRecord:
        """Trial state + campaign cache + audit event in ONE transaction.

        The runner calls this BEFORE submitting a job. If it raises, no engine
        call may follow -- that is the "DB write failed -> no new job" rule.
        """
        record.updated_at = _now()
        record.event_log.append(self._event_summary(event_type, trial.trial_id))
        position = next(
            (i for i, t in enumerate(record.trials) if t.trial_id == trial.trial_id),
            len(record.trials),
        )
        if position == len(record.trials):
            record.trials.append(trial)
        else:
            record.trials[position] = trial
        try:
            with self._session() as conn:
                self._write_trial_row(conn, record.campaign_id, trial, position)
                self._write_event_row(conn, record.campaign_id, trial.trial_id,
                                      event_type, payload or {})
                conn.execute(
                    """UPDATE campaigns SET status=?, updated_at=?, record_json=?
                       WHERE campaign_id=?""",
                    (record.status, record.updated_at, record.model_dump_json(),
                     record.campaign_id),
                )
        except sqlite3.Error as exc:
            raise CampaignStoreError(f"save_trial failed: {exc}") from exc
        return trial

    def append_event(self, campaign_id: str, event_type: str, *,
                     trial_id: str | None = None,
                     payload: dict[str, Any] | None = None) -> None:
        try:
            with self._session() as conn:
                self._write_event_row(conn, campaign_id, trial_id,
                                      event_type, payload or {})
        except sqlite3.Error as exc:
            raise CampaignStoreError(f"append_event failed: {exc}") from exc

    # --------------------------------------------------------------- reads
    def load_campaign(self, campaign_id: str) -> CampaignRecord | None:
        try:
            with self._session() as conn:
                row = conn.execute(
                    "SELECT record_json FROM campaigns WHERE campaign_id=?",
                    (campaign_id,),
                ).fetchone()
                if row is None:
                    return None
                record = CampaignRecord.model_validate_json(row["record_json"])
                trows = conn.execute(
                    """SELECT record_json FROM campaign_trials
                       WHERE campaign_id=? ORDER BY position""",
                    (campaign_id,),
                ).fetchall()
                erows = conn.execute(
                    """SELECT event_type, trial_id, created_at FROM campaign_events
                       WHERE campaign_id=? ORDER BY event_id""",
                    (campaign_id,),
                ).fetchall()
        except sqlite3.Error as exc:
            raise CampaignStoreError(f"load_campaign failed: {exc}") from exc
        record.trials = [TrialRecord.model_validate_json(r["record_json"])
                         for r in trows]
        record.event_log = [
            f"{e['created_at']} | {e['event_type']}"
            + (f" trial={e['trial_id']}" if e["trial_id"] else "")
            for e in erows
        ]
        return record

    def list_events(self, campaign_id: str) -> list[dict[str, Any]]:
        try:
            with self._session() as conn:
                rows = conn.execute(
                    """SELECT * FROM campaign_events
                       WHERE campaign_id=? ORDER BY event_id""",
                    (campaign_id,),
                ).fetchall()
        except sqlite3.Error as exc:
            raise CampaignStoreError(f"list_events failed: {exc}") from exc
        out = []
        for row in rows:
            item = dict(row)
            try:
                item["payload"] = json.loads(item.pop("payload_json"))
            except (ValueError, TypeError):
                item["payload"] = {}
            out.append(item)
        return out

    def campaign_ids(self) -> list[str]:
        with self._session() as conn:
            rows = conn.execute(
                "SELECT campaign_id FROM campaigns ORDER BY created_at"
            ).fetchall()
        return [r["campaign_id"] for r in rows]
