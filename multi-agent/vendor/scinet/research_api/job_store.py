"""SQLite-backed job registry -- survives process restarts and concurrent runs.

Why a file-backed store instead of an in-memory dict: an Agent may retry, be
restarted, or the chat session may close while a 30-minute training job is
still running. Job identity must outlive the submitting process.

Thread/process safety: every operation opens its own short-lived connection
(``check_same_thread`` is therefore irrelevant) and uses SQLite's own locking
with a busy timeout. ``claim_running`` is a conditional UPDATE so a job can
never be executed twice even if two workers race.
"""
from __future__ import annotations

import datetime
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

STATUSES = ("queued", "running", "succeeded", "failed", "cancelled")
TERMINAL = ("succeeded", "failed", "cancelled")

_UPDATABLE = {
    "status", "pid", "run_id", "log_path", "result_path", "exit_code",
    "error", "error_type", "started_at", "heartbeat_at", "finished_at",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id          TEXT PRIMARY KEY,
    experiment_id   TEXT NOT NULL,
    stage           TEXT NOT NULL,
    status          TEXT NOT NULL,
    engine          TEXT NOT NULL DEFAULT 'scinet++',
    spec            TEXT,
    command         TEXT,
    cwd             TEXT,
    pid             INTEGER,
    run_id          TEXT,
    log_path        TEXT,
    result_path     TEXT,
    exit_code       INTEGER,
    error           TEXT,
    error_type      TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    started_at      TEXT,
    heartbeat_at    TEXT,
    finished_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_experiment ON jobs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status     ON jobs(status);
"""


class DuplicateJobError(RuntimeError):
    """Raised when a job_id is submitted more than once."""


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _decode(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    out = dict(row)
    for key in ("spec", "command"):
        if out.get(key):
            try:
                out[key] = json.loads(out[key])
            except (ValueError, TypeError):
                pass
    return out


class JobStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._session() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @contextmanager
    def _session(self):
        """Commit + CLOSE the connection (sqlite3's own CM does not close it)."""
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ---------------------------------------------------------------- writes
    def create_job(self, *, job_id: str, experiment_id: str, stage: str,
                   spec: dict | None = None, command: list[str] | None = None,
                   cwd: str | None = None, run_id: str | None = None,
                   log_path: str | None = None, result_path: str | None = None,
                   engine: str = "scinet++") -> dict:
        ts = _now()
        try:
            with self._session() as conn:
                conn.execute(
                    """INSERT INTO jobs
                       (job_id, experiment_id, stage, status, engine, spec,
                        command, cwd, run_id, log_path, result_path,
                        created_at, updated_at, heartbeat_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        job_id, experiment_id, stage, "queued", engine,
                        json.dumps(spec or {}, ensure_ascii=False),
                        json.dumps(command or [], ensure_ascii=False),
                        cwd, run_id, log_path, result_path, ts, ts, ts,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateJobError(
                f"job_id {job_id!r} already exists; refusing to execute twice"
            ) from exc
        return self.get(job_id)

    def claim_running(self, job_id: str, pid: int) -> bool:
        """Atomically move queued -> running. False means someone else won."""
        ts = _now()
        with self._session() as conn:
            cur = conn.execute(
                """UPDATE jobs SET status='running', pid=?, started_at=?,
                          updated_at=?, heartbeat_at=?
                   WHERE job_id=? AND status='queued'""",
                (pid, ts, ts, ts, job_id),
            )
            return cur.rowcount == 1

    def update(self, job_id: str, /, **fields: Any) -> dict | None:
        # positional-only: a caller trying ``update(id, job_id=...)`` now lands
        # in ``fields`` and is rejected by the whitelist instead of silently
        # mutating the primary key (or blowing up with an opaque TypeError).
        unknown = set(fields) - _UPDATABLE
        if unknown:
            raise ValueError(f"non-updatable fields: {sorted(unknown)}")
        if "status" in fields and fields["status"] not in STATUSES:
            raise ValueError(f"invalid status: {fields['status']}")
        fields["updated_at"] = _now()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._session() as conn:
            conn.execute(f"UPDATE jobs SET {cols} WHERE job_id=?",
                         (*fields.values(), job_id))
        return self.get(job_id)

    def heartbeat(self, job_id: str) -> None:
        ts = _now()
        with self._session() as conn:
            conn.execute(
                "UPDATE jobs SET heartbeat_at=? WHERE job_id=? AND status='running'",
                (ts, job_id),
            )

    # ----------------------------------------------------------------- reads
    def get(self, job_id: str) -> dict | None:
        with self._session() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return _decode(row)

    def list_jobs(self, experiment_id: str | None = None,
                  status: str | None = None) -> list[dict]:
        sql = "SELECT * FROM jobs"
        clauses, params = [], []
        if experiment_id:
            clauses.append("experiment_id=?")
            params.append(experiment_id)
        if status:
            clauses.append("status=?")
            params.append(status)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC"
        with self._session() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_decode(r) for r in rows]

    def stale_running_jobs(self, stale_seconds: float) -> list[dict]:
        """Running jobs whose heartbeat is older than the threshold."""
        cutoff = datetime.datetime.now() - datetime.timedelta(seconds=stale_seconds)
        with self._session() as conn:
            rows = conn.execute(
                """SELECT * FROM jobs WHERE status IN ('queued','running')
                   AND heartbeat_at < ?""",
                (cutoff.isoformat(timespec="seconds"),),
            ).fetchall()
        return [_decode(r) for r in rows]

    def reconcile_stale(self, stale_seconds: float = 90.0) -> list[str]:
        """Mark orphaned jobs as failed. Returns the job_ids that were回收."""
        recovered = []
        for job in self.stale_running_jobs(stale_seconds):
            self.update(
                job["job_id"], status="failed",
                error_type="worker_interrupted",
                error="worker stopped sending heartbeats (process crashed or host restarted)",
                finished_at=_now(),
            )
            recovered.append(job["job_id"])
        return recovered
