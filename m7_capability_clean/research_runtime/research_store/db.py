"""Shared SQLite plumbing for the four P1-06 research stores.

Why ONE database and ONE transaction helper
-------------------------------------------
A plan, its Critic reviews, its approval and the session link must be written
atomically: a crash between "plan version 2 saved" and "approval invalidated"
would otherwise leave a state in which an old approval could authorise an
unreviewed plan. All four stores therefore share a single ``ResearchDatabase``;
every write uses ``BEGIN IMMEDIATE`` (a write lock taken BEFORE reading), so two
concurrent tool calls cannot both pass a check-then-write gate.

History is append-only. ``INSERT`` (never ``INSERT OR REPLACE``) is used for
version rows and approvals; a conflicting insert raises
``ResearchStoreConflict`` instead of silently overwriting evidence.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class ResearchStoreError(RuntimeError):
    pass


class ResearchStoreConflict(ResearchStoreError):
    """An immutable record already exists with a different identity."""


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str).encode("utf-8")


def content_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _now() -> str:
    import datetime
    return datetime.datetime.now().isoformat(timespec="seconds")


class ResearchDatabase:
    """Thread-safe, transaction-scoped access to the research state DB."""

    def __init__(self, db_path: str | Path, *, schema_sql: str = ""):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        # journal_mode is a connection-level pragma: it CANNOT be changed from
        # inside a transaction, and re-opening an existing WAL database used to
        # raise here. Set it once on a standalone connection instead.
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.commit()
        finally:
            conn.close()
        if schema_sql:
            with self.session() as conn:
                # executescript() implicitly COMMITs, which would break the
                # atomic "check + write" guarantee. Execute statements
                # individually so the whole schema stays in ONE transaction.
                for statement in schema_sql.split(";"):
                    if statement.strip():
                        conn.execute(statement)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def session(self):
        """One atomic transaction. Every store write MUST go through this."""
        with self._lock:
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

    def rows(self, query: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self.session() as conn:
            return conn.execute(query, params).fetchall()


def _dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _load(raw: str | None, default: Any) -> Any:
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return default
