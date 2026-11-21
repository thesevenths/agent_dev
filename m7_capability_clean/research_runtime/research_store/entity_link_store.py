"""M4.0-02 -- Entity Link Store: the unified ID relation layer.

Goal (task book §12): make ARR able to answer "where did a conclusion come
from, which objective drove it, which experiments, which evidence, which
experience". This module stores ONLY RELATIONS between EXISTING entity IDs;
it never invents an entity ID and never recomputes a verdict.

Design rules:
  * All writes go through ``ResearchDatabase.session()`` (BEGIN IMMEDIATE).
  * IDs are preserved verbatim (``hyp_fingerprint``, ``campaign_id``, ...).
  * UNIQUE(run_id, src_type, src_id, dst_type, dst_id, relation) prevents
    duplicate links while still allowing ONE entity to have MANY relations.
  * ``run_id`` is mandatory and is the isolation boundary; queries never
    silently mix runs.
  * Unconfirmed historical relations use an EXPLICIT sentinel id and
    status='unresolved' with a mandatory reason -- no guessing from filenames
    or similar text.
  * If ``source_path`` + ``source_sha256`` are supplied, the file is verified
    BEFORE the insert; a mismatch raises ResearchStoreConflict (fail-closed).
"""
from __future__ import annotations

import datetime
import hashlib
from pathlib import Path
from typing import Any

from .db import ResearchDatabase, ResearchStoreConflict

ENTITY_TYPES = frozenset({
    "objective", "hypothesis", "candidate", "experiment_plan",
    "campaign", "research_session", "evidence", "experience",
})
LINK_STATUS = frozenset({"confirmed", "reconstructed", "unresolved"})
UNRESOLVED_SENTINEL = "unresolved::<no-explicit-id>"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS entity_links (
    link_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT NOT NULL,
    objective_id   TEXT NOT NULL,
    src_type       TEXT NOT NULL,
    src_id         TEXT NOT NULL,
    dst_type       TEXT NOT NULL,
    dst_id         TEXT NOT NULL,
    relation       TEXT NOT NULL,
    status         TEXT NOT NULL,
    basis          TEXT NOT NULL,
    source_path    TEXT,
    source_sha256  TEXT,
    notes          TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL,
    UNIQUE(run_id, objective_id, src_type, src_id, dst_type, dst_id, relation)
);
CREATE INDEX IF NOT EXISTS idx_links_src ON entity_links(run_id, src_type, src_id);
CREATE INDEX IF NOT EXISTS idx_links_dst ON entity_links(run_id, dst_type, dst_id);
CREATE INDEX IF NOT EXISTS idx_links_objective ON entity_links(objective_id, run_id);
CREATE INDEX IF NOT EXISTS idx_links_status ON entity_links(run_id, status);
"""


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str | None:
    """Portable SHA-256; returns None when the file does not exist."""
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_link_schema(db: ResearchDatabase) -> dict[str, Any]:
    """Idempotently converge an EXISTING entity_links table to the current scope.

    M4.0-02 shipped with UNIQUE(run_id, src_type, ...). M4.0-03 found that the
    three domains share identical ``hyp_fingerprint`` values, so links for the
    same candidate in different objectives were silently merged. The scope now
    includes ``objective_id``. SQLite cannot ALTER a constraint, so an old EMPTY
    table is rebuilt atomically; a NON-EMPTY old table is refused (fail-closed)
    rather than having its history silently dropped.
    """
    with db.session() as conn:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name='entity_links'").fetchone()
        if row is None:
            for statement in SCHEMA_SQL.split(";"):
                if statement.strip():
                    conn.execute(statement)
            return {"status": "created"}
        ddl = row["sql"] or ""
        if "UNIQUE(run_id, objective_id" in ddl:
            return {"status": "current"}
        n = conn.execute("SELECT COUNT(*) AS n FROM entity_links").fetchone()["n"]
        if n:
            raise ResearchStoreConflict(
                "entity_links has an obsolete UNIQUE scope and is non-empty; "
                "refusing to silently rebuild historical relations")
        conn.execute("DROP TABLE entity_links")
        for statement in SCHEMA_SQL.split(";"):
            if statement.strip():
                conn.execute(statement)
        return {"status": "rebuilt_empty_legacy_table"}


class EntityLinkStore:
    def __init__(self, db: ResearchDatabase, *, workspace: Path | None = None):
        ensure_link_schema(db)
        self.db = db
        # source_path is stored POSIX-relative to this root for Windows/Linux
        # portability; a workspace makes SHA verification reproducible.
        self.workspace = Path(workspace) if workspace else None

    # -- writes -----------------------------------------------------------
    def add_link(self, *, run_id: str, objective_id: str,
                 src_type: str, src_id: str, dst_type: str, dst_id: str,
                 relation: str, status: str = "confirmed",
                 basis: str = "explicit_id", source_path: str | None = None,
                 source_sha256: str | None = None, notes: str = "",
                 verify_source_sha: bool = True) -> dict[str, Any]:
        if src_type not in ENTITY_TYPES or dst_type not in ENTITY_TYPES:
            raise ValueError(f"unknown entity type: {src_type!r} / {dst_type!r}")
        if status not in LINK_STATUS:
            raise ValueError(f"unknown link status: {status!r}")
        if not run_id or not objective_id or not src_id or not dst_id or not relation:
            raise ValueError("run_id / objective_id / src_id / dst_id / relation required")
        if status == "unresolved":
            # An unresolved link MUST be honest: sentinel id + explicit reason.
            if dst_id != UNRESOLVED_SENTINEL:
                raise ValueError(
                    "unresolved relations must use the explicit sentinel dst_id; "
                    "never guess a historical entity identity")
            if not notes.strip():
                raise ValueError("unresolved relations require a reason in notes")
        if source_sha256 and verify_source_sha and source_path and self.workspace:
            actual = sha256_file(self.workspace / Path(source_path))
            if actual is None or actual != source_sha256:
                raise ResearchStoreConflict(
                    f"source SHA mismatch for {source_path}: "
                    f"declared {source_sha256} != actual {actual}")

        # On Linux ``Path.as_posix()`` does NOT convert a literal backslash
        # (it is not a separator there), so normalise both conventions.
        posix_source = (str(source_path).replace("\\", "/")
                        if source_path else None)
        with self.db.session() as conn:
            existing = conn.execute(
                """SELECT link_id FROM entity_links
                   WHERE run_id=? AND objective_id=? AND src_type=? AND src_id=?
                     AND dst_type=? AND dst_id=? AND relation=?""",
                (run_id, objective_id, src_type, src_id, dst_type, dst_id,
                 relation)).fetchone()
            if existing is not None:
                return {"status": "duplicate", "link_id": existing["link_id"]}
            cur = conn.execute(
                """INSERT INTO entity_links
                   (run_id, objective_id, src_type, src_id, dst_type, dst_id,
                    relation, status, basis, source_path, source_sha256,
                    notes, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, objective_id, src_type, src_id, dst_type, dst_id,
                 relation, status, basis, posix_source, source_sha256,
                 notes, _now()))
            return {"status": "created", "link_id": cur.lastrowid}

    # -- reads ------------------------------------------------------------
    @staticmethod
    def _row(r: Any) -> dict[str, Any]:
        return {
            "link_id": r["link_id"], "run_id": r["run_id"],
            "objective_id": r["objective_id"], "src_type": r["src_type"],
            "src_id": r["src_id"], "dst_type": r["dst_type"],
            "dst_id": r["dst_id"], "relation": r["relation"],
            "status": r["status"], "basis": r["basis"],
            "source_path": r["source_path"], "source_sha256": r["source_sha256"],
            "notes": r["notes"], "created_at": r["created_at"],
        }

    def find_by_entity(self, entity_type: str, entity_id: str, *,
                       run_id: str | None = None,
                       objective_id: str | None = None,
                       direction: str = "both") -> list[dict[str, Any]]:
        """FORWARD (as src) and REVERSE (as dst) queries.

        ``direction``: 'out' | 'in' | 'both'. ``run_id`` scopes to one run;
        ``objective_id`` scopes to one objective graph. The latter matters
        because M3.x re-uses the SAME ``hyp_fingerprint`` in all three domains:
        preserving that original ID is required, so the OBJECTIVE is the
        isolation boundary that keeps per-domain chains from being mixed.
        """
        if direction not in ("out", "in", "both"):
            raise ValueError("direction must be out|in|both")
        clauses: list[str] = []
        params: list[Any] = []
        if direction in ("out", "both"):
            clauses.append("(src_type=? AND src_id=?)")
            params += [entity_type, entity_id]
        if direction in ("in", "both"):
            clauses.append("(dst_type=? AND dst_id=?)")
            params += [entity_type, entity_id]
        sql = "SELECT * FROM entity_links WHERE (" + " OR ".join(clauses) + ")"
        if run_id is not None:
            sql += " AND run_id=?"
            params.append(run_id)
        if objective_id is not None:
            sql += " AND objective_id=?"
            params.append(objective_id)
        sql += " ORDER BY link_id"
        return [self._row(r) for r in self.db.rows(sql, tuple(params))]

    def find_by_objective(self, objective_id: str, *, run_id: str | None = None,
                          include_unresolved: bool = True) -> list[dict[str, Any]]:
        sql = "SELECT * FROM entity_links WHERE objective_id=?"
        params: list[Any] = [objective_id]
        if run_id is not None:
            sql += " AND run_id=?"
            params.append(run_id)
        if not include_unresolved:
            sql += " AND status!='unresolved'"
        sql += " ORDER BY link_id"
        return [self._row(r) for r in self.db.rows(sql, tuple(params))]

    def list_objective_graph(self, objective_id: str, run_id: str
                             ) -> dict[str, Any]:
        """Deterministic, run-scoped graph for one objective."""
        links = self.find_by_objective(objective_id, run_id=run_id)
        nodes: dict[str, set[str]] = {}
        for l in links:
            nodes.setdefault(l["src_type"], set()).add(l["src_id"])
            nodes.setdefault(l["dst_type"], set()).add(l["dst_id"])
        return {
            "objective_id": objective_id,
            "run_id": run_id,
            "n_links": len(links),
            "n_confirmed": sum(1 for l in links if l["status"] == "confirmed"),
            "n_reconstructed": sum(1 for l in links if l["status"] == "reconstructed"),
            "n_unresolved": sum(1 for l in links if l["status"] == "unresolved"),
            "nodes": {t: sorted(ids) for t, ids in sorted(nodes.items())},
            "edges": [{"src_type": l["src_type"], "src_id": l["src_id"],
                       "dst_type": l["dst_type"], "dst_id": l["dst_id"],
                       "relation": l["relation"], "status": l["status"],
                       "basis": l["basis"]} for l in links],
            "unresolved": [l["notes"] for l in links if l["status"] == "unresolved"],
        }

    # -- integrity --------------------------------------------------------
    def validate_links(self, *, run_id: str | None = None,
                       workspace: Path | None = None) -> dict[str, Any]:
        """Fail-closed integrity check; NEVER mutates anything.

        Verifies: known entity types / statuses, unresolved honesty, run
        scoping, source SHA against the real file, and that no link points at
        the same (type,id) as both src and dst.
        """
        sql = "SELECT * FROM entity_links"
        params: tuple = ()
        if run_id is not None:
            sql += " WHERE run_id=?"
            params = (run_id,)
        sql += " ORDER BY link_id"
        links = [self._row(r) for r in self.db.rows(sql, params)]
        root = Path(workspace) if workspace else self.workspace
        violations: list[dict[str, Any]] = []
        checked_sha = 0
        for l in links:
            def bad(reason: str) -> None:
                violations.append({"link_id": l["link_id"], "reason": reason})
            if l["src_type"] not in ENTITY_TYPES or l["dst_type"] not in ENTITY_TYPES:
                bad("unknown entity type")
            if l["status"] not in LINK_STATUS:
                bad("unknown status")
            if l["status"] == "unresolved":
                if l["dst_id"] != UNRESOLVED_SENTINEL:
                    bad("unresolved link has a guessed dst_id")
                if not (l["notes"] or "").strip():
                    bad("unresolved link has no reason")
            if (l["src_type"], l["src_id"]) == (l["dst_type"], l["dst_id"]):
                bad("self-loop")
            if l["source_sha256"] and l["source_path"] and root is not None:
                actual = sha256_file(root / Path(l["source_path"]))
                checked_sha += 1
                if actual != l["source_sha256"]:
                    bad(f"source SHA mismatch: {l['source_path']}")
        return {
            "ok": not violations,
            "n_links": len(links),
            "n_sha_checked": checked_sha,
            "n_unresolved": sum(1 for l in links if l["status"] == "unresolved"),
            "n_confirmed": sum(1 for l in links if l["status"] == "confirmed"),
            "n_reconstructed": sum(1 for l in links if l["status"] == "reconstructed"),
            "violations": violations,
            "run_id": run_id,
        }

    def stats(self, *, run_id: str | None = None) -> dict[str, Any]:
        rows = self.db.rows(
            "SELECT status, COUNT(*) AS n FROM entity_links"
            + (" WHERE run_id=?" if run_id else "") + " GROUP BY status",
            (run_id,) if run_id else ())
        return {r["status"]: r["n"] for r in rows}
