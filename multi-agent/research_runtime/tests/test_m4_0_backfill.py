"""M4.0-03 -- backfill idempotency / read-only / unresolved tests.

Always operates on a COPY of the real DB + the REAL read-only historical files.
Never writes to M3_4_runs/arr_m34_001.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from M4_0_runs.backfill import (
    P106_RUN_ID, backfill, historical_file_list, sha_snapshot,
)
from research_runtime.research_store import ResearchStores, database_snapshot
from research_runtime.research_store.entity_link_store import (
    UNRESOLVED_SENTINEL, EntityLinkStore,
)

WS = Path(__file__).resolve().parents[2]      # multi-agent/
sys_path_root = WS
if str(sys_path_root) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(sys_path_root))
RUN_ROOT = WS / "M3_4_runs" / "arr_m34_001"
REAL_DB = WS / "research_runtime" / "_data" / "research.sqlite"


@pytest.fixture()
def env(tmp_path: Path):
    db = tmp_path / "research.sqlite"
    shutil.copy2(REAL_DB, db)
    # The real DB has already been backfilled once. Tests must start from a
    # CLEAN relation layer while KEEPING the P1-06 historical rows, so the
    # assertions are independent of the operator's local DB state.
    conn = sqlite3.connect(str(db))
    conn.execute("DELETE FROM entity_links")
    conn.execute("DELETE FROM objectives")
    conn.commit()
    conn.close()
    return tmp_path, db


def _run(env, name: str):
    tmp, db = env
    return backfill(run_root=RUN_ROOT, db_path=db, workspace=WS,
                    report_path=tmp / f"{name}.json")


def test_backfill_is_idempotent_no_new_links(env):
    r1 = _run(env, "pass1")
    r2 = _run(env, "pass2")
    assert sum(r1["links_created"].values()) > 0
    assert all(v == 0 for v in r2["links_created"].values())
    assert sum(r2["links_duplicated_on_repeat"].values()) == \
        sum(r1["links_created"].values())
    tmp, db = env
    n = sqlite3.connect(str(db)).execute(
        "SELECT COUNT(*) FROM entity_links").fetchone()[0]
    assert n == sum(r1["links_created"].values())


def test_historical_files_are_never_modified(env):
    files = historical_file_list(RUN_ROOT, WS)
    before = sha_snapshot(files, WS)
    _run(env, "x1")
    _run(env, "x2")
    after = sha_snapshot(files, WS)
    assert before == after
    assert len(before) == len(files)      # all 89 files hashed, none skipped


def test_unresolved_relations_are_explicit_and_complete(env):
    r = _run(env, "u")
    # 3 domains x 3 candidates x 4 missing entity types = 36
    assert r["n_unresolved"] == 36
    assert all("no explicit" in x["reason"] for x in r["unresolved"])
    assert all(x["missing_entity"] in
               ("hypothesis", "experiment_plan", "research_session", "experience")
               for x in r["unresolved"])
    tmp, db = env
    links = EntityLinkStore(ResearchStores(db).db, workspace=WS)
    v = links.validate_links(workspace=WS)
    assert v["ok"] is True
    assert v["n_unresolved"] == 36
    rows = sqlite3.connect(str(db)).execute(
        "SELECT COUNT(*) FROM entity_links WHERE status='unresolved' "
        "AND dst_id=?", (UNRESOLVED_SENTINEL,)).fetchone()[0]
    assert rows == 36


def test_no_guessed_entity_ids(env):
    tmp, db = env
    _run(env, "g")
    ids = [r[0] for r in sqlite3.connect(str(db)).execute(
        "SELECT DISTINCT dst_id FROM entity_links")]
    # No relation was inferred from a filename or similar text.
    assert not any("pendulum_smoke" in str(i) and "sha256:" not in str(i)
                   for i in ids)
    assert all((i.startswith("sha256:") or i == UNRESOLVED_SENTINEL
                or i.startswith(("obj_", "candidate:", "H_HORIZON", "plan_",
                                 "campaign_", "session_", "exp_",
                                 "arr_m34_001_")))
               for i in ids), [i for i in ids if not (
                   i.startswith("sha256:") or i == UNRESOLVED_SENTINEL)]


def test_cross_run_isolation_in_backfilled_data(env):
    tmp, db = env
    _run(env, "iso")
    links = EntityLinkStore(ResearchStores(db).db, workspace=WS)
    a = links.find_by_objective("obj_arr_m34_001_scinet", run_id="arr_m34_001")
    b = links.find_by_objective("obj_pendulum_multistep", run_id=P106_RUN_ID)
    # Each link lives in exactly ONE run namespace.
    p106 = [l for l in links.db.rows(
        "SELECT * FROM entity_links WHERE run_id=?", (P106_RUN_ID,))]
    assert len(p106) == 10 and len(b) == 10
    assert all(l["run_id"] == "arr_m34_001" for l in a)
    assert all(l["run_id"] == P106_RUN_ID for l in b)
    # Querying the P1-06 objective inside the M3.4 run returns NOTHING.
    assert links.find_by_objective("obj_pendulum_multistep",
                                   run_id="arr_m34_001") == []
    assert not any(l["run_id"] == P106_RUN_ID for l in a)
    assert not any(l["objective_id"] == "obj_arr_m34_001_scinet" for l in b)


def test_all_evidence_links_have_verifiable_source_sha(env):
    tmp, db = env
    r = _run(env, "sha")
    assert r["link_validation"]["ok"] is True
    assert r["link_validation"]["n_sha_checked"] > 80
    rows = sqlite3.connect(str(db)).execute(
        "SELECT source_path, source_sha256 FROM entity_links "
        "WHERE source_sha256 IS NOT NULL").fetchall()
    for path, digest in rows:
        p = WS / path
        assert p.is_file(), path
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest


def test_backfill_preserves_existing_db_rows(env):
    tmp, db = env
    before = database_snapshot(db)["statistics"]
    r = _run(env, "pres")
    after = database_snapshot(db)["statistics"]
    for table, n in before.items():
        if table not in ("entity_links", "objectives", "sqlite_sequence"):
            assert after[table] == n, f"historical table {table} changed"
    assert r["historical_files_unchanged"] is True


def test_source_paths_are_posix_paths(env):
    tmp, db = env
    _run(env, "posix")
    rows = sqlite3.connect(str(db)).execute(
        "SELECT source_path FROM entity_links WHERE source_path IS NOT NULL"
    ).fetchall()
    assert rows
    assert all("\\" not in p[0] for p in rows)
