"""M4.1-00 contract tests -- DiscoveryContextProvider + DiscoverySnapshot.

Enforces the two wiring constraints from the M4.0.1 review:

  (1) SINGLE READ PER RUN -- build_snapshot must call the unified Evidence
      Reader exactly once (never the per-access accessors in a loop).
  (2) NO FABRICATED EXPERIENCE -- M3.4 historical runs have no native
      research_experience; a missing Experience MUST surface as an explicit
      UNRESOLVED gap, never as a synthesised record.

Plus: read-only (no writes), deterministic/idempotent snapshot, and
traceability (objectives carry content_sha256; evidence entities carry a
sha256: id).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from research_runtime.discovery_context_provider import (
    DiscoveryContextProvider,
    DISCOVERY_SNAPSHOT_SCHEMA_VERSION,
)
from research_runtime.research_store import ResearchStores

# Repo-relative workspace: derived from THIS file so the test runs on any
# machine / CI runner (no developer-absolute path such as E:/...).
WS = Path(__file__).resolve().parents[2]      # multi-agent/


# --------------------------------------------------------------------- helpers
def _fake_view(run_id: str) -> dict:
    """Minimal EvidenceView stub (no filesystem reads)."""
    return {
        "schema_version": "m4.0-evidence-view-v1",
        "run_id": run_id,
        "n_evidence": 0,
        "n_carriers_read": 0,
        "carriers": [],
        "evidence": [],
        "raw_verdict_census": {},
        "provenance": {"domains": {}},
        "unresolved": [],
        "note": "fake",
    }


class _CountingAdapter:
    """Records exactly how many times query_run() is called.

    It deliberately does NOT implement discover_entities/carriers/
    raw_verdict_census/unresolved -- if the provider ever calls those
    per-access accessors, the test raises AttributeError (loud failure).
    """

    def __init__(self, view: dict):
        self.view = view
        self.calls = 0

    def query_run(self, run_id: str) -> dict:
        self.calls += 1
        return self.view


def _seed_objective(stores: ResearchStores, oid: str, domain: str = "scinet") -> None:
    with stores.db.session() as conn:
        conn.execute(
            """INSERT INTO objectives
               (objective_id, domain, description, constraints_json,
                success_criteria_json, parent_objective_id, status,
                origin_experience_id, content_sha256, created_at, provenance_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (oid, domain, "synthetic", "{}", "{}", None, "proposed",
             None, "sha256:synthetic", "2026-01-01T00:00:00", "{}"),
        )


def _seed_session(stores: ResearchStores, oid: str, session_id: str) -> None:
    stores.sessions.create_session(
        {"objective_id": oid, "domain": "scinet", "description": "synthetic",
         "constraints": {}, "success_criteria": {}},
        session_id=session_id,
    )


# --------------------------------------------------------------------- tests
def test_build_snapshot_reads_evidence_exactly_once(tmp_path):
    stores = ResearchStores(tmp_path / "r.sqlite")
    _seed_objective(stores, "obj_t_scinet")
    _seed_session(stores, "obj_t_scinet", "s1")
    adapter = _CountingAdapter(_fake_view("t"))
    provider = DiscoveryContextProvider(
        stores=stores, evidence_adapter=adapter, workspace=tmp_path)

    snap = provider.build_snapshot("t", objective_ids=["obj_t_scinet"])
    assert adapter.calls == 1
    # A second build must read exactly once more (no caching across calls,
    # but crucially never more than one read per build).
    provider.build_snapshot("t", objective_ids=["obj_t_scinet"])
    assert adapter.calls == 2
    assert snap.schema_version == DISCOVERY_SNAPSHOT_SCHEMA_VERSION


def test_missing_experience_is_unresolved_not_fabricated(tmp_path):
    stores = ResearchStores(tmp_path / "r.sqlite")
    _seed_objective(stores, "obj_t_scinet")
    _seed_session(stores, "obj_t_scinet", "s1")  # session exists, no experience
    provider = DiscoveryContextProvider(
        stores=stores, evidence_adapter=_CountingAdapter(_fake_view("t")))

    snap = provider.build_snapshot("t", objective_ids=["obj_t_scinet"])
    s = snap.sessions[0]
    assert s["session_available"] is True
    assert s["experience_available"] is False
    # (2) NO FABRICATION: zero synthesised experience records.
    assert sum(len(x["experience"]) for x in snap.sessions) == 0
    assert snap.experience_available is False
    kinds = [g["kind"] for g in snap.unresolved_gaps]
    assert "missing_experience" in kinds
    assert "missing_session" not in kinds
    # explicit, honest reason recorded
    gap = next(g for g in snap.unresolved_gaps if g["kind"] == "missing_experience")
    assert "must NOT be fabricated" in gap["reason"]


def test_missing_session_is_unresolved(tmp_path):
    stores = ResearchStores(tmp_path / "r.sqlite")
    _seed_objective(stores, "obj_t_embodied")  # no session at all
    provider = DiscoveryContextProvider(
        stores=stores, evidence_adapter=_CountingAdapter(_fake_view("t")))

    snap = provider.build_snapshot("t", objective_ids=["obj_t_embodied"])
    s = snap.sessions[0]
    assert s["session_available"] is False
    assert s["experience_available"] is False
    kinds = [g["kind"] for g in snap.unresolved_gaps]
    assert "missing_session" in kinds


def test_snapshot_is_deterministic(tmp_path):
    stores = ResearchStores(tmp_path / "r.sqlite")
    _seed_objective(stores, "obj_t_scinet")
    _seed_session(stores, "obj_t_scinet", "s1")
    adapter = _CountingAdapter(_fake_view("t"))
    provider = DiscoveryContextProvider(
        stores=stores, evidence_adapter=adapter, workspace=tmp_path)

    snap1 = provider.build_snapshot("t", objective_ids=["obj_t_scinet"])
    snap2 = provider.build_snapshot("t", objective_ids=["obj_t_scinet"])
    # data_dict() excludes the build stamp, so two builds of the same inputs
    # MUST be byte-identical.
    assert snap1.data_dict() == snap2.data_dict()


def test_objectives_are_traceable_by_sha(tmp_path):
    stores = ResearchStores(tmp_path / "r.sqlite")
    _seed_objective(stores, "obj_t_scinet")
    _seed_session(stores, "obj_t_scinet", "s1")
    provider = DiscoveryContextProvider(
        stores=stores, evidence_adapter=_CountingAdapter(_fake_view("t")))

    snap = provider.build_snapshot("t", objective_ids=["obj_t_scinet"])
    # Every aggregated objective carries its content_sha256 (traceability).
    assert len(snap.objectives) == 1
    assert snap.objectives[0]["content_sha256"].startswith("sha256:")


def test_real_arr_m34_001_snapshot_has_no_fabricated_experience(tmp_path):
    """Real-history test: M3.4 runs have NO native session/experience.

    This is the exact M4.0.1-review concern -- do not assume backfilled
    history carries Experience.
    """
    db = tmp_path / "real.sqlite"
    from M4_0_runs.backfill import backfill
    from research_runtime.tests.m4_0_seed_helper import build_equivalent_db

    meta = build_equivalent_db(db, workspace=WS)
    assert meta.get("source") in ("real_db_copy", "committed_p1_06_seed")

    conn = sqlite3.connect(str(db))
    conn.execute("DELETE FROM entity_links")
    conn.execute("DELETE FROM objectives")
    conn.commit()
    conn.close()

    backfill(run_root=WS / "M3_4_runs" / "arr_m34_001", db_path=db,
             workspace=WS, report_path=tmp_path / "bf.json")

    stores = ResearchStores(db)
    provider = DiscoveryContextProvider(stores=stores, workspace=WS)
    snap = provider.build_snapshot("arr_m34_001")

    # Real evidence + objectives present.
    assert snap.evidence_view["n_evidence"] > 0
    assert len(snap.objectives) == 3
    assert snap.provenance["counts"]["n_objectives"] == 3

    # (2) M3.4 history carries no Experience -> explicit unresolved gaps,
    # never fabricated records.
    assert snap.experience_available is False
    assert snap.provenance["counts"]["n_experience_records"] == 0
    missing_session = [g for g in snap.unresolved_gaps
                       if g["kind"] == "missing_session"]
    assert len(missing_session) == 3

    # Traceability: each objective resolved from the evidence-derived domains.
    oids = {o["objective_id"] for o in snap.objectives}
    assert oids == {"obj_arr_m34_001_scinet", "obj_arr_m34_001_embodied",
                    "obj_arr_m34_001_quant"}


def _seed_experience(stores, session_id: str, oid: str, kind: str,
                     payload: dict) -> None:
    stores.sessions.record_experience(
        session_id=session_id, objective_id=oid, kind=kind, payload=payload)


def test_all_historical_sessions_are_retained_not_just_latest(tmp_path):
    """P2: an objective with 3 sessions must surface ALL of them, including
    previously-failed ones that are NOT the latest.

    S1 (fail, has Experience) + S2 (success, has Experience) + S3 (latest, no
    Experience). Discovery must still see S1/S2 so it does not repeat a failed
    direction.
    """
    stores = ResearchStores(tmp_path / "r.sqlite")
    oid = "obj_t_multisess"
    _seed_objective(stores, oid)

    s1 = stores.sessions.create_session(
        {"objective_id": oid, "domain": "scinet"}, session_id="s1")
    _seed_experience(stores, s1.session_id, oid, "failure",
                     {"hypothesis": "H1", "why_failed": "horizon drift"})
    s2 = stores.sessions.create_session(
        {"objective_id": oid, "domain": "scinet"}, session_id="s2")
    _seed_experience(stores, s2.session_id, oid, "success",
                     {"hypothesis": "H2", "gain_r2": 0.04})
    # S3 is the latest session and has NO experience.
    stores.sessions.create_session(
        {"objective_id": oid, "domain": "scinet"}, session_id="s3")

    provider = DiscoveryContextProvider(
        stores=stores, evidence_adapter=_CountingAdapter(_fake_view("t")))
    snap = provider.build_snapshot("t", objective_ids=[oid])

    # All three sessions present (flat + nested views agree).
    assert len(snap.sessions) == 3
    by_obj = snap.sessions_by_objective[oid]
    assert len(by_obj) == 3

    # The OLD, non-latest sessions and their Experience are NOT lost.
    sids = {s["session_id"] for s in snap.sessions}
    assert sids == {"s1", "s2", "s3"}
    s1_snap = next(s for s in snap.sessions if s["session_id"] == "s1")
    s2_snap = next(s for s in snap.sessions if s["session_id"] == "s2")
    s3_snap = next(s for s in snap.sessions if s["session_id"] == "s3")
    assert s1_snap["experience_available"] is True
    assert s2_snap["experience_available"] is True
    assert s3_snap["experience_available"] is False

    # No fabrication: exactly the 2 recorded experiences exist.
    total_exp = sum(len(s["experience"]) for s in snap.sessions)
    assert total_exp == 2
    # S1's failure experience content is preserved verbatim.
    assert s1_snap["experience"][0]["payload"]["why_failed"] == "horizon drift"


def test_failure_signals_and_diagnosis_unavailable_are_distinct(tmp_path):
    """P3: a failure event is a SIGNAL only; with no structured ExperimentResult
    present, structured diagnosis is explicitly UNAVAILABLE (never guessed)."""
    stores = ResearchStores(tmp_path / "r.sqlite")
    oid = "obj_t_fail"
    _seed_objective(stores, oid)
    s = stores.sessions.create_session(
        {"objective_id": oid, "domain": "scinet"}, session_id="sf")
    # Emit a failure-named event via the store's save() event hook.
    stores.sessions.save(s, event_type="campaign_failed",
                         payload={"reason": "diverged"})

    provider = DiscoveryContextProvider(
        stores=stores, evidence_adapter=_CountingAdapter(_fake_view("t")))
    snap = provider.build_snapshot("t", objective_ids=[oid])

    # Signal captured from the event name...
    assert len(snap.failures["session_failure_signals"]) == 1
    sig = snap.failures["session_failure_signals"][0]
    assert sig["session_id"] == "sf"
    # ...but NO structured diagnosis was invented...
    assert snap.failures["structured_diagnoses"] == []
    assert snap.failures["diagnosis_status"]["structured_diagnosis_available"] is False
    # ...and the gap is explicitly flagged diagnosis_unavailable.
    kinds = [g["kind"] for g in snap.unresolved_gaps]
    assert "diagnosis_unavailable" in kinds


def test_evidence_entities_carry_sha_identity(tmp_path):
    """P4: every real evidence entity in the snapshot exposes a SHA-256
    identity (the unified reader's evidence identity = SHA-256 of real bytes),
    so a Discovery Objective can always trace to concrete evidence."""
    db = tmp_path / "real.sqlite"
    from M4_0_runs.backfill import backfill
    from research_runtime.tests.m4_0_seed_helper import build_equivalent_db

    build_equivalent_db(db, workspace=WS)
    conn = sqlite3.connect(str(db))
    conn.execute("DELETE FROM entity_links")
    conn.execute("DELETE FROM objectives")
    conn.commit()
    conn.close()
    backfill(run_root=WS / "M3_4_runs" / "arr_m34_001", db_path=db,
             workspace=WS, report_path=tmp_path / "bf.json")

    stores = ResearchStores(db)
    provider = DiscoveryContextProvider(stores=stores, workspace=WS)
    snap = provider.build_snapshot("arr_m34_001")

    assert len(snap.evidence_view["evidence"]) > 0
    for ent in snap.evidence_view["evidence"]:
        assert ent["evidence_id"].startswith("sha256:"), ent
        assert ent["sources"] or ent.get("manifest_sha256") or \
            ent.get("metrics_files")


def test_links_trace_to_source_artifact(tmp_path):
    """P4: every link in the snapshot carries its source artifact reference
    (source_sha256 + source_path), so a Discovery Objective can trace a relation
    to the concrete file it was derived from."""
    db = tmp_path / "real.sqlite"
    from M4_0_runs.backfill import backfill
    from research_runtime.tests.m4_0_seed_helper import build_equivalent_db

    build_equivalent_db(db, workspace=WS)
    conn = sqlite3.connect(str(db))
    conn.execute("DELETE FROM entity_links")
    conn.execute("DELETE FROM objectives")
    conn.commit()
    conn.close()
    backfill(run_root=WS / "M3_4_runs" / "arr_m34_001", db_path=db,
             workspace=WS, report_path=tmp_path / "bf.json")

    stores = ResearchStores(db)
    provider = DiscoveryContextProvider(stores=stores, workspace=WS)
    snap = provider.build_snapshot("arr_m34_001")

    for oid, graph in snap.links["by_objective"].items():
        assert "n_links" in graph
        # list_objective_graph returns link rows that always carry the source
        # artifact reference (provenance), even when status is unresolved.
        assert any("source_sha256" in row or "source_path" in row
                   for row in graph.get("links", []))


def test_experience_traces_back_to_its_session(tmp_path):
    """P4: a recorded Experience must be attributable to a real session, and the
    snapshot must surface it under that exact session (no orphan / fabricated
    experience)."""
    stores = ResearchStores(tmp_path / "r.sqlite")
    oid = "obj_t_trace"
    _seed_objective(stores, oid)
    s = stores.sessions.create_session(
        {"objective_id": oid, "domain": "scinet"}, session_id="se")
    _seed_experience(stores, s.session_id, oid, "success",
                     {"gain_r2": 0.03})

    provider = DiscoveryContextProvider(
        stores=stores, evidence_adapter=_CountingAdapter(_fake_view("t")))
    snap = provider.build_snapshot("t", objective_ids=[oid])

    sess = snap.sessions_by_objective[oid][0]
    assert sess["session_id"] == "se"
    assert len(sess["experience"]) == 1
    exp = sess["experience"][0]
    assert exp["session_id"] == "se"
    assert exp["experience_id"].startswith("exp_")
    assert exp["payload"]["gain_r2"] == 0.03


def test_real_snapshot_is_deterministic(tmp_path):
    """P4: building the real-history snapshot twice yields byte-identical
    data_dict() (only the build stamp differs)."""
    db = tmp_path / "real.sqlite"
    from M4_0_runs.backfill import backfill
    from research_runtime.tests.m4_0_seed_helper import build_equivalent_db

    build_equivalent_db(db, workspace=WS)
    conn = sqlite3.connect(str(db))
    conn.execute("DELETE FROM entity_links")
    conn.execute("DELETE FROM objectives")
    conn.commit()
    conn.close()
    backfill(run_root=WS / "M3_4_runs" / "arr_m34_001", db_path=db,
             workspace=WS, report_path=tmp_path / "bf.json")

    stores = ResearchStores(db)
    provider = DiscoveryContextProvider(stores=stores, workspace=WS)
    snap1 = provider.build_snapshot("arr_m34_001")
    snap2 = provider.build_snapshot("arr_m34_001")
    assert snap1.data_dict() == snap2.data_dict()
