"""M4.3-01.1 P1-03 -- REAL SciNet++ A/B/C engine E2E (OPT-IN, skipped in CI).

This is the ONLY P1-03 test that claims a REAL ENGINE run. It NEVER uses a
FakeRunner and NEVER falls back to a synthetic Snapshot: when the environment
is incomplete it SKIPS explicitly (never a fabricated pass).

Ten spec points, all with REAL persisted entities:
  1. REAL DiscoveryContextProvider over REAL historical Evidence.
  2. REAL A/B/C Objectives (three distinct persisted Objectives from the SAME
     real Snapshot / research opportunity).
  3. REAL out-of-band Approval via the separate ``approval_cli`` SUBPROCESS.
  4. ONE shared ``ResearchService`` (single SQLite db) for all three arms.
  5. REAL SciNet++ Engine (models are actually trained).
  6. REAL ResearchReport + Experience Store writeback.
  7. REAL Registry persistence (P0-03 append-only tables in the SAME db).
  8. REAL process-restart recovery (a genuine SUBPROCESS reopens the db).
  9. ROUND-2 Discovery reads the round-1 Experience.
 10. A COMPLETE E2E evidence bundle is saved (JSON, integrity-verifiable).

Gating (ALL required, otherwise an explicit SKIP):
  * ``M4_3_01_1_RUN_REAL=1`` (or the legacy ``M4_E2E_01_RUN_REAL=1``),
  * a SciNet++ checkout at ``settings.DEFAULT_SCINET_DIR``,
  * REAL historical Evidence (``n_evidence > 0`` from the real Provider).

Run manually (trains models for all three arms; minutes to hours)::

    M4_3_01_1_RUN_REAL=1 python -m pytest \\
        research_runtime/tests/test_m4_3_01_1_p1_03_real_engine.py -q -rs

ENGINE-FREE tests (P0-01/P0-02/P0-03/P1-01/P1-02) prove the orchestration,
gate, blinding, persistence and recovery MECHANISMS without an engine. This
file proves the same harness drives the REAL engine. The two are reported
SEPARATELY; no engine-free test claims a real-engine pass.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from research_runtime.discovery_context_provider import DiscoveryContextProvider
from research_runtime.objective_candidate_generator import ObjectiveCandidateGenerator
from research_runtime.objective_critic import ObjectiveCritic, _resolve_ref
from research_runtime.objective_discovery import ObjectiveDiscovery
from research_runtime.objective_ranker import ObjectiveRanker
from research_runtime.research_director import ResearchDirector
from research_runtime.research_environment import build_default_research_service
from research_runtime.research_store import ResearchStores
from research_runtime.settings import DEFAULT_SCINET_DIR

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_RUN_ID = "arr_m34_001"
SOURCE_EXPERIMENT_ID = "p0_demo_20261109"
ARMS = ("A", "B", "C")


# ------------------------------------------------------------------- gating
def _real_evidence_available() -> tuple[bool, str]:
    """READ-ONLY probe with the SAME real Provider used by the E2E itself."""
    try:
        probe = Path(tempfile.mkdtemp()) / "probe.sqlite"
        view = DiscoveryContextProvider(
            stores=ResearchStores(probe)).build_snapshot(
                HISTORICAL_RUN_ID, objective_ids=[]).evidence_view
        n = int(view.get("n_evidence") or 0)
        return (True, f"{n} real evidence entities") if n > 0 else \
            (False, f"no real evidence for {HISTORICAL_RUN_ID!r}")
    except Exception as exc:  # noqa: BLE001 - gating probe must never raise
        return False, f"probe failed: {type(exc).__name__}: {exc}"


def gate_state() -> dict:
    """Testable gate decision: ALL three conditions are mandatory."""
    scinet = bool(DEFAULT_SCINET_DIR) and os.path.isdir(str(DEFAULT_SCINET_DIR))
    optin = (os.environ.get("M4_3_01_1_RUN_REAL") == "1"
             or os.environ.get("M4_E2E_01_RUN_REAL") == "1")
    ev_ok, ev_reason = _real_evidence_available()
    return {"scinet_checkout": scinet, "operator_optin": optin,
            "real_historical_evidence": ev_ok, "evidence_reason": ev_reason,
            "ready": bool(scinet and optin and ev_ok)}


_GATE = gate_state()
SKIP_REASON = (
    "M4.3-01.1 P1-03 REAL engine E2E is opt-in and requires ALL of: "
    "M4_3_01_1_RUN_REAL=1, a SciNet++ checkout, and REAL historical Evidence "
    f"(checkout={_GATE['scinet_checkout']}, optin={_GATE['operator_optin']}, "
    f"real evidence={_GATE['evidence_reason']}). "
    "Explicit SKIP -- NEVER falls back to a FakeRunner or synthetic Snapshot.")


def imported_names(tree) -> set[str]:
    """All names bound by Import / ImportFrom nodes in a module AST."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update((a.asname or a.name.split(".")[0]) for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update((a.asname or a.name) for a in node.names)
    return names
    """Genuine OUT-OF-BAND human grant via the separate console PROCESS."""
    r = subprocess.run(
        [sys.executable, "-m", "research_runtime.research_store.approval_cli",
         "--db", str(db_path), "grant", approval_id,
         "--approved-by", "alice", "--yes"],
        cwd=str(ROOT), capture_output=True, text=True)
    assert r.returncode == 0, (
        f"approval_cli grant failed rc={r.returncode}\n"
        f"STDOUT={r.stdout}\nSTDERR={r.stderr}")


def _subprocess_recover(db_path: Path) -> dict:
    """(P1-03 #8) A GENUINE separate process reopens the SAME db."""
    code = (
        "import json,sys\n"
        "from research_runtime.m4_3.registry_store import RegistryStore\n"
        f"s=RegistryStore(db_path=r'{db_path}')\n"
        "print(json.dumps({'integrity':s.verify_integrity(),"
        "'n_runs':len(s.load_runs()),'n_blocks':len(s.load_blocks()),"
        "'n_reviews':len(s.load_reviews()),'n_events':len(s.load_events())}))\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                       capture_output=True, text=True)
    assert r.returncode == 0, (
        f"subprocess recovery failed rc={r.returncode}\n"
        f"STDOUT={r.stdout}\nSTDERR={r.stderr}")
    return json.loads(r.stdout.strip().splitlines()[-1])


# ---------------------------------------------- engine-free gating contracts
def test_gate_requires_all_three_conditions_and_never_falls_back():
    """ENGINE-FREE meta-test: the real-engine gate is honest."""
    import ast
    g = gate_state()
    assert set(g) >= {"scinet_checkout", "operator_optin",
                      "real_historical_evidence", "ready"}
    assert g["ready"] == (g["scinet_checkout"] and g["operator_optin"]
                          and g["real_historical_evidence"])
    assert "FakeRunner" in SKIP_REASON and "synthetic" in SKIP_REASON
    # AST-level proof: this module imports NO test double / fake runner and
    # defines NO synthetic snapshot helper. (String matching the source would
    # self-match on its own assertion, so inspect the imports instead.)
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    defined: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined.add(node.name)
    assert not any("research_fakes" in m or "campaign_fakes" in m
                   or "fake" in m.lower() for m in imported)
    assert not any("fake" in d.lower() for d in defined)
    assert "build_default_research_service" in imported_names(tree)


def test_real_evidence_probe_is_read_only_and_real():
    """ENGINE-FREE: the gate uses the REAL Provider, not a fabricated view."""
    ok, reason = _real_evidence_available()
    assert isinstance(ok, bool) and reason
    if ok:
        assert "real evidence entities" in reason


@pytest.mark.skipif(not _GATE["ready"], reason=SKIP_REASON)
def test_real_engine_abc_e2e(tmp_path):
    """The FULL 10-point REAL engine E2E. Skipped unless ALL gates pass."""
    from research_runtime.m4_3 import (
        ARMS as HARNESS_ARMS, BlockSpec, EvaluationHarness, FrozenProtocol,
        IndependentReviewer, ReviewPanel,
    )
    from research_runtime.m4_3.registry_store import RegistryStore
    from research_runtime.m4_3.research_service_gate import (
        GateRunRef, ResearchServiceGateAdapter,
    )
    from research_runtime.m4_3.metrics import frozen_metric_contract

    assert HARNESS_ARMS == ARMS

    # (#1/#4) ONE db, ONE service, ONE stores instance for EVERY entity.
    db_path = tmp_path / "research.sqlite"
    service = build_default_research_service("scinet", db_path=db_path)
    stores = service.stores
    provider = DiscoveryContextProvider(stores=stores)
    snapshot1 = provider.build_snapshot(HISTORICAL_RUN_ID)
    assert snapshot1.evidence_view.get("n_evidence", 0) > 0
    assert snapshot1.provenance.get("read_only") is True

    # (#2) REAL A/B/C Objectives from the SAME real Snapshot.
    discovery = ObjectiveDiscovery(
        generator=ObjectiveCandidateGenerator(domain="scinet"),
        critic=ObjectiveCritic(), ranker=ObjectiveRanker(), domain="scinet")
    res = discovery.discover(
        snapshot1, objective_store=stores.objectives, domain="scinet")
    assert len(res.proposed_objective_ids) >= 3, \
        "need three REAL persisted Objectives for A/B/C"
    oids = {arm: res.proposed_objective_ids[i] for i, arm in enumerate(ARMS)}
    assert len(set(oids.values())) == 3
    for oid in oids.values():
        assert stores.objectives.get(oid) is not None
    assert all("sha256:neg1" not in r
               for c in res.ranked for r in c["evidence_refs"])

    # P0-01: ONE real read-only Gate adapter over the SAME service.
    adapter = ResearchServiceGateAdapter(service)

    # P0-02: a 2-reviewer panel. The review is an ORCHESTRATION placeholder,
    # NOT a human quality judgement and NOT used for any value conclusion.
    def _orchestration_scorer(_b):
        return {d: 2 for d in frozen_metric_contract()["rubric_dimensions"]} | {
            "reason": ("E2E orchestration placeholder; NOT a human quality "
                       "judgement and NOT used for a value conclusion")}
    panel = ReviewPanel([
        IndependentReviewer(reviewer_id="e2e_reviewer_1",
                            scorer=_orchestration_scorer),
        IndependentReviewer(reviewer_id="e2e_reviewer_2",
                            scorer=_orchestration_scorer)])

    protocol = FrozenProtocol(
        snapshot_ids=(HISTORICAL_RUN_ID,), sample_size=0, seed=42,
        model_version="scinet-real", prompt_version="discovery-v1",
        ranker_version="deterministic-v1",
        budget={"paired_seeds": [42, 43]}, exclusion_rules=("none",),
        statistical_method="paired-block", stopping_rule="all blocks complete")
    # (#7) REAL Registry persistence in the SAME ARR SQLite database.
    harness = EvaluationHarness(
        protocol=protocol, review_panel=panel, gate_adapter=adapter,
        db=stores.db)
    assert harness.registry.store.db is stores.db
    harness.assert_pilot_review_ready()

    # (#3/#5/#6) three REAL arm chains: real engine + out-of-band approval.
    gate_refs: dict[str, GateRunRef] = {}
    session_ids: dict[str, str] = {}
    bid = "real_e2e_block_1"
    harness.add_block(
        BlockSpec(block_id=bid, snapshot_id=HISTORICAL_RUN_ID, seed=42,
                  current_evidence={"evidence_id": "real_snapshot_arr_m34_001",
                                    "n_evidence": snapshot1.evidence_view.get(
                                        "n_evidence", 0)},
                  current_evidence_ids=("real_snapshot_arr_m34_001",)),
        candidate_budget=1, human_time_budget_s=600.0)

    for arm in ARMS:
        oid = oids[arm]
        director = ResearchDirector(
            service=service, objective_store=stores.objectives,
            state_dir=tmp_path / "director" / arm, objective_id=oid,
            source_experiment_id=SOURCE_EXPERIMENT_ID)
        director.run_to_awaiting_approval()
        assert director.context.session_id and director.context.approval_id
        _grant_via_cli(db_path, director.context.approval_id)
        ap = stores.approvals.get(director.context.approval_id)
        assert ap.status == "approved" and ap.source == "human"
        final = director.run_to_completion()      # REAL SciNet++ training
        assert final["state"] == "COMPLETED", final
        ctx = director.context
        session_ids[arm] = ctx.session_id
        # Real Report + Experience writeback (nothing fabricated).
        rep = service.get_research_report(session_id=ctx.session_id)
        assert rep["ok"] and rep["report"]["experience_refs"]
        assert rep["report"]["frozen_test_used"] is False
        assert stores.sessions.experience(ctx.session_id)
        gate_refs[arm] = GateRunRef(
            session_id=ctx.session_id, plan_id=ctx.plan_id,
            approval_id=ctx.approval_id, campaign_id=ctx.campaign_id,
            report_id=ctx.report_id)

        # (#2/#7) register the REAL Objective, then (#P0-01) derive the gate
        # verdict from the REAL persisted records via the shared adapter.
        stored = stores.objectives.get(oid)
        objective = {
            "objective_id": oid,
            "description": (stored.description if stored else
                            "real discovered objective"),
            "evidence_refs": [c["evidence_refs"][0] for c in res.ranked[:1]
                              if c["evidence_refs"]],
        }
        harness.select_objective(bid, arm, objective=objective)
        rec = harness.record_outcome(
            bid, arm, outcome="valid_finding", gate_ref=gate_refs[arm],
            compute_units=1.0, wall_time_s=1.0)
        assert rec.gate_evidence["gate_passed"] is True
        assert rec.gate_evidence["external_approval"] is True
        assert rec.campaign_id == ctx.campaign_id
        assert rec.plan_id == ctx.plan_id and rec.approval_id == ctx.approval_id

    assert len(adapter.inspect_calls) >= 3
    assert harness.registry.complete_blocks() == [bid]
    assert harness.verify_integrity() is True

    # (#8) GENUINE process-restart recovery: a SUBPROCESS reopens the db.
    recovered = _subprocess_recover(db_path)
    assert recovered["integrity"] is True
    assert recovered["n_blocks"] == 1
    assert recovered["n_runs"] == 3
    assert recovered["n_reviews"] >= 6      # 2 reviewers x 3 arms
    assert recovered["n_events"] >= recovered["n_runs"]
    # In-process recovery must see the SAME finalized runs (no duplicate campaign).
    reg2 = __import__(
        "research_runtime.m4_3", fromlist=["ExperimentRegistry"]
    ).ExperimentRegistry.recover(protocol=protocol, db_path=db_path)
    assert len(reg2.all_runs()) == 3
    assert all(r.finalized for r in reg2.all_runs())
    assert {r.campaign_id for r in reg2.all_runs()} == \
        {gate_refs[a].campaign_id for a in ARMS}

    # (#9) ROUND-2 REAL Discovery reads the round-1 REAL Experience.
    snapshot2 = provider.build_snapshot(
        f"{HISTORICAL_RUN_ID}_round2", objective_ids=list(oids.values()))
    for arm in ARMS:
        sid = session_ids[arm]
        assert any(s.get("session_id") == sid and s.get("experience")
                   for s in snapshot2.sessions)
        assert _resolve_ref(f"experience:{sid}", snapshot2) is True
    assert _resolve_ref("experience:fabricated_session", snapshot2) is False
    res2 = discovery.discover(
        snapshot2, objective_store=stores.objectives, domain="scinet")
    cited = {arm: any(f"experience:{sid}" in c["evidence_refs"]
                      for c in res2.ranked)
             for arm, sid in session_ids.items()}
    assert all(cited.values()), "round-2 Discovery must cite round-1 Experience"

    # (#10) Save a COMPLETE, integrity-verifiable E2E evidence bundle.
    bundle = {
        "stage": "M4.3-01.1 P1-03",
        "engine": "REAL SciNet++ (no FakeRunner, no synthetic Snapshot)",
        "gate": gate_state(),
        "one_sqlite_db": str(db_path),
        "historical_run_id": HISTORICAL_RUN_ID,
        "n_round1_evidence": snapshot1.evidence_view.get("n_evidence", 0),
        "objectives": oids,
        "gate_refs": {a: r.to_dict() for a, r in gate_refs.items()},
        "session_ids": session_ids,
        "subprocess_recovery": recovered,
        "round2_experience_cited": cited,
        "harness_snapshot": harness.snapshot(),
        "registry_export": RegistryStore(db_path=db_path).export(),
        "registry_integrity": harness.verify_integrity(),
    }
    evidence_dir = Path(os.environ.get(
        "M4_3_01_1_E2E_EVIDENCE_DIR", str(tmp_path)))
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = evidence_dir / "p1_03_real_e2e_evidence.json"
    evidence_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True, default=str),
        encoding="utf-8")
    assert evidence_path.exists() and evidence_path.stat().st_size > 0
    saved = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert saved["registry_integrity"] is True
    assert saved["engine"].startswith("REAL SciNet++")
    assert set(saved["objectives"]) == set(ARMS)
