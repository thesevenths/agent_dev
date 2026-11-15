"""M4-E2E-01C -- REAL SciNet++ engine end-to-end (OPT-IN, skipped in CI).

This is the "no FakeRunner" path: it assembles the REAL ``ResearchService`` via
``build_default_research_service("scinet")`` (real SciNetAdapter + CampaignRunner
+ comparator) and drives the Director through the full chain, writing a REAL
Experience. It is deliberately NOT part of the engine-free CI gate:

  * it needs a SciNet++ checkout (``settings.DEFAULT_SCINET_DIR``),
  * it trains models (minutes-to-hours, depends on the on-disk dataset),
  * it is opt-in via ``M4_E2E_01_RUN_REAL=1``.

Per M4-E2E-01 engineering discipline, the engine-free suite (test_m4_e2e_01_*)
proves the state machine / approval / persistence / recovery and the Experience
writeback *mechanism* without any engine; this file proves the SAME Director
drives the REAL engine unchanged. Run it manually:

    M4_E2E_01_RUN_REAL=1 python -m pytest \\
        research_runtime/tests/test_m4_e2e_01_real_engine.py -q
"""
from __future__ import annotations

import os

import pytest

from research_runtime.discovery_context_provider import DiscoverySnapshot
from research_runtime.objective_discovery import ObjectiveDiscovery
from research_runtime.research_director import ResearchDirector
from research_runtime.research_environment import build_default_research_service
from research_runtime.settings import DEFAULT_SCINET_DIR
from research_runtime.research_store import ResearchStores

# ---- gating ---------------------------------------------------------------
_SCIENT_PRESENT = bool(DEFAULT_SCINET_DIR) and os.path.isdir(str(DEFAULT_SCINET_DIR))
_RUN_REAL = os.environ.get("M4_E2E_01_RUN_REAL") == "1"

pytestmark = pytest.mark.skipif(
    not (_SCIENT_PRESENT and _RUN_REAL),
    reason=("real SciNet++ E2E is opt-in: requires a SciNet++ checkout and "
            "M4_E2E_01_RUN_REAL=1 (trains models; not for CI)"))


def _make_snapshot(run_id: str) -> DiscoverySnapshot:
    evidence_view = {
        "schema_version": "m4.0-evidence-view-v1",
        "run_id": run_id,
        "n_evidence": 1,
        "evidence": [{"evidence_id": "sha256:neg1",
                      "verdicts": ["rejected"], "sources": []}],
        "provenance": {"domains": {"scinet": {"ood_dataset_version": "ood_v2"}}},
        "unresolved": [],
    }
    return DiscoverySnapshot(
        run_id=run_id,
        schema_version="m4.1-00-discovery-snapshot-v1",
        generated_at="2026-01-01T00:00:00",
        evidence_view=evidence_view, objectives=[],
        links={"run_stats": {}, "by_objective": {}},
        sessions=[], sessions_by_objective={}, experience_available=False,
        failures={"session_failure_signals": [],
                  "evidence_unresolved": [], "structured_diagnoses": []},
        provenance={"read_only": True, "run_id": run_id}, unresolved_gaps=[])


def test_real_engine_full_chain_writes_experience(tmp_path):
    # REAL service: real SciNet++ adapter / runner / comparator (no fakes).
    service = build_default_research_service(
        "scinet", db_path=tmp_path / "research.sqlite")

    # Discovery on a real-shaped snapshot -> a Proposed Objective.
    store = ResearchStores(tmp_path / "objectives.sqlite")
    res = ObjectiveDiscovery(domain="scinet").discover(
        _make_snapshot("real_run"), objective_store=store.objectives,
        domain="scinet")
    assert res.proposed_objective_ids, "Discovery must produce a Proposed Objective"
    oid = res.proposed_objective_ids[0]

    director = ResearchDirector(
        service=service, objective_store=store.objectives,
        state_dir=tmp_path / "director", objective_id=oid,
        source_experiment_id="exp_baseline")

    # Human grant is applied out-of-band; execute re-verifies the store (no bypass).
    final = director.run_to_completion(approved_by="human", source="human")
    assert final["state"] == "COMPLETED"

    rep = service.get_research_report(session_id=director.context.session_id)
    assert rep["ok"], rep
    report = rep["report"]
    assert report["experience_refs"], "REAL Experience must be recorded"
    assert report["frozen_test_used"] is False
    assert report["promotion"] == "not_requested"
