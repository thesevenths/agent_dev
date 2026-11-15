"""M4-E2E-01B/C -- Research Director engine-free contract tests.

Drives the FULL Director chain (IDLE -> ... -> COMPLETED) using the SAME
engine-free ``Harness`` that ``test_arr_0_1_e2e`` uses (FakeRunner /
FakeComparator / fake_evidence) -- NO real engine, LLM or model. This proves the
state machine, approval binding, persistence and recovery without SciNet++.

Verifies (spec M4-E2E-01B / M4-E2E-01C):
  * B. The Director reaches AWAITING_APPROVAL by reusing the existing
    ResearchService (no second gate, no second planner).
  * B. Illegal transitions are rejected; approval is bound to the exact Plan SHA
    and execution is refused until a human grants it.
  * B. Each transition is persisted; a "crashed" process reloads its state and
    recovers by reconciling the live engine (no gate bypass).
  * C. The full chain records a real ResearchReport with Experience and never
    promotes or touches the frozen split.
"""
from __future__ import annotations

from pathlib import Path

from research_runtime.discovery_context_provider import DiscoverySnapshot
from research_runtime.objective_discovery import ObjectiveDiscovery
from research_runtime.research_director import DirectorError, ResearchDirector
from research_runtime.director_state_machine import (
    IllegalTransitionError, S_AWAITING_APPROVAL, S_COMPLETED, S_IDLE,
)

from tests.rt_tests.research_fakes import Harness, SOURCE_EXPERIMENT_ID

WS = Path(__file__).resolve().parents[2]      # multi-agent/


# --------------------------------------------------------------------- helpers
def _make_snapshot(run_id: str = "r1") -> DiscoverySnapshot:
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
        evidence_view=evidence_view,
        objectives=[],
        links={"run_stats": {}, "by_objective": {}},
        sessions=[],
        sessions_by_objective={},
        experience_available=False,
        failures={"session_failure_signals": [],
                  "evidence_unresolved": [], "structured_diagnoses": []},
        provenance={"read_only": True, "run_id": run_id},
        unresolved_gaps=[],
    )


def _proposed_objective_id(harness: Harness) -> str:
    snap = _make_snapshot()
    res = ObjectiveDiscovery(domain="scinet").discover(
        snap, objective_store=harness.stores.objectives, domain="scinet")
    assert res.proposed_objective_ids, "Discovery must persist a Proposed Objective"
    return res.proposed_objective_ids[0]


def _granter(harness):
    def _g(aid: str, sha: str) -> None:
        harness.stores.approvals.grant(
            aid, approved_by="alice", source="human",
            current_resource_sha256=sha)
    return _g


# --------------------------------------------------------------------- B: reach AWAITING_APPROVAL
def test_director_reaches_awaiting_approval():
    harness = Harness()
    try:
        oid = _proposed_objective_id(harness)
        director = ResearchDirector(
            service=harness.service,
            objective_store=harness.stores.objectives,
            state_dir=harness.root / "director",
            objective_id=oid, source_experiment_id=SOURCE_EXPERIMENT_ID,
            approval_granter=_granter(harness))

        out = director.run_to_awaiting_approval()
        assert out["state"] == S_AWAITING_APPROVAL
        assert out["approval_id"]
        assert out["plan_id"]
        # The Director reused the existing service rather than re-implementing it.
        assert director.context.session_id
        # At least the four forward transitions were persisted.
        assert len(director.history()) >= 4
    finally:
        harness.close()


# --------------------------------------------------------------------- B: illegal transition rejected
def test_illegal_transition_is_rejected():
    harness = Harness()
    try:
        oid = _proposed_objective_id(harness)
        director = ResearchDirector(
            service=harness.service,
            objective_store=harness.stores.objectives,
            state_dir=harness.root / "director",
            objective_id=oid, source_experiment_id=SOURCE_EXPERIMENT_ID)
        # From IDLE, only SELECT_OBJECTIVE is legal; EXECUTE must be rejected.
        assert director.context.state == S_IDLE
        try:
            director.execute()
            raise AssertionError("execute() from IDLE should be illegal")
        except IllegalTransitionError:
            pass
    finally:
        harness.close()


# --------------------------------------------------------------------- B: approval bound to plan SHA
def test_execute_refused_until_approval_granted():
    harness = Harness()
    try:
        oid = _proposed_objective_id(harness)
        director = ResearchDirector(
            service=harness.service,
            objective_store=harness.stores.objectives,
            state_dir=harness.root / "director",
            objective_id=oid, source_experiment_id=SOURCE_EXPERIMENT_ID)
        director.run_to_awaiting_approval()
        # Approval is PENDING; executing must be refused (no gate bypass).
        try:
            director.execute()
            raise AssertionError("execute() without granted approval must fail")
        except DirectorError as exc:
            assert "grant" in str(exc).lower()
    finally:
        harness.close()


# --------------------------------------------------------------------- B: persistence + recovery
def test_recovery_reloads_state_and_reconciles():
    harness = Harness()
    try:
        oid = _proposed_objective_id(harness)
        state_dir = harness.root / "director"
        director = ResearchDirector(
            service=harness.service,
            objective_store=harness.stores.objectives,
            state_dir=state_dir,
            objective_id=oid, source_experiment_id=SOURCE_EXPERIMENT_ID,
            approval_granter=_granter(harness))
        director.run_to_awaiting_approval()
        director_id = director.director_id

        # Simulate a process crash: build a brand-new Director from the same
        # state_dir. It must reload AWAITING_APPROVAL and then reconcile against
        # the (pending) approval -- without granting anything itself.
        reloaded = ResearchDirector(
            service=harness.service,
            objective_store=harness.stores.objectives,
            state_dir=state_dir,
            objective_id=oid, director_id=director_id,
            source_experiment_id=SOURCE_EXPERIMENT_ID,
            approval_granter=_granter(harness))
        assert reloaded.context.state == S_AWAITING_APPROVAL
        resolved = reloaded.recover()
        assert resolved == S_AWAITING_APPROVAL

        # Finish the run after recovery.
        final = reloaded.run_to_completion()
        assert final["state"] == S_COMPLETED
        assert final["outcome"]
    finally:
        harness.close()


# --------------------------------------------------------------------- C: full chain to COMPLETED
def test_full_chain_records_experience_no_promotion():
    harness = Harness()
    try:
        oid = _proposed_objective_id(harness)
        director = ResearchDirector(
            service=harness.service,
            objective_store=harness.stores.objectives,
            state_dir=harness.root / "director",
            objective_id=oid, source_experiment_id=SOURCE_EXPERIMENT_ID,
            approval_granter=_granter(harness))

        final = director.run_to_completion()
        assert final["state"] == S_COMPLETED

        # Read the actual ResearchReport the service persisted and assert the
        # same invariants test_arr_0_1_e2e enforces.
        rep = director.service.get_research_report(
            session_id=director.context.session_id)
        assert rep["ok"], rep
        report = rep["report"]
        assert report["promotion"] == "not_requested"
        assert report["frozen_test_used"] is False
        assert report["experience_refs"], "Experience must be recorded"
        assert report["outcome"] in {
            "accepted_for_confirmation", "rejected", "inconclusive"}
    finally:
        harness.close()
