"""M4-E2E-01B/C -- Research Director engine-free contract + M4-E2E-01.1 P0 tests.

Drives the FULL Director chain (IDLE -> ... -> COMPLETED) using the SAME
engine-free ``Harness`` that ``test_arr_0_1_e2e`` uses (FakeRunner /
FakeComparator / fake_evidence) -- NO real engine, LLM or model. This proves the
state machine, approval binding, persistence and recovery without SciNet++.

M4-E2E-01.1 P0 hardening (approval permission boundary):
  * The Director has NO self-grant path: ``approve_approval`` / ``approval_granter``
    were removed. The human grant happens OUT OF BAND -- the real ``approval_cli``
    console (or, in these engine-free tests, the genuine ``ApprovalStore.grant``
    method that the console calls) writes a ``status='approved'`` record bound to
    the exact Plan SHA. The Director only *observes* the store.
  * ``run_to_completion()`` never auto-approves: with no external grant it stops
    at AWAITING_APPROVAL and launches no campaign.
  * ``execute()`` is fail-closed: missing approval / not-approved / revoked /
    plan-missing / SHA-missing / SHA-mismatch all refuse to execute.

NOTE on the engine-free grant: these tests call ``harness.stores.approvals.grant``
directly. That is the EXACT method ``approval_cli`` invokes -- it is a genuine
store write, not a Director-side callback impersonating a human. The P6 real-engine
test goes one step further and shells out to the ``approval_cli`` subprocess.
"""
from __future__ import annotations

import inspect
import pytest
from pathlib import Path

from research_runtime.discovery_context_provider import DiscoverySnapshot
from research_runtime.objective_discovery import ObjectiveDiscovery
from research_runtime.research_director import DirectorError, ResearchDirector
from research_runtime.director_state_machine import (
    IllegalTransitionError, S_AWAITING_APPROVAL, S_COMPLETED, S_IDLE,
)
from research_runtime.schemas import CandidateExperiment, ExperimentPlan

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


def _director(harness: Harness, oid: str, state_dir_name: str = "director"
              ) -> ResearchDirector:
    return ResearchDirector(
        service=harness.service,
        objective_store=harness.stores.objectives,
        state_dir=harness.root / state_dir_name,
        objective_id=oid, source_experiment_id=SOURCE_EXPERIMENT_ID)


def _grant_out_of_band(harness: Harness, approval_id: str, plan_sha256: str) -> None:
    """Simulate the genuine out-of-band human grant.

    This is the exact ``ApprovalStore.grant`` call the ``approval_cli`` console
    makes. It is a real store write bound to the Plan SHA -- NOT a Director-side
    self-grant (the Director no longer exposes any grant method).
    """
    harness.stores.approvals.grant(
        approval_id, approved_by="alice", source="human",
        current_resource_sha256=plan_sha256)


# --------------------------------------------------------------------- B: reach AWAITING_APPROVAL (no approval needed)
def test_director_reaches_awaiting_approval():
    harness = Harness()
    try:
        oid = _proposed_objective_id(harness)
        director = _director(harness, oid)
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
        director = _director(harness, oid)
        # From IDLE, only SELECT_OBJECTIVE is legal; EXECUTE must be rejected.
        assert director.context.state == S_IDLE
        try:
            director.execute()
            raise AssertionError("execute() from IDLE should be illegal")
        except IllegalTransitionError:
            pass
    finally:
        harness.close()


# --------------------------------------------------------------------- B: pending approval refuses execution
def test_execute_refused_until_approval_granted():
    harness = Harness()
    try:
        oid = _proposed_objective_id(harness)
        director = _director(harness, oid)
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
        director = _director(harness, oid, "director")
        director.run_to_awaiting_approval()
        director_id = director.director_id

        # Simulate a process crash: build a brand-new Director from the same
        # state_dir. It must reload AWAITING_APPROVAL and then reconcile against
        # the (pending) approval -- without granting anything itself.
        reloaded = _director(harness, oid, "director")
        assert reloaded.context.state == S_AWAITING_APPROVAL
        resolved = reloaded.recover()
        assert resolved == S_AWAITING_APPROVAL

        # Finish the run after recovery. The human grant is applied OUT OF BAND
        # (the genuine store write the approval_cli console performs).
        _grant_out_of_band(harness, director.context.approval_id,
                           director.context.plan_sha256)
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
        director = _director(harness, oid)
        director.run_to_awaiting_approval()

        # Out-of-band human grant (genuine store write, as the approval_cli
        # console would perform it). The Director never grants for itself.
        _grant_out_of_band(harness, director.context.approval_id,
                           director.context.plan_sha256)

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


# =====================================================================
# M4-E2E-01.1 P0 -- approval permission boundary (fail-closed)
# =====================================================================
def test_director_has_no_self_grant_path():
    """The Director must not even expose a self-grant entry point.

    P0 requirement: delete the Director's default direct ``ApprovalStore.grant``
    path and forbid the Director from manufacturing a human approval.
    """
    # (1) No method named approve_approval exists any more.
    assert not hasattr(ResearchDirector, "approve_approval")
    # (2) The constructor no longer accepts an approval_granter injection.
    params = inspect.signature(ResearchDirector.__init__).parameters
    assert "approval_granter" not in params
    # (3) No code path inside the module calls the store's grant().
    import inspect as _inspect, pathlib
    src = pathlib.Path(ResearchDirector.__module__.replace(".", "/") + ".py").read_text(
        encoding="utf-8")
    # The Director may only *read* approvals; it must never write them.
    assert "approvals.grant(" not in src, \
        "Director must not call ApprovalStore.grant anywhere"


def test_run_to_completion_stops_without_external_approval():
    """P0 #3/#4: run_to_completion() must NOT auto-approve.

    Without an external human grant the Director stops at AWAITING_APPROVAL and
    launches no campaign (no double-execution either).
    """
    harness = Harness()
    try:
        oid = _proposed_objective_id(harness)
        director = _director(harness, oid)
        out = director.run_to_completion()
        assert out["state"] == S_AWAITING_APPROVAL
        assert out.get("stopped_reason") == "awaiting external human approval"
        # No campaign was ever launched.
        assert director.context.campaign_id == ""
        assert director.context.state == S_AWAITING_APPROVAL
    finally:
        harness.close()


def test_execute_refused_revoked_approval():
    """P0: an approval that was granted then revoked cannot execute."""
    harness = Harness()
    try:
        oid = _proposed_objective_id(harness)
        director = _director(harness, oid)
        director.run_to_awaiting_approval()
        # Out-of-band grant, then revoke (e.g. scope abuse discovered).
        _grant_out_of_band(harness, director.context.approval_id,
                           director.context.plan_sha256)
        harness.stores.approvals.revoke(
            director.context.approval_id, decided_by="alice",
            notes="scope changed")
        with pytest.raises(DirectorError):
            director.execute()
    finally:
        harness.close()


def test_execute_refused_plan_modified_after_approval():
    """P0 #6/#7: editing the plan after approval invalidates the old approval."""
    harness = Harness()
    try:
        oid = _proposed_objective_id(harness)
        director = _director(harness, oid)
        director.run_to_awaiting_approval()
        pid = director.context.plan_id
        sha_a = director.context.plan_sha256
        # Out-of-band human grant bound to sha_A.
        _grant_out_of_band(harness, director.context.approval_id, sha_a)

        # The plan content changes -> a NEW version becomes current, with a
        # different sha (simulating an edit-after-approval).
        alt = ExperimentPlan(
            plan_id=pid, objective_id=oid,
            baseline_experiment_id=f"baseline_{pid}",
            dataset_version="ds_v1",
            candidates=[CandidateExperiment(
                candidate_id="c1",
                hypothesis_id=director.context.hypothesis_ids[0],
                config_overrides={"lr": 0.01}, seeds=[42],
                dataset_version="ds_v1")],
            primary_metric="mse@1", metric_direction="minimize",
            paired_seeds=[42])
        harness.stores.plans.save(alt)
        # Current plan sha is now sha_B != sha_A.
        assert harness.stores.plans.get(pid).plan_sha256 != sha_a

        with pytest.raises(DirectorError):
            director.execute()
    finally:
        harness.close()


def test_execute_refused_missing_plan():
    """P0 #7: a missing plan record must fail-closed."""
    harness = Harness()
    try:
        oid = _proposed_objective_id(harness)
        director = _director(harness, oid)
        director.run_to_awaiting_approval()
        # Point the director at a plan id that does not exist in the store.
        director.context.plan_id = "plan_does_not_exist"
        director._write_ctx(director.context)
        with pytest.raises(DirectorError):
            director.execute()
    finally:
        harness.close()


def test_execute_refused_empty_sha_binding():
    """P0 #7: an incomplete SHA binding (empty approval sha) must fail-closed.

    The store itself forbids an empty ``resource_sha256`` (the schema requires
    a non-empty binding), so a blank hash can only arise from a corrupt/partial
    record. ``execute()`` must still refuse it rather than trusting a blank hash.
    """
    harness = Harness()
    try:
        oid = _proposed_objective_id(harness)
        director = _director(harness, oid)
        director.run_to_awaiting_approval()
        pid = director.context.plan_id

        # Simulate a corrupt/partial approval: 'approved' but with an EMPTY
        # binding hash. Monkeypatch only the read used by execute().
        class _FakeApproval:
            status = "approved"
            resource_id = pid
            resource_sha256 = ""

        real_get = harness.service.stores.approvals.get
        harness.service.stores.approvals.get = lambda aid: _FakeApproval()
        try:
            with pytest.raises(DirectorError):
                director.execute()
        finally:
            harness.service.stores.approvals.get = real_get
    finally:
        harness.close()
