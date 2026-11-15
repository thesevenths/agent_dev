"""M4-E2E-01.1 P1 -- Objective -> Hypothesis / Plan real wiring (engine-free).

These tests drive the EXISTING ``ResearchService`` 12-step gate through the
``ResearchDirector`` using the SAME engine-free ``Harness`` (FakeRunner /
FakeComparator / fake_evidence) -- NO real engine, LLM or model. They prove the
Objective -> Hypothesis / Plan wiring required by M4-E2E-01.1 P1:

  * The Director reads the FULL Objective and the Session persists the REAL
    Objective Description + Success Criteria (not a Director placeholder).
  * The Hypothesis Generator RECEIVES the Objective's Evidence References /
    Hypothesis Hint / Proposed Intervention and threads them into the generated
    Hypothesis as provenance, so each hypothesis explicitly corresponds to the
    current Objective.
  * The Plan uses the Objective-consistent metric; an unsupported metric is
    REJECTED explicitly (never silently swapped to the domain default).
  * The original 12-step fail-closed gate is still valid after the wiring.

None of these tests rewrite the Hypothesis Manager / Critic / Plan Builder; they
only assert the new wiring yields the right artifacts.
"""
from __future__ import annotations

import json
import pytest

from research_runtime.schemas import ResearchObjective
from research_runtime.research_store.objective_store import from_research_objective
from research_runtime.research_director import DirectorError, ResearchDirector
from research_runtime.director_state_machine import S_AWAITING_APPROVAL, S_COMPLETED

from tests.rt_tests.research_fakes import Harness, SOURCE_EXPERIMENT_ID


# --------------------------------------------------------------------- helpers
def _save_objective(harness: Harness, *, objective_id: str,
                    description: str = "",
                    success_criteria: dict | None = None,
                    constraints: dict | None = None) -> None:
    """Persist a Proposed Objective directly into the shared ObjectiveStore.

    This is the SAME store the Director reads, so Discovery-equivalent Objectives
    can be asserted end-to-end without depending on the Discovery snapshot shape.
    """
    obj = ResearchObjective(
        objective_id=objective_id, domain="scinet",
        description=description,
        success_criteria=dict(success_criteria or {}),
        constraints=dict(constraints or {}),
        created_at="2026-01-01T00:00:00")
    record = from_research_objective(obj, status="proposed")
    harness.stores.objectives.save(record)


def _director_for(harness: Harness, oid: str) -> ResearchDirector:
    return ResearchDirector(
        service=harness.service,
        objective_store=harness.stores.objectives,
        state_dir=harness.root / f"dir_{oid}",
        objective_id=oid, source_experiment_id=SOURCE_EXPERIMENT_ID)


def _grant_out_of_band(harness: Harness, approval_id: str, plan_sha256: str) -> None:
    harness.stores.approvals.grant(
        approval_id, approved_by="alice", source="human",
        current_resource_sha256=plan_sha256)


# --------------------------------------------------------------------- P1: Session persists REAL Objective Description + Success Criteria
def test_director_session_persists_real_objective_description_and_criteria():
    harness = Harness()
    try:
        oid = "obj_session_real"
        _save_objective(
            harness, objective_id=oid,
            description="Real objective: cut validation MSE drift after horizon 3",
            success_criteria={"metric": "mse@1", "direction": "minimize",
                              "threshold": 0.05},
            constraints={"evidence_refs": ["experience:exp_alpha"],
                          "hypothesis_hint": "alpha drift suggests LR decay",
                          "proposed_intervention": {"axis": "lr_decay"}})
        director = _director_for(harness, oid)
        director.select_objective()

        session = harness.stores.sessions.get_session(director.context.session_id)
        assert session is not None
        # The Session must carry the REAL Objective, not the old placeholder.
        assert session.objective["description"] == (
            "Real objective: cut validation MSE drift after horizon 3")
        assert session.objective["success_criteria"]["metric"] == "mse@1"
        assert session.objective["success_criteria"]["direction"] == "minimize"
        assert session.objective["success_criteria"]["threshold"] == 0.05
    finally:
        harness.close()


# --------------------------------------------------------------------- P1: Evidence refs + hint thread into Hypothesis provenance
def test_objective_evidence_refs_threaded_into_hypothesis():
    harness = Harness()
    try:
        oid = "obj_evidence"
        _save_objective(
            harness, objective_id=oid,
            description="Objective with evidence references and a hypothesis hint",
            success_criteria={"metric": "mse@1", "direction": "minimize"},
            constraints={
                "evidence_refs": ["experience:exp_evidence_alpha",
                                  "baseline:obj_evidence"],
                "hypothesis_hint": "alpha drift suggests LR decay",
                "proposed_intervention": {"axis": "lr_decay"}})
        director = _director_for(harness, oid)
        director.select_objective()
        director.propose_and_review_hypotheses()

        assert director.context.hypothesis_ids, "expected an approved hypothesis"
        h = harness.stores.hypotheses.get(director.context.hypothesis_ids[0])
        assert h is not None
        hyp = h.hypothesis
        # The Objective's own evidence references must ride along in the Hypothesis.
        assert "experience:exp_evidence_alpha" in hyp["evidence_refs"]
        assert "baseline:obj_evidence" in hyp["evidence_refs"]
        # The generator must still anchor at least one ref to the source experiment.
        assert any(SOURCE_EXPERIMENT_ID in r for r in hyp["evidence_refs"])
        # The hint + proposed intervention are recorded as provenance.
        assert any("objective_hint: alpha drift suggests LR decay" in c
                   for c in hyp["confounders"])
        assert any("objective_proposed_intervention:" in c
                   for c in hyp["confounders"])
    finally:
        harness.close()


# --------------------------------------------------------------------- P1: different Objectives -> different (wired) contexts
def test_different_objectives_produce_different_hypothesis_contexts():
    harness = Harness()
    try:
        oid_a = "obj_ctx_A"
        oid_b = "obj_ctx_B"
        _save_objective(
            harness, objective_id=oid_a,
            description="Objective A",
            success_criteria={"metric": "mse@1", "direction": "minimize"},
            constraints={"evidence_refs": ["experience:exp_A"],
                          "hypothesis_hint": "hint for A"})
        _save_objective(
            harness, objective_id=oid_b,
            description="Objective B",
            success_criteria={"metric": "mse@1", "direction": "minimize"},
            constraints={"evidence_refs": ["experience:exp_B"],
                          "hypothesis_hint": "hint for B"})

        dir_a = _director_for(harness, oid_a)
        dir_a.select_objective()
        dir_a.propose_and_review_hypotheses()

        dir_b = _director_for(harness, oid_b)
        dir_b.select_objective()
        dir_b.propose_and_review_hypotheses()

        # Both Objectives map to the same family-based hypothesis id
        # ("H_HORIZON_MSE"); the store keeps every content version. Read ALL
        # versions and confirm each Objective produced a hypothesis carrying ONLY
        # its own evidence context.
        hid = dir_a.context.hypothesis_ids[0]
        versions = harness.stores.hypotheses.versions(hid)
        assert len(versions) >= 2, "expected one hypothesis version per Objective"

        refs_per_version = [v.hypothesis["evidence_refs"] for v in versions]
        conf_per_version = [v.hypothesis["confounders"] for v in versions]

        # Objective A's evidence appears in a version that does NOT carry B's.
        a_version = next(v for v in versions
                         if "experience:exp_A" in v.hypothesis["evidence_refs"])
        assert "experience:exp_B" not in a_version.hypothesis["evidence_refs"]
        assert any("hint for A" in c for c in a_version.hypothesis["confounders"])

        # Objective B's evidence appears in a (different) version that excludes A's.
        b_version = next(v for v in versions
                         if "experience:exp_B" in v.hypothesis["evidence_refs"])
        assert "experience:exp_A" not in b_version.hypothesis["evidence_refs"]
        assert any("hint for B" in c for c in b_version.hypothesis["confounders"])
        assert a_version.version != b_version.version
    finally:
        harness.close()


# --------------------------------------------------------------------- P1: Objective proposed_intervention (dotted) flows into the Plan
def test_objective_proposed_intervention_override_flows_to_plan():
    harness = Harness()
    try:
        oid = "obj_intervention"
        _save_objective(
            harness, objective_id=oid,
            description="Objective proposing a concrete config override",
            success_criteria={"metric": "mse@1", "direction": "minimize"},
            constraints={
                "evidence_refs": ["experience:exp_iv"],
                "hypothesis_hint": "widen the network",
                "proposed_intervention": {"model.width": 64}})
        director = _director_for(harness, oid)
        director.select_objective()
        director.propose_and_review_hypotheses()

        # The dotted config-path override must appear in the Hypothesis intervention.
        hyp = harness.stores.hypotheses.get(
            director.context.hypothesis_ids[0]).hypothesis
        assert hyp["intervention"].get("model.width") == 64, hyp["intervention"]

        # And therefore in the Plan candidate's config_overrides.
        director.create_plan()
        plan = harness.stores.plans.get(director.context.plan_id)
        assert plan is not None
        candidates = plan.plan["candidates"]
        assert candidates
        overrides = candidates[0]["config_overrides"]
        assert overrides.get("model.width") == 64, overrides
    finally:
        harness.close()


# --------------------------------------------------------------------- P1: modifying Success Criteria changes the Plan target metric
def test_modifying_success_criteria_changes_plan_metric():
    harness = Harness()
    try:
        # Two supported metrics so we can demonstrate a real change.
        harness.replace_domain(primary_metric="mse@1",
                               supported_metrics=("mse@1", "cosine@1"))
        oid_m1 = "obj_metric_mse"
        oid_m2 = "obj_metric_cosine"
        _save_objective(
            harness, objective_id=oid_m1, description="use mse@1",
            success_criteria={"metric": "mse@1", "direction": "minimize"})
        _save_objective(
            harness, objective_id=oid_m2, description="use cosine@1",
            success_criteria={"metric": "cosine@1", "direction": "maximize"})

        dir1 = _director_for(harness, oid_m1)
        dir1.select_objective()
        dir1.propose_and_review_hypotheses()
        dir1.create_plan()
        plan1 = harness.stores.plans.get(dir1.context.plan_id)
        assert plan1.plan["primary_metric"] == "mse@1"
        assert plan1.plan["metric_direction"] == "minimize"

        dir2 = _director_for(harness, oid_m2)
        dir2.select_objective()
        dir2.propose_and_review_hypotheses()
        dir2.create_plan()
        plan2 = harness.stores.plans.get(dir2.context.plan_id)
        assert plan2.plan["primary_metric"] == "cosine@1"
        assert plan2.plan["metric_direction"] == "maximize"
    finally:
        harness.close()


# --------------------------------------------------------------------- P1: unsupported metric REJECTED, never silently swapped
def test_unsupported_metric_rejected_not_silent_swap():
    harness = Harness()
    try:
        # Default domain: only the primary metric ("mse@1") is supported.
        oid = "obj_unsupported"
        _save_objective(
            harness, objective_id=oid, description="ask for an unsupported metric",
            success_criteria={"metric": "latent_mse@1", "direction": "minimize"})
        director = _director_for(harness, oid)
        director.select_objective()
        director.propose_and_review_hypotheses()

        # The Plan step must REFUSE the unsupported metric (fail-closed).
        with pytest.raises(DirectorError):
            director.create_plan()
        # Critically: NO plan row is created with a silently-swapped default metric.
        plan_id = f"plan_{director.director_id}"
        assert harness.stores.plans.get(plan_id) is None

        # And the service itself reports the explicit rejection reason.
        res = harness.service.create_research_plan(
            objective_id=oid,
            hypothesis_ids_json=json.dumps(director.context.hypothesis_ids),
            plan_id="plan_unsupported_explicit")
        assert res["ok"] is False
        assert res["error_code"] == "unsupported_metric"
        assert harness.stores.plans.get("plan_unsupported_explicit") is None
    finally:
        harness.close()


# --------------------------------------------------------------------- P1: original 12-step gate still valid after wiring
def test_twelve_step_gate_still_valid_with_wiring():
    harness = Harness()
    try:
        oid = "obj_gate"
        _save_objective(
            harness, objective_id=oid,
            description="end-to-end gate still valid",
            success_criteria={"metric": "mse@1", "direction": "minimize"},
            constraints={"evidence_refs": ["experience:exp_gate"],
                          "hypothesis_hint": "gate sanity",
                          "proposed_intervention": {"axis": "gate"}})
        director = _director_for(harness, oid)
        out = director.run_to_awaiting_approval()
        assert out["state"] == S_AWAITING_APPROVAL
        assert out["approval_id"] and out["plan_id"]

        _grant_out_of_band(harness, director.context.approval_id,
                           director.context.plan_sha256)
        final = director.run_to_completion()
        assert final["state"] == S_COMPLETED

        rep = director.service.get_research_report(
            session_id=director.context.session_id)
        assert rep["ok"], rep
        report = rep["report"]
        assert report["experience_refs"], "Experience must still be recorded"
        assert report["promotion"] == "not_requested"
    finally:
        harness.close()
