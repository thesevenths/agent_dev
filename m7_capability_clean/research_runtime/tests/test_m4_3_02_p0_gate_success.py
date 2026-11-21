"""M4.3-02 P0 -- terminal is NOT successful: counter-example gate tests.

Every case here mutates REAL persisted state (SQLite rows / a real
CampaignRecord) and then asks the READ-ONLY adapter for a verdict. Nothing is
faked at the evidence layer: the adapter must DERIVE failure from the records.

Covered (spec M4.3-02 P0):
  1. terminal vs successful (failed / cancelled never pass the gate);
  2. valid_finding needs a successful Campaign / Trial record;
  3. negative_finding is legal after a genuine completion (no positive result);
  4. failed / cancelled / interrupted stay failures (never gate success);
  5. Report / Experience / Session / Campaign / Plan / Approval are precisely
     associated;
  6. Experience refs must resolve to REAL Experience rows;
  7. one campaign's success evidence can not be reused (cross run);
  8. the adapter stays read-only.
"""
from __future__ import annotations

import json

import pytest

from research_runtime.m4_3 import (
    ARMS, BlockSpec, EvaluationHarness, FrozenProtocol, IndependentReviewer,
    ProtocolViolation,
)
from research_runtime.m4_3.metrics import frozen_metric_contract
from research_runtime.m4_3.research_service_gate import (
    CAMPAIGN_SUCCESS_STATUSES, GATE_STEP_COUNT, GateRunRef,
    ResearchServiceGateAdapter,
)
from tests.rt_tests.m4_3_gate import build_gate_fixture


def _protocol() -> FrozenProtocol:
    return FrozenProtocol(
        snapshot_ids=("snap_1",), sample_size=0, seed=42,
        model_version="m1", prompt_version="p1", ranker_version="deterministic-v1",
        budget={"paired_seeds": [42]}, exclusion_rules=("none",),
        statistical_method="paired-block", stopping_rule="all blocks complete")


def _reviewer() -> IndependentReviewer:
    def scorer(_b):
        return {d: 3 for d in frozen_metric_contract()["rubric_dimensions"]} | {
            "reason": "grounded and measurable"}
    return IndependentReviewer(reviewer_id="reviewer_1", scorer=scorer)


@pytest.fixture
def gate():
    fx = build_gate_fixture()
    yield fx
    fx.close()


def _harness(fx) -> EvaluationHarness:
    h = EvaluationHarness(protocol=_protocol(), reviewer=_reviewer(),
                          gate_adapter=fx.adapter, default_gate_ref=fx.ref,
                          db=fx.harness.stores.db)
    h.add_block(BlockSpec(block_id="b1", snapshot_id="snap_1", seed=42,
                          current_evidence={"n": 1}),
                candidate_budget=1, human_time_budget_s=10.0)
    h.select_objective("b1", "A", objective={
        "objective_id": "o_a", "description": "reduce error",
        "evidence_refs": ["sha256:real"],
        "success_criteria": {"metric": "mse@1", "measurable": True}})
    return h


# ---------------------------------------------------------------- 1. terminal != successful
@pytest.mark.parametrize("bad_status", ["failed", "cancelled"])
def test_failed_or_cancelled_campaign_never_passes_the_gate(gate, bad_status):
    gate.harness.campaign_store.records[gate.campaign_id].status = bad_status
    ev = gate.adapter.inspect(gate.ref)
    assert ev.campaign_found is True
    assert ev.campaign_terminal is True          # it IS terminal ...
    assert ev.campaign_successful is False       # ... but NOT successful
    assert ev.gate_passed is False
    assert ev.gate_steps_passed == 0
    assert "campaign_successful" in ev.missing_artefacts

    h = _harness(gate)
    for outcome in ("valid_finding", "negative_finding", "inconclusive"):
        with pytest.raises(ProtocolViolation):
            h.record_outcome("b1", "A", outcome=outcome)


def test_success_statuses_are_defined_once():
    assert CAMPAIGN_SUCCESS_STATUSES == {"completed", "completed_with_failures"}


# ---------------------------------------------------------------- 2. trial evidence
def test_campaign_whose_only_trials_failed_is_not_a_success(gate):
    from research_runtime.campaign.schemas import TrialRecord

    record = gate.harness.campaign_store.records[gate.campaign_id]
    record.trials = [
        TrialRecord(trial_id="t1", campaign_id=gate.campaign_id,
                    arm_type="baseline", arm_id="base", seed=42,
                    config_path="c.yaml", config_sha256="x" * 8,
                    dataset_version="ds", status="failed"),
        TrialRecord(trial_id="t2", campaign_id=gate.campaign_id,
                    arm_type="candidate", arm_id="cand", seed=42,
                    config_path="c.yaml", config_sha256="x" * 8,
                    dataset_version="ds", status="failed"),
    ]
    ev = gate.adapter.inspect(gate.ref)
    assert ev.campaign_trials == 2
    assert ev.campaign_succeeded_trials == 0
    assert ev.trial_evidence == "failed_only"
    assert ev.campaign_successful is False       # nothing succeeded
    assert ev.gate_passed is False


def test_campaign_with_a_succeeded_trial_is_a_success(gate):
    from research_runtime.campaign.schemas import TrialRecord

    record = gate.harness.campaign_store.records[gate.campaign_id]
    record.trials = [
        TrialRecord(trial_id="t1", campaign_id=gate.campaign_id,
                    arm_type="baseline", arm_id="base", seed=42,
                    config_path="c.yaml", config_sha256="x" * 8,
                    dataset_version="ds", status="succeeded"),
        TrialRecord(trial_id="t2", campaign_id=gate.campaign_id,
                    arm_type="candidate", arm_id="cand", seed=42,
                    config_path="c.yaml", config_sha256="x" * 8,
                    dataset_version="ds", status="succeeded"),
    ]
    ev = gate.adapter.inspect(gate.ref)
    assert ev.campaign_succeeded_trials == 2
    assert ev.trial_evidence == "succeeded"
    assert ev.campaign_successful is True
    assert ev.gate_passed is True
    assert ev.valid_finding_evidence is True      # report outcome is "rejected"


# ---------------------------------------------------------------- 3. negative finding
def test_negative_finding_is_allowed_after_a_real_completion(gate):
    h = _harness(gate)
    rec = h.record_outcome("b1", "A", outcome="negative_finding")
    assert rec.outcome == "negative_finding"
    assert rec.gate_steps_passed == GATE_STEP_COUNT
    assert rec.external_approval is True


def test_valid_finding_requires_a_real_scientific_outcome(gate):
    """A comparison that produced NO outcome cannot support a valid finding."""
    rep = gate.harness.service.get_research_report(session_id=gate.session_id)
    report = rep["report"]
    report["outcome"] = "not_evaluated"
    with gate.harness.stores.db.session() as conn:
        row = conn.execute(
            "SELECT record_json FROM research_reports WHERE report_id=?",
            (report["report_id"],)).fetchone()
        payload = json.loads(row["record_json"])
        payload["outcome"] = "not_evaluated"
        conn.execute("UPDATE research_reports SET record_json=? WHERE report_id=?",
                     (json.dumps(payload), report["report_id"]))
    ev = gate.adapter.inspect(gate.ref)
    assert ev.report_outcome == "not_evaluated"
    assert ev.gate_passed is True                 # the run DID complete
    assert ev.valid_finding_evidence is False     # but claims no finding
    h = _harness(gate)
    with pytest.raises(ProtocolViolation) as exc:
        h.record_outcome("b1", "A", outcome="valid_finding")
    assert "successful experiment" in str(exc.value)
    # A negative finding remains legal (genuine completion, no positive result).
    assert h.record_outcome("b1", "A", outcome="negative_finding").outcome == \
        "negative_finding"


# ---------------------------------------------------------------- 4. precise association
def test_report_from_another_session_is_rejected(gate):
    other = gate.harness.service.start_research_session(
        objective_id="obj_other", description="another objective")
    bad = GateRunRef(session_id=other["session_id"], plan_id=gate.plan_id,
                     approval_id=gate.approval_id,
                     campaign_id=gate.campaign_id)
    ev = gate.adapter.inspect(bad)
    assert ev.gate_passed is False
    assert ev.scope_conflicts, "cross-session reuse must be a scope conflict"


def test_campaign_from_another_plan_is_rejected(gate):
    out = gate.harness.chain(plan_id="plan_second")
    run = gate.harness.service.run_research_campaign(
        plan_id="plan_second", approval_id=out["approval_id"])
    assert run.get("ok")
    mixed = GateRunRef(session_id=gate.session_id, plan_id=gate.plan_id,
                       approval_id=gate.approval_id,
                       campaign_id=run["campaign_id"])
    ev = gate.adapter.inspect(mixed)
    assert ev.campaign_found is True
    assert ev.campaign_plan_id == "plan_second"
    assert ev.gate_passed is False
    assert any("another plan" in c for c in ev.scope_conflicts)


def test_approval_bound_to_a_different_plan_is_not_external(gate):
    other = gate.harness.chain(plan_id="plan_other")
    moved = GateRunRef(session_id=gate.session_id, plan_id=gate.plan_id,
                       approval_id=other["approval_id"],
                       campaign_id=gate.campaign_id)
    ev = gate.adapter.inspect(moved)
    assert ev.approval_found is True
    assert ev.external_approval is False
    assert ev.gate_passed is False
    assert any("another resource" in c for c in ev.scope_conflicts)


def test_wrong_plan_sha_invalidates_the_chain(gate):
    with gate.harness.stores.db.session() as conn:
        conn.execute("UPDATE approvals SET resource_sha256='deadbeef' "
                     "WHERE approval_id=?", (gate.approval_id,))
    ev = gate.adapter.inspect(gate.ref)
    assert ev.approval_hash_matches_plan is False
    assert ev.external_approval is False
    assert ev.gate_passed is False


def test_plan_of_another_objective_breaks_association(gate):
    """A plan for a DIFFERENT objective must not describe this run."""
    assert gate.harness.stores.plans.get(gate.plan_id).objective_id == "obj_p106"
    # Rewriting the persisted objective on the plan makes it another objective's
    # plan; the association check must catch it (no silent re-association).
    with gate.harness.stores.db.session() as conn:
        row = conn.execute(
            "SELECT record_json FROM plan_versions WHERE plan_id=?",
            (gate.plan_id,)).fetchone()
        payload = json.loads(row["record_json"])
        payload["objective_id"] = "obj_someone_else"
        conn.execute(
            "UPDATE plan_versions SET objective_id=?, record_json=? "
            "WHERE plan_id=?",
            ("obj_someone_else", json.dumps(payload), gate.plan_id))
    ev = gate.adapter.inspect(gate.ref)
    assert ev.plan_objective_id == "obj_someone_else"
    assert ev.association["plan_objective_matches_session"] is False
    assert ev.gate_passed is False


# ---------------------------------------------------------------- 5. experience refs
def test_experience_ref_must_resolve_to_a_real_row(gate):
    with gate.harness.stores.db.session() as conn:
        row = conn.execute(
            "SELECT record_json FROM research_reports WHERE session_id=?",
            (gate.session_id,)).fetchone()
        payload = json.loads(row["record_json"])
        payload["experience_refs"] = list(payload.get("experience_refs") or []) + [
            "exp_fabricated"]
        conn.execute("UPDATE research_reports SET record_json=? WHERE session_id=?",
                     (json.dumps(payload), gate.session_id))
    ev = gate.adapter.inspect(gate.ref)
    assert ev.experience_rows >= 1
    assert ev.experience_refs_unresolved == ("exp_fabricated",)
    assert ev.gate_passed is False
    assert "experience_refs_resolved" in ev.missing_artefacts
    h = _harness(gate)
    with pytest.raises(ProtocolViolation):
        h.record_outcome("b1", "A", outcome="valid_finding")


# ---------------------------------------------------------------- 6. real failures retained
def test_interrupted_run_is_retained_and_never_becomes_gate_success(gate):
    gate.harness.campaign_store.records[gate.campaign_id].status = "failed"
    h = _harness(gate)
    rec = h.record_outcome("b1", "A", outcome="interrupted")
    assert rec.outcome == "interrupted"
    assert rec.gate_steps_passed == 0
    assert rec.gate_evidence["gate_passed"] is False
    assert rec.gate_evidence["campaign_successful"] is False


# ---------------------------------------------------------------- 7. read-only
def test_adapter_never_mutates_persisted_records(gate):
    before = gate.harness.campaign_store.records[gate.campaign_id].model_dump()
    ev1 = gate.adapter.inspect(gate.ref)
    ev2 = ResearchServiceGateAdapter(gate.harness.service).inspect(gate.ref)
    after = gate.harness.campaign_store.records[gate.campaign_id].model_dump()
    assert before == after
    assert ev1 == ev2
    assert ev1.gate_passed is True


def test_arms_are_still_three_and_harness_cannot_forge_success(gate):
    assert ARMS == ("A", "B", "C")
    h = _harness(gate)
    with pytest.raises(ProtocolViolation):
        h.record_outcome("b1", "A", outcome="valid_finding",
                         gate_steps_passed=12)
