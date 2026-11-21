"""M4.3-01.1 P0-01 -- harness is wired to the REAL ResearchService gate.

Engine-free: the ResearchService / SQLite stores / Approval / Campaign /
Report / Experience records are REAL; only the training runner is an injected
double. The gate verdict is DERIVED from persisted records by the read-only
``ResearchServiceGateAdapter``. No caller claim, no mocked verdict, no second
Gate. The real SciNet++ engine E2E is a separate gated stage (P1-03).
"""
from __future__ import annotations

import pytest

from research_runtime.m4_3 import (
    ARMS, BlockSpec, EvaluationHarness, FrozenProtocol, IndependentReviewer,
    ProtocolViolation,
)
from research_runtime.m4_3.research_service_gate import (
    FORGED_GATE_KEYS, GATE_STEP_COUNT, GateRunRef,
    ResearchServiceGateAdapter,
)
from research_runtime.m4_3.metrics import frozen_metric_contract
from tests.rt_tests.m4_3_gate import build_gate_fixture, build_multi_arm_fixture


def _protocol() -> FrozenProtocol:
    return FrozenProtocol(
        snapshot_ids=("snap_1",), sample_size=0, seed=42,
        model_version="m1", prompt_version="p1",
        ranker_version="deterministic-v1",
        budget={"paired_seeds": [42, 43]}, exclusion_rules=("none",),
        statistical_method="paired-block", stopping_rule="all blocks complete")


def _scorer(_b):
    return {d: 3 for d in frozen_metric_contract()["rubric_dimensions"]} | {
        "reason": "grounded, falsifiable, measurable"}


def _reviewer() -> IndependentReviewer:
    return IndependentReviewer(reviewer_id="independent_reviewer_1",
                               scorer=_scorer)


def _objective(oid: str) -> dict:
    return {"objective_id": oid, "description": "reduce prediction error",
            "evidence_refs": ["sha256:real"],
            "success_criteria": {"metric": "mse@1", "measurable": True}}


def _block(bid: str = "block_1") -> BlockSpec:
    return BlockSpec(block_id=bid, snapshot_id="snap_1", seed=42,
                     current_evidence={"n": 1})


@pytest.fixture
def gate():
    fx = build_gate_fixture()
    yield fx
    fx.close()


def _harness(fx) -> EvaluationHarness:
    return EvaluationHarness(
        protocol=_protocol(), reviewer=_reviewer(),
        gate_adapter=fx.adapter, default_gate_ref=fx.ref)


def _seed(h: EvaluationHarness, bid: str = "block_1"):
    h.add_block(_block(bid), candidate_budget=1, human_time_budget_s=10.0)
    for arm in ARMS:
        h.select_objective(bid, arm, objective=_objective(f"{bid}_{arm}"))


# ------------------------------------------------------------------ P0-01 #1/#2/#3
def test_adapter_queries_real_persisted_records_not_a_claim(gate):
    ev = gate.adapter.inspect(gate.ref)
    assert ev.gate_passed is True
    assert ev.gate_steps_passed == GATE_STEP_COUNT == 12
    assert ev.external_approval is True
    assert "approval_record" in ev.verified_artefacts
    assert "campaign_record" in ev.verified_artefacts
    assert "research_report" in ev.verified_artefacts
    assert "experience_rows" in ev.verified_artefacts
    assert ev.missing_artefacts == ()
    assert ev.errors == ()
    # Removing ANY single persisted artefact flips the derived verdict: proves
    # the value comes from the records, not from a stored/caller claim.
    gate.break_experience_rows()
    assert gate.adapter.inspect(gate.ref).gate_passed is False
    assert gate.adapter.inspect(gate.ref).gate_steps_passed == 0


def test_real_full_chain_allows_valid_finding_and_records_derived_gate(gate):
    h = _harness(gate)
    _seed(h)
    rec = h.record_outcome("block_1", "A", outcome="valid_finding",
                           compute_units=2.0)
    assert rec.outcome == "valid_finding"
    # Derived from real records, NOT from a caller argument.
    assert rec.gate_steps_passed == 12
    assert rec.external_approval is True
    assert rec.approval_bypass is False
    assert rec.gate_evidence["gate_passed"] is True
    assert rec.gate_evidence["external_approval"] is True
    assert rec.campaign_id == gate.campaign_id  # real persisted campaign id
    assert len(gate.adapter.inspect_calls) == 1  # adapter really executed


# ------------------------------------------------------------------ P0-01 #4 forging
def test_manual_gate_steps_passed_12_cannot_forge_gate(gate):
    h = _harness(gate)
    _seed(h)
    with pytest.raises(ProtocolViolation) as exc:
        h.record_outcome("block_1", "A", outcome="valid_finding",
                         gate_steps_passed=12)
    assert "forbidden" in str(exc.value).lower()
    assert h.registry.runs["block_1_A"].outcome == "interrupted"  # unchanged


@pytest.mark.parametrize("forged_key", sorted(FORGED_GATE_KEYS))
def test_every_caller_gate_verdict_key_is_rejected(gate, forged_key):
    h = _harness(gate)
    _seed(h)
    with pytest.raises(ProtocolViolation):
        h.record_outcome("block_1", "A", outcome="valid_finding",
                         **{forged_key: 12})


def test_external_approval_has_no_optimistic_default(gate):
    """external_approval=True is gone: the flag is derived, never defaulted."""
    import inspect
    sig = inspect.signature(EvaluationHarness.record_outcome)
    assert "external_approval" not in sig.parameters
    assert "gate_steps_passed" not in sig.parameters
    assert "gate_ref" in sig.parameters
    # With NO adapter at all, fail-closed: no completed outcome is possible.
    bare = EvaluationHarness(protocol=_protocol(), reviewer=_reviewer())
    bare.add_block(_block(), candidate_budget=1, human_time_budget_s=1.0)
    bare.select_objective("block_1", "A", objective=_objective("o_A"))
    with pytest.raises(ProtocolViolation) as exc:
        bare.record_outcome("block_1", "A", outcome="valid_finding")
    assert "gate evidence incomplete" in str(exc.value)


# ------------------------------------------------------------------ P0-01 #6 approval
def test_no_real_approval_rejects_success_record(gate):
    gate.break_approval()
    ev = gate.adapter.inspect(gate.ref)
    assert ev.approval_found is False
    assert ev.external_approval is False
    assert ev.gate_passed is False
    h = _harness(gate)
    _seed(h)
    with pytest.raises(ProtocolViolation) as exc:
        h.record_outcome("block_1", "A", outcome="valid_finding")
    assert "gate evidence incomplete" in str(exc.value)
    assert "approval" in str(exc.value)


def test_pending_approval_is_not_external_and_records_bypass(gate):
    gate.downgrade_approval_to_pending()
    ev = gate.adapter.inspect(gate.ref)
    assert ev.approval_status == "pending"
    assert ev.external_approval is False
    # A REAL interruption is retained and the missing approval is flagged.
    h = _harness(gate)
    _seed(h)
    rec = h.record_outcome("block_1", "A", outcome="interrupted")
    assert rec.outcome == "interrupted"
    assert rec.approval_bypass is True
    assert rec.external_approval is False


def test_system_policy_source_is_not_a_human_external_approval(gate):
    gate.set_approval_source("system_policy")
    ev = gate.adapter.inspect(gate.ref)
    assert ev.approval_source == "system_policy"
    assert ev.external_approval is False
    assert ev.gate_passed is False


def test_hash_mismatch_invalidates_the_approval(gate):
    """Post-approval plan edit breaks the hash binding -> not external."""
    with gate.harness.stores.db.session() as conn:
        conn.execute("UPDATE approvals SET resource_sha256='tampered' "
                     "WHERE approval_id=?", (gate.approval_id,))
    ev = gate.adapter.inspect(gate.ref)
    assert ev.approval_hash_matches_plan is False
    assert ev.external_approval is False


# ------------------------------------------------------------------ P0-01 #7 campaign/report/experience
def test_no_real_campaign_rejects_success_record(gate):
    gate.break_campaign()
    ev = gate.adapter.inspect(gate.ref)
    assert ev.campaign_found is False
    assert ev.gate_passed is False
    h = _harness(gate)
    _seed(h)
    with pytest.raises(ProtocolViolation) as exc:
        h.record_outcome("block_1", "A", outcome="valid_finding")
    assert "campaign" in str(exc.value)


def test_report_missing_cannot_record_valid_finding(gate):
    gate.break_report()
    ev = gate.adapter.inspect(gate.ref)
    assert ev.report_found is False
    assert ev.gate_passed is False
    h = _harness(gate)
    _seed(h)
    with pytest.raises(ProtocolViolation) as exc:
        h.record_outcome("block_1", "A", outcome="valid_finding")
    assert "research_report" in str(exc.value) or "gate evidence" in str(exc.value)


def test_experience_missing_cannot_record_valid_finding(gate):
    gate.break_experience_rows()
    ev = gate.adapter.inspect(gate.ref)
    assert ev.experience_rows == 0
    assert ev.gate_passed is False
    h = _harness(gate)
    _seed(h)
    with pytest.raises(ProtocolViolation):
        h.record_outcome("block_1", "A", outcome="valid_finding")


def test_negative_and_inconclusive_outcomes_also_need_real_gate(gate):
    gate.break_report()
    h = _harness(gate)
    _seed(h)
    for outcome in ("negative_finding", "inconclusive"):
        with pytest.raises(ProtocolViolation):
            h.record_outcome("block_1", "A", outcome=outcome)


# ------------------------------------------------------------------ P0-01 #8 same adapter / entry point
def test_abc_use_the_same_real_gate_adapter():
    """ONE adapter, ONE service, THREE genuinely distinct arm chains.

    M4.3-02 P0 #7: a campaign's success evidence can never be reused by a
    second run, so each arm owns its own persisted plan/approval/campaign/
    session. Sharing the ADAPTER (hence the same ResearchService entry point)
    is what proves the three arms went through the SAME gate.
    """
    fx = build_multi_arm_fixture()
    try:
        h = EvaluationHarness(
            protocol=_protocol(), reviewer=_reviewer(),
            gate_adapter=fx.adapter, db=fx.harness.stores.db)
        h.add_block(_block("block_1"), candidate_budget=1, human_time_budget_s=1.0)
        for arm in ARMS:
            h.select_objective("block_1", arm, objective=_objective(f"o_{arm}"))
        for arm in ARMS:
            rec = h.record_outcome("block_1", arm, outcome="valid_finding",
                                   gate_ref=fx.refs[arm], compute_units=1.0)
            assert rec.gate_evidence["gate_passed"] is True
            assert rec.campaign_id == fx.campaign_ids[arm]
            assert rec.plan_id == fx.plan_ids[arm]
        # ONE adapter instance, ONE underlying service, once per arm.
        assert h.gate_adapter is fx.adapter
        assert len(fx.adapter.inspect_calls) == 3
        assert fx.adapter.service is fx.harness.service
        assert len({r.campaign_id for r in h.registry.all_runs()}) == 3
    finally:
        fx.close()


def test_one_campaign_cannot_supply_success_to_a_second_run(gate):
    """M4.3-02 P0 #7: cross-run reuse of one campaign is refused."""
    from research_runtime.m4_3.registry_store import RegistryConflict

    h = _harness(gate)
    h.add_block(_block("b1"), candidate_budget=1, human_time_budget_s=1.0)
    h.add_block(_block("b2"), candidate_budget=1, human_time_budget_s=1.0)
    h.select_objective("b1", "A", objective=_objective("o_b1"))
    h.select_objective("b2", "A", objective=_objective("o_b2"))
    rec = h.record_outcome("b1", "A", outcome="valid_finding")
    assert rec.gate_evidence["gate_passed"] is True
    with pytest.raises(RegistryConflict) as exc:
        h.record_outcome("b2", "A", outcome="valid_finding")
    assert "already bound" in str(exc.value)
    assert h.registry.runs["b2_A"].finalized is False



def test_gate_ref_points_at_records_not_verdicts(gate):
    """A caller may only supply IDs; there is no field for a verdict."""
    import dataclasses
    fields = {f.name for f in dataclasses.fields(GateRunRef)}
    assert fields == {"session_id", "plan_id", "approval_id", "campaign_id",
                      "report_id"}
    assert not (fields & FORGED_GATE_KEYS)


# ------------------------------------------------------------------ P0-01 #9 real failures retained
def test_real_failures_and_interrupts_are_retained_even_without_gate(gate):
    gate.break_approval()
    gate.break_campaign()
    gate.break_report()
    h = _harness(gate)
    _seed(h)
    for arm, outcome in (("A", "failed"), ("B", "interrupted"),
                         ("C", "invalid")):
        rec = h.record_outcome("block_1", arm, outcome=outcome)
        assert rec.outcome == outcome
        assert rec.gate_steps_passed == 0
        assert rec.approval_bypass is True
    m = h.compute_metrics()
    assert m["approval_bypass_count"] == 3
    assert m["plan_validity_rate"] == 0.0
    assert h.registry.complete_blocks() == ["block_1"]


def test_adapter_is_read_only(gate):
    """inspect() never writes, grants, resumes or executes anything."""
    before = gate.harness.runner.run_calls[:]
    ev_before = gate.adapter.inspect(gate.ref)
    gate.adapter.inspect(gate.ref)
    ev_after = gate.adapter.inspect(gate.ref)
    assert ev_before == ev_after
    assert gate.harness.runner.run_calls == before
    ap = gate.harness.stores.approvals.get(gate.approval_id)
    assert ap.status == "approved"  # not modified by inspection
    # A fresh adapter over the SAME real service sees the SAME evidence.
    assert gate.fresh_adapter().inspect(gate.ref) == ev_after
    assert isinstance(gate.adapter, ResearchServiceGateAdapter)
    assert isinstance(gate.ref, GateRunRef)
