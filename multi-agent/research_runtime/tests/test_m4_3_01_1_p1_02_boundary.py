"""M4.3-01.1 P1-02 -- executable protocol & execution boundaries.

Verifies: A reads ONLY current evidence; B/C read the SAME history; the three
arms share one Snapshot; current vs historical are separated by EXPLICIT,
DISJOINT ID sets; no arm reads a frozen final split; per-block budgets are
fixed; all three arms' complete records (incl. failed/interrupted/invalid/
INCONCLUSIVE) are retained.
"""
from __future__ import annotations

import pytest

from research_runtime.m4_3 import (
    ARMS, BlockSpec, EvaluationHarness, FrozenProtocol, IndependentReviewer,
    ProtocolViolation,
)
from research_runtime.m4_3.evidence_boundary import (
    FROZEN_SPLIT_TOKENS, EvidenceBoundaryViolation, record_id,
    validate_block_budget, validate_evidence_partition,
)
from research_runtime.m4_3.metrics import frozen_metric_contract
from research_runtime.m4_3.research_service_gate import GateEvidence, GateRunRef


def _protocol() -> FrozenProtocol:
    return FrozenProtocol(
        snapshot_ids=("snap_1",), sample_size=0, seed=42,
        model_version="m1", prompt_version="p1",
        ranker_version="deterministic-v1",
        budget={"paired_seeds": [42, 43]}, exclusion_rules=("none",),
        statistical_method="paired-block", stopping_rule="all blocks complete")


def _reviewer() -> IndependentReviewer:
    def scorer(_b):
        return {d: 3 for d in frozen_metric_contract()["rubric_dimensions"]} | {
            "reason": "grounded, falsifiable, measurable"}
    return IndependentReviewer(reviewer_id="r1", scorer=scorer)


class _CompleteGate:
    def inspect(self, ref):
        return GateEvidence(
            ref=ref, plan_found=True, plan_sha256="sha", approval_found=True,
            approval_status="approved", approval_source="human",
            approval_approved_by="alice", approval_hash_matches_plan=True,
            approval_resource_sha256="sha", campaign_found=True,
            campaign_status="completed", campaign_terminal=True,
            report_found=True, report_experience_refs=("exp_1",),
            experience_rows=1)


def _harness() -> EvaluationHarness:
    return EvaluationHarness(
        protocol=_protocol(), reviewer=_reviewer(),
        gate_adapter=_CompleteGate(),
        default_gate_ref=GateRunRef(campaign_id="c1", plan_id="p1",
                                    approval_id="a1", session_id="s1"))


def _objective(oid: str) -> dict:
    return {"objective_id": oid, "description": "reduce prediction error",
            "evidence_refs": ["sha256:real"],
            "success_criteria": {"metric": "mse@1", "measurable": True}}


def _block(**kw) -> BlockSpec:
    base = dict(
        block_id="block_1", snapshot_id="snap_1", seed=42,
        current_evidence={"evidence_id": "cur_1", "split": "validation"},
        historical_evidence=[{"evidence_id": "hist_ev_1",
                              "split": "validation"}],
        historical_experience=[{"experience_id": "hist_xp_1",
                                "split": "validation"}],
        historical_failure=[{"session_id": "hist_fail_1",
                             "split": "validation"}])
    base.update(kw)
    return BlockSpec(**base)


def _add(h: EvaluationHarness, spec: BlockSpec):
    return h.add_block(spec, candidate_budget=1, human_time_budget_s=300.0)


# ------------------------------------------------------- A / B / C contexts
def test_arm_A_reads_only_current_evidence():
    h = _harness()
    c = _add(h, _block())
    assert c["A"].current_evidence["evidence_id"] == "cur_1"
    assert c["A"].historical_experience == ()
    assert c["A"].historical_evidence == ()
    assert c["A"].historical_failure == ()


def test_B_and_C_read_identical_history_on_the_same_snapshot():
    h = _harness()
    c = _add(h, _block())
    assert c["B"].historical_evidence == c["C"].historical_evidence
    assert c["B"].historical_experience == c["C"].historical_experience
    assert c["B"].historical_failure == c["C"].historical_failure
    assert c["B"].historical_evidence  # non-empty history
    for arm in ARMS:
        assert c[arm].snapshot_id == "snap_1" and c[arm].seed == 42


def test_all_arms_share_one_research_opportunity_snapshot():
    h = _harness()
    c = _add(h, _block())
    assert len({c[a].snapshot_id for a in ARMS}) == 1
    part = h.evidence_partition("block_1")
    assert part.snapshot_id == "snap_1"


# ------------------------------------------------------- explicit ID sets
def test_current_and_historical_ids_are_explicit_and_disjoint():
    h = _harness()
    _add(h, _block())
    part = h.evidence_partition("block_1")
    assert part.current_ids == ("cur_1",)
    assert set(part.historical_ids) == {"hist_ev_1", "hist_xp_1",
                                        "hist_fail_1"}
    assert not (set(part.current_ids) & set(part.historical_ids))


def test_explicit_id_sets_are_used_when_provided():
    h = _harness()
    _add(h, _block(
        current_evidence_ids=("cur_1",),
        historical_ids=("hist_ev_1", "hist_xp_1", "hist_fail_1")))
    assert h.evidence_partition("block_1").current_ids == ("cur_1",)


def test_overlapping_current_and_historical_ids_are_rejected():
    h = _harness()
    # Same evidence_id on both sides -> boundary breach.
    spec = _block(historical_evidence=[
        {"evidence_id": "cur_1", "split": "validation"}],
        historical_experience=[], historical_failure=[])
    with pytest.raises(EvidenceBoundaryViolation) as exc:
        _add(h, spec)
    assert "disjoint" in str(exc.value)


@pytest.mark.parametrize("bad_set", [
    ("does_not_exist",), ("cur_1", "phantom"), ("wrong_id", "cur_1"),
])
def test_explicit_sets_must_match_the_actual_records(bad_set):
    h = _harness()
    spec = _block(current_evidence_ids=bad_set)
    with pytest.raises(EvidenceBoundaryViolation) as exc:
        _add(h, spec)
    assert "do not match" in str(exc.value)


def test_empty_id_sets_mean_derive_from_records():
    """An empty set = NOT explicitly specified; the harness derives IDs."""
    h = _harness()
    _add(h, _block())   # current_evidence_ids=() by default
    part = h.evidence_partition("block_1")
    assert part.current_ids == ("cur_1",)   # derived, still explicit output


def test_record_id_is_explicit_or_a_stable_hash():
    assert record_id({"evidence_id": "E1"}) == "E1"
    assert record_id({"x": 1}) == record_id({"x": 1})
    assert record_id({"x": 1}).startswith("sha256:")


# ------------------------------------------------------- frozen final split
@pytest.mark.parametrize("split_value", sorted(FROZEN_SPLIT_TOKENS))
def test_no_arm_may_read_a_frozen_final_split_in_current_evidence(split_value):
    h = _harness()
    spec = _block(current_evidence={"evidence_id": "cur_1",
                                    "split": split_value})
    with pytest.raises(EvidenceBoundaryViolation) as exc:
        _add(h, spec)
    assert "FROZEN FINAL split" in str(exc.value)


@pytest.mark.parametrize("field", [
    "historical_evidence", "historical_experience", "historical_failure",
])
def test_frozen_final_split_in_history_is_rejected(field):
    h = _harness()
    kw = {"historical_evidence": [], "historical_experience": [],
          "historical_failure": []}
    kw[field] = [{"evidence_id": f"hist_{field}", "split": "test"}]
    with pytest.raises(EvidenceBoundaryViolation):
        _add(h, _block(**kw))


def test_frozen_split_in_nested_metadata_is_rejected():
    h = _harness()
    spec = _block(current_evidence={
        "evidence_id": "cur_1",
        "metrics": {"evaluation_split": "final_test"}})
    with pytest.raises(EvidenceBoundaryViolation):
        _add(h, spec)


def test_validation_split_is_allowed():
    h = _harness()
    _add(h, _block())   # split="validation" everywhere -> OK


def test_frozen_split_rejected_by_the_pure_validator():
    with pytest.raises(EvidenceBoundaryViolation):
        validate_evidence_partition(
            snapshot_id="snap_1",
            current_evidence={"evidence_id": "c", "split": "test"},
            historical_experience=[], historical_evidence=[],
            historical_failure=[])


# ------------------------------------------------------- fixed budgets
def test_per_block_budgets_are_fixed_and_recorded():
    h = _harness()
    _add(h, _block())
    b = h.block_budget("block_1")
    assert b.candidate_budget == 1
    assert b.human_time_budget_s == 300.0
    for arm in ARMS:
        c = h._contexts["block_1"][arm]
        assert c.candidate_budget == 1
        assert c.human_time_budget_s == 300.0


@pytest.mark.parametrize("kw", [
    {"candidate_budget": 0}, {"candidate_budget": -1},
    {"human_time_budget_s": 0.0}, {"human_time_budget_s": -5.0},
])
def test_invalid_budgets_are_rejected(kw):
    base = dict(candidate_budget=1, human_time_budget_s=300.0,
                campaign_budget={"max_jobs": 10})
    base.update(kw)
    with pytest.raises(EvidenceBoundaryViolation):
        validate_block_budget(**base)


def test_campaign_budget_must_match_frozen_protocol():
    with pytest.raises(EvidenceBoundaryViolation) as exc:
        validate_block_budget(
            candidate_budget=1, human_time_budget_s=300.0,
            campaign_budget={"max_jobs": 999},
            protocol_campaign_budget={"max_jobs": 10})
    assert "FROZEN protocol budget" in str(exc.value)


def test_a_block_cannot_be_added_twice_with_different_budget():
    h = _harness()
    _add(h, _block())
    with pytest.raises(ProtocolViolation):
        h.add_block(_block(), candidate_budget=99, human_time_budget_s=1.0)


# ------------------------------------------------------- complete retention
def test_all_three_arms_and_all_outcomes_are_retained():
    h = _harness()
    _add(h, _block())
    outcomes = {"A": "failed", "B": "interrupted", "C": "invalid"}
    for arm in ARMS:
        h.select_objective("block_1", arm, objective=_objective(f"o_{arm}"))
        h.record_outcome("block_1", arm, outcome=outcomes[arm])
    assert set(h.registry.runs_for_block("block_1")) == set(ARMS)
    assert {r.outcome for r in h.registry.all_runs()} == {
        "failed", "interrupted", "invalid"}
    assert h.registry.complete_blocks() == ["block_1"]
    assert h.compute_metrics()["invalid_experiment_rate"] == \
        round(1 / 3, 6)


def test_protocol_yaml_boundary_controls_match_code():
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "M4_3_runs" / "protocol.yaml")
        .read_text(encoding="utf-8"))
    c = cfg["controls"]
    assert c["arm_A_reads_current_evidence_only"] is True
    assert c["arm_BC_read_identical_historical_context"] is True
    assert c["three_arms_share_one_snapshot"] is True
    assert c["current_and_historical_separated_by_explicit_disjoint_id_sets"] \
        is True
    assert c["no_arm_reads_frozen_final_split"] is True
    assert c["fixed_campaign_budget_per_block"] is True
    assert c["retain_all_three_arms_complete_records"] is True
    assert set(c["frozen_final_split_tokens"]) == set(FROZEN_SPLIT_TOKENS)
    doc = (Path(__file__).resolve().parents[2] /
           "M4_3_EXPERIMENT_PROTOCOL.md").read_text(encoding="utf-8")
    assert "EXPLICIT, DISJOINT ID" in doc
    assert "frozen" in doc.lower() and "NOT" in doc


def test_inconclusive_outcome_is_retained():
    h = _harness()
    _add(h, _block())
    for arm in ARMS:
        h.select_objective("block_1", arm, objective=_objective(f"o_{arm}"))
    for arm in ARMS:
        h.record_outcome("block_1", arm, outcome="inconclusive")
    assert {r.outcome for r in h.registry.all_runs()} == {"inconclusive"}
