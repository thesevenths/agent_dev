"""M4.3-01 -- A/B/C Evaluation Harness contract tests (engine-free).

Verifies the harness WITHOUT running any engine, Pilot or Confirmatory
experiment. It proves: group definitions + context controls, blocked pairing,
fixed metrics, independent BLIND review (Ranker cannot score itself), complete
outcome retention, frozen protocol, and that no conclusion is drawn.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from research_runtime.m4_3 import (
    ALL_METRICS, PRIMARY_METRICS, SAFETY_METRICS, SECONDARY_METRICS,
    ARMS, ArmContext, BlockSpec, EvaluationHarness, ExperimentRegistry,
    FrozenProtocol, IndependentReviewer, blind, build_arm_context,
    frozen_metric_contract,
)
from research_runtime.m4_3.evaluation_harness import (
    ARM_CONTEXT_RULES, ProtocolViolation,
)

ROOT = Path(__file__).resolve().parents[2]


def _scorer(_blinded):
    return {dim: 3 for dim in frozen_metric_contract()["rubric_dimensions"]} | {
        "reason": "grounded, falsifiable, measurable on held-out split"}


def _reviewer() -> IndependentReviewer:
    return IndependentReviewer(reviewer_id="independent_reviewer_1",
                               scorer=_scorer)


def _protocol() -> FrozenProtocol:
    return FrozenProtocol(
        snapshot_ids=("snap_1",), sample_size=0, seed=42,
        model_version="m1", prompt_version="p1",
        ranker_version="m4-e2e-01a-deterministic-v1",
        budget={"paired_seeds": [42, 43], "max_jobs": 30},
        exclusion_rules=("none",), statistical_method="paired-block",
        stopping_rule="all blocks complete", confirmatory=False)


def _objective(oid: str = "obj_1") -> dict:
    return {"objective_id": oid, "description": "reduce prediction error",
            "evidence_refs": ["sha256:real"],
            "success_criteria": {"metric": "mse@1", "measurable": True},
            "proposed_intervention": {"axis": "regularisation"}}


def _block(spec_id: str = "block_1", snapshot: str = "snap_1") -> BlockSpec:
    return BlockSpec(
        block_id=spec_id, snapshot_id=snapshot, seed=42,
        current_evidence={"n": 1},
        historical_experience=[{"kind": "research_outcome", "outcome": "rejected"}],
        historical_evidence=[{"evidence_id": "sha256:real"}],
        historical_failure=[{"event_type": "train_failed"}])


@pytest.fixture
def gate():
    """REAL engine-free ResearchService gate chain (P0-01)."""
    from tests.rt_tests.m4_3_gate import build_gate_fixture
    fx = build_gate_fixture()
    yield fx
    fx.close()


def _harness(gate=None) -> EvaluationHarness:
    # P0-01: completed outcomes MUST be backed by a real read-only gate
    # adapter over real persisted records; there is no optimistic default.
    kw = (dict(gate_adapter=gate.adapter, default_gate_ref=gate.ref)
          if gate is not None else {})
    return EvaluationHarness(protocol=_protocol(), reviewer=_reviewer(), **kw)


# --------------------------------------------------------------------- groups/context
def test_three_group_definitions_and_context_rules():
    assert ARMS == ("A", "B", "C")
    assert ARM_CONTEXT_RULES["A"]["read_historical_experience"] is False
    assert ARM_CONTEXT_RULES["B"]["read_historical_experience"] is True
    assert ARM_CONTEXT_RULES["C"]["use_autonomous_discovery"] is True
    assert ARM_CONTEXT_RULES["C"]["read_historical_evidence"] is True


def test_arm_A_must_not_see_any_history():
    spec = _block()
    # (1) The HARNESS masks history for A while giving B/C the same history.
    contexts = _harness().add_block(
        spec, candidate_budget=1, human_time_budget_s=300.0)
    a = contexts["A"]
    assert a.historical_experience == ()
    assert a.historical_evidence == ()
    assert a.historical_failure == ()
    assert contexts["B"].historical_experience == \
        contexts["C"].historical_experience
    # (2) Defense in depth: passing history to A DIRECTLY is a violation.
    with pytest.raises(ProtocolViolation):
        build_arm_context(
            arm="A", snapshot_id=spec.snapshot_id, seed=spec.seed,
            current_evidence=spec.current_evidence,
            historical_experience=[{"x": 1}], historical_evidence=[],
            historical_failure=[], candidate_budget=1,
            human_time_budget_s=300.0)
    # (3) An A context built with NO history is valid and empty.
    clean = build_arm_context(
        arm="A", snapshot_id=spec.snapshot_id, seed=spec.seed,
        current_evidence=spec.current_evidence,
        historical_experience=[], historical_evidence=[],
        historical_failure=[], candidate_budget=1,
        human_time_budget_s=300.0)
    assert clean.historical_experience == ()


def test_B_and_C_share_identical_historical_context():
    h = _harness()
    contexts = h.add_block(_block(), candidate_budget=1,
                           human_time_budget_s=300.0)
    assert contexts["B"].historical_experience == \
        contexts["C"].historical_experience
    assert contexts["B"].historical_evidence == contexts["C"].historical_evidence
    assert contexts["C"].historical_experience  # C DOES read history
    # Rule #1/#4: same snapshot, seed, budget for all arms.
    for arm in ARMS:
        c = contexts[arm]
        assert c.snapshot_id == "snap_1" and c.seed == 42
        assert c.candidate_budget == 1 and c.human_time_budget_s == 300.0


def test_unknown_arm_is_rejected():
    with pytest.raises(ProtocolViolation):
        build_arm_context(
            arm="D", snapshot_id="s", seed=1, current_evidence={},
            historical_experience=[], historical_evidence=[],
            historical_failure=[], candidate_budget=1,
            human_time_budget_s=0.0)


# --------------------------------------------------------------------- independent blind review
def test_blind_strips_arm_and_ranker_signals_or_rejects():
    # P0-02: the blinded payload carries ONLY an opaque review_id; the raw
    # objective_id (which can encode an arm, e.g. block_1_A) NEVER appears.
    b = blind(_objective())
    assert b.review_id.startswith("rev_") and len(b.review_id) > 20
    assert "objective_id" not in b.to_dict()
    assert "arm" not in b.to_dict() and "rank_score" not in b.to_dict()
    assert "obj_1" not in str(b.to_dict())
    leaked = _objective() | {"arm": "C", "rank_score": 0.9}
    with pytest.raises(ValueError) as exc:
        blind(leaked)
    assert "arm" in str(exc.value) or "Ranker" in str(exc.value)


def test_ranker_cannot_score_itself_and_reason_is_mandatory():
    # A scorer returning a Ranker-derived signal instead of the fixed rubric
    # must fail closed (missing rubric dimensions).
    bad = IndependentReviewer(
        reviewer_id="r", scorer=lambda _b: {"rank_score": 1.0,
                                            "reason": "from ranker"})
    with pytest.raises(ValueError):
        bad.review(_objective())
    # Missing reason fails closed even with valid scores.
    no_reason = IndependentReviewer(
        reviewer_id="r", scorer=lambda _b: {
            **{d: 3 for d in frozen_metric_contract()["rubric_dimensions"]}})
    with pytest.raises(ValueError):
        no_reason.review(_objective())


def test_independent_review_uses_fixed_rubric_and_records_reason():
    r = _reviewer().review(_objective())
    assert r.reviewer_id == "independent_reviewer_1"
    assert set(r.scores) == set(frozen_metric_contract()["rubric_dimensions"])
    assert r.reason and r.total == 3.0
    assert r.rubric_version == "m4_3_objective_quality_rubric_v1"


def test_selector_ranker_signals_are_stripped_before_review():
    """Arm C may return a ranked candidate; the review must stay blind."""
    h = _harness()
    h.add_block(_block(), candidate_budget=1, human_time_budget_s=1.0)
    ranked = _objective() | {"rank_score": 0.87,
                             "ranker_version": "v1", "arm": "C"}
    obj = h.select_objective("block_1", "C", objective=ranked)
    assert "rank_score" not in obj
    rec = h.registry.runs["block_1_C"]
    assert rec.quality_review["total"] == 3.0
    assert "arm" not in rec.quality_review


# --------------------------------------------------------------------- metrics frozen
def test_metric_vocabulary_is_complete_and_frozen():
    assert set(PRIMARY_METRICS) == {
        "independently_reviewed_objective_quality",
        "valid_research_findings_per_compute_budget"}
    assert len(SECONDARY_METRICS) == 9
    assert len(SAFETY_METRICS) == 5
    assert set(ALL_METRICS) == set(PRIMARY_METRICS) | set(SECONDARY_METRICS) | \
        set(SAFETY_METRICS)
    c = frozen_metric_contract()
    assert len(c["contract_sha256"]) == 64
    # Deterministic hash (no post-hoc drift).
    assert frozen_metric_contract()["contract_sha256"] == c["contract_sha256"]


def test_primary_metric_cannot_be_changed_post_hoc():
    h = _harness()
    h.assert_no_posthoc_metric_change()
    # Simulate a post-hoc swap of the primary metric.
    h.protocol = FrozenProtocol(
        primary_metrics=("a_different_metric",), confirmatory=False)
    with pytest.raises(ProtocolViolation):
        h.assert_no_posthoc_metric_change()


# --------------------------------------------------------------------- complete retention
def test_all_outcomes_are_retained_and_gate_approval_recorded(gate):
    h = _harness(gate)
    # Three blocks with different outcomes, all paired A/B/C.
    outcomes = ("valid_finding", "inconclusive", "failed")
    for i, oc in enumerate(outcomes):
        bid = f"block_{i}"
        spec = _block(bid, "snap_1")
        h.add_block(spec, candidate_budget=1, human_time_budget_s=10.0)
        for arm in ARMS:
            h.select_objective(bid, arm, objective=_objective(f"{bid}_{arm}"))
            # P0-01: no caller-supplied gate verdict; derived from persisted
            # Approval/Campaign/Report/Experience via the real adapter.
            h.record_outcome(
                bid, arm, outcome=oc, compute_units=2.0, wall_time_s=5.0)
    runs = h.registry.all_runs()
    assert len(runs) == 9
    # NOTHING was dropped: inconclusive + failed are present in the denominator.
    assert {r.outcome for r in runs} == set(outcomes)
    metrics = h.compute_metrics()
    assert set(metrics) == set(ALL_METRICS)
    assert metrics["invalid_experiment_rate"] == 0.0
    assert metrics["approval_bypass_count"] == 0
    assert metrics["valid_research_findings_per_compute_budget"] == \
        round(3 / 18.0, 6)
    assert metrics["independently_reviewed_objective_quality"] == 3.0
    assert h.registry.complete_blocks() == ["block_0", "block_1", "block_2"]


def test_approval_bypass_and_frozen_split_are_safety_counts(gate):
    h = _harness(gate)
    h.add_block(_block(), candidate_budget=1, human_time_budget_s=1.0)
    for arm in ARMS:
        h.select_objective("block_1", arm, objective=_objective(f"o_{arm}"))
    from research_runtime.m4_3.research_service_gate import GateRunRef
    # P0-01: approval_bypass is DERIVED from a genuinely missing Approval
    # record (a gate_ref pointing at a non-existent approval), not claimed.
    no_approval = GateRunRef(
        session_id="no_such_session", plan_id="no_such_plan",
        approval_id="no_such_approval", campaign_id="no_such_campaign")
    # A real interruption is retained even with no gate evidence; the missing
    # approval is recorded as a bypass (fail-closed for completed outcomes).
    h.record_outcome("block_1", "A", outcome="interrupted",
                     gate_ref=no_approval, frozen_split_violation=True)
    h.record_outcome("block_1", "B", outcome="invalid")   # real gate ref
    h.record_outcome("block_1", "C", outcome="interrupted")  # real gate ref
    m = h.compute_metrics()
    assert m["approval_bypass_count"] == 1   # only A lacks a real Approval
    assert m["frozen_split_violation_count"] == 1
    assert m["invalid_experiment_rate"] == round(1 / 3, 6)
    # The forged-key path is gone entirely.
    with pytest.raises(ProtocolViolation):
        h.record_outcome("block_1", "A", outcome="valid_finding",
                         external_approval=False)


def test_completed_arm_must_pass_all_12_gate_steps(gate):
    # P0-01: a completed arm needs the FULL real gate chain; a caller cannot
    # claim a partial/passing step count at all.
    h = _harness(gate)
    h.add_block(_block(), candidate_budget=1, human_time_budget_s=1.0)
    h.select_objective("block_1", "A", objective=_objective())
    with pytest.raises(ProtocolViolation):
        h.record_outcome("block_1", "A", outcome="valid_finding",
                         gate_steps_passed=11)
    # A missing persisted artefact also blocks a completed outcome.
    gate.break_experience_rows()
    with pytest.raises(ProtocolViolation):
        h.record_outcome("block_1", "A", outcome="valid_finding")


# --------------------------------------------------------------------- frozen protocol / stage gates
def test_confirmatory_refuses_unfrozen_protocol():
    p = _protocol()
    with pytest.raises(ValueError) as exc:
        p.assert_confirmatory_ready()
    assert "frozen" in str(exc.value).lower()
    frozen = FrozenProtocol(
        snapshot_ids=("snap_1", "snap_2"), sample_size=30, seed=7,
        model_version="m1", prompt_version="p1", ranker_version="r1",
        budget={"paired_seeds": [42, 43]},
        exclusion_rules=("pre-registered",), statistical_method="paired-block",
        stopping_rule="all blocks complete", confirmatory=True)
    frozen.assert_confirmatory_ready()  # no exception
    assert len(frozen.freeze_sha256()) == 64


def test_registry_preserves_snapshot_seed_hierarchy(gate):
    h = _harness(gate)
    h.add_block(_block(), candidate_budget=1, human_time_budget_s=1.0)
    for arm in ARMS:
        h.select_objective("block_1", arm, objective=_objective(f"o_{arm}"))
        h.record_outcome("block_1", arm, outcome="valid_finding",
                         compute_units=1.0)
    paired = h.registry.runs_for_block("block_1")
    assert set(paired) == {"A", "B", "C"}
    assert {r.snapshot_id for r in paired.values()} == {"snap_1"}
    assert {r.seed for r in paired.values()} == {42}
    assert {tuple(r.paired_seeds) for r in paired.values()} == {(42, 43)}
    by_arm = h.metrics_by_arm()
    assert set(by_arm) == {"A", "B", "C"}
    assert all(by_arm[a]["n_runs"] == 1 for a in ARMS)


def test_protocol_yaml_matches_frozen_code_contract_no_drift():
    """The YAML config must not drift from the code's frozen metric contract."""
    import yaml
    from research_runtime.m4_3.metrics import (
        REVIEW_RUBRIC_VERSION, RUBRIC_DIMENSIONS,
    )
    cfg = yaml.safe_load((ROOT / "M4_3_runs" / "protocol.yaml").read_text(
        encoding="utf-8"))
    assert tuple(cfg["primary_metrics"]) == PRIMARY_METRICS
    assert tuple(cfg["secondary_metrics"]) == SECONDARY_METRICS
    assert tuple(cfg["safety_metrics"]) == SAFETY_METRICS
    assert cfg["independent_review"]["fixed_rubric_version"] == \
        REVIEW_RUBRIC_VERSION
    assert tuple(cfg["independent_review"]["rubric_dimensions"]) == \
        RUBRIC_DIMENSIONS
    assert list(cfg["groups"]) == list(ARMS)


def test_protocol_yaml_and_doc_explicitly_do_not_start_pilot():
    import yaml
    cfg = yaml.safe_load((ROOT / "M4_3_runs" / "protocol.yaml").read_text(
        encoding="utf-8"))
    assert cfg["status"] == "shell_only_no_runs_started"
    assert cfg["pre_registration"]["confirmatory"] is False
    assert cfg["pre_registration"]["sample_size"] == 0
    assert cfg["stage_gates"]["M4_3_02_pilot"].startswith("NOT_STARTED")
    assert cfg["stage_gates"]["M4_3_03_confirmatory"] == "NOT_STARTED"
    assert len(cfg["allowed_conclusions"]) == 4
    doc = (ROOT / "M4_3_EXPERIMENT_PROTOCOL.md").read_text(encoding="utf-8")
    assert "NOT STARTED" in doc and "NOT made here" in doc
    assert "forcing" in doc.lower() or "Forcing" in doc
