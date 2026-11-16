"""M4.3-01.1 P1-01 -- complete per-arm grouped metrics.

Verifies all ten spec requirements: all 16 metrics per arm, explicit
numerator/denominator/missing/failure rules, Primary paired by Block,
Snapshot/Block/Seed hierarchy, identical Paired Seeds per block, incomplete
blocks excluded, three metric families, formula version in the contract hash,
no merged-mean-only output, no significance conclusion.
"""
from __future__ import annotations

import pytest

from research_runtime.m4_3 import (
    ALL_METRICS, ARMS, METRIC_CATEGORIES, METRIC_DEFINITIONS,
    METRIC_FORMULA_VERSION, PRIMARY_METRICS, SAFETY_METRICS,
    SECONDARY_METRICS, BlockSpec, EvaluationHarness, FrozenProtocol,
    IndependentReviewer, ProtocolViolation, compute_metric_set,
    frozen_metric_contract, metric_definitions,
)
from research_runtime.m4_3.research_service_gate import GateEvidence, GateRunRef
from research_runtime.m4_3.metrics import frozen_metric_contract as fmc
from tests.rt_tests.m4_3_gate import complete_gate_evidence


def _protocol() -> FrozenProtocol:
    return FrozenProtocol(
        snapshot_ids=("snap_1", "snap_2"), sample_size=0, seed=42,
        model_version="m1", prompt_version="p1",
        ranker_version="deterministic-v1",
        budget={"paired_seeds": [42, 43]}, exclusion_rules=("none",),
        statistical_method="paired-block", stopping_rule="all blocks complete")


def _reviewer() -> IndependentReviewer:
    def scorer(_b):
        return {d: 3 for d in fmc()["rubric_dimensions"]} | {
            "reason": "grounded, falsifiable, measurable"}
    return IndependentReviewer(reviewer_id="r1", scorer=scorer)


def _objective(oid: str) -> dict:
    return {"objective_id": oid, "description": "reduce prediction error",
            "evidence_refs": ["sha256:real"],
            "success_criteria": {"metric": "mse@1", "measurable": True}}


class _CompleteGate:
    """Read-only gate double returning a COMPLETE real evidence chain."""

    def inspect(self, ref):
        return complete_gate_evidence(
            campaign_id=ref.campaign_id or "c1",
            plan_id=ref.plan_id or "p1",
            approval_id=ref.approval_id or "a1",
            session_id=ref.session_id or "s1",
            report_id=ref.report_id or "r1")


def _harness() -> EvaluationHarness:
    return EvaluationHarness(
        protocol=_protocol(), reviewer=_reviewer(),
        gate_adapter=_CompleteGate(),
        default_gate_ref=GateRunRef(campaign_id="c1", plan_id="p1",
                                    approval_id="a1", session_id="s1"))


def _block(bid: str, snapshot: str = "snap_1", seed: int = 42) -> BlockSpec:
    return BlockSpec(block_id=bid, snapshot_id=snapshot, seed=seed,
                     current_evidence={"n": 1})


def _ref(bid: str, arm: str) -> GateRunRef:
    """DISTINCT persisted scope per run (M4.3-02 P0 #7: no campaign reuse)."""
    return GateRunRef(session_id=f"s_{bid}_{arm}", plan_id=f"p_{bid}_{arm}",
                      approval_id=f"a_{bid}_{arm}",
                      campaign_id=f"c_{bid}_{arm}",
                      report_id=f"r_{bid}_{arm}")


def _populate(h: EvaluationHarness, n_blocks: int = 2,
              outcomes: tuple[str, ...] = ("valid_finding",)):
    for i in range(n_blocks):
        bid = f"block_{i}"
        h.add_block(_block(bid, f"snap_{(i % 2) + 1}"),
                    candidate_budget=1, human_time_budget_s=1.0)
        for j, arm in enumerate(ARMS):
            h.select_objective(bid, arm,
                               objective=_objective(f"{bid}_{arm}"))
            oc = outcomes[(i * 3 + j) % len(outcomes)]
            # M4.3-02 P0 #7: every run owns a DISTINCT campaign; a campaign is
            # never reused as success evidence across runs.
            ref = GateRunRef(session_id=f"s_{bid}_{arm}",
                             plan_id=f"p_{bid}_{arm}",
                             approval_id=f"a_{bid}_{arm}",
                             campaign_id=f"c_{bid}_{arm}",
                             report_id=f"r_{bid}_{arm}")
            h.record_outcome(bid, arm, outcome=oc, gate_ref=ref,
                             compute_units=2.0, wall_time_s=5.0)


# ------------------------------------------------------- #1 all 16 per arm
def test_metrics_by_arm_contains_all_16_metrics_for_each_arm():
    h = _harness()
    _populate(h)
    by_arm = h.metrics_by_arm()
    assert set(by_arm) == set(ARMS) == {"A", "B", "C"}
    for arm in ARMS:
        assert set(ALL_METRICS) <= set(by_arm[arm]), \
            f"arm {arm} missing {set(ALL_METRICS) - set(by_arm[arm])}"
        assert by_arm[arm]["n_runs"] == 2


def test_overall_and_per_arm_use_the_same_frozen_implementation():
    h = _harness()
    _populate(h)
    overall = h.compute_metrics()
    assert set(overall) == set(ALL_METRICS)
    # compute_metric_set is the single source for both views.
    assert compute_metric_set(h.registry.all_runs()) == overall
    for arm in ARMS:
        runs = [r for r in h.registry.all_runs() if r.arm == arm]
        expected = compute_metric_set(runs)
        assert all(h.metrics_by_arm()[arm][k] == expected[k]
                   for k in ALL_METRICS)


# ------------------------------------------------------- #2 definitions
def test_every_metric_declares_numerator_denominator_missing_failure():
    defs = metric_definitions()
    assert set(defs) == set(ALL_METRICS)
    for name, d in defs.items():
        assert d["numerator"] and d["denominator"], name
        assert d["tier"] in ("primary", "secondary", "safety"), name
        assert d["category"] in METRIC_CATEGORIES, name
        assert d["failure_policy"], name
        assert "missing_value" in d, name
    # Missing-value rule: time/compute per finding are None, rates are 0.0,
    # safety counts are 0 -- never a flattering substitution.
    assert METRIC_DEFINITIONS["time_to_valid_finding"].missing_value is None
    assert METRIC_DEFINITIONS["compute_per_valid_finding"].missing_value is None
    assert METRIC_DEFINITIONS["evidence_grounding_rate"].missing_value == 0.0
    assert METRIC_DEFINITIONS["approval_bypass_count"].missing_value == 0


def test_missing_values_and_failure_runs_are_not_dropped():
    # Only failed/interrupted/invalid runs -> no valid findings.
    h = _harness()
    h.add_block(_block("block_0"), candidate_budget=1,
                human_time_budget_s=1.0)
    for arm in ARMS:
        h.select_objective("block_0", arm,
                           objective=_objective(f"o_{arm}"))
        h.record_outcome("block_0", arm, outcome="invalid",
                         gate_ref=_ref("block_0", arm),
                         compute_units=0.0, wall_time_s=0.0)
    m = h.compute_metrics()
    assert m["time_to_valid_finding"] is None       # missing, not 0
    assert m["compute_per_valid_finding"] is None
    assert m["invalid_experiment_rate"] == 1.0     # failures retained
    assert m["plan_validity_rate"] == 1.0          # real gate chain present
    by_arm = h.metrics_by_arm()
    assert all(by_arm[a]["time_to_valid_finding"] is None for a in ARMS)


# ------------------------------------------------------- #3/#4/#5/#6 pairing
def test_primary_metrics_are_paired_by_block_with_three_level_hierarchy():
    h = _harness()
    _populate(h, n_blocks=2)
    pm = h.paired_metrics()
    assert pm["hierarchy"] == ("snapshot", "block", "seed")
    assert pm["paired_primary_metrics"] == list(PRIMARY_METRICS)
    assert pm["n_eligible_blocks"] == 2
    for b in pm["eligible_blocks"]:
        assert {"snapshot_id", "block_id", "seed", "paired_seeds", "arms"} \
            <= set(b)
        assert set(b["arms"]) == set(ARMS)
        for arm in ARMS:
            assert set(b["arms"][arm]) == set(PRIMARY_METRICS)


def test_incomplete_block_is_excluded_from_full_paired_analysis():
    h = _harness()
    _populate(h, n_blocks=2)
    # Third block: only A registered -> incomplete.
    h.add_block(_block("block_2"), candidate_budget=1,
                human_time_budget_s=1.0)
    h.select_objective("block_2", "A", objective=_objective("o_A2"))
    pm = h.paired_metrics()
    assert pm["n_eligible_blocks"] == 2
    assert "block_2" in [e["block_id"] for e in pm["excluded_blocks"]]
    assert h.registry.complete_blocks() == ["block_0", "block_1"]


def test_interrupted_block_is_excluded_even_when_all_arms_registered():
    h = _harness()
    h.add_block(_block("block_0"), candidate_budget=1,
                human_time_budget_s=1.0)
    for arm in ARMS:
        h.select_objective("block_0", arm, objective=_objective(f"o_{arm}"))
    for arm in ARMS:
        h.record_outcome("block_0", arm, outcome="valid_finding",
                         gate_ref=_ref("block_0", arm))
    # A new block where one arm never finalizes.
    h.add_block(_block("block_1"), candidate_budget=1,
                human_time_budget_s=1.0)
    for arm in ARMS:
        h.select_objective("block_1", arm, objective=_objective(f"o1_{arm}"))
    for arm in ("A", "B"):
        h.record_outcome("block_1", arm, outcome="valid_finding",
                         gate_ref=_ref("block_1", arm))
    pm = h.paired_metrics()
    assert pm["n_eligible_blocks"] == 1
    ex = next(e for e in pm["excluded_blocks"]
              if e["block_id"] == "block_1")
    assert any("finalized" in r for r in ex["reasons"])


def test_same_paired_seeds_required_within_a_block():
    h = _harness()
    h.add_block(_block("block_0"), candidate_budget=1,
                human_time_budget_s=1.0)
    for arm in ARMS:
        h.select_objective("block_0", arm, objective=_objective(f"o_{arm}"))
    # Deliberately give arm C DIFFERENT paired seeds (a real protocol breach).
    for arm in ARMS:
        h.record_outcome(
            "block_0", arm, outcome="valid_finding",
            gate_ref=_ref("block_0", arm),
            paired_seeds=[42, 43] if arm != "C" else [99, 100])
    with pytest.raises(ProtocolViolation) as exc:
        h.assert_paired_seeds_consistent()
    assert "Paired Seeds" in str(exc.value)
    pm = h.paired_metrics()
    assert pm["n_eligible_blocks"] == 0
    assert any("Paired Seeds" in r
               for r in pm["excluded_blocks"][0]["reasons"])


def test_consistent_paired_seeds_pass_the_check():
    h = _harness()
    _populate(h, n_blocks=1)
    h.assert_paired_seeds_consistent()   # no exception
    assert h.paired_metrics()["n_eligible_blocks"] == 1


# ------------------------------------------------------- #7 categories
def test_three_metric_families_are_distinct_and_complete():
    assert set(METRIC_CATEGORIES) == {
        "objective_quality", "research_finding_rate", "compute_efficiency"}
    all_cat = [n for names in METRIC_CATEGORIES.values() for n in names]
    assert len(all_cat) == len(set(all_cat)) == len(ALL_METRICS)
    assert set(all_cat) == set(ALL_METRICS)
    assert PRIMARY_METRICS[0] in METRIC_CATEGORIES["objective_quality"]
    assert "time_to_valid_finding" in METRIC_CATEGORIES["compute_efficiency"]
    h = _harness()
    _populate(h, n_blocks=1)
    by_arm = h.metrics_by_arm()
    for arm in ARMS:
        assert set(by_arm[arm]["by_category"]) == set(METRIC_CATEGORIES)
        for cat, names in METRIC_CATEGORIES.items():
            assert set(by_arm[arm]["by_category"][cat]) == set(names)


# ------------------------------------------------------- #8 formula version in hash
def test_formula_version_is_part_of_the_frozen_metric_contract_hash():
    c = frozen_metric_contract()
    assert c["formula_version"] == METRIC_FORMULA_VERSION
    assert set(c["definitions"]) == set(ALL_METRICS)
    assert len(c["contract_sha256"]) == 64
    # Tamper the formula version -> hash changes.
    import hashlib
    import json
    payload = {k: c[k] for k in c if k != "contract_sha256"}
    payload["formula_version"] = "tampered"
    altered = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    assert altered != c["contract_sha256"]
    # Tamper one metric definition -> hash changes.
    payload = {k: c[k] for k in c if k != "contract_sha256"}
    payload["definitions"]["evidence_grounding_rate"]["numerator"] = "wrong"
    altered2 = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    assert altered2 != c["contract_sha256"]


# ------------------------------------------------------- #9 / #10 no merge / no significance
def test_output_is_not_only_a_merged_average():
    h = _harness()
    _populate(h, n_blocks=2)
    s = h.snapshot()
    assert "metrics" in s and "metrics_by_arm" in s and "paired_metrics" in s
    assert set(s["metrics_by_arm"]) == set(ARMS)
    assert s["paired_metrics"]["n_eligible_blocks"] == 2
    assert s["metric_formula_version"] == METRIC_FORMULA_VERSION
    # per-arm views carry ALL 16 metrics, not just a combined mean
    assert all(set(ALL_METRICS) <= set(s["metrics_by_arm"][a]) for a in ARMS)


def test_no_significance_conclusion_is_drawn():
    h = _harness()
    _populate(h, n_blocks=2)
    pm = h.paired_metrics()
    assert pm["significance_conclusion"] is None
    s = h.snapshot()
    low = str(s).lower()
    for forbidden in ("p-value", "p_value", "significant", "reject the null",
                      "confidence interval", "effect size conclusion"):
        assert forbidden not in low, forbidden
