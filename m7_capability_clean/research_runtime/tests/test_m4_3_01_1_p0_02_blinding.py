"""M4.3-01.1 P0-02 -- GENUINELY anonymous, blind, independent review.

Covers the spec's seven required behaviours:
  * objective_id does not leak the arm
  * review_id cannot be reversed to an arm
  * metadata / description leaks are intercepted
  * Ranker scores cannot reach the reviewer
  * arm mapping is unreadable before unblinding
  * two reviewers' results are stored separately
  * a completed review cannot be silently overwritten
plus: >= 2 independent reviewers (Pilot default), human-blind vs automated
separation, required reviewer_id / rubric_version / reason / timestamp.
"""
from __future__ import annotations

import pytest

from research_runtime.m4_3 import (
    ARMS, BlockSpec, EvaluationHarness, FrozenProtocol, ProtocolViolation,
)
from research_runtime.m4_3.blind_review import (
    FORBIDDEN_BLIND_FIELDS, REVIEW_KIND_AUTOMATED, REVIEW_KIND_HUMAN_BLIND,
    REVIEW_RUBRIC_VERSION, BlindedObjective, BlindingLedger, BlindingViolation,
    IndependentReviewer, QualityReview, ReviewPanel, blind,
)
from research_runtime.m4_3.metrics import frozen_metric_contract
from research_runtime.m4_3.research_service_gate import GateRunRef
from tests.rt_tests.m4_3_gate import complete_gate_evidence


def _dims():
    return frozen_metric_contract()["rubric_dimensions"]


def _scorer(value: int = 3):
    def score(_b):
        return {d: value for d in _dims()} | {
            "reason": "grounded, falsifiable, measurable"}
    return score


def _reviewer(rid: str, value: int = 3,
              kind: str = REVIEW_KIND_HUMAN_BLIND) -> IndependentReviewer:
    return IndependentReviewer(reviewer_id=rid, scorer=_scorer(value),
                               review_kind=kind)


def _objective(oid: str = "obj_1", description: str = "reduce prediction error",
               **extra) -> dict:
    return {"objective_id": oid, "description": description,
            "evidence_refs": ["sha256:real"],
            "success_criteria": {"metric": "mse@1", "measurable": True},
            "proposed_intervention": {"axis": "regularisation"}, **extra}


def _protocol() -> FrozenProtocol:
    return FrozenProtocol(
        snapshot_ids=("snap_1",), sample_size=0, seed=42,
        model_version="m1", prompt_version="p1",
        ranker_version="deterministic-v1",
        budget={"paired_seeds": [42, 43]}, exclusion_rules=("none",),
        statistical_method="paired-block", stopping_rule="all blocks complete")


def _block() -> BlockSpec:
    return BlockSpec(block_id="block_1", snapshot_id="snap_1", seed=42,
                     current_evidence={"n": 1})


class _NullGate:
    """Read-only gate stub that always reports complete evidence.

    Used only to let harness-level blinding tests record completed outcomes;
    the REAL gate adapter is covered exhaustively by P0-01 tests.
    """

    def __init__(self):
        from research_runtime.m4_3.research_service_gate import GateEvidence
        self._ev_cls = GateEvidence
        self.inspect_calls = []

    def inspect(self, ref):
        from research_runtime.m4_3.research_service_gate import GateEvidence
        self.inspect_calls.append(ref)
        return complete_gate_evidence(
            campaign_id=ref.campaign_id or "c1",
            plan_id=ref.plan_id or "p1",
            approval_id=ref.approval_id or "a1",
            session_id=ref.session_id or "s1",
            report_id=ref.report_id or "r1")


def _harness(**kw) -> EvaluationHarness:
    base = dict(protocol=_protocol(), reviewer=_reviewer("r1"),
                gate_adapter=_NullGate(), default_gate_ref=GateRunRef())
    base.update(kw)
    return EvaluationHarness(**base)


# ------------------------------------------------------- #1 objective_id / #2 review_id
def test_blinded_payload_never_contains_objective_id():
    b = blind(_objective("block_1_A"))
    assert "objective_id" not in b.to_dict()
    assert "block_1_A" not in str(b.to_dict())
    for forbidden in FORBIDDEN_BLIND_FIELDS:
        assert forbidden not in b.to_dict()


def test_review_id_is_opaque_and_cannot_reverse_to_arm():
    ids = set()
    for arm in ARMS:
        for _ in range(20):
            b = blind(_objective(f"block_7_{arm}"))
            assert b.review_id.startswith("rev_")
            assert arm not in b.review_id
            assert "block_7" not in b.review_id
            assert b.review_id not in ids
            ids.add(b.review_id)
    assert len(ids) == 60   # no collisions, no arm encoding


def test_reviewer_sees_only_blinded_objective():
    seen = []

    def scorer(b):
        seen.append(b)
        assert isinstance(b, BlindedObjective)
        assert "objective_id" not in b.to_dict()
        return _scorer(3)(b)

    r = IndependentReviewer(reviewer_id="r1", scorer=scorer)
    r.review(_objective("block_1_C"))
    assert len(seen) == 1


# ------------------------------------------------------- #3 leak interception
@pytest.mark.parametrize("leak", [
    "block_1_A", "this is arm C", "group: B", "objective_9_A",
    "rank_score was high", "ranker_version v2", "generation_source x",
    "snapshot_id s1", "block_id b1",
])
def test_description_and_metadata_leaks_are_intercepted(leak):
    with pytest.raises(BlindingViolation):
        blind(_objective(description=f"investigate {leak} on held-out split"))


@pytest.mark.parametrize("field", [
    "success_criteria", "proposed_intervention", "evidence_refs",
])
def test_nested_metadata_leaks_are_intercepted(field):
    payload = {
        "success_criteria": {"metric": "mse@1"},
        "proposed_intervention": {"axis": "reg"},
        "evidence_refs": ["sha256:real"],
    }
    if field == "evidence_refs":
        payload[field] = ["block_2_C"]
    else:
        payload[field] = {"note": "arm A", "value": 1}
    with pytest.raises(BlindingViolation):
        blind(_objective(**payload))


def test_leak_in_metadata_key_is_intercepted():
    with pytest.raises(BlindingViolation):
        blind(_objective(success_criteria={"rank_score": 0.9}))


# ------------------------------------------------------- #4 Ranker separation
@pytest.mark.parametrize("field", [
    "arm", "group", "rank_score", "rank_breakdown", "ranker_version",
    "generation_source", "snapshot_id", "block_id",
])
def test_ranker_and_group_fields_never_reach_review(field):
    with pytest.raises(BlindingViolation):
        blind(_objective(**{field: "leak"}))


def test_ranker_score_cannot_become_a_review_score():
    bad = IndependentReviewer(
        reviewer_id="r", scorer=lambda _b: {"rank_score": 1.0,
                                            "reason": "from ranker"})
    with pytest.raises(ValueError):
        bad.review(_objective())


def test_automated_scores_are_never_mixed_into_human_blind_metric():
    contract = frozen_metric_contract()
    assert contract["review_kinds"] == [REVIEW_KIND_HUMAN_BLIND,
                                        REVIEW_KIND_AUTOMATED]
    panel = ReviewPanel([
        _reviewer("human_1", 3),
        _reviewer("human_2", 1),
        _reviewer("ranker_auto", 4, REVIEW_KIND_AUTOMATED),
    ])
    led = BlindingLedger()
    blinded = led.seal(_objective(), arm="C", block_id="b1",
                       snapshot_id="s1")
    reviews = panel.review(blinded)
    for r in reviews:
        led.record_review(r)
    humans = led.human_blind_reviews(blinded.review_id)
    assert len(humans) == 2
    assert all(r.is_human_blind for r in humans)
    automated = [r for r in reviews if r.review_kind == REVIEW_KIND_AUTOMATED]
    assert automated and not any(r.is_human_blind for r in automated)
    # The automated score (4.0) is NOT in the human-blind average (2.0).
    human_mean = sum(r.total for r in humans) / len(humans)
    assert human_mean == 2.0
    assert all(r.total != human_mean for r in automated)
    # Harness keeps them in SEPARATE fields.
    h = _harness(review_panel=panel, blinding=BlindingLedger())
    h.add_block(_block(), candidate_budget=1, human_time_budget_s=1.0)
    h.select_objective("block_1", "A", objective=_objective())
    rec = h.registry.runs["block_1_A"]
    assert len(rec.independent_reviews) == 2
    assert len(rec.automated_scores) == 1
    assert rec.quality_review["review_kind"] == REVIEW_KIND_HUMAN_BLIND


# ------------------------------------------------------- #5 unblinding
def test_arm_mapping_is_unreadable_before_unblinding():
    led = BlindingLedger()
    b = led.seal(_objective("obj_secret"), arm="C", block_id="b1",
                 snapshot_id="snap_1")
    with pytest.raises(BlindingViolation) as exc:
        led.arm_mapping(b.review_id)
    assert "sealed" in str(exc.value).lower()
    assert led.export()["entries"] == "SEALED"
    assert led.unblinded is False


def test_unblind_is_refused_before_review_completion():
    led = BlindingLedger()
    b = led.seal(_objective(), arm="A")
    r1 = _reviewer("r1").review(b, review_id=b.review_id)
    led.record_review(r1)
    with pytest.raises(BlindingViolation) as exc:
        led.unblind()
    assert "before review completion" in str(exc.value)


def test_unblind_after_two_independent_reviews_reveals_mapping():
    led = BlindingLedger()
    b = led.seal(_objective("obj_x"), arm="B", block_id="blk_9",
                 snapshot_id="snap_1")
    for rid in ("alice", "bob"):
        led.record_review(
            _reviewer(rid).review(b, review_id=b.review_id))
    led.unblind()
    assert led.unblinded is True
    mapping = led.arm_mapping(b.review_id)
    assert mapping["objective_id"] == "obj_x" and mapping["arm"] == "B"
    assert mapping["block_id"] == "blk_9"


def test_harness_unblind_requires_completed_reviews():
    h = _harness(review_panel=ReviewPanel([_reviewer("r1"), _reviewer("r2")]),
                 blinding=BlindingLedger())
    h.add_block(_block(), candidate_budget=1, human_time_budget_s=1.0)
    obj = h.select_objective("block_1", "A", objective=_objective())
    rid = h.registry.runs["block_1_A"].review_id
    with pytest.raises(BlindingViolation):
        h.arm_mapping(rid)          # sealed
    h.unblind()
    assert h.arm_mapping(rid)["arm"] == "A"
    assert obj["objective_id"] == "obj_1"


# ------------------------------------------------------- #6 two reviewers separate
def test_two_reviewers_results_are_stored_separately():
    panel = ReviewPanel([_reviewer("alice", 4), _reviewer("bob", 1)])
    led = BlindingLedger()
    b = led.seal(_objective(), arm="C")
    for r in panel.review(b):
        led.record_review(r)
    reviews = led.reviews_for(b.review_id)
    assert len(reviews) == 2
    assert {r.reviewer_id for r in reviews} == {"alice", "bob"}
    totals = {r.reviewer_id: r.total for r in reviews}
    assert totals == {"alice": 4.0, "bob": 1.0}
    assert panel.is_independent


def test_pilot_default_requires_two_independent_reviewers():
    single = _harness(reviewer=_reviewer("r1"))
    with pytest.raises(ProtocolViolation):
        single.assert_pilot_review_ready()
    dual = _harness(review_panel=ReviewPanel(
        [_reviewer("alice"), _reviewer("bob")]))
    dual.assert_pilot_review_ready()   # no exception
    with pytest.raises(ValueError):
        ReviewPanel([_reviewer("same"), _reviewer("same")])
    with pytest.raises(ValueError):
        ReviewPanel([_reviewer("a")], min_reviewers=1)


def test_automated_only_panel_is_not_independent():
    panel = ReviewPanel([
        _reviewer("auto1", 4, REVIEW_KIND_AUTOMATED),
        _reviewer("auto2", 4, REVIEW_KIND_AUTOMATED)])
    assert panel.is_independent is False


# ------------------------------------------------------- #7 append-only reviews
def test_completed_review_cannot_be_silently_overwritten():
    led = BlindingLedger()
    b = led.seal(_objective(), arm="A")
    original = _reviewer("alice", 4).review(b, review_id=b.review_id)
    led.record_review(original)
    duplicate = QualityReview(
        review_id=original.review_id, reviewer_id="alice",
        scores={d: 0 for d in _dims()}, reason="changed my mind",
        reviewed_at="later")
    with pytest.raises(BlindingViolation) as exc:
        led.record_review(duplicate)
    assert "cannot be overwritten" in str(exc.value)
    # Original result intact.
    assert led.reviews_for(b.review_id) == (original,)
    assert led.reviews_for(b.review_id)[0].total == 4.0


def test_review_records_required_audit_fields():
    r = _reviewer("alice").review(_objective())
    assert r.reviewer_id == "alice"
    assert r.rubric_version == REVIEW_RUBRIC_VERSION
    assert r.reason
    assert r.reviewed_at and "T" in r.reviewed_at    # ISO timestamp
    assert r.review_kind == REVIEW_KIND_HUMAN_BLIND
    assert set(r.to_dict()) >= {"review_id", "reviewer_id", "rubric_version",
                                "reason", "reviewed_at", "review_kind"}


def test_protocol_yaml_blinding_contract_matches_code():
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "M4_3_runs" / "protocol.yaml")
        .read_text(encoding="utf-8"))
    ir = cfg["independent_review"]
    assert ir["opaque_review_id"] is True
    assert ir["objective_id_excluded_from_payload"] is True
    assert ir["arm_mapping_separately_sealed"] is True
    assert ir["unblind_only_after_review_completion"] is True
    assert ir["min_independent_reviewers"] == ReviewPanel(
        [_reviewer("a"), _reviewer("b")]).min_reviewers
    assert ir["append_only_reviews"] is True
    assert list(ir["review_kinds"]) == frozen_metric_contract()["review_kinds"]
    assert set(ir["review_record_fields"]) == {
        "reviewer_id", "rubric_version", "reason", "reviewed_at"}
    assert ir["human_blind_and_automated_metrics_separate"] is True


def test_blinding_is_additive_no_researchservice_change():
    """The sealed mapping lives in the harness, not in ResearchService."""
    from research_runtime.m4_3 import BlindingLedger as Exported
    assert Exported is BlindingLedger
    # No objective_id appears anywhere in a reviewer-visible payload.
    h = _harness(review_panel=ReviewPanel([_reviewer("a"), _reviewer("b")]))
    h.add_block(_block(), candidate_budget=1, human_time_budget_s=1.0)
    h.select_objective("block_1", "A", objective=_objective("block_1_A"))
    rec = h.registry.runs["block_1_A"]
    assert rec.review_id.startswith("rev_")
    assert "objective_id" not in rec.quality_review
    assert all("objective_id" not in r for r in rec.independent_reviews)
