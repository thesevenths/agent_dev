"""M4.3-02.1 P1-02 -- fixed test family + conclusion consistency.

ENGINE-FREE: verifies the pre-registered correction family never shrinks when a
contrast is missing, the Primary/Secondary hierarchy is explicit, and neither a
secondary contrast nor a secondary metric can make significance_claimed True.
"""
from __future__ import annotations

import pytest

from research_runtime.m4_3 import ExperimentRegistry, FrozenProtocol
from research_runtime.m4_3.metrics import PRIMARY_METRICS, SECONDARY_METRICS
from research_runtime.m4_3.statistical_analysis import analyse_primary_contrasts
from research_runtime.m4_3.value_report import (
    DECISION_PRIMARY_METRIC, PRIMARY_CONTRAST_LABELS, SECONDARY_CONTRAST_LABELS,
    TEST_FAMILY_LABELS, VALUE_OUTCOMES, generate_value_report,
)
from research_runtime.research_store.db import ResearchDatabase

FAMILY = list(TEST_FAMILY_LABELS)


# ----------------------------------------------------------- fixed family
def test_missing_contrast_does_not_shrink_the_family():
    full = analyse_primary_contrasts(
        differences={"C-B": [0.5, 0.6, 0.55, 0.52],
                     "C-A": [0.1, 0.05, 0.2, 0.0],
                     "B-A": [0.0, 0.0, 0.0, 0.0]},
        family=FAMILY)
    partial = analyse_primary_contrasts(
        differences={"C-B": [0.5, 0.6, 0.55, 0.52]},  # C-A / B-A missing
        family=FAMILY)
    assert full["n_contrasts_in_family"] == partial["n_contrasts_in_family"] == 3
    assert partial["missing_contrasts"] == ["B-A", "C-A"] or \
        set(partial["missing_contrasts"]) == {"B-A", "C-A"}
    # A missing contrast is explicitly non-significant but STILL in family.
    for label in ("B-A", "C-A"):
        assert partial["contrasts"][label]["significant"] is False
        assert partial["contrasts"][label]["p_value"] is None
    # Effective alpha for C-B must be identical: family size did not change.
    assert full["contrasts"]["C-B"]["correction"]["effective_alpha"] == \
        partial["contrasts"]["C-B"]["correction"]["effective_alpha"]


def test_family_is_pre_registered_and_fixed_not_data_driven():
    # With NO data at all the family is still the frozen 3 contrasts.
    r = analyse_primary_contrasts(differences={}, family=FAMILY)
    assert r["family"] == FAMILY
    assert r["n_contrasts_in_family"] == 3
    assert r["n_contrasts_with_statistic"] == 0


# -------------------------------------------------------- hierarchy / gating
def _populate(db, *, quality_sig_metric=None, secondary_only=False):
    """3 complete blocks; secondary contrast C-A can be made the only signal."""
    proto = FrozenProtocol(snapshot_ids=("s1",), sample_size=0, seed=42,
                           model_version="m", prompt_version="p",
                           ranker_version="r", paired_seeds=(42, 43),
                           exclusion_rules=("none",),
                           statistical_method="paired-block",
                           stopping_rule="all blocks")
    reg = ExperimentRegistry(protocol=proto, db=db)
    for i in range(1, 6):
        bid = f"b{i}"
        reg.create_block(block_id=bid, snapshot_id="s1", seed=40 + i)
        for arm in ("A", "B", "C"):
            reg.register_run(block_id=bid, arm=arm)
            rid = f"{bid}_{arm}"
            reg.finalize(rid, outcome="valid_finding",
                         campaign_id=f"c_{rid}", plan_id=f"p_{rid}",
                         approval_id=f"a_{rid}", report_id=f"rep_{rid}",
                         experience_id=f"e_{rid}", gate_steps_passed=12,
                         external_approval=True, compute_units=1.0,
                         wall_time_s=10.0, paired_seeds=[42, 43],
                         gate_evidence={"gate_passed": True})
    return reg


def test_hierarchy_is_declared_in_the_design(tmp_path):
    db = ResearchDatabase(tmp_path / "r.sqlite")
    _populate(db)
    r = generate_value_report(db=db, out_dir=tmp_path, stage="final",
                              write=False,
                              protocol={"protocol_sha256": "b" * 64})
    h = r["design"]["testing_hierarchy"]
    assert h["primary_metrics"] == list(PRIMARY_METRICS)
    assert h["secondary_metrics"] == list(SECONDARY_METRICS)
    assert h["primary_contrasts"] == list(PRIMARY_CONTRAST_LABELS)
    assert h["secondary_contrasts"] == list(SECONDARY_CONTRAST_LABELS)
    assert h["decision_primary_contrast"] == PRIMARY_CONTRAST_LABELS[0]
    assert r["design"]["test_family"] == FAMILY
    assert r["design"]["correction_family_size"] == 3


def test_significance_uses_only_primary_contrast_and_primary_metrics(tmp_path):
    db = ResearchDatabase(tmp_path / "r.sqlite")
    _populate(db)
    r = generate_value_report(db=db, out_dir=tmp_path, stage="final",
                              write=False)
    basis = r["primary"]["significance_basis"]
    assert basis["allowed_contrasts"] == list(PRIMARY_CONTRAST_LABELS)
    assert set(basis["excluded_contrasts"]) == set(SECONDARY_CONTRAST_LABELS)
    assert set(basis["excluded_metrics"]) == set(SECONDARY_METRICS)
    assert set(basis["allowed_metrics"]) == set(PRIMARY_METRICS)
    # The report-level and conclusion-level claims agree.
    assert r["primary"]["significance_claimed"] == \
        r["conclusion"]["significance_claimed"]


def test_conclusion_references_the_primary_contrast(tmp_path):
    db = ResearchDatabase(tmp_path / "r.sqlite")
    _populate(db)
    r = generate_value_report(db=db, out_dir=tmp_path, stage="final",
                              write=False)
    assert r["conclusion"]["decision_basis"]["decision_contrast"] == \
        PRIMARY_CONTRAST_LABELS[0]
    assert r["conclusion"]["outcome"] in VALUE_OUTCOMES


def test_secondary_only_significance_cannot_claim_a_value(tmp_path):
    """C-A / B-A significant but the PRIMARY contrast C-B is zero.

    significance_claimed MUST stay False and the outcome must NOT be one of
    the two positive value outcomes -- a secondary contrast is not a decision
    driver (P1-02).
    """
    db = ResearchDatabase(tmp_path / "r.sqlite")
    proto = FrozenProtocol(snapshot_ids=("s1",), sample_size=0, seed=42,
                           model_version="m", prompt_version="p",
                           ranker_version="r", paired_seeds=(42, 43),
                           exclusion_rules=("none",),
                           statistical_method="paired-block",
                           stopping_rule="all blocks")
    reg = ExperimentRegistry(protocol=proto, db=db)
    for i in range(1, 6):                       # 5 paired blocks
        bid = f"b{i}"
        reg.create_block(block_id=bid, snapshot_id="s1", seed=40 + i)
        for arm, total in (("A", 1.0), ("B", 3.0), ("C", 3.0)):  # C == B
            reg.register_run(block_id=bid, arm=arm)
            rid = f"{bid}_{arm}"
            qr = {"total": total, "reviewer_id": "r1",
                  "review_kind": "human_blind", "scores": {}, "reason": "x"}
            reg.finalize(rid, outcome="negative_finding", quality_review=qr,
                         campaign_id=f"c_{rid}", plan_id=f"p_{rid}",
                         approval_id=f"a_{rid}", report_id=f"rep_{rid}",
                         experience_id=f"e_{rid}", gate_steps_passed=12,
                         external_approval=True, compute_units=1.0,
                         wall_time_s=10.0, paired_seeds=[42, 43],
                         gate_evidence={"gate_passed": True})
    r = generate_value_report(db=db, out_dir=tmp_path, stage="final",
                              write=False)
    m = DECISION_PRIMARY_METRIC
    assert r["primary"]["contrasts"][m]["contrasts"]["C-B"]["significant"] is False
    assert r["primary"]["contrasts"][m]["contrasts"]["C-A"]["significant"] is True
    assert r["primary"]["significance_claimed"] is False
    assert r["conclusion"]["significance_claimed"] is False
    assert r["conclusion"]["outcome"] == VALUE_OUTCOMES[2]   # no clear value
