"""M4.3-02.1 P1-01 -- dual primary-metric sample size.

ENGINE-FREE: verifies both primary metrics get their OWN paired differences,
variance, SEOI and sample size; zero variance / missing pairs / no human review
have explicit statuses; the recommendation is the most conservative across
metrics; and inability to estimate stays evidence_insufficient.
"""
from __future__ import annotations

import pytest

from research_runtime.m4_3.metrics import PRIMARY_METRICS
from research_runtime.m4_3.pilot_analysis import analyze_pilot
from research_runtime.m4_3.sample_size import (
    VARIANCE_ESTIMATED, VARIANCE_INSUFFICIENT, VARIANCE_NO_HUMAN_REVIEW,
    VARIANCE_ZERO, estimate_dual_primary_sample_size,
)
from research_runtime.research_store.db import ResearchDatabase

Q = "independently_reviewed_objective_quality"
F = "valid_research_findings_per_compute_budget"
CS = ("B-A", "C-B", "C-A")
SEOI = {Q: 0.5, F: 0.01}


def _est(paired, *, seoi=SEOI, human=True):
    return estimate_dual_primary_sample_size(
        paired_by_metric=paired, seoi_by_metric=dict(seoi),
        required_metrics=(Q, F), contrasts=CS,
        human_reviews_present=human)


def _per_contrast(vals):
    return {c: list(vals) for c in CS}


def test_seoi_valid_findings_is_actually_used():
    """Changing the findings SEOI must change the findings sample size."""
    paired = {Q: _per_contrast([0.1, 0.2, 0.3]),
              F: _per_contrast([0.001, 0.004, 0.009])}
    a = _est(paired, seoi={Q: 0.5, F: 0.01})
    b = _est(paired, seoi={Q: 0.5, F: 0.05})
    na = a["per_metric"][F]["per_contrast"]["C-B"]["n_blocks"]
    nb = b["per_metric"][F]["per_contrast"]["C-B"]["n_blocks"]
    assert na is not None and nb is not None
    assert nb < na                       # larger SEOI -> fewer blocks
    assert a["seoi_by_metric"][F] == 0.01
    assert b["seoi_by_metric"][F] == 0.05


def test_each_metric_uses_its_own_variance_not_the_others():
    paired = {Q: _per_contrast([0.1, 0.3, 0.5]),   # var ~0.04
              F: _per_contrast([0.001, 0.002, 0.003])}  # var ~1e-6
    r = _est(paired)
    qv = r["per_metric"][Q]["per_contrast"]["C-B"]["variance_used"]
    fv = r["per_metric"][F]["per_contrast"]["C-B"]["variance_used"]
    assert qv != fv
    assert qv > fv
    # findings' variance is its OWN, not copied from quality
    assert fv == pytest.approx(1e-6, rel=1e-3)


def test_zero_variance_is_not_a_known_true_variance():
    paired = {Q: _per_contrast([0.1, 0.2, 0.3]),
              F: _per_contrast([0.0, 0.0, 0.0])}
    r = _est(paired)
    row = r["per_metric"][F]["per_contrast"]["C-B"]
    assert row["variance_status"] == VARIANCE_ZERO
    assert row["n_blocks"] is None
    assert r["per_metric"][F]["status"] == "evidence_insufficient"
    assert r["status"] == "evidence_insufficient"
    assert r["n_blocks_recommended"] is None


def test_missing_paired_data_gets_explicit_status():
    paired = {Q: _per_contrast([0.1, 0.2, 0.3]),
              F: {"B-A": [], "C-B": [0.1], "C-A": []}}
    r = _est(paired)
    assert r["per_metric"][F]["per_contrast"]["C-B"]["variance_status"] == \
        VARIANCE_INSUFFICIENT
    assert r["status"] == "evidence_insufficient"
    assert r["n_blocks_recommended"] is None


def test_no_human_review_blocks_quality_metric():
    paired = {Q: {c: [] for c in CS}, F: _per_contrast([0.1, 0.2, 0.3])}
    r = _est(paired, human=False)
    qrow = r["per_metric"][Q]["per_contrast"]["C-B"]
    assert qrow["variance_status"] == VARIANCE_NO_HUMAN_REVIEW
    assert r["per_metric"][Q]["status"] == "evidence_insufficient"
    # findings CAN be estimated but the overall stays insufficient
    assert r["per_metric"][F]["status"] == "estimated"
    assert r["status"] == "evidence_insufficient"


def test_overall_recommendation_is_most_conservative():
    # quality needs many blocks (tiny SEOI); findings needs few
    paired = {Q: _per_contrast([0.2, 0.4, 0.6]),
              F: _per_contrast([0.002, 0.004, 0.008])}
    r = _est(paired, seoi={Q: 0.05, F: 0.5})
    assert r["status"] == "estimated"
    qn = r["per_metric"][Q]["n_blocks_recommended"]
    fn = r["per_metric"][F]["n_blocks_recommended"]
    assert qn is not None and fn is not None
    assert r["n_blocks_recommended"] == max(qn, fn)
    assert r["recommendation_rule"] == "max over all required primary metrics"


def test_variance_source_assumptions_and_sensitivity_present():
    paired = {Q: _per_contrast([0.1, 0.2, 0.3]),
              F: _per_contrast([0.001, 0.002, 0.003])}
    r = _est(paired)
    for metric in (Q, F):
        m = r["per_metric"][metric]
        assert m["variance_source"] == "pilot_paired_differences"
        assert m["assumptions"] and any("SEOI" in a for a in m["assumptions"])
        assert m["per_contrast"]["C-B"]["sensitivity"]
    assert f"{F}::C-B" in r["sensitivity_grid"]
    assert VARIANCE_ESTIMATED in r["variance_statuses"]


def test_analyze_pilot_uses_both_seois_and_reports_per_metric(tmp_path):
    """The full analysis pipeline exposes per-metric sample sizes."""
    from research_runtime.m4_3 import ExperimentRegistry, FrozenProtocol
    proto = FrozenProtocol(snapshot_ids=("s1",), sample_size=0, seed=42,
                           model_version="m", prompt_version="p",
                           ranker_version="r", paired_seeds=(42, 43),
                           exclusion_rules=("none",),
                           statistical_method="paired-block",
                           stopping_rule="all blocks")
    reg = ExperimentRegistry(protocol=proto, db=ResearchDatabase(
        tmp_path / "r.sqlite"))
    for i in range(1, 4):
        bid = f"b{i}"
        reg.create_block(block_id=bid, snapshot_id="s1", seed=40 + i)
        for arm in ("A", "B", "C"):
            reg.register_run(block_id=bid, arm=arm)
            rid = f"{bid}_{arm}"
            quality = {"total": 2.0 + (0.1 * i if arm == "C" else 0.0),
                       "reviewer_id": "r1", "review_kind": "human_blind",
                       "scores": {}, "reason": "x"}
            reg.finalize(rid, outcome="negative_finding", quality_review=quality,
                         campaign_id=f"c_{rid}", plan_id=f"p_{rid}",
                         approval_id=f"a_{rid}", report_id=f"rep_{rid}",
                         experience_id=f"e_{rid}", gate_steps_passed=12,
                         external_approval=True, compute_units=1.0,
                         wall_time_s=10.0, paired_seeds=[42, 43],
                         gate_evidence={"gate_passed": True})
    db = reg.store.db
    a = analyze_pilot(db=db, out_dir=tmp_path, write=False,
                      seoi_quality=0.5, seoi_valid_findings=0.01)
    s = a["sample_size"]
    assert set(s["primary_metrics"]) == set(PRIMARY_METRICS)
    assert s["seoi_by_metric"][Q] == 0.5
    assert s["seoi_by_metric"][F] == 0.01
    assert set(s["per_metric"]) == set(PRIMARY_METRICS)
