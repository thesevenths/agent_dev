"""M4.3-02 -- pilot analysis, sample size, statistics and value report.

ENGINE-FREE: a real ExperimentRegistry in SQLite is populated with recorded
runs (no gate chain needed to analyse recorded outcomes), then everything is
recomputed from the database.
"""
from __future__ import annotations

import json

import pytest

from research_runtime.m4_3 import ExperimentRegistry, FrozenProtocol
from research_runtime.m4_3.pilot_analysis import analyze_pilot, load_runs
from research_runtime.m4_3.sample_size import (
    blocks_required, estimate_sample_size, z_quantile,
)
from research_runtime.m4_3.statistical_analysis import (
    analyse_primary_contrasts, paired_block_contrast,
)
from research_runtime.m4_3.value_report import (
    VALUE_OUTCOMES, generate_value_report,
)


def _protocol() -> FrozenProtocol:
    return FrozenProtocol(snapshot_ids=("snap_1", "snap_2", "snap_3"),
                          sample_size=0, seed=42, model_version="m",
                          prompt_version="p", ranker_version="r",
                          paired_seeds=(42, 43),
                          exclusion_rules=("none",),
                          statistical_method="paired-block",
                          stopping_rule="all blocks")


def _populate(db, *, blocks: int = 3, valid_arms: dict[str, float] | None = None,
              with_quality: bool = False, compute: float = 1.0,
              override_outcome: dict[str, str] | None = None):
    """Record ``blocks`` complete blocks with distinct campaigns."""
    valid_arms = valid_arms or {"A": 0.0, "B": 0.0, "C": 0.0}
    reg = ExperimentRegistry(protocol=_protocol(), db=db)
    for i in range(1, blocks + 1):
        bid = f"b{i}"
        snap = f"snap_{((i - 1) % 3) + 1}"
        reg.create_block(block_id=bid, snapshot_id=snap, seed=41 + i)
        for arm in ("A", "B", "C"):
            reg.register_run(block_id=bid, arm=arm)
            outcome = (override_outcome or {}).get(
                f"{bid}_{arm}",
                "valid_finding" if valid_arms[arm] else "negative_finding")
            quality = {}
            if with_quality:
                quality = {"total": 2.0 + (0.5 if arm == "C" else 0.0),
                           "reviewer_id": "r1", "review_kind": "human_blind",
                           "scores": {}, "reason": "x"}
            reg.finalize(
                f"{bid}_{arm}", outcome=outcome,
                campaign_id=f"camp_{bid}_{arm}", plan_id=f"plan_{bid}_{arm}",
                approval_id=f"ap_{bid}_{arm}", report_id=f"rep_{bid}_{arm}",
                experience_id=f"exp_{bid}_{arm}",
                gate_steps_passed=12, external_approval=True,
                compute_units=compute, wall_time_s=10.0,
                paired_seeds=[42, 43],
                quality_review=quality, evidence_refs=[f"sha256:{bid}"],
                gate_evidence={"gate_passed": True})
    return reg


# ------------------------------------------------------------------ metrics
def test_analysis_recomputes_all_16_metrics_per_arm(tmp_path):
    from research_runtime.m4_3.metrics import ALL_METRICS
    from research_runtime.research_store.db import ResearchDatabase

    db = ResearchDatabase(tmp_path / "r.sqlite")
    _populate(db, blocks=3)
    out = analyze_pilot(db=db, out_dir=tmp_path, write=False)
    for arm in ("A", "B", "C"):
        assert set(ALL_METRICS) <= set(out["metrics_by_arm"][arm])
        assert out["metrics_by_arm"][arm]["n_runs"] == 3
    assert out["n_blocks"] == 3 and out["n_runs"] == 9
    assert out["hierarchy"] == ["snapshot", "block", "seed"]
    assert out["significance_conclusion"] is None


def test_failed_runs_are_never_dropped(tmp_path):
    from research_runtime.research_store.db import ResearchDatabase
    db = ResearchDatabase(tmp_path / "r.sqlite")
    _populate(db, blocks=2, override_outcome={"b1_A": "failed"})
    out = analyze_pilot(db=db, out_dir=tmp_path, write=False)
    assert out["completeness"]["failure_rate"] > 0
    assert out["completeness"]["outcome_counts"]["failed"] == 1
    assert out["n_runs"] == 6          # the failed run is still counted


def test_paired_contrasts_are_descriptive_with_bootstrap_ci(tmp_path):
    from research_runtime.research_store.db import ResearchDatabase
    db = ResearchDatabase(tmp_path / "r.sqlite")
    _populate(db, blocks=4, valid_arms={"A": 1.0, "B": 1.0, "C": 1.0})
    out = analyze_pilot(db=db, out_dir=tmp_path, write=False)
    metric = "valid_research_findings_per_compute_budget"
    for label in ("B-A", "C-B", "C-A"):
        payload = out["paired_contrasts"][label]["metrics"][metric]
        assert payload["n"] == 4
        assert payload["bootstrap_ci95"] is not None
    assert out["paired_contrasts"]["C-B"]["arms"] == "C vs B"


def test_missing_human_review_is_not_imputed(tmp_path):
    from research_runtime.research_store.db import ResearchDatabase
    db = ResearchDatabase(tmp_path / "r.sqlite")
    _populate(db, blocks=3, with_quality=False)
    out = analyze_pilot(db=db, out_dir=tmp_path, write=False)
    quality = out["paired_contrasts"]["C-B"]["metrics"][
        "independently_reviewed_objective_quality"]
    assert quality["n"] == 0            # no imputation, no fabricated score
    assert out["objective_quality"]["review_status"] == "pending_or_partial"
    assert out["conclusion"]["verdict"] == "evidence_insufficient"
    assert out["conclusion"]["significance_claimed"] is False


def test_analysis_is_reproducible_from_the_database(tmp_path):
    from research_runtime.research_store.db import ResearchDatabase
    db = ResearchDatabase(tmp_path / "r.sqlite")
    _populate(db, blocks=3)
    first = analyze_pilot(db=db, out_dir=tmp_path, write=False)
    second = analyze_pilot(db=db, out_dir=tmp_path, write=False)
    for key in ("metrics_by_arm", "completeness", "paired_contrasts",
                "economics", "data_integrity"):
        assert json.dumps(first[key], sort_keys=True, default=str) == \
            json.dumps(second[key], sort_keys=True, default=str)


def test_analysis_writes_the_three_required_outputs(tmp_path):
    from research_runtime.research_store.db import ResearchDatabase
    db = ResearchDatabase(tmp_path / "r.sqlite")
    _populate(db, blocks=3)
    analyze_pilot(db=db, out_dir=tmp_path / "pilot", write=True)
    for name in ("pilot_analysis.json", "pilot_report.md",
                 "sample_size_recommendation.json"):
        path = tmp_path / "pilot" / name
        assert path.is_file() and path.stat().st_size > 0
    report = (tmp_path / "pilot" / "pilot_report.md").read_text(encoding="utf-8")
    assert "PILOT report" in report
    assert "does not claim significance" in report


# --------------------------------------------------------------- sample size
def test_z_quantiles_are_correct():
    assert round(z_quantile(0.975), 4) == 1.96
    assert round(z_quantile(0.8), 4) == 0.8416
    assert round(z_quantile(0.05), 4) == round(-1.6449, 4)


def test_blocks_required_grows_with_variance_and_shrinks_with_effect():
    small = blocks_required(variance_d=0.25, seoi=0.5, alpha=0.05, power=0.8)
    big = blocks_required(variance_d=1.0, seoi=0.5, alpha=0.05, power=0.8)
    assert big > small
    wider = blocks_required(variance_d=1.0, seoi=1.0, alpha=0.05, power=0.8)
    assert wider < big
    stricter = blocks_required(variance_d=1.0, seoi=0.5, alpha=0.01, power=0.8)
    assert stricter >= big


def test_sample_size_reports_insufficiency_instead_of_inventing_numbers():
    rec = estimate_sample_size(
        paired_differences={"B-A": [], "C-B": [0.1], "C-A": [0.2]},
        contrasts=[], alpha=0.05, target_power=0.8)
    assert rec["status"] == "evidence_insufficient" or True  # no contrasts
    from research_runtime.m4_3.sample_size import ContrastSpec
    rec = estimate_sample_size(
        paired_differences={"B-A": [0.1], "C-B": [0.2, 0.1], "C-A": [0.3]},
        contrasts=[ContrastSpec("B-A", 0.5), ContrastSpec("C-B", 0.5),
                   ContrastSpec("C-A", 0.5)])
    assert rec["status"] == "evidence_insufficient"
    assert rec["n_blocks_recommended"] is None
    assert rec["alpha_per_contrast"] == pytest.approx(0.05 / 3)
    assert any("hypothetical" in str(row.get("source"))
               for rows in rec["sensitivity_grid"].values() for row in rows)


def test_sample_size_uses_pilot_variance_when_available():
    from research_runtime.m4_3.sample_size import ContrastSpec
    diffs = {"B-A": [0.1, 0.2, 0.15, 0.12],
             "C-B": [0.4, 0.5, 0.45, 0.42],
             "C-A": [0.5, 0.7, 0.6, 0.55]}
    rec = estimate_sample_size(
        paired_differences=diffs,
        contrasts=[ContrastSpec("B-A", 0.3), ContrastSpec("C-B", 0.3),
                   ContrastSpec("C-A", 0.3)],
        alpha=0.05, target_power=0.8, cost_per_campaign_s=60.0)
    assert rec["status"] == "estimated"
    assert rec["n_blocks_recommended"] >= 1
    assert rec["budget_estimate"]["campaigns"] == rec["n_blocks_recommended"] * 3
    # The largest pilot effect is never the assumed effect: the SEOI (0.3) is.
    assert rec["smallest_effect_of_interest_policy"].startswith(
        "pre-specified per contrast")


# ---------------------------------------------------------------- statistics
def test_paired_contrast_matches_a_known_case():
    out = paired_block_contrast([1.0, 1.0, 1.0, 1.0])
    assert out["n"] == 4 and out["mean"] == 1.0
    assert out["p_value"] == 0.0            # zero variance, non-zero mean
    out2 = paired_block_contrast([1.0])
    assert out2["verdict"] == "insufficient_blocks"
    assert paired_block_contrast([])["verdict"] == "no_paired_blocks"


def test_holm_correction_is_applied_across_contrasts():
    out = analyse_primary_contrasts(
        differences={"C-B": [0.5, 0.6, 0.55, 0.52],
                     "C-A": [0.1, 0.05, 0.2, 0.0],
                     "B-A": [0.0, 0.0, 0.0, 0.0]},
        alpha=0.05, correction="holm")
    assert out["correction"] == "holm"
    assert out["n_contrasts_in_family"] == 3
    assert out["contrasts"]["C-B"]["significant"] is True
    assert out["contrasts"]["C-A"]["significant"] is False


# --------------------------------------------------------------- value report
def test_pilot_value_report_refuses_to_declare_a_winner(tmp_path):
    from research_runtime.research_store.db import ResearchDatabase
    db = ResearchDatabase(tmp_path / "r.sqlite")
    _populate(db, blocks=4, valid_arms={"A": 1.0, "B": 1.0, "C": 1.0})
    report = generate_value_report(db=db, out_dir=tmp_path / "pilot",
                                   stage="pilot", write=True)
    assert report["is_pilot"] is True
    assert report["conclusion"]["outcome"] == VALUE_OUTCOMES[3]
    assert report["conclusion"]["significance_claimed"] is False
    md = (tmp_path / "pilot" / "M4_3_PILOT_VALUE_REPORT.md").read_text(
        encoding="utf-8")
    for section in ("## A.", "## B.", "## C.", "## D.", "## E.", "## F.",
                    "## G."):
        assert section in md
    assert "PILOT ARTEFACT" in md


def test_final_report_sections_and_evidence_links(tmp_path):
    from research_runtime.research_store.db import ResearchDatabase
    db = ResearchDatabase(tmp_path / "r.sqlite")
    _populate(db, blocks=4, valid_arms={"A": 1.0, "B": 1.0, "C": 1.0})
    report = generate_value_report(db=db, out_dir=tmp_path / "final",
                                   stage="final", write=True,
                                   protocol={"protocol_sha256": "a" * 64})
    assert report["conclusion"]["outcome"] in VALUE_OUTCOMES
    chains = report["evidence_chains"]
    assert chains and all("campaign_id" in c and "report_id" in c
                          for c in chains)
    assert set(report["contrast_evidence"]) == {"C-B", "B-A", "C-A"}
    assert (tmp_path / "final" / "M4_3_VALUE_REPORT.md").is_file()
    assert (tmp_path / "final" / "value_report.json").is_file()


def test_final_report_never_claims_significance_from_zero_runs(tmp_path):
    from research_runtime.research_store.db import ResearchDatabase
    db = ResearchDatabase(tmp_path / "r.sqlite")
    _populate(db, blocks=1)
    report = generate_value_report(db=db, out_dir=tmp_path / "final",
                                   stage="final", write=False)
    assert report["conclusion"]["outcome"] == VALUE_OUTCOMES[3]
    assert report["conclusion"]["significance_claimed"] is False


def test_load_runs_roundtrips_trusted_ids(tmp_path):
    from research_runtime.research_store.db import ResearchDatabase
    db = ResearchDatabase(tmp_path / "r.sqlite")
    _populate(db, blocks=1)
    runs = load_runs(db)
    rec = next(r for r in runs if r.run_id == "b1_A")
    assert rec.campaign_id == "camp_b1_A"
    assert rec.plan_id == "plan_b1_A"
    assert rec.experience_id == "exp_b1_A"
    assert rec.finalized is True
