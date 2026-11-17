"""M7.2 benchmark isolation and verdict tests."""
from __future__ import annotations

from research_runtime.capability_bench.scientific_judgment import (
    compute_metrics, load_cases, load_ground_truth, make_prompt)
from research_runtime.harness_rsi.identity import freeze_json


def test_benchmark_cases_do_not_contain_ground_truth():
    cases = load_cases("M7_2_BENCHMARK")
    assert len(cases) == 50
    for case in cases:
        prompt = make_prompt(case)
        assert "abstention_expected" not in prompt
        assert "ground_truth" not in prompt
        assert "reviewer_annotations" not in prompt


def test_inconclusive_without_expert_review(tmp_path):
    manifest = {"expert_blind_review": "NOT_PERFORMED"}
    result = compute_metrics(results_dir="M7_2_BENCHMARK/pilot_run",
                             ground_truth=load_ground_truth("M7_2_BENCHMARK"),
                             benchmark_manifest=manifest)
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["n_scored"] == 14
    assert result["abstention"]["f1"] is not None
