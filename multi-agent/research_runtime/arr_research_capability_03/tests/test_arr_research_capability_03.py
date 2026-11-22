"""ARR-03 engine-free unit tests (no simulator, no engine, no LLM network)."""
from __future__ import annotations

import pytest

from research_runtime.arr_research_capability_03 import (
    benchmark, evaluation, integrity, reacher, statistics as st,
)
from research_runtime.arr_research_capability_03.integrity import (
    IntegrityError, assert_no_ground_truth, sha256_json, SealedGroundTruthStore,
)
from research_runtime.arr_research_capability_03.knowledge import ResearchKnowledge
from research_runtime.arr_research_capability_03.prereg import freeze_prereg


def test_ground_truth_leak_detection():
    with pytest.raises(IntegrityError):
        assert_no_ground_truth({"ground_truth": {"cause": "x"}})
    with pytest.raises(IntegrityError):
        assert_no_ground_truth({"nested": {"failure_id": "reacher::E01"}})
    assert_no_ground_truth({"metrics": {"one_step_mse": 0.1}})


def test_sealed_ground_truth_outside_root(tmp_path):
    seal = SealedGroundTruthStore(tmp_path / "sealed")
    info = seal.seal({"reacher::E01": {"cause": "insufficient_model_capacity"}})
    loaded = seal.load(info["ground_truth_sha256"])
    assert loaded["reacher::E01"]["cause"] == "insufficient_model_capacity"
    with pytest.raises(IntegrityError):
        seal.load("0" * 64)


def test_blind_views_have_no_ground_truth():
    for r in benchmark.reacher_regimes():
        view = r.blind_view()
        assert_no_ground_truth(view)
        assert "cause" not in view and "oracle" not in str(view).lower()


def test_m2_ten_regimes_and_m3_six():
    m2 = benchmark.m2_state_regimes()
    assert len(m2) == 10
    assert not any(r["regime_id"] in benchmark.M2_EXCLUDED for r in m2)
    assert len(benchmark.reacher_regimes()) == 6


def test_all_m3_regimes_oracle_sensitive():
    for r in benchmark.reacher_regimes():
        assert r.oracle_sensitivity >= 0.05, r.regime_id


def test_intervention_registry_is_real():
    for cause, iv in benchmark.INTERVENTION_REGISTRY.items():
        assert iv["id"] and iv["parameter"] and iv["minimum_effect"] == 0.05
        assert iv["expected_direction"] in ("decrease", "increase")
    cfg = benchmark.intervention_to_reacher_config(
        "insufficient_model_capacity",
        {"n_episodes": 8, "action_mode": "random"}, {"hidden": [4], "lr": 0.002})
    assert cfg["model"]["hidden"] == [64, 64]
    assert cfg["train"]["n_episodes"] == 8


def test_intervention_changes_real_engine_config():
    base_train = {"n_episodes": 2, "action_mode": "zero", "action_scale": 1.0}
    base_model = {"hidden": [4], "lr": 0.3}
    checks = {
        "insufficient_model_capacity": ("model", "hidden", [64, 64]),
        "insufficient_state_action_coverage": ("train", "n_episodes", 16),
        "optimization_instability": ("model", "lr", 0.001),
        "action_conditioning_failure": ("train", "action_mode", "random"),
    }
    for cause, (section, key, expected) in checks.items():
        cfg = benchmark.intervention_to_reacher_config(cause, base_train, base_model)
        assert cfg[section][key] == expected


def test_agent_evidence_is_blind():
    from research_runtime.arr_research_capability_03.diagnosis import build_agent_evidence
    ev = build_agent_evidence(
        metrics={"one_step_mse": 0.5},
        config_view={"model.hidden": [4]},
        artifacts=["artifact:x:model_signature"])
    assert_no_ground_truth(ev)
    assert "one_step_mse" in ev["metrics"]


def test_statistics_paired_and_holm():
    s = st.paired_stats([0.1, 0.2, 0.05, 0.15, 0.12], n_boot=100, n_perm=100)
    assert s["n"] == 5 and 0 <= s["p"] <= 1
    assert s["bootstrap_ci95"][0] <= s["bootstrap_ci95"][1]
    h = st.holm([0.01, 0.04, 0.03], alpha=0.05)
    assert len(h) == 3 and all(0 <= x["holm_p"] <= 1 for x in h)


def test_wilson_interval():
    lo, hi = st.wilson(8, 10)
    assert 0 <= lo <= 0.8 <= hi <= 1


def test_diagnosis_metrics_and_calibration():
    cases = []
    for i in range(10):
        cases.append({"diagnoses": {
            "arr": {"primary": "insufficient_model_capacity", "confidence": 0.9,
                    "candidates": ["insufficient_model_capacity"], "abstain": False},
            "rule": {"primary": "insufficient_model_capacity", "confidence": 0.6,
                     "candidates": [], "abstain": False},
            "random": {"primary": "optimization_instability", "confidence": 0.5,
                       "candidates": [], "abstain": False},
            "shuffled": {"primary": "dynamics_distribution_shift", "confidence": 0.5,
                         "candidates": [], "abstain": False},
            "oracle": {"primary": "insufficient_model_capacity", "confidence": 1.0,
                       "candidates": ["insufficient_model_capacity"], "abstain": False},
        }, "truth": {"cause": "insufficient_model_capacity"}})
    m = evaluation.diagnosis_metrics(cases)
    assert m["arr"]["accuracy"] == 1.0
    assert m["random"]["accuracy"] == 0.0
    assert 0 <= m["arr"]["ece"] <= 1
    assert 0 <= m["arr"]["brier"] <= 1
    assert m["arr"]["top2_accuracy"] == 1.0


def test_evidence_quality_and_verdicts():
    prereg = freeze_prereg("ARR_03_TEST_PREREG.json")
    stat = {k: {"mean": 0.1, "ci95": [0.02, 0.18], "significant": True,
                "p": 0.01, "holm_p": 0.02}
            for k in ("candidate_vs_base", "candidate_vs_random", "oracle_vs_base")}
    stat["n_units"] = 10
    diag = {k: {"accuracy": 0.8 if k == "arr" else 0.3,
                "ece": 0.05 if k == "arr" else 0.4,
                "brier": 0.1 if k == "arr" else 0.5,
                "top2_accuracy": 0.9 if k == "arr" else 0.4}
            for k in ("arr", "rule", "random", "shuffled", "oracle")}
    transfer = {k: {"mean": 0.1, "ci95": [0.02, 0.18], "significant": True,
                    "holm_p": 0.02} for k in
                ("arr_transfer_vs_fresh", "arr_transfer_vs_random", "oracle_vs_base")}
    v = evaluation.verdicts(
        engineering={"n_cases": 48, "ok": True}, integrity={"gt": True},
        diag=diag, causal=stat, transfer=transfer, prereg=prereg)
    assert v["engineering"] == "PASS"
    assert v["m1_diagnosis"] == "PASS"
    assert v["overall_research_capability"] == "PASS"


def test_knowledge_transfer_excludes_unverified():
    k = ResearchKnowledge(
        knowledge_id="K_test", scientific_mechanism="insufficient_model_capacity",
        evidence=["e1"], intervention={"model.hidden": [64, 64]},
        before_after={"before": 1.0, "after": 0.5}, confidence=0.8,
        verification_status="M2_VERIFIED", applicability="dynamics",
        limitations=[])
    av = k.agent_view()
    assert_no_ground_truth(av)
    assert k.provenance_hash
    assert len(k.provenance_hash) == 64


def test_reacher_hash_and_dataclass():
    b = reacher.TrajectoryBatch(
        states=__import__("numpy").zeros((3, reacher.OBS_DIM)),
        actions=__import__("numpy").zeros((3, reacher.ACTION_DIM)),
        next_states=__import__("numpy").zeros((3, reacher.OBS_DIM)))
    assert b.n == 3 and len(b.digest()) == 64
