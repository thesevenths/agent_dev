"""ARR_EMB_RSI_02 engine-free tests."""
from __future__ import annotations

import pytest

from research_runtime.arr_emb_rsi_02.benchmark import (ALLOWED_KEYS,
                                                       build_regimes)
from research_runtime.arr_emb_rsi_02.campaign import (_paired_stats,
                                                      _valid_agent_intervention)
from research_runtime.arr_emb_rsi_02.evidence import (ArtifactClaim,
                                                      ArtifactResolver,
                                                      build_evidence,
                                                      evidence_audit)
from research_runtime.arr_emb_rsi_02.integrity import (IntegrityError,
                                                       assert_no_ground_truth)
from research_runtime.arr_emb_rsi_02.prereg import freeze_prereg


def test_10_valid_regimes_and_blind_view():
    regimes = build_regimes()
    assert len(regimes) == 10
    assert {r.generator for r in regimes} == {
        "pendulum", "newton", "double_pendulum"}
    assert not any("E02" in r.regime_id and r.generator != "pendulum"
                   for r in regimes)
    for r in regimes:
        view = r.blind_view()
        assert_no_ground_truth(view)
        assert "generator" not in view and "mechanism" not in view
        assert r.regime_id not in str(view)


def test_gt_leak_detection():
    with pytest.raises(IntegrityError):
        assert_no_ground_truth({"correct_intervention": {"train.lr": 0.001}})
    assert_no_ground_truth({"metrics": {"latent_mse@1": 0.01}})


def test_agent_intervention_validation():
    assert _valid_agent_intervention({"train.lr": 0.0008}) == {
        "train.lr": 0.0008}
    assert _valid_agent_intervention({"model.predictor_hidden": [512, 512]}) is None
    assert _valid_agent_intervention({"unknown.key": 1}) is None
    assert set(ALLOWED_KEYS) >= {"model.predictor_hidden", "data.train_samples",
                                 "train.lr"}


def test_paired_stats_small_sample():
    s = _paired_stats([0.1, 0.2, 0.05, 0.15, 0.12], n_boot=100)
    assert s["n"] == 5 and s["p"] >= 0 and len(s["ci95"]) == 2
    assert s["bootstrap_ci95"][0] <= s["bootstrap_ci95"][1]


def test_artifact_resolver_and_evidence_audit(tmp_path):
    f = tmp_path / "result.json"
    f.write_text('{"latent_mse@1":0.01}', encoding="utf-8")
    from research_runtime.arr_emb_rsi_01.adapter import sha256_file
    h = sha256_file(f)
    claim = ArtifactClaim("artifact:x:result_json", str(f), h,
                          "engine artifact result_json", run_id="r1")
    res = ArtifactResolver().verify([claim])
    assert res["integrity_pass"] is True
    audit = evidence_audit(resolver_result=res,
                           agent_citations=["artifact:x:result_json"])
    assert audit["citation_validity"] == 1.0
    assert audit["evidence_sufficiency"] is False  # one claim is insufficient


def test_prereg_frozen():
    pr = freeze_prereg("ARR_EMB_RSI_02_TEST_PREREG.json")
    assert pr["independent_units"] == 10
    assert pr["mme_normalized"] == 0.05
    assert pr["power_analysis"]["computed_power"] == pytest.approx(0.8031)
    assert pr["preregistration_sha256"]
