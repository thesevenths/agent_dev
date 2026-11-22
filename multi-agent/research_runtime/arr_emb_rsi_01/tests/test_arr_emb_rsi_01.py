"""ARR_EMB_RSI_01 engine-free integrity tests."""
from __future__ import annotations

import pytest

from research_runtime.arr_emb_rsi_01.benchmark import (FAILURES, build_cases,
                                                       sealed_ground_truth)
from research_runtime.arr_emb_rsi_01.diagnosis import (INTERVENTION_REGISTRY,
                                                       diagnose)
from research_runtime.arr_emb_rsi_01.integrity import (ArtifactClaim,
                                                       ArtifactResolver,
                                                       IntegrityError,
                                                       SealedGroundTruthStore,
                                                       assert_no_ground_truth,
                                                       sha256_file)
from research_runtime.arr_emb_rsi_01.prereg import freeze_prereg


def test_gt_sealed_outside_run_root(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    truth = {"x": {"case_id": "x", "ground_truth": "secret"}}
    store = SealedGroundTruthStore(run_root=root)
    seal = store.seal(truth)
    assert seal["outside_run_root"] is True
    assert store.load(seal["ground_truth_sha256"]) == truth
    with pytest.raises(IntegrityError):
        store.load("bad")


def test_agent_view_and_evidence_never_leak_gt(tmp_path):
    cases = build_cases(seeds=(42,))
    case = cases[0]
    view = case.agent_view()
    assert_no_ground_truth(view)
    assert "failure_id" not in view
    truth = sealed_ground_truth(cases)
    with pytest.raises(IntegrityError):
        assert_no_ground_truth(truth[case.case_id])


def test_artifact_resolver_requires_hash_and_claim_match(tmp_path):
    f = tmp_path / "metric.json"
    f.write_text('{"latent_mse@1": 0.01}', encoding="utf-8")
    h = sha256_file(f)
    claim = ArtifactClaim("c1", str(f), h, "real metric")
    ok = ArtifactResolver().verify([claim])
    assert ok["pass"] is True
    bad = ArtifactClaim("c1", str(f), h, "tampered claim",
                        claim_sha256=claim.to_dict()["claim_sha256"])
    assert ArtifactResolver().verify([bad])["pass"] is False


def test_oracle_and_random_diagnosis_schema():
    cases = build_cases(seeds=(42,))
    case = cases[0]
    truth = sealed_ground_truth(cases)[case.case_id]
    evidence = {"case_id": case.case_id, "metrics": {"cosine@1": 0.02},
                "config_snapshot": {}, "budget_usage": {}, "claims": []}
    oracle = diagnose(controller="oracle", evidence=evidence, sealed=truth)
    rand = diagnose(controller="random", evidence=evidence)
    assert oracle.root_cause == truth["ground_truth"]
    assert oracle.proposed_intervention["overrides"] == truth[
        "correct_intervention"]
    assert rand.controller == "random" and rand.diagnosis_id


def test_registry_and_prereg_frozen(tmp_path):
    assert all(v["engine_parameters"] and v["falsification"]
               for v in INTERVENTION_REGISTRY.values())
    assert len(FAILURES) == 4
    p = freeze_prereg(tmp_path / "prereg.json")
    assert p["full_experiment_authorized"] is False
    assert p["mme_absolute"] == 0.02
    assert p["preregistration_sha256"]
