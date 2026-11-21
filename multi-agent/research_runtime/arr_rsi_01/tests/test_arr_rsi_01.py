"""ARR-RSI-01 P0 unit tests (engine-free, deterministic)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_runtime.arr_rsi_01.arms import ARMS, derive_arms
from research_runtime.arr_rsi_01.diagnosis import (
    DiagnosisOutput,
    EvidenceGroundedDiagnoser,
    ShuffledDiagnoser,
)
from research_runtime.arr_rsi_01.evaluation import (
    Preregistration,
    decide_verdict,
    evaluate_task,
)
from research_runtime.arr_rsi_01.tasks import (
    FAMILIES,
    GroundTruthLeak,
    GroundTruthStore,
    PublicTask,
    assert_no_ground_truth,
    build_ground_truth,
    build_tasks,
    sha256_json,
    task_manifest,
)
from research_runtime.settings import DEFAULT_SCINET_DIR

TASKS = build_tasks(DEFAULT_SCINET_DIR)
TRUTH = build_ground_truth(TASKS)
TASK = TASKS[0]


def _diagnosis(task=TASK, refs=True):
    traj = ({"tool_calls": [{"call_id": "tc_1"}],
             "evidence": {"baseline_id": "b1"},
             "experiment_results": []} if refs else {})
    results = [{
        "experiment_id": "exp_1", "job_id": "job_1", "status": "succeeded",
        "stage": "evaluate", "engine": "scinet++",
        "metrics": {"latent_mse@1": 0.05, "latent_mse@3": 0.09},
        "result_path": "outputs/result_1.json",
    }] if refs else []
    return EvidenceGroundedDiagnoser().diagnose(
        task=task.agent_view(), trajectory=traj, experiment_results=results,
        evidence={"baseline_id": "b1"} if refs else {},
        budget=task.budget)


def test_20_tasks_deterministic_and_complete():
    again = build_tasks(DEFAULT_SCINET_DIR)
    assert len(TASKS) == 20 == len(again)
    assert [t.task_id for t in TASKS] == [t.task_id for t in again]
    assert sha256_json([t.to_dict() for t in TASKS]) == sha256_json(
        [t.to_dict() for t in again])
    assert set(t.task_family for t in TASKS) == set(FAMILIES)
    for task in TASKS:
        for field in ("task_id", "task_family", "task_description",
                      "dataset_reference", "train_split", "validation_split",
                      "hidden_test_split", "failure_mode", "valid_interventions",
                      "primary_metric", "budget"):
            assert getattr(task, field) not in (None, "", [])
    for family in FAMILIES:
        assert len([t for t in TASKS if t.task_family == family]) == 4


def test_ground_truth_isolation():
    view = TASK.agent_view()
    assert "hidden_test_split" not in view
    assert not any(k in view for k in (
        "ground_truth_root_cause", "correct_intervention",
        "distractor_intervention", "falsification_condition"))
    with pytest.raises(GroundTruthLeak):
        assert_no_ground_truth({"ground_truth_root_cause": "x"})
    with pytest.raises(GroundTruthLeak):
        assert_no_ground_truth({"nested": [{"correct_intervention": "x"}]})
    assert_no_ground_truth(view)
    manifest = task_manifest(TASKS, ground_truth_sha256="abc")
    assert manifest["ground_truth_separately_sealed"] is True
    assert_no_ground_truth(manifest)


def test_sealed_truth_roundtrip_and_tamper_detection(tmp_path):
    store = GroundTruthStore(tmp_path / "gt.json")
    seal = store.seal(TRUTH)
    assert seal["n_records"] == 20
    assert set(store.load(seal["ground_truth_sha256"])) == set(TRUTH)
    with pytest.raises(GroundTruthLeak):
        store.load("deadbeef")


def test_diagnosis_requires_evidence_and_has_all_fields():
    d = _diagnosis()
    assert d.validate() == []
    assert d.evidence_refs
    assert d.root_cause
    assert d.confidence >= 0.5
    empty = _diagnosis(refs=False)
    assert empty.proposed_intervention is None
    assert empty.root_cause == "evidence_gap"
    # A diagnosis produced from evidence only never carries GT field names.
    assert_no_ground_truth(d.to_dict())


def test_four_arms_matched_and_traceable():
    d = _diagnosis()
    arms = derive_arms(task=TASK.agent_view(), diagnosis=d)
    assert set(arms) == set(ARMS)
    assert arms["base"].harness.planner_strategy == "fixed_first_knob"
    assert arms["candidate"].diagnosis_sha256 == d.to_dict()["diagnosis_sha256"]
    assert arms["candidate"].intervention.get("train.lr") == 0.0008
    assert arms["placebo"].intervention != arms["candidate"].intervention
    for arm in arms.values():
        assert arm.harness.max_candidates == 1
        assert arm.harness.campaign_rounds == 1
        assert arm.harness.experience_top_k == 0


def test_shuffled_diagnosis_is_traceable_control():
    d1, d2 = _diagnosis(TASKS[0]), _diagnosis(TASKS[1])
    shuffled = ShuffledDiagnoser(pool={TASKS[0].task_id: d1,
                                       TASKS[1].task_id: d2}).diagnose(
        task=TASKS[0].agent_view(), trajectory={}, experiment_results=[],
        evidence={}, budget=TASKS[0].budget)
    assert shuffled.task_id == TASKS[0].task_id
    assert shuffled.root_cause == d2.root_cause
    assert shuffled.evidence_refs and shuffled.evidence_refs[0].startswith(
        "shuffled_from:")


def _results(ok=True, delta=0.05):
    return {name: {
        "ok": ok, "artifact_sha256": "sha", "evidence": {
            "holdout": {"mean_holdout_delta": delta},
            "ood": {"mean_holdout_delta": delta}},
        "telemetry": {"jobs_used": 4}, "quality": {"Q": 0.8},
        "provenance": {"harness_sha256": "h"},
    } for name in ARMS}


def test_evaluation_uses_ground_truth_only_in_evaluator():
    d = _diagnosis()
    arms = derive_arms(task=TASK.agent_view(), diagnosis=d)
    ev = evaluate_task(
        task=TASK, truth=TRUTH[TASK.task_id], arms=arms, arm_results=_results(),
        diagnoses={TASK.task_id: {"base": d.to_dict()}},
        trajectory={"evidence": {"baseline_id": "b1"}}, prereg=Preregistration())
    assert ev["scientific_correctness"]["root_cause_correct"] is True
    assert ev["scientific_correctness"]["intervention_correct"] is True
    assert ev["experimental_validity"]["valid"] is True
    assert set(ev["dimensions"]) == {
        "scientific_correctness", "ood_performance", "evidence_completeness",
        "experimental_validity", "reproducibility", "budget_efficiency"}


def test_verdict_never_promotes_missing_evidence():
    prereg = Preregistration()
    stats = {"comparisons": {
        "candidate_vs_base": {"contrasts": [{"holm_corrected_p": 0.01,
                                              "significant_after_correction": True,
                                              "mean_difference": 0.1}]},
        "candidate_vs_placebo": {"contrasts": [{"mean_difference": 0.05}]},
        "candidate_vs_shuffled": {"contrasts": [{"mean_difference": 0.05}]}}}
    blocked_eval = {"task_id": "t", "task_family": "f",
                    "dimensions": {k: 1.0 for k in (
                        "scientific_correctness", "ood_performance",
                        "evidence_completeness", "experimental_validity",
                        "reproducibility", "budget_efficiency")},
                    "experimental_validity": {"valid": False,
                                              "all_arms_present": False,
                                              "all_arms_ok": False,
                                              "infrastructure_failures": [],
                                              "ground_truth_leak": False}}
    assert decide_verdict(task_evaluations=[blocked_eval] * 5,
                          statistics=stats, prereg=prereg)["verdict"] == "BLOCKED"
    small = [dict(blocked_eval, experimental_validity={
        "valid": True, "all_arms_present": True, "all_arms_ok": True,
        "infrastructure_failures": [], "ground_truth_leak": False})]
    assert decide_verdict(task_evaluations=small, statistics=stats,
                          prereg=prereg)["verdict"] == "INCONCLUSIVE"
