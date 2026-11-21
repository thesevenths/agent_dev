"""ARR-RSI-01 P0.1 unit tests (engine-free, deterministic)."""
from __future__ import annotations

import pytest

from research_runtime.arr_rsi_01.arms import (
    ARMS, derive_arms, finalize_intervention_record,
)
from research_runtime.arr_rsi_01.diagnosis import (
    DiagnosisOutput, EvidenceGroundedDiagnoser, INSUFFICIENT,
    ShuffledDiagnoser, CAUSE_HORIZON,
)
from research_runtime.arr_rsi_01.evaluation import (
    Preregistration, analyse_statistics, decide_verdict, evaluate_task,
)
from research_runtime.arr_rsi_01.runner import run_pilot
from research_runtime.arr_rsi_01.tasks import (
    FAMILIES, GroundTruthLeak, GroundTruthStore, LEAK_KEYS,
    assert_no_ground_truth, build_ground_truth, build_tasks, sha256_json,
    task_manifest,
)
from research_runtime.settings import DEFAULT_SCINET_DIR

TASKS = build_tasks(DEFAULT_SCINET_DIR)
TRUTH = build_ground_truth(TASKS)
TASK = TASKS[0]
BASELINE = {"train": {"lr": 0.001, "weight_decay": 0.0},
            "model": {"latent_dim": 4, "predictor_hidden": [256, 256]},
            "jepa": {"pred_weight": 1.0, "ema_momentum": 0.996}}


def _traj(evidence: bool = True):
    return {
        "run_id": "run123",
        "evidence_refs": {"baseline_id": "exp_b", "report_path": "report.json"},
        "baseline_config": BASELINE, "budget_usage": {"jobs_used": 4},
        "experiment_results": [{
            "experiment_id": "exp_b", "job_id": "job_b", "status": "succeeded",
            "metrics": {"latent_mse@1": 0.05, "latent_mse@3": 0.09},
            "result_path": "out/result.json"}] if evidence else [],
    }


def _diagnosis(evidence: bool = True):
    traj = _traj(evidence)
    return EvidenceGroundedDiagnoser().diagnose(
        task=TASK.agent_view(), trajectory=traj,
        experiment_results=traj["experiment_results"],
        evidence=traj["evidence_refs"], budget=TASK.budget)


def test_20_tasks_deterministic_and_5_independent_units():
    again = build_tasks(DEFAULT_SCINET_DIR)
    assert [t.task_id for t in TASKS] == [t.task_id for t in again]
    assert sha256_json([t.to_dict() for t in TASKS]) == sha256_json(
        [t.to_dict() for t in again])
    assert len(TASKS) == 20 and set(t.task_family for t in TASKS) == set(FAMILIES)
    for family in FAMILIES:
        assert len([t for t in TASKS if t.task_family == family]) == 4
    assert len({t.independence_unit for t in TASKS}) == 5
    assert all(t.agent_task_id.startswith("t_") for t in TASKS)


def test_agent_view_has_no_solution_information():
    view = TASK.agent_view()
    for key in LEAK_KEYS:
        assert key not in view, key
    for key in ("task_family", "failure_mode", "valid_interventions",
                "hidden_test_split", "engine_config", "seed", "task_id"):
        assert key not in view, key
    for hint in ("pendulum", "newton", "ood", "regime", "symbolic", "robot"):
        assert hint not in view["objective"].lower(), hint
    assert len(view["allowed_config_schema"]) == 6
    assert all("value" not in s for s in view["allowed_config_schema"])
    assert_no_ground_truth(view)
    for key in ("failure_mode", "correct_intervention", "hidden_test_split"):
        with pytest.raises(GroundTruthLeak):
            assert_no_ground_truth({"wrapped": [{key: "x"}]})
    manifest = task_manifest(TASKS, ground_truth_sha256="abc")
    assert manifest["n_independent_units"] == 5
    assert manifest["n_seed_replicates_per_unit"] == 4


def test_sealed_ground_truth_roundtrip_and_tamper(tmp_path):
    store = GroundTruthStore(tmp_path / "gt.json")
    seal = store.seal(TRUTH)
    assert seal["n_records"] == 20
    assert set(store.load(seal["ground_truth_sha256"])) == set(TRUTH)
    with pytest.raises(GroundTruthLeak):
        store.load("deadbeef")


def test_diagnosis_uses_evidence_not_task_labels():
    d = _diagnosis()
    assert d.validate() == []
    assert d.root_cause == CAUSE_HORIZON
    assert d.insufficient_evidence is False
    assert d.proposed_intervention["key"] == "train.lr"
    assert d.proposed_intervention["value"] == pytest.approx(0.0008)
    assert d.diagnosis_id.startswith("dg_")
    assert d.evidence_refs
    # Same evidence + a DIFFERENT task label must yield the SAME diagnosis.
    other_view = dict(TASKS[5].agent_view())
    traj = _traj()
    d2 = EvidenceGroundedDiagnoser().diagnose(
        task=other_view, trajectory=traj,
        experiment_results=traj["experiment_results"],
        evidence=traj["evidence_refs"], budget=TASKS[5].budget)
    assert d2.root_cause == d.root_cause
    assert d2.proposed_intervention == d.proposed_intervention
    assert_no_ground_truth(d.to_dict())


def test_insufficient_evidence_forces_no_intervention():
    d = _diagnosis(evidence=False)
    assert d.insufficient_evidence is True
    assert d.proposed_intervention is None
    assert d.root_cause == INSUFFICIENT
    assert d.confidence == 0.0


def test_shuffled_diagnosis_is_distinct_and_traceable():
    d = _diagnosis()
    shuffled = ShuffledDiagnoser(pool={TASKS[0].agent_task_id: d,
                                       TASKS[1].agent_task_id: d}).diagnose(
        task=TASK.agent_view(), trajectory=_traj(),
        experiment_results=_traj()["experiment_results"],
        evidence={}, budget=TASK.budget)
    assert shuffled.diagnosis_id.startswith("dg_shuffle_")
    assert shuffled.shuffled_from
    assert shuffled.task_id == TASK.agent_task_id


def test_four_arms_record_full_intervention_trace():
    d = _diagnosis()
    arms = derive_arms(task=TASK.agent_view(), diagnosis=d,
                       baseline_config=BASELINE)
    assert set(arms) == set(ARMS)
    rec = arms["candidate"].intervention_record
    for key in ("diagnosis_id", "diagnosis_evidence_refs",
                "intervention_hypothesis", "expected_effect",
                "falsification_condition", "before_config", "after_config",
                "config_diff_hash", "status"):
        assert key in rec, key
    assert rec["diagnosis_id"] == d.diagnosis_id
    assert rec["before_config"]["train.lr"] == 0.001
    assert rec["after_config"]["train.lr"] == pytest.approx(0.0008)
    assert rec["status"] == "PENDING_APPLICATION"
    applied = finalize_intervention_record(
        record=rec, applied_config=BASELINE, applied_config_path="none")
    assert applied["status"] == "NOT_APPLIED"
    with_lr = {"train": {"lr": 0.0008}}
    applied_ok = finalize_intervention_record(
        record=rec, applied_config=with_lr, applied_config_path="config.yaml")
    assert applied_ok["status"] == "APPLIED"
    assert applied_ok["application_verified"] is True


def test_inert_candidate_is_marked_inert():
    d = _diagnosis(evidence=False)
    arms = derive_arms(task=TASK.agent_view(), diagnosis=d,
                       baseline_config=BASELINE)
    assert arms["candidate"].intervention_record["status"] == "INERT"
    assert arms["candidate"].intervention_policy == {}


def _results(ok=True, delta=0.05):
    return {name: {
        "ok": ok, "artifact_sha256": "sha", "evidence": {
            "holdout": {"mean_holdout_delta": delta},
            "ood": {"mean_holdout_delta": delta}},
        "telemetry": {"jobs_used": 4}, "quality": {"Q": 0.8},
        "provenance": {"harness_sha256": "h"}, "error": "" if ok else "KeyError: x",
    } for name in ARMS}


def _trajectories():
    out = {}
    for name in ARMS:
        t = _traj()
        t["infrastructure_failures"] = ([] if name != "candidate" else
                                         [{"arm": name, "error": "x",
                                           "error_type": "KeyError"}])
        out[name] = t
    return out


def test_evaluation_integrity_and_independence_units():
    d = _diagnosis()
    arms = derive_arms(task=TASK.agent_view(), diagnosis=d,
                       baseline_config=BASELINE)
    ev = evaluate_task(
        task=TASK, truth=TRUTH[TASK.task_id], arms=arms,
        arm_results=_results(), diagnoses={TASK.task_id: {"base": d.to_dict()}},
        trajectories=_trajectories(), prereg=Preregistration())
    assert ev["scientific_correctness"]["root_cause_correct"] is True
    assert ev["independence_unit"] == TASK.independence_unit
    assert ev["infrastructure"]["n_failures"] == 1
    assert ev["complete_case"] is False
    assert "evidence" in ev and "n_unresolved" in ev["evidence"]


def test_verdict_blocks_missing_evidence_and_small_n():
    prereg = Preregistration()
    stats = {"n_independent_units": 5, "comparisons": {
        "candidate_vs_base": {"contrasts": [{"holm_corrected_p": 0.01,
                                              "significant_after_correction": True,
                                              "mean_difference": 0.1}]},
        "candidate_vs_placebo": {"contrasts": [{"mean_difference": 0.05}]},
        "candidate_vs_shuffled": {"contrasts": [{"mean_difference": 0.05}]}}}
    good_dims = {k: 1.0 for k in (
        "scientific_correctness", "ood_performance", "evidence_completeness",
        "experimental_validity", "reproducibility", "budget_efficiency")}
    base = {"task_id": "t", "task_family": "f", "independence_unit": "f|ds",
            "dimensions": good_dims, "complete_case": True,
            "intervention_validity": {"applied_status": "APPLIED",
                                      "shuffled_distinct": True},
            "infrastructure": {"n_failures": 0},
            "experimental_validity": {"valid": True, "all_arms_present": True,
                                      "all_arms_ok": True}}
    assert decide_verdict(task_evaluations=[base] * 4, statistics=stats,
                          prereg=prereg)["verdict"] in ("INCONCLUSIVE", "BLOCKED")
    assert decide_verdict(task_evaluations=[base] * 5, statistics=stats,
                          prereg=prereg)["verdict"] == "PASS"
    blocked = dict(base, infrastructure={"n_failures": 1})
    assert decide_verdict(task_evaluations=[blocked] * 5, statistics=stats,
                          prereg=prereg)["verdict"] == "BLOCKED"


def test_concurrency_guard():
    with pytest.raises(ValueError, match="concurrency=1"):
        run_pilot(n_tasks=1, run_root="unused", concurrency=6)
