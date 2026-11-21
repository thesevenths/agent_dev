"""M6 -- engine-free tests incl. ALL mandatory adversarial cases.

These tests exercise REAL production modules (identity, benchmark, harness,
generation, paired execution, independent evaluation, formal promotion, real
adoption/rollback, experience). No engine is required and no success is faked.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_runtime.harness_rsi import (
    AdoptionError, AdoptionManager, ContaminationLedger, EvidenceBoundary,
    ExperienceBank, GenerationBoundaryViolation, PreRegistration,
    ResearchExperience, RollbackError, TaskResult, analyse_paired,
    base_harness, build_benchmark, derive_candidate, deterministic_method_runner,
    formal_promote, freeze_bundle, freeze_json, freeze_preregistration,
    generate_candidate, load_frozen_bundle, make_artifact_reader,
    rollback, run_paired, run_retrieval_ablation, verify_frozen,
)
from research_runtime.research_store.db import (
    ResearchDatabase, ResearchStoreConflict,
)
from research_runtime.rrsi import EvolutionRegistry
from research_runtime.runtime_state import RuntimeState, RuntimeStateError
from research_runtime.runtime_state import config_sha as rt_config_sha


@pytest.fixture()
def world(tmp_path):
    bench = build_benchmark(tmp_path / "bench")
    base = base_harness()
    bf = freeze_bundle(base, tmp_path / "base.json")
    cand = derive_candidate(base, component="hypothesis_critic", version="v2",
                            config={"require_falsifying_observation": True},
                            candidate_id="H_CAND")
    cf = freeze_bundle(cand, tmp_path / "cand.json")
    db = ResearchDatabase(tmp_path / "runtime.sqlite")
    state = RuntimeState(db=db)
    cfg = SimpleNamespace(run_id="M6TEST", max_rounds=4, global_budget=8)
    state.initialize(cfg.run_id, config=cfg, dataset_shas={},
                     max_rounds=4, global_budget=8)
    return SimpleNamespace(tmp=tmp_path, bench=bench, base=base, bf=bf,
                           cand=cand, cf=cf, state=state, cfg=cfg,
                           registry=EvolutionRegistry(
                               db=ResearchDatabase(tmp_path / "evo.sqlite")))


@pytest.fixture()
def w(world):
    return world


def _evidence(w, *, cand_sha=None, parent_sha=None, result_files=None,
              result_shas=None, sel_ids=None, eval_ids=None, sel_fams=None,
              eval_fams=None, overrides=None):
    ev = {
        "dataset_manifest_sha": w.bench.manifest_sha, "split_id": "holdout",
        "selection_split_id": "selection",
        "candidate_sha": cand_sha or w.cand.sha256,
        "parent_sha": parent_sha if parent_sha is not None else w.base.sha256,
        "evaluation_task_ids": eval_ids if eval_ids is not None else [
            t.task_id for t in w.bench.by_split("holdout")[:3]],
        "selection_task_ids": sel_ids if sel_ids is not None else [
            t.task_id for t in w.bench.by_split("selection")[:3]],
        "result_shas": result_shas or [], "result_file_paths": result_files or [],
        "evaluation_families": eval_fams if eval_fams is not None else sorted(
            w.bench.families("holdout")),
        "selection_families": sel_fams if sel_fams is not None else sorted(
            w.bench.families("selection")),
        "metrics": {"objective_quality": 0.9},
        "statistics": {"n_holdout_seeds": 3, "holdout_direction_fraction": 1.0},
        "independent": True,
    }
    ev.update(overrides or {})
    return ev


def _reader(w, cand_path=None, parent_path=None):
    return make_artifact_reader(
        candidate_sha=w.cand.sha256,
        candidate_artifact_path=str(cand_path or w.cf["path"]),
        parent_sha=w.base.sha256,
        parent_artifact_path=str(parent_path or w.bf["path"]))


def _real_result_files(w, arm="candidate"):
    files, shas = [], []
    for i, t in enumerate(w.bench.by_split("holdout")[:3]):
        f = freeze_json({"task": t.task_id, "arm": arm,
                         "metrics": {"falsifiability_rate": 0.9}},
                        w.tmp / f"result_{arm}_{i}.json")
        files.append(f["path"])
        shas.append(f["sha256"])
    return files, shas


def _gate(**kwargs):
    d = {"decision": "ACCEPTED", "primary_meets_mme": True,
         "significant_after_correction": True, "infrastructure_failures": 0,
         "evaluator_sha256": "a" * 64}
    d.update(kwargs)
    return d


_SENTINEL = object()


def _promote(w, ev, *, gate=None, reader=_SENTINEL, approval=None, eid="M6_E"):
    return formal_promote(
        registry=w.registry, experiment_id=eid, candidate=w.cand,
        parent_sha=w.base.sha256, holdout_evidence=ev,
        manifest_path=w.bench.manifest_path,
        artifact_reader=(_reader(w) if reader is _SENTINEL else reader),
        preregistration_sha="a" * 64,
        statistical_gate=gate or _gate(),
        promotable_view=w.bench.promotable_view(),
        human_approval=approval if approval is not None else {
            "approved": True, "by": "unit_test"})


# --------------------------------------------------------------------------- #
# Happy paths / core contracts                                                #
# --------------------------------------------------------------------------- #
def test_benchmark_family_disjoint_and_sealed(w):
    from research_runtime.harness_rsi.benchmark import validate_benchmark
    v = validate_benchmark(w.bench)
    assert v["ok"] and len(w.bench.tasks) == 60
    assert not set(v["families_selection"]) & set(v["families_holdout"])
    gv = w.bench.generator_view()
    assert len(gv["tasks"]) == 36 and "gold_ref" not in json.dumps(gv)


def test_bundle_sha_covers_complete_artifact(w):
    assert w.cand.sha256 != w.base.sha256
    assert w.cf["candidate_sha"] == w.cand.sha256
    load = load_frozen_bundle(w.cf["path"],
                              expected_candidate_sha=w.cand.sha256,
                              expected_file_sha=w.cf["file_sha256"])
    assert load["ok"]


def test_failure_driven_generation_single_component(w):
    led = ContaminationLedger(w.tmp / "contam.json")
    gen = generate_candidate(
        base=w.base, failures=[{
            "task_id": w.bench.by_split("dev")[0].task_id,
            "reason": "hypothesis unfalsifiable"}],
        benchmark=w.bench, ledger=led, out_dir=w.tmp / "cands",
        candidate_id="H_GEN")
    assert gen["change"]["changed_component"] == "hypothesis_critic"
    assert gen["record"].parent_sha == w.base.sha256
    assert gen["record"].candidate_sha == gen["candidate"].sha256
    clean = led.assert_clean(candidate_sha=gen["record"].candidate_sha,
                             holdout_task_ids=[t.task_id for t in
                                               w.bench.by_split("holdout")])
    assert clean["ok"]


def test_paired_execution_retains_all_outcomes(w):
    tasks = w.bench.by_split("selection")[:3]
    paired = run_paired(base=w.base, candidate=w.cand, tasks=tasks,
                        runner=deterministic_method_runner,
                        out_dir=w.tmp / "paired", experiment_id="E")
    assert paired["n_results"] == 6
    assert all(r.task_id for r in paired["results"])
    assert verify_frozen(paired["trajectory_path"],
                         paired["trajectory_sha256"])["ok"]


def test_independent_evaluation_detects_real_improvement(w):
    pre = PreRegistration(experiment_id="E", primary_metric="falsifiability_rate",
                          direction="maximize", minimum_meaningful_effect=0.05,
                          n_bootstrap=200, n_permutation=200)
    tasks = w.bench.by_split("selection")[:3]
    paired = run_paired(base=w.base, candidate=w.cand, tasks=tasks,
                        runner=deterministic_method_runner,
                        out_dir=w.tmp / "paired2", experiment_id="E")
    res = analyse_paired(prereg=pre,
                         base=[r for r in paired["results"] if r.arm == "base"],
                         cand=[r for r in paired["results"] if r.arm == "candidate"],
                         families={t.task_id: t.family_id for t in tasks})
    primary = next(c for c in res["contrasts"] if c["is_primary"])
    assert primary["mean_difference"] > 0
    assert res["decision"] == "ACCEPTED"


def test_formal_promotion_and_adoption_and_rollback(w):
    files, shas = _real_result_files(w)
    ev = _evidence(w, result_files=files, result_shas=shas)
    verdict = _promote(w, ev)
    assert verdict["status"] == "PROMOTED"
    am = AdoptionManager(state=w.state, run_id="M6TEST", round_no=2,
                         record_path=w.tmp / "adoption.json")
    ad = am.adopt(approved_record=verdict, candidate_sha=w.cand.sha256,
                  artifact_path=w.cf["path"],
                  file_sha256=w.cf["file_sha256"],
                  human_approval={"approved": True, "by": "unit_test"})
    assert ad["ok"] and am.active_sha() == w.cand.sha256
    am.bind_campaign("camp_test")
    rb = rollback(manager=am, target_sha=w.base.sha256,
                  target_artifact_path=w.bf["path"],
                  target_file_sha256=w.bf["file_sha256"], reason="unit rollback",
                  smoke=lambda sha: am.verify_loaded())
    assert rb["status"] == "ROLLED_BACK" and am.active_sha() == w.base.sha256
    assert w.state.verify_journal("M6TEST")["ok"]


# --------------------------------------------------------------------------- #
# MANDATORY ADVERSARIAL CASES                                                 #
# --------------------------------------------------------------------------- #
def test_adv_holdout_leakage_blocked(w):
    boundary = EvidenceBoundary(w.bench, ContaminationLedger(w.tmp / "c.json"))
    with pytest.raises(GenerationBoundaryViolation):
        boundary.assert_allowed([{"task_id":
                                  w.bench.by_split("holdout")[0].task_id}])
    with pytest.raises(GenerationBoundaryViolation):
        boundary.assert_allowed([{"gold_ref": "gold::x"}])


def test_adv_empty_selection_rejected(w):
    files, shas = _real_result_files(w)
    ev = _evidence(w, result_files=files, result_shas=shas, sel_ids=[])
    v = _promote(w, ev)["verification"]
    assert not v["verified"]
    assert any("selection" in r for r in v["reasons"])


def test_adv_empty_holdout_rejected(w):
    files, shas = _real_result_files(w)
    ev = _evidence(w, result_files=files, result_shas=shas, eval_ids=[])
    v = _promote(w, ev)["verification"]
    assert not v["verified"]


def test_adv_modified_candidate_after_freeze(w):
    frozen = freeze_bundle(w.cand, w.tmp / "mod_cand.json")
    p = Path(frozen["path"])
    p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    check = load_frozen_bundle(p, expected_candidate_sha=frozen["candidate_sha"],
                               expected_file_sha=frozen["file_sha256"])
    assert not check["ok"]


def test_adv_modified_result_after_evaluation(w):
    files, shas = _real_result_files(w)
    Path(files[0]).write_text("{}", encoding="utf-8")
    ev = _evidence(w, result_files=files, result_shas=shas)
    v = _promote(w, ev)["verification"]
    assert not v["verified"]
    assert any("result" in r for r in v["reasons"])


def test_adv_incorrect_parent_sha(w):
    files, shas = _real_result_files(w)
    ev = _evidence(w, result_files=files, result_shas=shas, parent_sha="f" * 64)
    v = _promote(w, ev)["verification"]
    assert not v["verified"]
    assert any("parent_sha" in r for r in v["reasons"])


def test_adv_candidate_sha_mismatch(w):
    files, shas = _real_result_files(w)
    ev = _evidence(w, result_files=files, result_shas=shas, cand_sha="e" * 64)
    v = _promote(w, ev)["verification"]
    assert not v["verified"]


def test_adv_reused_holdout(w):
    files, shas = _real_result_files(w)
    ev = _evidence(w, result_files=files, result_shas=shas,
                   sel_ids=[t.task_id for t in w.bench.by_split("holdout")[:3]])
    v = _promote(w, ev)["verification"]
    assert not v["verified"]
    assert any("overlap" in r or "not_from_benchmark" in r
               for r in v["reasons"])


def test_adv_candidate_self_grading_rejected(w):
    files, shas = _real_result_files(w)
    ev = _evidence(w, result_files=files, result_shas=shas)
    # A candidate-supplied evaluator SHA cannot substitute the preregistered one.
    v = _promote(w, ev, gate=_gate(evaluator_sha256="b" * 64))["verification"]
    assert not v["verified"]
    assert any("evaluator_version" in r for r in v["reasons"])


def test_adv_budget_mismatch(w):
    with pytest.raises(RuntimeStateError):
        w.state.reserve_execution("M6TEST", "bad", config_sha="0" * 64)
    r = w.state.reserve_execution("M6TEST", "ok",
                                  config_sha=rt_config_sha(w.cfg))
    assert r["state"] == "reserved"


def test_adv_missing_evaluator(w):
    files, shas = _real_result_files(w)
    ev = _evidence(w, result_files=files, result_shas=shas)
    gate = _gate()
    del gate["significant_after_correction"]
    v = _promote(w, ev, gate=gate)["verification"]
    assert not v["verified"]


def test_adv_missing_artifact_reader(w):
    files, shas = _real_result_files(w)
    ev = _evidence(w, result_files=files, result_shas=shas)
    v = _promote(w, ev, reader=None)["verification"]
    assert not v["verified"]
    assert "missing_artifact_reader" in v["reasons"]


def test_adv_partial_task_execution(w):
    from research_runtime.harness_rsi.evaluation import _paired_differences
    base = [TaskResult("t1", "f", "b", "base", {"objective_quality": 0.5}),
            TaskResult("t2", "f", "b", "base", {"objective_quality": 0.5})]
    cand = [TaskResult("t1", "f", "c", "candidate", {"objective_quality": 0.8})]
    ids, diffs = _paired_differences(base, cand, "objective_quality", "maximize")
    assert len(ids) == 2 and len(diffs) == 2   # missing pair retained as 0


def test_adv_crash_during_paired_evaluation(w):
    def broken(h, t):
        if t.task_id.endswith("000"):
            raise TimeoutError("subprocess timeout")
        return {"ok": True, "metrics": {"falsifiability_rate": 0.5}}
    paired = run_paired(base=w.base, candidate=w.cand,
                        tasks=w.bench.by_split("selection")[:3],
                        runner=broken, out_dir=w.tmp / "crash",
                        experiment_id="E")
    assert paired["n_infra_failures"] > 0
    assert len(paired["results"]) == 6   # no silent exclusion


def test_adv_crash_during_promotion_registry_immutable(w):
    files, shas = _real_result_files(w)
    ev = _evidence(w, result_files=files, result_shas=shas)
    _promote(w, ev, eid="M6_IMMUT")
    with pytest.raises(ResearchStoreConflict):
        _promote(w, {**ev, "metrics": {"objective_quality": 0.1}},
                 eid="M6_IMMUT")


def test_adv_concurrent_promotion_single_winner(w):
    files, shas = _real_result_files(w)
    ev = _evidence(w, result_files=files, result_shas=shas)
    _promote(w, ev, eid="M6_CONC")
    from research_runtime.harness_rsi.harness import derive_candidate as dc
    other = dc(w.base, component="objective_generator", version="v9",
               config={"x": 1}, candidate_id="OTHER")
    with pytest.raises(ResearchStoreConflict):
        formal_promote(
            registry=w.registry, experiment_id="M6_CONC", candidate=other,
            parent_sha=w.base.sha256, holdout_evidence=ev,
            manifest_path=w.bench.manifest_path, artifact_reader=_reader(w),
            preregistration_sha="a" * 64, statistical_gate=_gate(),
            promotable_view=w.bench.promotable_view(),
            human_approval={"approved": True})


def test_adv_crash_during_adoption(w):
    verdict = {"status": "PROMOTED", "candidate_sha": w.cand.sha256,
               "experiment_id": "E"}
    am = AdoptionManager(state=w.state, run_id="M6TEST", round_no=2,
                         record_path=w.tmp / "no_adoption.json")
    with pytest.raises(AdoptionError):
        am.adopt(approved_record=verdict, candidate_sha=w.cand.sha256,
                 artifact_path=str(w.tmp / "missing.json"),
                 file_sha256="0" * 64,
                 human_approval={"approved": True})
    assert not am.active_sha()


def test_adv_failed_rollback_stays_stopped(w):
    files, shas = _real_result_files(w)
    verdict = _promote(w, _evidence(w, result_files=files, result_shas=shas))
    am = AdoptionManager(state=w.state, run_id="M6TEST", round_no=2,
                         record_path=w.tmp / "rb_fail.json")
    am.adopt(approved_record=verdict, candidate_sha=w.cand.sha256,
             artifact_path=w.cf["path"], file_sha256=w.cf["file_sha256"],
             human_approval={"approved": True, "by": "unit_test"})
    rb = rollback(manager=am, target_sha=w.base.sha256,
                  target_artifact_path=w.bf["path"],
                  target_file_sha256=w.bf["file_sha256"], reason="failed smoke",
                  smoke=lambda sha: {"ok": False, "error": "smoke failed"})
    assert rb["status"] == "FAILED" and am.is_dispatch_stopped()


def test_adv_incorrect_loaded_sha(w):
    files, shas = _real_result_files(w)
    verdict = _promote(w, _evidence(w, result_files=files, result_shas=shas))
    am = AdoptionManager(state=w.state, run_id="M6TEST", round_no=2,
                         record_path=w.tmp / "bad_sha.json")
    am.adopt(approved_record=verdict, candidate_sha=w.cand.sha256,
             artifact_path=w.cf["path"], file_sha256=w.cf["file_sha256"],
             human_approval={"approved": True, "by": "unit_test"})
    rec = json.loads(am.path.read_text(encoding="utf-8"))
    rec["loaded_sha"] = "0" * 64
    am._write(rec)
    assert not am.verify_loaded()["ok"]


def test_adv_missing_human_approval_inconclusive(w):
    files, shas = _real_result_files(w)
    verdict = _promote(w, _evidence(w, result_files=files, result_shas=shas),
                       approval={"approved": False})
    assert verdict["status"] == "INCONCLUSIVE"
    assert verdict["verification"]["verified"] is True


def test_adv_legacy_boolean_holdout_forbidden(w):
    files, shas = _real_result_files(w)
    from research_runtime.harness_rsi.promotion import verify_formal_promotion
    v = verify_formal_promotion(
        candidate=w.cand, parent_sha=w.base.sha256,
        holdout_evidence=_evidence(w, result_files=files, result_shas=shas),
        manifest_path=w.bench.manifest_path, artifact_reader=_reader(w),
        preregistration_sha="a" * 64, statistical_gate=_gate(),
        promotable_view=w.bench.promotable_view(),
        legacy_boolean_holdout=True)
    assert not v["verified"] and "legacy_boolean_holdout_path_forbidden" in v["reasons"]


def test_adv_holdout_family_overlap(w):
    files, shas = _real_result_files(w)
    ev = _evidence(w, result_files=files, result_shas=shas,
                   eval_fams=sorted(w.bench.families("selection")))
    v = _promote(w, ev)["verification"]
    assert not v["verified"]
    assert "task_family_overlap_selection_holdout" in v["reasons"]


# --------------------------------------------------------------------------- #
# Experience + ablation                                                       #
# --------------------------------------------------------------------------- #
def test_experience_ablation_changes_behaviour(w):
    bank = ExperienceBank(w.tmp / "experience.json")
    bank.add(ResearchExperience(
        experience_id="e1", kind="IMPROVEMENT", problem="falsifiability critic",
        claim="require falsifying observation", component="hypothesis_critic",
        source_run_id="M6TEST", source_task_ids=["t1"],
        evidence_shas=["a" * 64], artifact_shas=[w.cand.sha256]))
    out = run_retrieval_ablation(
        bank=bank, problem="falsifiability critic", harness=w.cand,
        tasks=w.bench.by_split("dev")[:1],
        runner=lambda h, t: {"ok": True, "injected":
                             [e["experience_id"] for e in h["experience_context"]]},
        out_dir=w.tmp / "ablation")
    assert out["behavior_changed"]
    assert out["arms"]["with_retrieval"]["n_injected"] == 1
    assert out["arms"]["without_retrieval"]["n_injected"] == 0
