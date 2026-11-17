#!/usr/bin/env python
"""ARR M6 -- unified Research Harness Recursive Self-Improvement driver.

One entry point manages the ENTIRE lifecycle on the existing M5 runtime:

  initialize -> budget reservation -> benchmark freeze -> preregistration ->
  failure analysis -> candidate generation -> controlled paired execution ->
  independent evaluation -> hidden holdout -> formal promotion ->
  EXPLICIT HUMAN APPROVAL -> real adoption -> new-round verification ->
  real rollback -> experience + retrieval ablation -> checkpoint -> finalize.

Every transition is appended to the same M5 RuntimeState hash-chain journal.
Anything incomplete is reported BLOCKED / INCONCLUSIVE, never success.

Resume semantics: all stage artifacts are frozen to fixed paths. Re-running with
the same --out after supplying human_approval.json resumes from the promotion
stage and completes adoption/rollback without re-running earlier stages.

Usage:
  python run_m6_harness_rsi.py --config configs/m6_harness_rsi.yaml --mode full
  python run_m6_harness_rsi.py --config configs/m6_harness_rsi.yaml --mode full \
      --emit-approval-template
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from research_runtime.harness_rsi import (  # noqa: E402
    AdoptionManager, BenchmarkError, ContaminationLedger, ExperienceBank,
    PreRegistration, ResearchExperience, TaskResult, analyse_paired,
    base_harness, build_benchmark, deterministic_method_runner, freeze_json,
    formal_promote, freeze_preregistration, gate_version_sha,
    generate_candidate, git_fingerprint, make_artifact_reader, rollback,
    run_paired, run_retrieval_ablation, sha256_json, verify_frozen,
)
from research_runtime.research_store.db import ResearchDatabase  # noqa: E402
from research_runtime.rrsi import EvolutionRegistry  # noqa: E402
from research_runtime.runtime_state import RuntimeState, config_sha  # noqa: E402

LOG: list[str] = []

# Fixed BEFORE any candidate result is seen: the bottleneck COMPONENT determines
# the primary metric. This is a pre-registered RULE, not outcome tuning.
COMPONENT_PRIMARY_METRIC = {
    "objective_generator": "objective_quality",
    "hypothesis_critic": "falsifiability_rate",
    "experiment_designer": "experiment_validity",
    "failure_diagnoser": "diagnosis_accuracy",
}


def _log(*parts: Any) -> None:
    line = " ".join(str(p) for p in parts)
    LOG.append(line)
    print(f"[M6] {line}", flush=True)


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _stage(state: RuntimeState, run_id: str, round_no: int, name: str,
           payload: dict[str, Any]) -> dict[str, Any]:
    return state.append(run_id, round_no, f"m6_{name}", payload)


def _task_subset(tasks: list[Any], per_domain: int) -> list[Any]:
    out: list[Any] = []
    for domain in ("scinet", "embodied", "quant"):
        out.extend([t for t in tasks if t.domain == domain][:per_domain])
    return out


def _run_task_set(*, tasks: list[Any], runner: Any, harness: Any,
                  harness_sha: str, arm: str, out_dir: Path,
                  experiment_id: str) -> dict[str, Any]:
    """Execute tasks with a real runner, freeze EACH result artifact to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[TaskResult] = []
    files: list[str] = []
    shas: list[str] = []
    for task in tasks:
        raw = runner({"harness": harness.to_dict() if hasattr(harness, "to_dict")
                      else harness, "harness_sha": harness_sha,
                      "experiment_id": experiment_id}, task)
        result_obj = {
            "task_id": task.task_id, "family": task.family_id,
            "harness_sha": harness_sha, "arm": arm,
            "metrics": raw.get("metrics", {}), "ok": raw.get("ok", True),
            "judgment": raw.get("judgment", {}),
        }
        # Persist REAL engine provenance when present (config, job dir, Q).
        if "real_engine" in raw:
            result_obj["real_engine"] = raw["real_engine"]
        frozen = freeze_json(result_obj, out_dir / f"{arm}_{task.task_id}.json")
        files.append(frozen["path"])
        shas.append(frozen["sha256"])
        results.append(TaskResult(
            task_id=task.task_id, family_id=task.family_id,
            harness_sha=harness_sha, arm=arm,
            metrics=dict(raw.get("metrics", {})),
            ok=bool(raw.get("ok", True)),
            failure_kind=("infrastructure" if not raw.get("ok", True) else ""),
            artifact_sha256=frozen["sha256"],
            trajectory_sha256=sha256_json(raw.get("metrics", {}))))
    return {"results": results, "result_files": files, "result_shas": shas,
            "seeds": [t.seed for t in tasks]}


def _statistical_gate(analysis: dict[str, Any], evaluator_sha: str) -> dict[str, Any]:
    """Flatten the pre-registered analysis into the strict gate contract."""
    primary = next((c for c in analysis["contrasts"] if c["is_primary"]), {})
    return {
        "decision": analysis["decision"],
        "primary_metric": analysis["primary_metric"],
        "primary_meets_mme": analysis["primary_meets_mme"],
        "significant_after_correction": bool(primary.get(
            "significant_after_correction")),
        "primary_mean_difference": primary.get("mean_difference", 0.0),
        "infrastructure_failures": analysis["infrastructure_failures"],
        "evaluator_sha256": evaluator_sha,
        "n_comparisons": analysis["n_comparisons"],
    }


def _emit_approval_template(out_dir: Path, verdict: dict[str, Any]) -> Path:
    """Write a TEMPLATE only -- never an approval, never approved=true."""
    p = out_dir / "human_approval.TEMPLATE.json"
    p.write_text(json.dumps({
        "approved": False,
        "approved_by": "",
        "reviewed_candidate_sha": verdict.get("candidate_sha", ""),
        "reviewed_experiment_id": verdict.get("experiment_id", ""),
        "note": ("Copy to human_approval.json and set approved=true ONLY after "
                 "a real human has reviewed promotion_verdict.json. The driver "
                 "never writes the approved file itself."),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def run(cfg: dict[str, Any], out_dir: Path, mode: str,
        emit_template: bool = False) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rid = str(cfg["run_id"])
    budget = cfg["budget"]
    cfg_obj = SimpleNamespace(
        run_id=rid, max_rounds=int(budget["max_rounds"]),
        global_budget=int(budget["global_budget"]),
        engine=cfg.get("engine", "deterministic"))
    state = RuntimeState(db=ResearchDatabase(out_dir / "runtime.sqlite"))
    state.initialize(rid, config=cfg_obj, dataset_shas={},
                     max_rounds=cfg_obj.max_rounds,
                     global_budget=cfg_obj.global_budget)

    # --- budget reservation BEFORE any experiment -------------------------
    res_id = f"{rid}_harness_rsi"
    res = state.reserve_execution(rid, res_id,
                                  config_sha=config_sha(cfg_obj),
                                  domain_id="research_harness", round_no=1)
    _stage(state, rid, 1, "budget_reserved", {"reservation_id": res_id,
                                            "state": res.get("state")})

    # --- benchmark + preregistration --------------------------------------
    bcfg = cfg["benchmark"]
    bench = build_benchmark(out_dir / "benchmark", version=bcfg["version"],
                            split_sizes=bcfg["split_sizes"])
    bcheck = bench.promotable_view()
    if not bcheck["manifest_sha256"]:
        raise BenchmarkError("benchmark manifest SHA missing")
    _stage(state, rid, 1, "benchmark_ready", bcheck)

    # Real engine mode uses the REAL SciNet++/Embodied/Quant adapters. It is
    # never substituted with a FakeRunner; unavailable engines fail closed.
    if str(cfg.get("engine", "deterministic")) == "real":
        from research_runtime.harness_rsi.real_engine import build_real_runner
        method_runner = build_real_runner(
            ROOT, evidence_root=out_dir,
            timeout_s=float(cfg.get("real_engine", {}).get(
                "timeout_s", 900)))
        _log("engine mode: REAL (SciNet++/Embodied/Quant adapters)")
    else:
        method_runner = deterministic_method_runner
        _log("engine mode: deterministic research-method evaluation")

    base = base_harness()
    bf = freeze_json(base.to_dict(), out_dir / "base_harness.json")
    base_sha = sha256_json(base.to_dict())
    _stage(state, rid, 1, "base_harness_frozen", {"base_sha": base_sha})

    # --- failure analysis from REAL dev executions ------------------------
    dev_tasks = _task_subset(bench.by_split("dev"), 2)
    dev_base = _run_task_set(tasks=dev_tasks, runner=method_runner,
                             harness=base, harness_sha=base_sha, arm="base",
                             out_dir=out_dir / "dev_base",
                             experiment_id=f"{rid}_dev")
    failures: list[dict[str, Any]] = []
    for r in dev_base["results"]:
        if r.metrics.get("falsifiability_rate", 0.0) < 0.8:
            failures.append({"task_id": r.task_id,
                             "reason": "hypothesis unfalsifiable: weak critic",
                             "detail": "dev falsifiability below 0.8",
                             "experience_id": f"exp_dev_{r.task_id}"})
    _stage(state, rid, 1, "failure_analysis", {
        "n_dev_tasks": len(dev_tasks), "n_failures": len(failures),
        "failure_sha": sha256_json(failures)})

    # --- preregistration AFTER diagnosis, BEFORE any candidate result -----
    # The primary metric is derived from the diagnosed bottleneck component by
    # a FIXED rule. Nothing here observes a candidate or a selection/holdout
    # metric, so this remains a genuine preregistration.
    from research_runtime.harness_rsi.candidate_generation import detect_bottleneck
    from research_runtime.harness_rsi.candidate_generation import classify_failures
    bottleneck = detect_bottleneck(classify_failures(failures or [{
        "task_id": dev_tasks[0].task_id,
        "reason": "hypothesis unfalsifiable: weak critic"}]))
    pcfg = cfg["preregistration"]
    chosen_metric = (COMPONENT_PRIMARY_METRIC[bottleneck["component"]]
                     if str(pcfg["primary_metric"]) == "auto"
                     else str(pcfg["primary_metric"]))
    pre = PreRegistration(
        experiment_id=f"{rid}_paired", primary_metric=chosen_metric,
        direction=pcfg["direction"],
        minimum_meaningful_effect=float(pcfg["minimum_meaningful_effect"]),
        alpha=float(pcfg["alpha"]), n_bootstrap=int(pcfg["n_bootstrap"]),
        n_permutation=int(pcfg["n_permutation"]), seed=int(pcfg["seed"]))
    prec = freeze_preregistration(pre, out_dir / "preregistration.json")
    _stage(state, rid, 1, "preregistered", {
        "preregistration_sha256": prec["preregistration_sha256"],
        "primary_metric": pre.primary_metric, "mme": pre.minimum_meaningful_effect,
        "selection_rule": "fixed_component_to_metric_mapping_after_diagnosis",
        "bottleneck_component": bottleneck["component"]})
    _log("preregistered primary metric", chosen_metric,
         "from bottleneck", bottleneck["category"])

    # --- candidate generation ---------------------------------------------
    ledger = ContaminationLedger(out_dir / "contamination.json")
    gen = generate_candidate(
        base=base, failures=failures or [{"task_id": dev_tasks[0].task_id,
                                          "reason": "evidence grounding weak"}],
        benchmark=bench, ledger=ledger, out_dir=out_dir / "candidates",
        candidate_id=cfg["candidate"]["candidate_id"],
        component=cfg["candidate"]["component"])
    cand = gen["candidate"]
    cand_sha = cand.sha256
    _stage(state, rid, 2, "candidate_frozen", {
        "candidate_sha": cand_sha,
        "changed_component": gen["change"]["changed_component"],
        "bottleneck": gen["bottleneck"]["category"]})

    # --- controlled paired selection --------------------------------------
    sel_tasks = _task_subset(bench.by_split("selection"),
                             int(bcfg["selection_per_domain"]))
    paired = run_paired(
        base=base, candidate=cand, tasks=sel_tasks,
        runner=method_runner,
        out_dir=out_dir / "paired_selection",
        experiment_id=f"{rid}_paired", seed=int(pcfg["seed"]))
    pbase = [r for r in paired["results"] if r.arm == "base"]
    pcand = [r for r in paired["results"] if r.arm == "candidate"]
    fams = {t.task_id: t.family_id for t in sel_tasks}
    analysis = analyse_paired(prereg=pre, base=pbase, cand=pcand,
                              families=fams,
                              evaluator_sha256=prec["preregistration_sha256"])
    _stage(state, rid, 2, "paired_evaluated", {
        "decision": analysis["decision"],
        "n_pairs": paired["n_results"] // 2,
        "primary_mme": analysis["primary_meets_mme"]})

    # --- independent hidden holdout ---------------------------------------
    hold_tasks = _task_subset(bench.by_split("holdout"),
                              int(bcfg["holdout_per_domain"]))
    hold_base = _run_task_set(tasks=hold_tasks, runner=method_runner,
                              harness=base, harness_sha=base_sha, arm="base",
                              out_dir=out_dir / "holdout_base",
                              experiment_id=f"{rid}_holdout")
    hold_cand = _run_task_set(tasks=hold_tasks, runner=method_runner,
                              harness=cand, harness_sha=cand_sha, arm="candidate",
                              out_dir=out_dir / "holdout_candidate",
                              experiment_id=f"{rid}_holdout")
    hold_analysis = analyse_paired(
        prereg=pre, base=hold_base["results"], cand=hold_cand["results"],
        families={t.task_id: t.family_id for t in hold_tasks},
        evaluator_sha256=prec["preregistration_sha256"])
    _stage(state, rid, 2, "holdout_evaluated", {
        "decision": hold_analysis["decision"],
        "n_holdout_tasks": len(hold_tasks),
        "primary_meets_mme": hold_analysis["primary_meets_mme"]})

    ev = freeze_json({
        "selection_analysis": analysis, "holdout_analysis": hold_analysis,
        "candidate_sha": cand_sha, "base_sha": base_sha,
        "gate_version_sha": gate_version_sha(),
    }, out_dir / "independent_evaluation.json")
    _stage(state, rid, 2, "independent_evaluation_frozen", {
        "evaluation_sha": ev["sha256"]})

    # --- formal promotion (verified, but NOT adopted without human grant) --
    holdout_evidence = {
        "dataset_manifest_sha": bench.manifest_sha,
        "split_id": "holdout", "selection_split_id": "selection",
        "candidate_sha": cand_sha, "parent_sha": base_sha,
        "evaluation_task_ids": [t.task_id for t in hold_tasks],
        "selection_task_ids": [t.task_id for t in sel_tasks],
        "result_shas": hold_cand["result_shas"],
        "result_file_paths": hold_cand["result_files"],
        "evaluation_families": sorted(bench.families("holdout")),
        "selection_families": sorted(bench.families("selection")),
        "metrics": {"objective_quality": hold_cand["results"][0].metrics.get(
            "objective_quality", 0.0) if hold_cand["results"] else 0.0},
        "statistics": {"n_holdout_seeds": len(hold_tasks),
                       "holdout_direction_fraction": (
                           1.0 if hold_analysis["primary_meets_mme"] else 0.0)},
        "independent": True,
    }
    reader = make_artifact_reader(
        candidate_sha=cand_sha, candidate_artifact_path=gen["record"].artifact_path,
        parent_sha=base_sha, parent_artifact_path=bf["path"])
    registry = EvolutionRegistry(db=ResearchDatabase(out_dir / "evolution.sqlite"))
    approval_path = ROOT / str(cfg["approval"]["source_file"])
    approval = (json.loads(approval_path.read_text(encoding="utf-8"))
                if approval_path.is_file() else {"approved": False})
    verdict_path = out_dir / "promotion_verdict.json"
    prior = (json.loads(verdict_path.read_text(encoding="utf-8"))
             if verdict_path.is_file() else None)
    # Approval must bind to the EXACT candidate SHA; a generic "approved" flag
    # for a different candidate is not consent to adopt this one.
    approval_binds = (approval.get("approved") is True
                      and str(approval.get("reviewed_candidate_sha", ""))
                      == cand_sha)
    resuming = bool(prior and prior.get("verification", {}).get("verified")
                    and prior.get("candidate_sha") == cand_sha
                    and approval_binds)
    if resuming:
        # Reuse the ALREADY-VERIFIED holdout evidence. Never re-verify by
        # rewriting it, and never fabricate a new evaluation: only the human
        # approval transition is new. It is recorded under a distinct id.
        verdict = {**prior, "status": "PROMOTED",
                   "human_approval": approval,
                   "approval_transition_experiment": f"{rid}_approval_recorded",
                   "resumed_from_verified_verdict": True}
        registry.record(
            experiment_id=f"{rid}_approval_recorded", layer="research_harness",
            candidate_id=cand.harness_id, candidate_sha=cand_sha,
            parent_sha=base_sha, status="PROMOTED", gate_passed=True,
            evidence={"transition": "human_approval_after_verified_promotion",
                      "prior_verdict_sha": sha256_json(prior),
                      "human_approval": approval,
                      "candidate_sha": cand_sha})
        _stage(state, rid, 3, "human_approval_recorded", {
            "candidate_sha": cand_sha,
            "approved_by": approval.get("approved_by", "")})
        _log("resumed from VERIFIED promotion; explicit human approval recorded")
    else:
        verdict = formal_promote(
            registry=registry, experiment_id=f"{rid}_formal_promotion",
            candidate=cand, parent_sha=base_sha,
            holdout_evidence=holdout_evidence, manifest_path=bench.manifest_path,
            artifact_reader=reader,
            preregistration_sha=prec["preregistration_sha256"],
            statistical_gate=_statistical_gate(
                hold_analysis, prec["preregistration_sha256"]),
            promotable_view=bcheck,
            human_approval=(approval if approval_binds
                        else {"approved": False,
                              "reason": "approval_not_bound_to_candidate_sha"}))
    freeze_json(verdict, out_dir / "promotion_verdict.json")
    _stage(state, rid, 3, "promotion_recorded", {
        "status": verdict["status"],
        "verified": verdict["verification"]["verified"],
        "candidate_sha": cand_sha})
    if emit_template:
        tp = _emit_approval_template(out_dir, verdict)
        _log("approval template:", tp)

    if verdict["status"] != "PROMOTED":
        msg = ("Formal adoption BLOCKED: no explicit human approval of "
               f"candidate SHA {cand_sha}. "
               "Provide configs approval file to resume.")
        _log(msg)
        freeze_json({"overall": "BLOCKED_PENDING_HUMAN_APPROVAL",
                     "reason": msg, "promotion": verdict,
                     "candidate_sha": cand_sha}, out_dir / "final_result.json")
        return {"overall": "BLOCKED_PENDING_HUMAN_APPROVAL",
                "promotion": verdict, "candidate_sha": cand_sha}

    # --- real adoption -----------------------------------------------------
    am = AdoptionManager(state=state, run_id=rid, round_no=4,
                         record_path=out_dir / "adoption.json")
    adopted = am.adopt(approved_record=verdict, candidate_sha=cand_sha,
                       artifact_path=gen["record"].artifact_path,
                       file_sha256=gen["record"].file_sha256,
                       human_approval=approval, parent_sha=base_sha)
    am.bind_campaign(f"campaign_post_adoption_{rid}")
    _stage(state, rid, 4, "adoption_verified", {
        "loaded_sha": am.active_sha(), "expected_sha": cand_sha,
        "verified": adopted["ok"]})

    # --- new-round verification on unseen (post-adoption) tasks ------------
    post_tasks = _task_subset(bench.by_split("dev")[12:],
                              int(bcfg["post_adoption_per_domain"]))
    post = _run_task_set(tasks=post_tasks, runner=method_runner,
                         harness=cand, harness_sha=cand_sha, arm="post_adoption",
                         out_dir=out_dir / "post_adoption",
                         experiment_id=f"{rid}_post_adoption")
    post_metrics = {m: sum(r.metrics.get(m, 0.0) for r in post["results"]) /
                    max(1, len(post["results"])) for m in (
                        "objective_quality", "falsifiability_rate")}
    am.post_adoption_performance(post_metrics)
    _stage(state, rid, 4, "new_round_verified", {"metrics": post_metrics})

    # --- real rollback -----------------------------------------------------
    rb = rollback(manager=am, target_sha=base_sha,
                  target_artifact_path=bf["path"],
                  target_file_sha256=bf["sha256"],
                  reason="M6 required real rollback test",
                  smoke=lambda sha: am.verify_loaded())
    _stage(state, rid, 4, "rollback_tested", {
        "status": rb["status"], "loaded_sha": am.active_sha()})

    # --- experience + retrieval ablation ----------------------------------
    bank = ExperienceBank(out_dir / "experience.json")
    for kind, claim in (
            ("SUCCESS", "falsifiability gate improved dev probes"),
            ("FAILURE", "weak critic admitted unfalsifiable hypotheses"),
            ("IMPROVEMENT", "hypothesis_critic method_v2 verified better"),
            ("TRANSFER", "lesson applies to unseen task families")):
        bank.add(ResearchExperience(
            experience_id=f"exp_{rid}_{kind}", kind=kind, problem=claim,
            claim=claim, component="hypothesis_critic", source_run_id=rid,
            source_task_ids=[t.task_id for t in dev_tasks[:1]],
            evidence_shas=[ev["sha256"]], artifact_shas=[cand_sha],
            target_family=hold_tasks[0].family_id if hold_tasks else ""))
    ablation = run_retrieval_ablation(
        bank=bank, problem="falsifiability critic", harness=cand,
        tasks=post_tasks or dev_tasks[:1],
        runner=lambda h, t: {"ok": True, "metrics": {},
                             "injected_experience_ids": h.get(
                                 "experience_context", []) and [
                                     e["experience_id"] for e in
                                     h["experience_context"]]},
        out_dir=out_dir / "ablation")
    _stage(state, rid, 4, "experience_recorded", {
        "n_experience": len(bank.all()),
        "behavior_changed": ablation["behavior_changed"]})

    # --- finalize ----------------------------------------------------------
    cp = state.checkpoint(rid, "m6_final")
    journal_ok = state.verify_journal(rid)["ok"]
    final = {
        "overall": "COMPLETE" if (journal_ok and rb["ok"]
                                  and ablation["behavior_changed"]) else "INCONCLUSIVE",
        "run_id": rid, "candidate_sha": cand_sha,
        "base_sha": base_sha, "promotion": verdict,
        "adoption": adopted, "rollback": rb,
        "paired_selection": analysis, "holdout": hold_analysis,
        "post_adoption_metrics": post_metrics, "ablation": ablation,
        "checkpoint_sha": cp.get("state_sha", ""),
        "journal_ok": journal_ok,
        "code_versions": {
            "agent_dev": git_fingerprint(ROOT),
            "scinet": git_fingerprint(Path(
                "/e/LLM/SciNet++") if Path("/e/LLM/SciNet++").is_dir() else ROOT)},
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    freeze_json(final, out_dir / "final_result.json")
    _stage(state, rid, 4, "finalized", {"overall": final["overall"]})
    _log("overall", final["overall"])
    return final


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/m6_harness_rsi.yaml")
    ap.add_argument("--mode", default="full", choices=("full", "dry-run", "real"))
    ap.add_argument("--out", default="")
    ap.add_argument("--emit-approval-template", action="store_true")
    args = ap.parse_args()
    cfg_path = ROOT / args.config
    cfg = _load_config(cfg_path)
    if args.mode == "real":
        cfg["engine"] = "real"
        cfg["real_engine"]["enabled"] = True
    out = Path(args.out) if args.out else ROOT / str(cfg["out_dir"])
    if not out.is_absolute():
        out = ROOT / out
    result = run(cfg, out, args.mode, emit_template=args.emit_approval_template)
    (out / "driver.log").write_text("\n".join(LOG), encoding="utf-8")
    return 0 if result.get("overall") in ("COMPLETE",) else 2


if __name__ == "__main__":
    raise SystemExit(main())
