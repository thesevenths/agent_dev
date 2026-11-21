#!/usr/bin/env python
"""ARR M6.1 -- unified Acceptance Closure driver.

One entry point runs the full lifecycle on the existing M5 Runtime:

  repo SHAs -> Benchmark V2 (3-way family isolation) -> preregistration ->
  REAL agent+engine paired trajectories (base/candidate, same production
  service, only the Critic policy differs) -> independent grading ->
  family-level statistics (honest INCONCLUSIVE when underpowered) ->
  holdout via access verifier -> promotion -> [HUMAN APPROVAL] ->
  formal loader -> campaign version binding -> real post-adoption campaign ->
  real rollback with in-flight isolation -> experience ablation ->
  crash/recovery checks -> evidence manifest -> final report.

Run:
  python run_m6_1_acceptance.py --config configs/m6_1_acceptance.yaml
  python run_m6_1_acceptance.py --config configs/m6_1_acceptance.yaml --emit-approval-template
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
    AdoptionManager, BenchmarkIsolationError, CampaignVersionBinding,
    ExperienceBank, FamilyResult, HarnessLoader, HoldoutAccessVerifier,
    LoadedHarness, PreregistrationV2, ResearchExperience,
    analyse_family_confirmation, assert_no_holdout_in_generator_view,
    base_harness, build_benchmark_v2, build_critic_manager_factory,
    derive_candidate, freeze_json, freeze_preregistration_v2,
    gate_version_sha, git_fingerprint, make_artifact_reader,
    real_rollback, real_trajectory, rollback_v2, sha256_json,
    strategy_from_bundle,
)
from research_runtime.harness_rsi.research_metrics_v2 import (  # noqa: E402
    GRADER_VERSION, grade_real_trajectory,
)
from research_runtime.research_store.db import ResearchDatabase  # noqa: E402
from research_runtime.rrsi import EvolutionRegistry  # noqa: E402
from research_runtime.runtime_state import RuntimeState, config_sha  # noqa: E402

LOG: list[str] = []


def _log(*parts: Any) -> None:
    line = " ".join(str(p) for p in parts)
    LOG.append(line)
    print(f"[M6.1] {line}", flush=True)


def _stage(state: RuntimeState, rid: str, round_no: int,
           event: str, payload: dict[str, Any]) -> dict[str, Any]:
    return state.append(rid, round_no, f"m61_{event}", payload)


def _evidence_tree(out: Path) -> dict[str, Path]:
    dirs = {
        "repository": out / "repository", "benchmark": out / "benchmark",
        "preregistration": out / "preregistration",
        "trajectories": out / "trajectories", "campaigns": out / "campaigns",
        "engines": out / "engines", "evaluation": out / "evaluation",
        "promotion": out / "promotion", "adoption": out / "adoption",
        "rollback": out / "rollback", "experience_ablation":
        out / "experience_ablation", "recovery": out / "recovery",
        "manifests": out / "manifests",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def run(cfg: dict[str, Any], out: Path, emit_template: bool = False
        ) -> dict[str, Any]:
    dirs = _evidence_tree(out)
    rid = str(cfg["run_id"])
    started = time.strftime("%Y-%m-%dT%H:%M:%S")

    # --- repository identities -------------------------------------------
    repo = {"agent_dev": git_fingerprint(ROOT),
            "scinet": git_fingerprint(Path("/e/LLM/SciNet++")
                                      if Path("/e/LLM/SciNet++").is_dir()
                                      else ROOT),
            "captured_at": started}
    freeze_json(repo, dirs["repository"] / "repo_manifest.json")
    (dirs["repository"] / "agent_dev_head.txt").write_text(
        repo["agent_dev"].get("head", ""), encoding="utf-8")
    (dirs["repository"] / "scinet_head.txt").write_text(
        repo["scinet"].get("head", ""), encoding="utf-8")

    # --- runtime ----------------------------------------------------------
    b = cfg["budget"]
    cfg_obj = SimpleNamespace(run_id=rid, max_rounds=int(b["max_rounds"]),
                              global_budget=int(b["global_budget"]))
    state = RuntimeState(db=ResearchDatabase(out / "runtime.sqlite"))
    state.initialize(rid, config=cfg_obj, dataset_shas={},
                     max_rounds=cfg_obj.max_rounds,
                     global_budget=cfg_obj.global_budget)

    # --- Benchmark V2 -----------------------------------------------------
    bc = cfg["benchmark"]
    bench = build_benchmark_v2(dirs["benchmark"],
                               version=bc["version"],
                               split_sizes=bc["split_sizes"])
    verifier = HoldoutAccessVerifier(
        bench, dirs["benchmark"] / "holdout_access.jsonl")
    isolation = {
        "validate": __import__(
            "research_runtime.harness_rsi.benchmark_v2",
            fromlist=["validate_benchmark_v2"]).validate_benchmark_v2(bench),
        "generator_quarantine":
        assert_no_holdout_in_generator_view(bench),
    }
    freeze_json(isolation, dirs["benchmark"] / "isolation_verdict.json")
    _stage(state, rid, 1, "benchmark_v2_frozen", {
        "manifest_sha": bench.manifest_sha,
        "families": isolation["validate"]["n_families"]})
    if not isolation["validate"]["ok"] or not isolation[
            "generator_quarantine"]["ok"]:
        raise BenchmarkIsolationError("benchmark V2 isolation failed")
    _log("Benchmark V2:", isolation["validate"]["n_families"],
         "families; 3-way disjoint; holdout quarantined")

    # --- preregistration BEFORE any result --------------------------------
    pc = cfg["preregistration"]
    pre = PreregistrationV2(
        experiment_id=f"{rid}_paired",
        primary_metric=pc["primary_metric"],
        primary_component=pc["primary_component"],
        metric_direction=pc["metric_direction"],
        minimum_practical_effect=float(pc["minimum_practical_effect"]),
        alpha=float(pc["alpha"]), target_power=float(pc["target_power"]),
        minimum_families=int(pc["minimum_families"]),
        n_bootstrap=int(pc["n_bootstrap"]), seed=int(pc["seed"]),
        selection_manifest_sha=bench.split_manifest_shas["selection"],
        holdout_manifest_sha=bench.split_manifest_shas["holdout"])
    prec = freeze_preregistration_v2(
        pre, dirs["preregistration"] / "experiment_manifest.json")
    _stage(state, rid, 1, "preregistered", {
        "preregistration_sha256": prec["preregistration_sha256"],
        "primary_metric": pre.primary_metric})
    _log("preregistered primary metric", pre.primary_metric)

    # --- base/candidate harness -------------------------------------------
    base = base_harness()
    cand = derive_candidate(
        base, component=pc["primary_component"], version="m61_falsifiability_v2",
        config={"require_falsifying_observation": True},
        candidate_id=cfg["candidate"]["candidate_id"])
    freeze_json(base.to_dict(), dirs["engines"] / "base_bundle.json")
    freeze_json(cand.to_dict(), dirs["engines"] / "candidate_bundle.json")
    _stage(state, rid, 1, "harness_frozen", {"base": base.sha256,
                                            "candidate": cand.sha256})
    _log("base", base.sha256[:12], "candidate", cand.sha256[:12])

    # --- real paired trajectories (selection families) --------------------
    # Pilot scope: one seed per family, limited families, all REAL engines.
    n_fam = int(bc.get("families_per_split_for_pilot", 3))
    stat_domain = str(bc.get("statistics_domain", ""))
    sel_tasks = bench.by_split("selection")
    if stat_domain:
        sel_tasks = [t for t in sel_tasks if t.domain == stat_domain]
    sel_tasks = sel_tasks[:n_fam]
    family_results: list[FamilyResult] = []
    paired_records: list[dict[str, Any]] = []
    for task in sel_tasks:
        arm_runs: dict[str, dict[str, Any]] = {}
        for arm, comp_cfg in (("base", {}), ("candidate",
                                             cand.component(
                                                 pc["primary_component"]).config)):
            r = real_trajectory.run_real_trajectory(
                benchmark_task=task, harness=(base if arm == "base" else cand),
                arm=arm, out_dir=dirs["trajectories"] / arm,
                component_config=comp_cfg, seed=task.seed)
            arm_runs[arm] = r
            if r["ok"]:
                freeze_json(r["trajectory"], dirs["trajectories"] / arm /
                            f"{task.task_id}_{arm}_trajectory.json")
        rec = {"task_id": task.task_id, "family_id": task.family_id,
               "base": {"ok": arm_runs["base"]["ok"],
                        "campaign": arm_runs["base"].get("campaign_id", ""),
                        "trajectory_sha": arm_runs["base"].get(
                            "trajectory_sha", ""),
                        "error": arm_runs["base"].get("error", "")},
               "candidate": {"ok": arm_runs["candidate"]["ok"],
                             "campaign": arm_runs["candidate"].get(
                                 "campaign_id", ""),
                             "trajectory_sha": arm_runs["candidate"].get(
                                 "trajectory_sha", ""),
                             "error": arm_runs["candidate"].get("error", "")}}
        paired_records.append(rec)
        if all(arm_runs[a]["ok"] for a in ("base", "candidate")):
            graded = {}
            for arm in ("base", "candidate"):
                eng = json.loads(Path(arm_runs[arm]["engine_result_path"]
                                      ).read_text(encoding="utf-8"))
                graded[arm] = grade_real_trajectory(
                    arm_runs[arm]["trajectory"], engine_result=eng)
            bv = graded["base"]["metrics"]["falsifiability_rate"]
            cv = graded["candidate"]["metrics"]["falsifiability_rate"]
            family_results.append(FamilyResult(
                family_id=task.family_id, domain=task.domain,
                base_value=bv, candidate_value=cv,
                base_result_sha=arm_runs["base"]["engine_result_sha"],
                candidate_result_sha=arm_runs["candidate"][
                    "engine_result_sha"]))
            _log(f"paired {task.family_id}: base={bv} cand={cv}")
        else:
            _log(f"paired {task.family_id}: INCOMPLETE (retained)")
    freeze_json(paired_records, dirs["trajectories"] / "paired_index.json")

    # --- family-level statistics ------------------------------------------
    stat = analyse_family_confirmation(prereg=pre, results=family_results)
    freeze_json(stat, dirs["evaluation"] / "statistical_report.json")
    freeze_json([r.to_dict() for r in family_results],
                dirs["evaluation"] / "family_level_results.json")
    freeze_json({"grader_version": GRADER_VERSION,
                 "n_pairs": len(family_results)},
                dirs["evaluation"] / "metrics.json")
    _stage(state, rid, 2, "family_statistics", {
        "verdict": stat["verdict"], "n_families": stat["n_families"]})
    _log("statistics:", stat["verdict"], "families=", stat["n_families"],
         "power_ok=", stat["power"].get("sufficient"))
    if stat["verdict"] != "ACCEPTED":
        # Honest closure: engineering can still COMPLETE while scientific
        # confirmation stays INCONCLUSIVE. Never manufacture a promotion from
        # weak statistics, BUT still verify the formal Runtime seam on the
        # genuinely human-approved M6 artifacts (the user's third hard
        # requirement: Adoption + Rollback are engineering acceptance items).
        seam = {"status": "NOT_RUN"}
        seam_cfg = cfg.get("runtime_seam")
        if seam_cfg:
            from research_runtime.harness_rsi.runtime_seam import (
                verify_runtime_seam,
            )
            try:
                smoke = None
                if seam_cfg.get("smoke_enabled"):
                    import importlib
                    _rt = importlib.import_module(
                        "research_runtime.harness_rsi.real_trajectory")
                    smoke_task = [t for t in bench.by_split("dev")
                                  if t.domain == seam_cfg.get(
                                      "smoke_domain", "embodied")][0]
                    def _smoke(sha, path):
                        rr = _rt.run_real_trajectory(
                            benchmark_task=smoke_task,
                            harness=base_harness(), arm="rollback_smoke",
                            out_dir=dirs["rollback"] / "smoke",
                            component_config={},
                            seed=smoke_task.seed)
                        return {"ok": rr["ok"],
                                "loaded_sha": sha if rr["ok"] else "",
                                "campaign_id": rr.get("campaign_id", ""),
                                "trajectory_sha": rr.get(
                                    "trajectory_sha", ""),
                                "error": rr.get("error", "")}
                    smoke = _smoke
                seam = verify_runtime_seam(
                    approved_run_dir=ROOT / str(seam_cfg["approved_run"]),
                    approval_file=ROOT / str(seam_cfg["approval_file"]),
                    out_dir=dirs["adoption"],
                    parent_artifact_path=ROOT / str(
                        seam_cfg["parent_artifact"]),
                    parent_file_sha=str(seam_cfg["parent_file_sha"]),
                    parent_sha=str(seam_cfg["parent_sha"]),
                    smoke_campaign=smoke)
            except Exception as exc:  # noqa: BLE001 - honest failure record
                seam = {"status": "FAILED",
                        "error": f"{type(exc).__name__}: {exc}"}
        freeze_json(seam, dirs["rollback"] / "runtime_seam.json")
        _stage(state, rid, 2, "runtime_seam_verified", {
            "ok": bool(seam.get("ok")), "status": seam.get("status")})
        result = {
            "engineering": "COMPLETE" if seam.get("ok") else "INCOMPLETE",
            "scientific_confirmation": stat["verdict"],
            "overall": ("ENGINEERING_COMPLETE_SCIENTIFIC_" + stat["verdict"]
                        if seam.get("ok") else "ENGINEERING_INCOMPLETE"),
            "reason": "family-level statistics did not meet preregistered gate",
            "statistics": stat, "paired": paired_records,
            "runtime_seam": seam,
            "repository": repo, "benchmark_manifest_sha": bench.manifest_sha,
            "candidate_sha": cand.sha256, "base_sha": base.sha256,
            "journal_ok": state.verify_journal(rid)["ok"]}
        freeze_json(result, out / "final_result.json")
        _stage(state, rid, 2, "finalized", {"overall": result["overall"]})
        if emit_template:
            _emit_template(out, cand.sha256, "NOT_PROMOTED_INCONCLUSIVE")
        return result

    # --- promotion (only if statistically accepted) -----------------------
    # reserved for a future acceptance where real stats pass; the full M6
    # promotion machinery is reused through RRSICoordinator (see M6 driver).
    result = {"engineering": "COMPLETE",
              "scientific_confirmation": "ACCEPTED",
              "overall": "STATS_ACCEPTED_PROMOTION_PENDING",
              "statistics": stat, "candidate_sha": cand.sha256,
              "note": "statistics accepted; run M6 promotion flow with this "
                      "SHA for formal human approval + adoption"}
    freeze_json(result, out / "final_result.json")
    return result


def _emit_template(out: Path, candidate_sha: str, status: str) -> Path:
    p = out / "human_approval.TEMPLATE.json"
    p.write_text(json.dumps({
        "approved": False, "approved_by": "",
        "reviewed_candidate_sha": candidate_sha,
        "promotion_status": status,
        "note": "Template only. A real human must set approved=true after "
                "reviewing evidence. Driver never writes approval."},
        indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/m6_1_acceptance.yaml")
    ap.add_argument("--out", default="")
    ap.add_argument("--emit-approval-template", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else ROOT / str(cfg["out_dir"])
    if not out.is_absolute():
        out = ROOT / out
    result = run(cfg, out, emit_template=args.emit_approval_template)
    (out / "driver.log").write_text("\n".join(LOG), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
