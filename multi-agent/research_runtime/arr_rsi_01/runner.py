"""ARR-RSI-01 -- real-runtime orchestration and artifact emission.

No FakeRunner: every arm calls ``research_runtime.arr_runtime.run_research_task``
which drives the existing ResearchService, SciNet++ adapter, CampaignRunner and
comparator. This module only wires a task-specific domain bundle, records the
trajectory, and evaluates AFTER all arms have finished.
"""
from __future__ import annotations

import json
import subprocess
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..arr_runtime import run_research_task
from ..research_task import ResearchTask
from ..settings import DEFAULT_SCINET_DIR
from .arms import ARMS, ArmSpec, derive_arms
from .diagnosis import DiagnosisOutput, EvidenceGroundedDiagnoser, ShuffledDiagnoser
from .evaluation import (
    DIMENSIONS,
    Preregistration,
    analyse_statistics,
    decide_verdict,
    evaluate_task,
)
from .tasks import (
    BENCHMARK_ID,
    BENCHMARK_VERSION,
    PublicTask,
    assert_no_ground_truth,
    build_ground_truth,
    build_tasks,
    sha256_json,
    task_manifest,
)

ARTIFACTS = (
    "experiment_manifest.json", "task_manifest.json", "trajectory.json",
    "diagnosis.json", "intervention.json", "results.json",
    "statistical_analysis.json", "ARR_RSI_01_REPORT.md",
)


def _git_head(path: str | Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _register_task_bundle(*, task: PublicTask, arm: ArmSpec,
                          run_root: Path) -> dict[str, Any]:
    """Register a per-task/per-arm SciNet bundle in the process registry.

    Reuses the production components; only the policy factory is task-scoped.
    The engine config, dataset version, seeds and intervention policy are
    frozen BEFORE the arm is executed.
    """
    from ..adapters import SciNetAdapter
    from ..campaign.campaign_store import CampaignStore
    from ..campaign.scinet import (
        build_scinet_campaign_runner,
        scinet_campaign_policy,
    )
    from ..candidate_config import CandidateConfigBuilder
    from ..comparison.report_store import ComparisonReportStore
    from ..comparison.scinet import build_scinet_comparator
    from ..domain_registry import DomainBundle, register_domain
    from ..domains.scinet import scinet_research_policy

    project = Path(DEFAULT_SCINET_DIR)
    data_dir = run_root / "store"
    data_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(s) for s in task.budget.get("paired_seeds", [task.seed]))
    policy = scinet_research_policy(
        baseline_config_path=task.engine_config,
        dataset_version=task.dataset_version,
        allowed_dataset_versions=(task.dataset_version,),
        paired_seeds=seeds,
        max_seeds=max(1, len(seeds)),
        max_campaign_rounds=arm.harness.campaign_rounds,
        intervention_policy={k: dict(v) for k, v in arm.intervention_policy.items()},
        epochs=int(task.budget.get("epochs", 1)),
        max_jobs=int(task.budget.get("max_jobs", 8)),
        max_total_seconds=float(task.budget.get("max_total_seconds", 900.0)),
        evaluation_split="validation",
    )
    campaign_store = CampaignStore(data_dir / "campaigns.sqlite")
    report_store = ComparisonReportStore(project / policy.campaign_root)
    config_builder = CandidateConfigBuilder(
        project, allowed_keys=policy.allowed_config_keys,
        value_rules=policy.value_rules, campaign_root=policy.campaign_root)
    campaign_policy = scinet_campaign_policy(
        epochs=policy.epochs, timeout_per_job_s=policy.job_timeout_s,
        max_jobs=policy.max_jobs,
        max_candidates=arm.harness.max_candidates,
        max_seeds=len(seeds),
    )

    def runner_factory():
        return build_scinet_campaign_runner(
            store=campaign_store, config_builder=config_builder,
            policy=campaign_policy, adapter=SciNetAdapter())

    def comparator_factory():
        return build_scinet_comparator(
            campaign_store=campaign_store, report_store=report_store,
            project_dir=project)

    def bundle_policy_factory(**_: Any):
        return policy

    register_domain(DomainBundle(
        name="scinet", policy_factory=bundle_policy_factory,
        campaign_store=campaign_store, report_store=report_store,
        campaign_runner_factory=runner_factory,
        comparator_factory=comparator_factory,
        dataset_verifier=lambda cfg, version: config_builder.verify_dataset(cfg, version),
        campaign_root=str(project / policy.campaign_root),
        adapter_factory=SciNetAdapter,
        description=f"ARR-RSI-01 task-scoped bundle {task.task_id}/{arm.arm}",
    ))
    return {"dataset_version": policy.dataset_version,
            "baseline_config_path": policy.baseline_config_path,
            "intervention_policy": policy.intervention_policy,
            "paired_seeds": list(policy.paired_seeds)}


def build_research_task(task: PublicTask, arm: ArmSpec,
                        run_id: str = "") -> ResearchTask:
    safe_id = task.task_id.replace("::", "__")
    # SciNet++ derives job IDs from experiment_id. A run-scoped nonce prevents
    # a later experiment from colliding with an earlier run of the same task.
    suffix = f"__{run_id}" if run_id else ""
    return ResearchTask(
        task_id=f"{safe_id}__{arm.arm}{suffix}",
        domain="scinet",
        objective=task.objective,
        dataset_refs=[task.dataset_reference],
        budget={k: v for k, v in task.budget.items() if k != "paired_seeds"},
        evaluation_protocol={
            "primary_metric": task.primary_metric,
            "metric_direction": task.metric_direction,
            "decision_split": "validation",
            "reserved_final_split": "test",
        },
        heldout_policy={
            "independent_holdout": True,
            "holdout_split": "test",
            "ood_task": task.task_family == "scientific_ml",
            "paired_seeds": list(task.budget.get("paired_seeds", [task.seed])),
        },
    )


def _read_evidence(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
    return records


def run_arm(*, task: PublicTask, arm: ArmSpec, run_root: Path,
            run_id: str = "") -> dict[str, Any]:
    """Execute ONE arm through the REAL research runtime."""
    started = time.time()
    arm_root = run_root / arm.arm
    arm_root.mkdir(parents=True, exist_ok=True)
    frozen = _register_task_bundle(task=task, arm=arm, run_root=arm_root)
    research_task = build_research_task(task, arm, run_id=run_id)
    result = run_research_task(
        research_task, harness=arm.harness, run_root=arm_root,
        approved_by="arr-rsi-01-operator")
    payload = result.model_dump(mode="json")
    # run_research_task creates its own nested run_root; the real evidence
    # ledger lives there. Never synthesize tool calls from prose.
    reported_root = Path((payload.get("provenance") or {}).get("run_root") or arm_root)
    evidence_path = reported_root / "store" / "evidence.jsonl"
    evidence_records = _read_evidence(evidence_path)
    trajectory = {
        "task_id": task.task_id,
        "arm": arm.arm,
        "task_view": task.agent_view(),
        "research_task_sha256": research_task.fingerprint(),
        "harness": arm.harness.to_dict(),
        "harness_sha256": arm.harness.fingerprint(),
        "frozen": frozen,
        "hypotheses": payload.get("hypotheses", []),
        "experiment_plans": {
            "campaign": payload.get("campaign", {}),
            "critic_reviews": (payload.get("provenance") or {}).get("critic_reviews", []),
        },
        "tool_calls": [
            {"event_type": e.get("event_type"), "at": e.get("at"),
             "payload": e.get("payload")}
            for e in evidence_records
        ],
        "experiment_results": [
            (e.get("payload") or {}).get("result")
            for e in evidence_records
            if e.get("event_type") == "result" and (e.get("payload") or {}).get("result")
        ],
        "evidence_refs": {
            "baseline_id": (payload.get("evidence") or {}).get("baseline_id", ""),
            "report_path": (payload.get("evidence") or {}).get("report_path", ""),
            "n_evidence_records": len(evidence_records),
        },
        "budget_usage": payload.get("telemetry", {}),
        "final_conclusion": {
            "outcome": payload.get("outcome", ""),
            "quality": payload.get("quality", {}),
            "comparisons": payload.get("comparisons", []),
            "holdout": (payload.get("evidence") or {}).get("holdout", {}),
            "ood": (payload.get("evidence") or {}).get("ood", {}),
        },
        "ok": payload.get("ok", False),
        "error": payload.get("error", ""),
        "artifact_sha256": payload.get("artifact_sha256", ""),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    assert_no_ground_truth(trajectory)
    (arm_root / "trajectory.json").write_text(
        json.dumps(trajectory, indent=2, sort_keys=True, ensure_ascii=False,
                   default=str), encoding="utf-8")
    return {"result": payload, "trajectory": trajectory}


def run_task(*, task: PublicTask, truth: Any, run_root: Path,
             prereg: Preregistration, run_id: str = "",
             shuffled_pool: dict[str, DiagnosisOutput] | None = None,
             verbose: bool = False) -> dict[str, Any]:
    """Base research -> diagnosis -> 4 matched arms -> independent evaluation."""
    task_root = run_root / "tasks" / task.task_id.replace("::", "__")
    task_root.mkdir(parents=True, exist_ok=True)
    base_spec = ArmSpec(
        arm="base",
        harness=derive_arms(task=task.agent_view(),
                            diagnosis=DiagnosisOutput(
                                task_id=task.task_id, failure_modes=[task.failure_mode],
                                root_cause="", evidence_refs=[], confidence=0.0,
                                alternative_explanations=[],
                                improvement_hypothesis="", proposed_intervention=None,
                                expected_metric={}, falsification_condition=""))["base"].harness,
        intervention_policy={}, diagnosis_sha256="", intervention=None,
        source_task_id="", rationale="Frozen H0.",
    )
    if verbose:
        print(f"[{task.task_id}] base run", flush=True)
    executed = {"base": run_arm(task=task, arm=base_spec, run_root=task_root,
                                run_id=run_id)}
    base_traj = executed["base"]["trajectory"]
    base_result = executed["base"]["result"]

    diagnoser = EvidenceGroundedDiagnoser()
    diagnosis = diagnoser.diagnose(
        task=task.agent_view(), trajectory=base_traj,
        experiment_results=base_traj["experiment_results"],
        evidence=base_traj["evidence_refs"],
        budget=task.budget)
    diagnoses = {"base": diagnosis.to_dict()}

    shuffled = None
    if shuffled_pool:
        try:
            shuffled = ShuffledDiagnoser(pool=shuffled_pool).diagnose(
                task=task.agent_view(), trajectory=base_traj,
                experiment_results=base_traj["experiment_results"],
                evidence=base_traj["evidence_refs"], budget=task.budget)
            diagnoses["shuffled"] = shuffled.to_dict()
        except ValueError:
            pass

    arms = derive_arms(task=task.agent_view(), diagnosis=diagnosis,
                       shuffled=shuffled)
    for arm_name in ("candidate", "placebo", "shuffled"):
        if verbose:
            print(f"[{task.task_id}] {arm_name} run", flush=True)
        executed[arm_name] = run_arm(task=task, arm=arms[arm_name],
                                     run_root=task_root, run_id=run_id)

    arm_results = {name: value["result"] for name, value in executed.items()}
    evaluation = evaluate_task(
        task=task, truth=truth, arms=arms, arm_results=arm_results,
        diagnoses={task.task_id: diagnoses},
        trajectory=base_traj, prereg=prereg)
    (task_root / "diagnosis.json").write_text(
        json.dumps(diagnoses, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8")
    (task_root / "intervention.json").write_text(
        json.dumps({k: v.to_dict() for k, v in arms.items()},
                   indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8")
    return {
        "task": task.to_dict(),
        "diagnoses": diagnoses,
        "arms": {k: v.to_dict() for k, v in arms.items()},
        "arm_results": arm_results,
        "trajectories": {k: v["trajectory"] for k, v in executed.items()},
        "evaluation": evaluation,
        "task_root": str(task_root),
    }


def finalize_run(*, run_root: str | Path,
                 scinet_dir: str | Path | None = None) -> dict[str, Any]:
    """Re-run ONLY evaluation/artifact emission from already-recorded arms.

    Used to recover a run whose engine arms completed but whose aggregation
    failed. No experiment is re-executed and no result is fabricated.
    """
    from .tasks import GroundTruthStore
    root = Path(run_root).resolve()
    manifest = json.loads((root / "experiment_manifest.json").read_text(
        encoding="utf-8"))
    prereg_raw = dict(manifest["preregistration"])
    prereg_raw.pop("preregistration_sha256", None)
    prereg = Preregistration(**prereg_raw)
    task_ids = list(prereg.freeze["tasks"])
    all_tasks = {t.task_id: t for t in build_tasks(
        scinet_dir or DEFAULT_SCINET_DIR)}
    truth = GroundTruthStore(root / "ground_truth.sealed.json").load(
        manifest["frozen"]["ground_truth_sha256"])
    diagnoser = EvidenceGroundedDiagnoser()
    executions = []
    for task_id in task_ids:
        task = all_tasks[task_id]
        task_dir = root / "tasks" / task_id.replace("::", "__")
        trajectories = {arm: json.loads((task_dir / arm / "trajectory.json").read_text(
            encoding="utf-8")) for arm in ARMS}
        arm_results = {}
        for arm in ARMS:
            result_files = sorted((task_dir / arm).glob("*/task_result.json"))
            if not result_files:
                raise FileNotFoundError(f"missing real arm result: {task_id}/{arm}")
            arm_results[arm] = json.loads(result_files[-1].read_text(encoding="utf-8"))
        base_traj = trajectories["base"]
        diagnosis = diagnoser.diagnose(
            task=task.agent_view(), trajectory=base_traj,
            experiment_results=base_traj.get("experiment_results", []),
            evidence=base_traj.get("evidence_refs", {}), budget=task.budget)
        diagnoses = {"base": diagnosis.to_dict()}
        shuffled = None
        if len(task_ids) > 1:
            from .diagnosis import ShuffledDiagnoser
            pool = {tid: diagnosis for tid in task_ids}
            try:
                shuffled = ShuffledDiagnoser(pool=pool).diagnose(
                    task=task.agent_view(), trajectory=base_traj,
                    experiment_results=base_traj.get("experiment_results", []),
                    evidence=base_traj.get("evidence_refs", {}), budget=task.budget)
                diagnoses["shuffled"] = shuffled.to_dict()
            except ValueError:
                pass
        arms = derive_arms(task=task.agent_view(), diagnosis=diagnosis,
                           shuffled=shuffled)
        evaluation = evaluate_task(
            task=task, truth=truth[task_id], arms=arms, arm_results=arm_results,
            diagnoses={task_id: diagnoses}, trajectory=base_traj, prereg=prereg)
        (task_dir / "diagnosis.json").write_text(
            json.dumps(diagnoses, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8")
        (task_dir / "intervention.json").write_text(
            json.dumps({k: v.to_dict() for k, v in arms.items()},
                       indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8")
        executions.append({
            "task": task.to_dict(), "diagnoses": diagnoses,
            "arms": {k: v.to_dict() for k, v in arms.items()},
            "arm_results": arm_results, "trajectories": trajectories,
            "evaluation": evaluation,
        })

    evaluations = [e["evaluation"] for e in executions]
    arm_results_all = {e["task"]["task_id"]: e["arm_results"] for e in executions}
    statistics = analyse_statistics(task_evaluations=evaluations,
                                    arm_results=arm_results_all, prereg=prereg)
    verdict = decide_verdict(task_evaluations=evaluations,
                             statistics=statistics, prereg=prereg)
    payloads = {
        "trajectory.json": {e["task"]["task_id"]: e["trajectories"] for e in executions},
        "diagnosis.json": {e["task"]["task_id"]: e["diagnoses"] for e in executions},
        "intervention.json": {e["task"]["task_id"]: e["arms"] for e in executions},
        "results.json": {"n_tasks": len(executions),
                         "tasks": [e["task"] for e in executions],
                         "evaluations": evaluations,
                         "arm_results": arm_results_all, "verdict": verdict},
        "statistical_analysis.json": statistics,
    }
    for name, payload in payloads.items():
        (root / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False,
                       default=str), encoding="utf-8")
    write_report(run_root=root, verdict=verdict, statistics=statistics,
                 evaluations=evaluations, manifest=manifest)
    return {"run_root": str(root), "n_tasks": len(executions),
            "verdict": verdict["verdict"],
            "artifacts": [str(root / name) for name in ARTIFACTS]}


def freeze_manifest(*, run_root: Path, tasks: list[PublicTask],
                    truth_sha: str, prereg: Preregistration) -> dict[str, Any]:
    manifest = {
        "benchmark_id": BENCHMARK_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "experiment_id": "arr_rsi_01_p0",
        "preregistration": prereg.to_dict(),
        "frozen": {
            "model_version": "frozen-by-SciNet-config; no weight modification",
            "runtime_version": _git_head(Path(__file__).resolve().parents[2]),
            "engine_version": _git_head(DEFAULT_SCINET_DIR),
            "dataset": sorted({t.dataset_reference for t in tasks}),
            "seeds": sorted({s for t in tasks for s in t.budget["paired_seeds"]}),
            "metrics": sorted({t.primary_metric for t in tasks}),
            "research_budget": tasks[0].budget,
            "tool_budget": {"engine": "research_api.cli",
                            "allowed_stages": ["train", "evaluate"],
                            "max_jobs_per_arm": tasks[0].budget["max_jobs"]},
            "ground_truth_sha256": truth_sha,
        },
        "artifact_contract": list(ARTIFACTS),
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    (run_root / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8")
    return manifest


def write_report(*, run_root: Path, verdict: dict[str, Any],
                 statistics: dict[str, Any], evaluations: list[dict[str, Any]],
                 manifest: dict[str, Any]) -> Path:
    reason_lines = [f"- {r}" for r in verdict.get("reasons", [])]
    if not reason_lines:
        reason_lines = ["- (no reasons recorded)"]
    lines = [
        "# ARR-RSI-01 P0 — Experimental Report", "",
        "## Verdict", "",
        f"**{verdict['verdict']}**", "",
        *reason_lines,
        "", "## Dimension means", "",
        "| Dimension | Mean |", "|---|---|",
    ]
    for dim, value in (verdict.get("dimension_means") or {}).items():
        lines.append(f"| {dim} | {value if value is not None else 'n/a'} |")
    lines += ["", "## Gates", ""]
    for key, value in (verdict.get("gates") or {}).items():
        lines.append(f"- `{key}`: {value}")
    lines += ["", "## Task outcomes", "",
              "| task_id | cause correct | intervention correct | validity |",
              "|---|---|---|---|"]
    for e in evaluations:
        sc = e.get("scientific_correctness") or {}
        val = (e.get("experimental_validity") or {}).get("valid")
        lines.append(f"| {e['task_id']} | {sc.get('root_cause_correct')} | "
                     f"{sc.get('intervention_correct')} | {val} |")
    lines += ["", "## Evidence boundary", "",
              "- Ground truth was sealed separately and read only by the evaluator.",
              "- Research/Diagnosis agents received an agent view with no GT and no hidden-test reference.",
              "- Missing evidence was never converted into PASS.", "",
              "## Scope statement", "",
              verdict.get("verdict_note", verdict.get("note", "")),
              "", f"Preregistration SHA: `{manifest['preregistration']['preregistration_sha256']}`",
              f"Manifest SHA: `{manifest['manifest_sha256']}`",
              f"Statistical analysis SHA: `{statistics.get('statistical_analysis_sha256', '')}`",
              ""]
    path = run_root / "ARR_RSI_01_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _worker_run_task_with_shuffle(task: PublicTask, truth: Any, root: str,
                                  prereg: Preregistration,
                                  shuffled_pool: dict[str, DiagnosisOutput],
                                  run_id: str):
    return run_task(task=task, truth=truth, run_root=Path(root), prereg=prereg,
                    run_id=run_id, shuffled_pool=shuffled_pool, verbose=False)


def _worker_run_task(args: tuple[PublicTask, Any, str, Preregistration, str, str]):
    """Top-level picklable worker for parallel task execution."""
    task, truth, root, prereg, task_root_name, run_id = args
    return run_task(task=task, truth=truth, run_root=Path(root),
                    prereg=prereg, run_id=run_id, verbose=False)


def run_pilot(*, n_tasks: int = 1, n_arms: int | None = None,
              run_root: str | Path = "ARR_RSI_01_RUN",
              scinet_dir: str | Path | None = None,
              verbose: bool = False,
              parallel: int = 1) -> dict[str, Any]:
    """Run the real loop for the first N tasks; all 4 matched arms per task."""
    root = Path(run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:8]
    tasks = build_tasks(scinet_dir or DEFAULT_SCINET_DIR)
    tasks = tasks[:max(1, min(int(n_tasks), len(tasks)))]
    truth_all = build_ground_truth(tasks)
    from .tasks import GroundTruthStore
    store = GroundTruthStore(root / "ground_truth.sealed.json")
    seal = store.seal(truth_all)
    prereg = Preregistration(freeze={
        "run_id": run_id,
        "tasks": [t.task_id for t in tasks],
        "truth_sha256": seal["ground_truth_sha256"],
        "arms": ["base", "candidate", "placebo", "shuffled"],
    })
    manifest = freeze_manifest(run_root=root, tasks=tasks,
                               truth_sha=seal["ground_truth_sha256"],
                               prereg=prereg)
    (root / "task_manifest.json").write_text(
        json.dumps(task_manifest(tasks, ground_truth_sha256=seal[
            "ground_truth_sha256"]), indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8")

    if parallel > 1:
        from concurrent.futures import ProcessPoolExecutor
        # Attribution control without cross-process diagnosis sharing: shuffle
        # the candidate intervention deterministically to another task's knob.
        shuffled_pool: dict[str, DiagnosisOutput] = {}
        for index, task in enumerate(tasks):
            donor = tasks[(index + 1) % len(tasks)]
            donor_ivs = [iv for iv in donor.valid_interventions
                         if isinstance(iv, dict) and iv.get("role") == "candidate"]
            shuffled_pool[task.task_id] = DiagnosisOutput(
                task_id=donor.task_id, failure_modes=[donor.failure_mode],
                root_cause="shuffled_control", evidence_refs=[], confidence=0.0,
                alternative_explanations=[], improvement_hypothesis="",
                proposed_intervention=donor_ivs[0] if donor_ivs else None,
                expected_metric={}, falsification_condition="")
        jobs = [(t, truth_all[t.task_id], str(root), prereg,
                 t.task_id.replace("::", "__"), run_id) for t in tasks]
        with ProcessPoolExecutor(max_workers=min(parallel, len(jobs))) as pool:
            futures = []
            for t in tasks:
                futures.append(pool.submit(
                    _worker_run_task_with_shuffle,
                    t, truth_all[t.task_id], str(root), prereg,
                    shuffled_pool, run_id))
            executions = [f.result() for f in futures]
    else:
        shuffled_pool: dict[str, DiagnosisOutput] = {}
        executions = []
        for task in tasks:
            result = run_task(
                task=task, truth=truth_all[task.task_id], run_root=root,
                prereg=prereg, run_id=run_id,
                shuffled_pool=shuffled_pool or None, verbose=verbose)
            executions.append(result)
            if "base" in result["diagnoses"]:
                shuffled_pool[task.task_id] = DiagnosisOutput(
                    **{k: v for k, v in result["diagnoses"]["base"].items()
                       if k in DiagnosisOutput.__dataclass_fields__})

    truth = store.load(expected_sha256=seal["ground_truth_sha256"])
    evaluations = [e["evaluation"] for e in executions]
    arm_results = {e["task"]["task_id"]: e["arm_results"] for e in executions}
    statistics = analyse_statistics(task_evaluations=evaluations,
                                    arm_results=arm_results, prereg=prereg)
    verdict = decide_verdict(task_evaluations=evaluations,
                             statistics=statistics, prereg=prereg)
    trajectory_all = {e["task"]["task_id"]: e["trajectories"] for e in executions}
    results = {
        "n_tasks": len(executions),
        "tasks": [e["task"] for e in executions],
        "evaluations": evaluations,
        "arm_results": arm_results,
        "verdict": verdict,
    }
    diagnosis_all = {e["task"]["task_id"]: e["diagnoses"] for e in executions}
    intervention_all = {e["task"]["task_id"]: e["arms"] for e in executions}
    for name, payload in (("trajectory.json", trajectory_all),
                          ("diagnosis.json", diagnosis_all),
                          ("intervention.json", intervention_all),
                          ("results.json", results),
                          ("statistical_analysis.json", statistics)):
        (root / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False,
                       default=str), encoding="utf-8")
    write_report(run_root=root, verdict=verdict, statistics=statistics,
                 evaluations=evaluations, manifest=manifest)

    # Persist verified improvement (or failure) for subsequent research.
    from ..harness_rsi.experience import ExperienceBank, ResearchExperience
    bank = ExperienceBank(root / "experience.jsonl")
    kind = "IMPROVEMENT" if verdict["verdict"] == "PASS" else "FAILURE"
    exp = ResearchExperience(
        experience_id=f"arr_rsi_01_{kind.lower()}_{sha256_json(verdict)[:12]}",
        kind=kind, problem="ARR-RSI-01 recursive self-improvement loop",
        claim=f"verdict={verdict['verdict']}; "
              f"primary_delta={verdict['gates'].get('primary_mean_difference')}",
        component="failure_diagnoser", source_run_id="arr_rsi_01_p0",
        source_task_ids=[t.task_id for t in tasks],
        evidence_shas=[statistics.get("statistical_analysis_sha256", ""),
                       manifest["manifest_sha256"]],
        confidence=(verdict["dimension_means"].get("scientific_correctness") or 0.0),
    )
    bank.add(exp)
    return {"run_root": str(root), "n_tasks": len(executions),
            "verdict": verdict["verdict"],
            "artifacts": [str(root / name) for name in ARTIFACTS]}


__all__ = ["ARTIFACTS", "run_arm", "run_task", "run_pilot",
           "build_research_task", "freeze_manifest"]
