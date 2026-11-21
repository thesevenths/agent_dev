"""ARR-RSI-01 P0.1 -- real-runtime orchestration with integrity guards.

No FakeRunner. Every arm calls ``research_runtime.arr_runtime.run_research_task``.
P0.1 adds forced concurrency=1, run-scoped IDs, resume-in-place, applied-config
verification, per-arm infrastructure records and independent protocol freezing.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
import uuid
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any

import yaml

from ..arr_runtime import run_research_task
from ..research_task import ResearchTask
from ..settings import DEFAULT_SCINET_DIR
from .arms import (
    ARMS, ArmSpec, derive_arms, finalize_intervention_record,
)
from .diagnosis import (
    DiagnosisOutput, EvidenceGroundedDiagnoser, INFRA_ERROR_TOKENS,
    ShuffledDiagnoser, _perturb,
)
from .evaluation import (
    DIMENSIONS, Preregistration, analyse_statistics, decide_verdict,
    evaluate_task,
)
from .tasks import (
    ALLOWED_CONFIG_SCHEMA, BENCHMARK_ID, BENCHMARK_VERSION, EXPERIMENT_ID,
    PublicTask, assert_no_ground_truth, build_ground_truth, build_tasks,
    sha256_json, task_manifest,
)

ARTIFACTS = (
    "experiment_manifest.json", "task_manifest.json", "trajectory.json",
    "diagnosis.json", "intervention.json", "results.json",
    "statistical_analysis.json", "ARR_RSI_01_P0_1_REPORT.md",
)
# P0.1-4: ONE CLI timeout for every arm. The 60s default proved too short for
# `research_api.cli list` under sustained engine load.
CLI_TIMEOUT_S = 180.0


def _git_head(path: str | Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _read_config(path: str | Path) -> dict[str, Any]:
    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _find_applied_config(*, campaign_id: str, research_task_id: str,
                         root: Path, arm: str = "") -> tuple[dict[str, Any], str]:
    """Read the ACTUAL config the engine executed, never a candidate copy."""
    base = Path(DEFAULT_SCINET_DIR) / "outputs" / "research_campaigns"
    search_roots = [base / campaign_id] if campaign_id else []
    search_roots.append(root)
    want_candidate = arm in ("candidate", "placebo", "shuffled")
    for search in search_roots:
        if not search.exists():
            continue
        paths = sorted(search.rglob("config.yaml"))
        preferred = [p for p in paths if
                     (("cand_" in str(p)) if want_candidate
                      else ("baseline" in str(p)))]
        for path in preferred or paths:
            if campaign_id and campaign_id in str(path):
                return _read_config(path), str(path)
            if not campaign_id and research_task_id in str(path):
                return _read_config(path), str(path)
    return {}, ""


def _ensure_distinct_shuffle(*, shuffled: DiagnosisOutput | None,
                             diagnosis: DiagnosisOutput,
                             baseline_config: dict[str, Any],
                             agent_task_id: str) -> DiagnosisOutput | None:
    """A shuffle landing on the candidate's knob is not a valid control."""
    if shuffled is None:
        return None
    cand_key = (diagnosis.proposed_intervention or {}).get("key")
    sh_key = (shuffled.proposed_intervention or {}).get("key")
    if not cand_key or sh_key != cand_key:
        return shuffled
    others = [s["key"] for s in ALLOWED_CONFIG_SCHEMA if s["key"] != cand_key]
    pick = int.from_bytes(hashlib.sha256(
        f"{agent_task_id}|shuffle".encode()).digest()[:4], "big") % len(others)
    alt_key = others[pick]
    schema = next(s for s in ALLOWED_CONFIG_SCHEMA if s["key"] == alt_key)
    base = baseline_config
    for part in alt_key.split("."):
        base = base.get(part) if isinstance(base, dict) else None
    direction = "increase"
    if isinstance(base, (int, float)) and not isinstance(base, bool):
        if ((schema["range"] in ("(0, 1]", "(0, 1)")
             and float(base) * 1.25 >= 1.0) or
                (schema["range"] == "(0, 1)" and float(base) >= 0.9)):
            direction = "decrease"
    alt_value = _perturb(key=alt_key, baseline_config=baseline_config,
                         direction=direction)
    return dc_replace(
        shuffled,
        diagnosis_id="dg_shuffle_distinct_" + sha256_json(
            {"own": agent_task_id, "key": alt_key})[:12],
        proposed_intervention={"key": alt_key, "value": alt_value,
                               "direction": direction,
                               "mechanism": "shuffled control (distinct knob)"},
        shuffled_from=(shuffled.shuffled_from or "synthetic_shuffle")
        + "|distinct_knob", diagnosis_sha256="")


def _infra_failures(result: dict[str, Any], arm: str) -> list[dict[str, Any]]:
    error = str(result.get("error") or "")
    if result.get("ok") and not error:
        return []
    if not any(token in error for token in INFRA_ERROR_TOKENS):
        if result.get("ok"):
            return []
    return [{"arm": arm, "error": error[:500],
             "error_type": next((t for t in INFRA_ERROR_TOKENS if t in error),
                                "research_failure" if result.get("ok") else "unknown")}]


def _register_task_bundle(*, task: PublicTask, arm: ArmSpec,
                          run_root: Path) -> dict[str, Any]:
    from ..adapters import SciNetAdapter
    from ..campaign.campaign_store import CampaignStore
    from ..campaign.scinet import build_scinet_campaign_runner, scinet_campaign_policy
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
        paired_seeds=seeds, max_seeds=max(1, len(seeds)),
        max_campaign_rounds=arm.harness.campaign_rounds,
        intervention_policy={k: dict(v) for k, v in arm.intervention_policy.items()},
        epochs=int(task.budget.get("epochs", 1)),
        max_jobs=int(task.budget.get("max_jobs", 8)),
        max_total_seconds=float(task.budget.get("max_total_seconds", 900.0)),
        evaluation_split="validation")
    campaign_store = CampaignStore(data_dir / "campaigns.sqlite")
    report_store = ComparisonReportStore(project / policy.campaign_root)
    config_builder = CandidateConfigBuilder(
        project, allowed_keys=policy.allowed_config_keys,
        value_rules=policy.value_rules, campaign_root=policy.campaign_root)
    campaign_policy = scinet_campaign_policy(
        epochs=policy.epochs, timeout_per_job_s=policy.job_timeout_s,
        max_jobs=policy.max_jobs, max_candidates=arm.harness.max_candidates,
        max_seeds=len(seeds))

    def runner_factory():
        return build_scinet_campaign_runner(
            store=campaign_store, config_builder=config_builder,
            policy=campaign_policy,
            adapter=SciNetAdapter(cli_timeout=CLI_TIMEOUT_S))

    def comparator_factory():
        return build_scinet_comparator(
            campaign_store=campaign_store, report_store=report_store,
            project_dir=project)

    register_domain(DomainBundle(
        name="scinet", policy_factory=lambda **_: policy,
        campaign_store=campaign_store, report_store=report_store,
        campaign_runner_factory=runner_factory,
        comparator_factory=comparator_factory,
        dataset_verifier=lambda cfg, v: config_builder.verify_dataset(cfg, v),
        campaign_root=str(project / policy.campaign_root),
        adapter_factory=lambda: SciNetAdapter(cli_timeout=CLI_TIMEOUT_S),
        description=f"{EXPERIMENT_ID} task-scoped bundle"))
    return {"dataset_version": policy.dataset_version,
            "baseline_config_path": policy.baseline_config_path,
            "intervention_policy": policy.intervention_policy,
            "paired_seeds": list(policy.paired_seeds),
            "cli_timeout_s": CLI_TIMEOUT_S,
            "timeout_s": policy.max_total_seconds}


def build_research_task(task: PublicTask, arm: ArmSpec, run_id: str) -> ResearchTask:
    safe_id = task.task_id.replace("::", "__")
    return ResearchTask(
        task_id=f"{safe_id}__{arm.arm}__{run_id}", domain="scinet",
        objective=task.objective, dataset_refs=[task.dataset_reference],
        budget={k: v for k, v in task.budget.items() if k != "paired_seeds"},
        evaluation_protocol={"primary_metric": task.primary_metric,
                             "metric_direction": task.metric_direction,
                             "decision_split": "validation",
                             "reserved_final_split": "test"},
        heldout_policy={"independent_holdout": True, "holdout_split": "test",
                        "ood_task": task.task_family == "scientific_ml",
                        "paired_seeds": list(task.budget.get("paired_seeds", [task.seed]))})


def _read_evidence(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def run_arm(*, task: PublicTask, arm: ArmSpec, run_root: Path, run_id: str,
            baseline_config: dict[str, Any],
            resume: bool = False) -> dict[str, Any]:
    """Execute one arm through the real runtime, or resume recorded evidence."""
    started = time.time()
    arm_root = run_root / arm.arm
    arm_root.mkdir(parents=True, exist_ok=True)
    trajectory_path = arm_root / "trajectory.json"
    result_files = sorted(arm_root.glob("*/task_result.json"))
    if resume and trajectory_path.is_file() and result_files:
        traj = json.loads(trajectory_path.read_text(encoding="utf-8"))
        result = json.loads(result_files[-1].read_text(encoding="utf-8"))
        record = traj.get("intervention_record") or {}
        bad_status = record.get("status") == "NOT_APPLIED"
        if result.get("ok") and not traj.get("infrastructure_failures") \
                and not bad_status:
            traj["reused_from_existing_evidence"] = True
            return {"result": result, "trajectory": traj}
        # A failed/incomplete/unapplied arm is re-executed explicitly with a
        # NEW run-scoped ID so no SciNet job ID collides with the earlier run.
        traj["retry_after_failure"] = True
        for old in result_files:
            old_dir = old.parent
            archived = old_dir.parent / f"failed_attempt_{uuid.uuid4().hex[:6]}_{old_dir.name}"
            old_dir.rename(archived)
            traj.setdefault("archived_failed_attempts", []).append(str(archived))
        exec_run_id = f"{run_id}_r{uuid.uuid4().hex[:4]}"
    else:
        exec_run_id = run_id

    frozen = _register_task_bundle(task=task, arm=arm, run_root=arm_root)
    research_task = build_research_task(task, arm, exec_run_id)
    result_obj = run_research_task(
        research_task, harness=arm.harness, run_root=arm_root,
        approved_by="arr-rsi-01-p0.1-operator")
    result = result_obj.model_dump(mode="json")
    reported_root = Path((result.get("provenance") or {}).get("run_root") or arm_root)
    records = _read_evidence(reported_root / "store" / "evidence.jsonl")
    campaign_id = str((result.get("campaign") or {}).get("campaign_id") or "")
    applied_config, applied_path = _find_applied_config(
        campaign_id=campaign_id, research_task_id=research_task.task_id,
        root=reported_root, arm=arm.arm)
    record = finalize_intervention_record(
        record=arm.intervention_record, applied_config=applied_config,
        applied_config_path=applied_path)
    infra = _infra_failures(result, arm.arm)
    trajectory = {
        "experiment_id": EXPERIMENT_ID, "run_id": exec_run_id,
        "task_id": task.task_id, "agent_task_id": task.agent_task_id,
        "task_view": task.agent_view(), "arm": arm.arm,
        "research_task_sha256": research_task.fingerprint(),
        "harness": arm.harness.to_dict(), "harness_sha256": arm.harness.fingerprint(),
        "frozen": frozen, "baseline_config": baseline_config,
        "baseline_config_sha256": sha256_json(baseline_config),
        "intervention_record": record,
        "applied_config_path": applied_path,
        "infrastructure_failures": infra,
        "concurrency": 1, "timeout_s": float(task.budget.get(
            "max_total_seconds", 900.0)),
        "tool_calls": [{"event_type": e.get("event_type"), "at": e.get("at"),
                        "payload": e.get("payload")} for e in records],
        "experiment_results": [(e.get("payload") or {}).get("result") for e in records
                               if e.get("event_type") == "result"],
        "evidence_refs": {"baseline_id": (result.get("evidence") or {}).get(
            "baseline_id", ""), "report_path": (result.get("evidence") or {}).get(
                "report_path", ""), "applied_config_path": applied_path,
            "n_evidence_records": len(records)},
        "budget_usage": result.get("telemetry", {}),
        "final_conclusion": {"outcome": result.get("outcome", ""),
                             "quality": result.get("quality", {}),
                             "comparisons": result.get("comparisons", []),
                             "holdout": (result.get("evidence") or {}).get("holdout", {}),
                             "ood": (result.get("evidence") or {}).get("ood", {})},
        "ok": result.get("ok", False), "error": result.get("error", ""),
        "artifact_sha256": result.get("artifact_sha256", ""),
        "elapsed_seconds": round(time.time() - started, 3),
        "reused_from_existing_evidence": False,
    }
    assert_no_ground_truth(trajectory)
    trajectory_path.write_text(json.dumps(
        trajectory, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8")
    return {"result": result, "trajectory": trajectory}


def _dummy_diagnosis(task_id: str) -> DiagnosisOutput:
    return DiagnosisOutput(
        task_id=task_id, failure_modes=[], root_cause="", evidence_refs=[],
        confidence=0.0, alternative_explanations=[], improvement_hypothesis="",
        proposed_intervention=None, expected_metric={},
        falsification_condition="", diagnosis_id="", insufficient_evidence=True,
        evidence_basis={})


def run_task(*, task: PublicTask, truth: Any, run_root: Path, run_id: str,
             prereg: Preregistration, baseline_config: dict[str, Any],
             shuffled_pool: dict[str, DiagnosisOutput] | None = None,
             resume: bool = False, verbose: bool = False) -> dict[str, Any]:
    task_root = run_root / "tasks" / task.task_id.replace("::", "__")
    task_root.mkdir(parents=True, exist_ok=True)
    base_arms = derive_arms(task=task.agent_view(), diagnosis=_dummy_diagnosis(
        task.agent_task_id), baseline_config=baseline_config)
    if verbose:
        print(f"[{task.task_id}] base", flush=True)
    executed = {"base": run_arm(task=task, arm=base_arms["base"],
                                run_root=task_root, run_id=run_id,
                                baseline_config=baseline_config, resume=resume)}
    base_traj = executed["base"]["trajectory"]
    diagnoser = EvidenceGroundedDiagnoser()
    diagnosis = diagnoser.diagnose(
        task=task.agent_view(), trajectory=base_traj,
        experiment_results=base_traj["experiment_results"],
        evidence=base_traj["evidence_refs"], budget=task.budget)
    diagnoses = {"base": diagnosis.to_dict()}
    shuffled = None
    try:
        shuffled = ShuffledDiagnoser(pool=shuffled_pool or {}).diagnose(
            task=task.agent_view(), trajectory=base_traj,
            experiment_results=base_traj["experiment_results"],
            evidence=base_traj["evidence_refs"], budget=task.budget)
        diagnoses["shuffled"] = shuffled.to_dict()
    except ValueError:
        pass
    shuffled = _ensure_distinct_shuffle(
        shuffled=shuffled, diagnosis=diagnosis,
        baseline_config=baseline_config,
        agent_task_id=task.agent_task_id)
    if shuffled:
        diagnoses["shuffled"] = shuffled.to_dict()
    arms = derive_arms(task=task.agent_view(), diagnosis=diagnosis,
                       shuffled=shuffled, baseline_config=baseline_config)
    for name in ("candidate", "placebo", "shuffled"):
        if verbose:
            print(f"[{task.task_id}] {name}", flush=True)
        executed[name] = run_arm(task=task, arm=arms[name], run_root=task_root,
                                 run_id=run_id, baseline_config=baseline_config,
                                 resume=resume)
    arm_results = {k: v["result"] for k, v in executed.items()}
    # Bind the VERIFIED applied-config record into the arm spec so evaluation
    # sees APPLIED / NOT_APPLIED, never the pre-execution PENDING placeholder.
    for name, arm in arms.items():
        arms[name] = dc_replace(
            arm, intervention_record=executed[name]["trajectory"][
                "intervention_record"])
    evaluation = evaluate_task(
        task=task, truth=truth, arms=arms, arm_results=arm_results,
        diagnoses={task.task_id: diagnoses},
        trajectories={k: v["trajectory"] for k, v in executed.items()},
        prereg=prereg)
    for name, payload in (("diagnosis.json", diagnoses),
                          ("intervention.json", {k: v.to_dict() for k, v in arms.items()})):
        (task_root / name).write_text(json.dumps(
            payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return {"task": task.to_dict(), "diagnoses": diagnoses,
            "arms": {k: v.to_dict() for k, v in arms.items()},
            "arm_results": arm_results,
            "trajectories": {k: v["trajectory"] for k, v in executed.items()},
            "evaluation": evaluation, "task_root": str(task_root)}


def freeze_protocol(*, root: Path, tasks: list[PublicTask], prereg: Preregistration,
                    truth_sha: str, run_id: str, stage: str) -> dict[str, Any]:
    protocol = {
        "experiment_id": EXPERIMENT_ID, "stage": stage, "run_id": run_id,
        "preregistration": prereg.to_dict(),
        "frozen": {"model_version": "frozen-by-SciNet-config; no weight change",
                   "runtime_version": _git_head(Path(__file__).resolve().parents[2]),
                   "engine_version": _git_head(DEFAULT_SCINET_DIR),
                   "dataset": sorted({t.dataset_reference for t in tasks}),
                   "seeds": sorted({s for t in tasks for s in t.budget["paired_seeds"]}),
                   "metrics": sorted({t.primary_metric for t in tasks}),
                   "research_budget": tasks[0].budget,
                   "tool_budget": {"engine": "research_api.cli",
                                   "allowed_stages": ["train", "evaluate"],
                                   "max_jobs_per_arm": tasks[0].budget["max_jobs"]},
                   "concurrency": 1,
                   "independence_unit": "family|frozen_dataset; seeds are replicates",
                   "ground_truth_sha256": truth_sha},
        "artifact_contract": list(ARTIFACTS),
    }
    protocol["protocol_sha256"] = sha256_json(protocol)
    (root / "p0_1_protocol.json").write_text(json.dumps(
        protocol, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return protocol


def write_report(*, root: Path, verdict: dict[str, Any], statistics: dict[str, Any],
                 evaluations: list[dict[str, Any]], protocol: dict[str, Any],
                 stage: str) -> Path:
    lines = [f"# {EXPERIMENT_ID} — {stage} Report", "",
             "## Verdict", "", f"**{verdict['verdict']}**", ""]
    lines += [f"- {r}" for r in verdict.get("reasons", [])] or ["- (none)"]
    lines += ["", "## Dimension means", "", "| Dimension | Mean |", "|---|---|"]
    for k, v in (verdict.get("dimension_means") or {}).items():
        lines.append(f"| {k} | {v if v is not None else 'n/a'} |")
    lines += ["", "## Gates", ""]
    for k, v in (verdict.get("gates") or {}).items():
        lines.append(f"- `{k}`: {v}")
    lines += ["", "## Per-task integrity", "",
              "| task | cause correct | intervention valid | applied | infra | complete-case |",
              "|---|---|---|---|---|---|"]
    for e in evaluations:
        sc = e.get("scientific_correctness") or {}
        iv = e.get("intervention_validity") or {}
        lines.append(
            f"| {e['task_id']} | {sc.get('root_cause_correct')} | "
            f"{iv.get('valid')} | {iv.get('applied_status')} | "
            f"{len(e.get('infrastructure', {}).get('per_arm', []))} | "
            f"{e.get('complete_case')} |")
    lines += ["", "## Scope", "",
              "Original P0 verdict remains BLOCKED and is untouched. This report "
              "covers only the P0.1 integrity repair. " + verdict.get("note", ""),
              "", f"Protocol SHA: `{protocol['protocol_sha256']}`",
              f"Statistical SHA: `{statistics.get('statistical_analysis_sha256', '')}`", ""]
    path = root / "ARR_RSI_01_P0_1_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_pilot(*, n_tasks: int = 1, run_root: str | Path = "ARR_RSI_01_P0_1_RUN",
              scinet_dir: str | Path | None = None, verbose: bool = False,
              concurrency: int = 1, resume: bool = False,
              stage: str = "p0_1_run") -> dict[str, Any]:
    """Run the P0.1 experiment with STRICT concurrency=1."""
    if concurrency != 1:
        raise ValueError("P0.1 requires concurrency=1; got "
                         f"{concurrency}. Parallel execution is out of scope.")
    root = Path(run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:8]
    tasks = build_tasks(scinet_dir or DEFAULT_SCINET_DIR)
    tasks = tasks[:max(1, min(int(n_tasks), len(tasks)))]
    configs = {}
    for t in {t.engine_config for t in tasks}:
        configs[t] = _read_config(Path(DEFAULT_SCINET_DIR) / t)
    truth_all = build_ground_truth(tasks)
    from .tasks import GroundTruthStore
    store = GroundTruthStore(root / "ground_truth.sealed.json")
    protocol_path = root / "p0_1_protocol.json"
    if resume and protocol_path.is_file():
        # Resume MUST NOT re-freeze the protocol: MME, seeds, tasks and GT SHA
        # stay exactly as recorded before the failed arm.
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        prereg_raw = dict(protocol["preregistration"])
        prereg_raw.pop("preregistration_sha256", None)
        prereg = Preregistration(**prereg_raw)
        run_id = str(protocol["run_id"])
        truth_sha = str(protocol["frozen"]["ground_truth_sha256"])
    else:
        seal = store.seal(truth_all)
        truth_sha = seal["ground_truth_sha256"]
        prereg = Preregistration(freeze={
            "experiment_id": EXPERIMENT_ID, "run_id": run_id,
            "stage": stage, "tasks": [t.task_id for t in tasks],
            "truth_sha256": truth_sha, "arms": list(ARMS),
            "concurrency": 1,
            "independence_units": sorted({t.independence_unit for t in tasks})})
        protocol = freeze_protocol(root=root, tasks=tasks, prereg=prereg,
                                   truth_sha=truth_sha, run_id=run_id,
                                   stage=stage)
    (root / "task_manifest.json").write_text(json.dumps(
        task_manifest(tasks, ground_truth_sha256=truth_sha,
                      experiment_id=EXPERIMENT_ID),
        indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")

    shuffled_pool: dict[str, DiagnosisOutput] = {}
    executions = []
    for task in tasks:
        result = run_task(
            task=task, truth=truth_all[task.task_id], run_root=root,
            run_id=run_id, prereg=prereg,
            baseline_config=configs.get(task.engine_config, {}),
            shuffled_pool=shuffled_pool or None, resume=resume, verbose=verbose)
        executions.append(result)
        base_d = result["diagnoses"].get("base")
        if base_d:
            shuffled_pool[task.task_id] = DiagnosisOutput(
                **{k: v for k, v in base_d.items()
                   if k in DiagnosisOutput.__dataclass_fields__})

    truth = store.load(expected_sha256=truth_sha)
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
        "results.json": {"experiment_id": EXPERIMENT_ID, "stage": stage,
                         "n_tasks": len(executions), "tasks": [e["task"] for e in executions],
                         "evaluations": evaluations, "arm_results": arm_results_all,
                         "verdict": verdict},
        "statistical_analysis.json": statistics,
    }
    for name, payload in payloads.items():
        (root / name).write_text(json.dumps(
            payload, indent=2, sort_keys=True, ensure_ascii=False, default=str),
            encoding="utf-8")
    write_report(root=root, verdict=verdict, statistics=statistics,
                 evaluations=evaluations, protocol=protocol, stage=stage)
    return {"experiment_id": EXPERIMENT_ID, "stage": stage, "run_root": str(root),
            "run_id": run_id, "n_tasks": len(executions),
            "verdict": verdict["verdict"],
            "artifacts": [str(root / n) for n in ARTIFACTS]}


def finalize_run(*, run_root: str | Path,
                 scinet_dir: str | Path | None = None) -> dict[str, Any]:
    """Recompute evaluation/artifacts from recorded arms. No engine re-run."""
    root = Path(run_root).resolve()
    protocol = json.loads((root / "p0_1_protocol.json").read_text(encoding="utf-8"))
    prereg_raw = dict(protocol["preregistration"])
    prereg_raw.pop("preregistration_sha256", None)
    prereg = Preregistration(**prereg_raw)
    task_ids = list(prereg.freeze["tasks"])
    all_tasks = {t.task_id: t for t in build_tasks(scinet_dir or DEFAULT_SCINET_DIR)}
    from .tasks import GroundTruthStore
    truth = GroundTruthStore(root / "ground_truth.sealed.json").load(
        protocol["frozen"]["ground_truth_sha256"])
    configs = {}
    for task_id in task_ids:
        cfg_name = all_tasks[task_id].engine_config
        configs[task_id] = _read_config(Path(DEFAULT_SCINET_DIR) / cfg_name)
    executions = []
    for task_id in task_ids:
        task = all_tasks[task_id]
        task_root = root / "tasks" / task_id.replace("::", "__")
        trajectories = {a: json.loads((task_root / a / "trajectory.json").read_text(
            encoding="utf-8")) for a in ARMS}
        arm_results = {}
        for a in ARMS:
            files = sorted((task_root / a).glob("*/task_result.json"))
            if not files:
                raise FileNotFoundError(f"missing real arm result {task_id}/{a}")
            arm_results[a] = json.loads(files[-1].read_text(encoding="utf-8"))
        base_traj = trajectories["base"]
        diagnosis = EvidenceGroundedDiagnoser().diagnose(
            task=task.agent_view(), trajectory=base_traj,
            experiment_results=base_traj.get("experiment_results", []),
            evidence=base_traj.get("evidence_refs", {}), budget=task.budget)
        diagnoses = {"base": diagnosis.to_dict()}
        shuffled = None
        try:
            shuffled = ShuffledDiagnoser(
                pool={tid: diagnosis for tid in task_ids}).diagnose(
                    task=task.agent_view(), trajectory=base_traj,
                    experiment_results=base_traj.get("experiment_results", []),
                    evidence=base_traj.get("evidence_refs", {}), budget=task.budget)
            diagnoses["shuffled"] = shuffled.to_dict()
        except ValueError:
            pass
        shuffled = _ensure_distinct_shuffle(
            shuffled=shuffled, diagnosis=diagnosis,
            baseline_config=configs[task_id], agent_task_id=task.agent_task_id)
        if shuffled:
            diagnoses["shuffled"] = shuffled.to_dict()
        arms = derive_arms(task=task.agent_view(), diagnosis=diagnosis,
                           shuffled=shuffled,
                           baseline_config=configs[task_id])
        for arm_name, arm_obj in arms.items():
            arms[arm_name] = dc_replace(
                arm_obj, intervention_record=trajectories[arm_name][
                    "intervention_record"])
        evaluation = evaluate_task(
            task=task, truth=truth[task_id], arms=arms, arm_results=arm_results,
            diagnoses={task_id: diagnoses}, trajectories=trajectories,
            prereg=prereg)
        executions.append({"task": task.to_dict(), "diagnoses": diagnoses,
                           "arms": {k: v.to_dict() for k, v in arms.items()},
                           "arm_results": arm_results, "trajectories": trajectories,
                           "evaluation": evaluation})
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
        "results.json": {"experiment_id": EXPERIMENT_ID,
                         "stage": prereg.freeze.get("stage", "finalize"),
                         "n_tasks": len(executions), "tasks": [e["task"] for e in executions],
                         "evaluations": evaluations, "arm_results": arm_results_all,
                         "verdict": verdict},
        "statistical_analysis.json": statistics,
    }
    for name, payload in payloads.items():
        (root / name).write_text(json.dumps(
            payload, indent=2, sort_keys=True, ensure_ascii=False, default=str),
            encoding="utf-8")
    write_report(root=root, verdict=verdict, statistics=statistics,
                 evaluations=evaluations, protocol=protocol,
                 stage=prereg.freeze.get("stage", "finalize"))
    return {"experiment_id": EXPERIMENT_ID, "run_root": str(root),
            "n_tasks": len(executions), "verdict": verdict["verdict"]}


__all__ = ["ARTIFACTS", "run_arm", "run_task", "run_pilot", "finalize_run",
           "build_research_task", "freeze_protocol"]
