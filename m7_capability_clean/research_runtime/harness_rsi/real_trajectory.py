"""M6.1 -- REAL Agent+Engine trajectory evaluation.

Closes the M6 P0 where a rule-level probe runner and a separately-running
engine were stitched into one report. Here Base and Candidate run through the
SAME production ``run_research_task`` (planner -> REAL HypothesisCritic ->
plan -> human approval boundary -> REAL campaign engine -> report -> quality).
The ONLY difference is the injected Critic manager factory, derived from the
frozen Harness component config.

Every arm records a trajectory whose identity binds:
  harness/candidate/component SHAs, prompt/tool/runtime SHAs, real engine
  result SHA, critic decisions with reasons, hypothesis IDs and campaign ID.
A fabricated engine_Q or a mismatched SHA fails closed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from ..arr_runtime import run_research_task
from ..research_task import ResearchTask
from .critic_strategies import (
    CRITIC_VERSION, build_critic_manager_factory, strategy_from_bundle,
)
from .harness import HarnessBundle
from .identity import freeze_json, sha256_json

AGENT_MODEL = "deterministic-planner-v1"
TRAJECTORY_VERSION = "m6_1_trajectory_v1"


class TrajectoryError(RuntimeError):
    """Raised when a real trajectory cannot be established."""


def _evidence_resolver_from_store(data_dir: Path) -> Callable[[str], Any]:
    """Resolver over the SAME evidence.jsonl that run_research_task writes.

    run_research_task derives ``data_dir = run_root/<domain>_<task>_<fp>/store``;
    the caller must pass that exact path (not the arm root).
    """
    """Resolve experiment ids to REAL recorded results (same as service)."""
    from ..research_service import default_evidence_provider
    from ..evidence_store import EvidenceStore

    store = EvidenceStore(data_dir / "evidence.jsonl")

    def resolver(experiment_id: str):
        try:
            result, _ = default_evidence_provider(experiment_id, store)
            return result
        except Exception:
            return None

    return resolver


def _task_for(benchmark_task: Any, *, domain: str, seed: int) -> ResearchTask:
    """Build a REAL ResearchTask from a benchmark task's frozen source."""
    src = dict(getattr(benchmark_task, "source", {}) or {})
    objective = str(getattr(benchmark_task, "objective", "") or src.get(
        "research_objective", "") or benchmark_task.task_id)
    primary = getattr(benchmark_task, "primary_metric", "") or src.get(
        "primary_metric", "")
    # The production ResearchTask reserves "test" as the FINAL split and uses
    # "validation" for decisions. A V2 hidden-holdout task whose source split
    # is "test"/"ood" must map to validation for the decision phase; its
    # hidden status is enforced by HoldoutAccessVerifier + family isolation,
    # never by illegally reusing the frozen final split as a decision split.
    source_split = str(src.get("split", "validation"))
    decision_split = "validation" if source_split in ("test", "ood", "final") \
        else source_split
    return ResearchTask(
        task_id=f"{getattr(benchmark_task, 'task_id', 't')}@s{seed}",
        domain=domain,
        objective=objective,
        objective_id=f"obj_{getattr(benchmark_task, 'family_id', 'fam')}",
        constraints={"benchmark_task_sha": getattr(
            benchmark_task, "task_sha", ""),
            "family_id": getattr(benchmark_task, "family_id", ""),
            "task_config": dict(getattr(benchmark_task, "task_config", {}) or {})},
        dataset_refs=[src.get("dataset", "")] if src.get("dataset") else [],
        budget={"experiments": 4, "max_jobs": 8, "max_total_seconds": 900.0,
                "epochs": 1},
        evaluation_protocol={
            "primary_metric": primary,
            "metric_direction": getattr(benchmark_task, "direction", "maximize"),
            "decision_split": decision_split,
            "reserved_final_split": "test",
        },
        heldout_policy={"independent_holdout": True, "holdout_split": "test",
                        "ood_task": False, "paired_seeds": [int(seed)]},
    )


def run_real_trajectory(*, benchmark_task: Any, harness: HarnessBundle,
                        arm: str, out_dir: str | Path,
                        component_config: dict[str, Any],
                        seed: int | None = None,
                        domain_override: str = "",
                        harness_loader: Any = None) -> dict[str, Any]:
    """Execute ONE real arm; return a SHA-bound trajectory + research result."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    domain = domain_override or str(getattr(
        benchmark_task, "domain", "scinet"))
    task_seed = int(seed if seed is not None else getattr(
        benchmark_task, "seed", 42))
    task = _task_for(benchmark_task, domain=domain, seed=task_seed)
    strategy = strategy_from_bundle(component_config)
    arm_root = root / f"{arm}_{getattr(benchmark_task, 'task_id', 'task')}_s{task_seed}"
    arm_root.mkdir(parents=True, exist_ok=True)
    data_dir = arm_root / "store"
    data_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    # Build the SAME real proposer the production service uses (harness-
    # controlled deterministic planner over the domain's real baseline config),
    # so Base vs Candidate differ ONLY in the Critic policy.
    from ..arr_runtime import make_harness_generate
    from ..hypothesis_manager import LLMProposer
    from ..research_environment import build_local_bundle, get_domain, register_domain
    import yaml
    from ..domain_registry import default_registry
    from ..settings import DEFAULT_SCINET_DIR, RUNTIME_DIR
    from ..research_environment import ensure_domain_registered
    if not default_registry.contains(domain):
        if domain == "scinet":
            # SciNet uses an external engine adapter; it is registered via the
            # same helper the production run_research_task uses (no local
            # bundle builder exists for it).
            ensure_domain_registered(domain)
        else:
            register_domain(build_local_bundle(
                domain, data_dir=data_dir))
    policy = get_domain(domain).policy_factory()
    baseline_cfg: dict[str, Any] = {}
    try:
        cfg_path = Path(policy.baseline_config_path)
        if not cfg_path.is_absolute():
            project = (Path(DEFAULT_SCINET_DIR) if domain == "scinet"
                       else RUNTIME_DIR / "domains" / "projects" / domain)
            cfg_path = project / cfg_path
        if cfg_path.is_file():
            baseline_cfg = yaml.safe_load(
                cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        baseline_cfg = {}
    planner_harness = __import__(
        "research_runtime.arr_runtime", fromlist=["BASE_HARNESS"]
    ).BASE_HARNESS
    generate = make_harness_generate(planner_harness, policy, None,
                                     baseline_config=baseline_cfg)
    proposer = LLMProposer(
        generate=generate,
        allowed_intervention_keys=list(policy.allowed_config_keys))
    # run_research_task places its store at
    #   run_root / "<domain>_<task_id>_<harness.fp8>" / "store"
    # with harness=None -> BASE_HARNESS. Bind the resolver to that exact file.
    from ..arr_runtime import BASE_HARNESS
    svc_data_dir = (arm_root /
                    f"{domain}_{task.task_id}_{BASE_HARNESS.fingerprint()[:8]}"
                    / "store")
    factory = build_critic_manager_factory(
        strategy=strategy,
        evidence_resolver=_evidence_resolver_from_store(svc_data_dir),
        config=dict(component_config), proposer=proposer)
    result = run_research_task(
        task, harness=None, run_root=str(arm_root),
        approved_by=f"m61_smoke_{arm}", experiences=None,
        hypothesis_manager_factory=factory,
        harness_loader=harness_loader)
    elapsed = time.time() - started

    # The research harness for the production planner is the M6 base harness
    # (frozen, outside candidate control); the ONLY candidate-controlled change
    # is the Critic strategy above. Record that provenance explicitly.
    # Join the REAL Critic verdicts (passed/errors, from the production
    # ResearchService review) with the REAL generated hypothesis content.
    # Both come from this run's recorded output -- neither is a probe fixture.
    hypotheses_by_id = {h.get("hypothesis_id", ""): h
                        for h in (result.hypotheses or [])}
    critic_reviews: list[dict[str, Any]] = []
    for review in (result.provenance.get("critic_reviews") or []):
        hid = str(review.get("hypothesis_id", ""))
        h = hypotheses_by_id.get(hid, {})
        critic_reviews.append({
            "hypothesis_id": hid,
            "passed": bool(review.get("passed")),
            "errors": list(review.get("errors") or []),
            "predicted_effects": h.get("predicted_effects", []),
            "falsification_criteria": h.get("falsification_criteria", []),
            "hypothesis_sha256": review.get("hypothesis_sha256", ""),
        })
    campaign = result.campaign or {}
    campaign_id = str(campaign.get("campaign_id", "") or "")
    if not campaign_id:
        # No real campaign -> infrastructure failure, NOT a fake success.
        return {"ok": False, "arm": arm, "strategy": strategy,
                "failure_kind": "infrastructure",
                "error": result.error or "no_real_campaign",
                "task_sha": getattr(benchmark_task, "task_sha", "")}

    engine_result = freeze_json({
        "arm": arm, "campaign_id": campaign_id,
        "domain": domain, "task_id": task.task_id,
        "outcome": result.outcome, "report": result.report,
        "quality": result.quality, "telemetry": result.telemetry,
        "comparisons": result.comparisons,
    }, arm_root / "real_engine_result.json")

    trajectory = {
        "trajectory_version": TRAJECTORY_VERSION,
        "trajectory_id": sha256_json({
            "arm": arm, "task_id": task.task_id, "campaign_id": campaign_id,
            "harness_sha": harness.sha256}),
        "task_id": task.task_id,
        "task_sha256": getattr(benchmark_task, "task_sha", ""),
        "family_id": getattr(benchmark_task, "family_id", ""),
        "domain": domain, "campaign_id": campaign_id,
        "round_id": "m61_arm", "agent_id": "production_research_service",
        "agent_model": AGENT_MODEL, "agent_model_version": "v1",
        # M6.1-fix-C: when a formal loader is wired, the authority is the REAL
        # SHA pinned into the campaign by the Dispatcher (read back from the
        # persisted campaign record), NOT the caller's bundle object.
        "harness_sha": (str(result.provenance.get("harness_sha256", ""))
                        if harness_loader is not None
                        and result.provenance.get("harness_sha256")
                        else harness.sha256),
        "harness_sha_source": ("formal_dispatcher"
                               if harness_loader is not None
                               and result.provenance.get("harness_sha256")
                               else "caller_bundle"),
        "harness_activation_id": str(
            result.provenance.get("harness_activation_id", "")),
        "component_name": "hypothesis_critic",
        "component_sha": harness.component("hypothesis_critic").sha256,
        "parent_component_sha": "",
        "critic_strategy": strategy,
        "critic_version": CRITIC_VERSION,
        "prompt_sha": sha256_json({"harness": harness.sha256,
                                    "objective": task.objective}),
        "toolset_sha": sha256_json(sorted(["research_api", "local_engine",
                                            "evidence_store"])),
        "runtime_sha": sha256_json({"protocol": "arr-1.0",
                                     "trajectory_version": TRAJECTORY_VERSION}),
        "engine_result_sha": engine_result["sha256"],
        "input_sha": task.fingerprint(),
        "output_sha": sha256_json(critic_reviews),
        "decision": result.outcome,
        "decision_reason": str(result.report.get("rationale", "")) if isinstance(
            result.report, dict) else "",
        "hypothesis_ids": [h.get("hypothesis_id", "")
                           for h in (result.hypotheses or [])],
        "critic_reviews": critic_reviews,
        "evidence_ids": [e.get("experiment_id", "") for e in
                         (result.evidence.get("experiments", [])
                          if isinstance(result.evidence, dict) else [])],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S",
                                    time.localtime(started)),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_s": round(elapsed, 3),
        "token_usage": int((result.telemetry or {}).get("tokens", 0)),
        "budget_consumed": dict((result.telemetry or {}).get("budget", {}) or {}),
        "result_status": "succeeded" if result.ok else "failed",
        "ok": bool(result.ok),
    }
    frozen = freeze_json(trajectory, arm_root / "trajectory.json")
    return {"ok": bool(result.ok), "arm": arm, "strategy": strategy,
            "trajectory": trajectory, "trajectory_sha": frozen["sha256"],
            "trajectory_path": frozen["path"],
            "engine_result_sha": engine_result["sha256"],
            "engine_result_path": engine_result["path"],
            "campaign_id": campaign_id,
            "decision": result.outcome,
            "quality": result.quality,
            "telemetry": result.telemetry,
            "failure_kind": "" if result.ok else "research"}


__all__ = ["TRAJECTORY_VERSION", "AGENT_MODEL", "TrajectoryError",
           "run_real_trajectory"]
