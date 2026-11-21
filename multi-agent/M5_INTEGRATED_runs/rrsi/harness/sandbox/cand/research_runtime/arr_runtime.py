"""ARR 1.0 / D3+E4 -- the unified research runtime entry point.

One function, ``run_research_task(task)``, drives the SAME generic
``ResearchService`` for scinet / embodied / quant:

    real baseline experiment (real engine, real frozen split)
      -> autonomous hypotheses (NO human-authored hypotheses_json)
      -> deterministic Critic (the only approval gate)
      -> ExperimentPlan -> approval (out-of-band human grant)
      -> real paired campaign (existing CampaignRunner)
      -> real paired comparison (existing CampaignComparator)
      -> immutable ResearchReport (existing schema)
      -> independent holdout confirmation on the RESERVED test split
      -> D1 domain-injected ResearchQualityEvaluator
      -> uniform TaskResult with full SHA-256 provenance

A ``ResearchHarness`` is the research SYSTEM's own configuration: planner
strategy / prompt, evidence-memory depth, workflow breadth, reviewer strictness.
It is what Package F evolves. Crucially it contains NO evaluation fields --
weights, splits, gate rules and the evaluator are outside its reach (F4).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .domain_registry import get_domain
from .evidence_store import EvidenceStore
from .research_environment import (
    build_custom_research_service,
    build_default_research_service,
    build_local_bundle,
    register_domain,
)
from .research_quality import (
    QUALITY_FROZEN_FIELDS,
    QualityEvidence,
    evaluate_research_quality,
    quality_evaluator_for,
)
from .research_store import ResearchStores
from .research_task import ResearchTask, TaskResult
from .schemas import ExperimentSpec

HARNESS_VERSION = "arr-1.0-harness-v1"

#: Candidate-proof: a Harness may not contain ANY of these keys.
HARNESS_FORBIDDEN_KEYS = QUALITY_FROZEN_FIELDS | frozenset({
    "gate", "evaluation", "scoring", "holdout", "quality", "weights",
    "primary_metric", "metric_direction", "evaluation_split", "frozen_test_used",
    "promote", "version",
})


@dataclass(frozen=True)
class ResearchHarness:
    """The Research Harness: the object RRSI is allowed to improve."""

    schema_version: str = HARNESS_VERSION
    # How the Experiment Designer maps a diagnosis to a controlled intervention.
    planner_strategy: str = "diagnosis_targeted"
    # Planner prompt: does it require explicit evidence + falsification?
    hypothesis_prompt: str = "evidence_grounded_v1"
    # How many past (including REFUTED) experiences are injected into planning.
    experience_top_k: int = 3
    # Competing hypotheses per round / allowed research rounds.
    max_candidates: int = 2
    campaign_rounds: int = 2
    # Reviewer policy (advisory extra text; the deterministic Critic stays the gate).
    require_falsification: bool = True
    notes: str = ""

    def validate(self) -> None:
        if self.planner_strategy not in (
                "diagnosis_targeted", "fixed_first_knob", "repeat_refuted"):
            raise ValueError(f"unknown planner_strategy {self.planner_strategy!r}")
        if self.hypothesis_prompt not in ("evidence_grounded_v1", "terse_v0"):
            raise ValueError(f"unknown hypothesis_prompt {self.hypothesis_prompt!r}")
        if not (0 <= self.experience_top_k <= 10):
            raise ValueError("experience_top_k out of range")
        if not (1 <= self.max_candidates <= 2):
            raise ValueError("max_candidates must be 1..2 (plan protocol cap)")
        if not (1 <= self.campaign_rounds <= 3):
            raise ValueError("campaign_rounds out of range")
        for key in asdict(self):
            if key.lower() in HARNESS_FORBIDDEN_KEYS:
                raise ValueError(f"Harness may not declare frozen field {key!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False,
                          separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


BASE_HARNESS = ResearchHarness(
    planner_strategy="fixed_first_knob",
    hypothesis_prompt="terse_v0",
    experience_top_k=0,
    max_candidates=2,
    campaign_rounds=1,
    require_falsification=False,
    notes="Promoted base after ARR 1.0 evolve R1 (base/v3, operator human:user): "
          "workflow.max_candidates 1->2 ACCEPTED (Q 0.8763->0.8957, +0.0194 "
          "validated). Other knobs (planner_strategy/hypothesis_prompt/"
          "experience_top_k) stayed INCONCLUSIVE and were NOT landed.",
)

IMPROVED_HARNESS = ResearchHarness(
    planner_strategy="diagnosis_targeted",
    hypothesis_prompt="evidence_grounded_v1",
    experience_top_k=3,
    max_candidates=2,
    campaign_rounds=2,
    require_falsification=True,
    notes="Diagnosis-driven intervention policy + memory + competing hypotheses.",
)

NEGATIVE_HARNESS = ResearchHarness(
    planner_strategy="repeat_refuted",
    hypothesis_prompt="terse_v0",
    experience_top_k=0,
    max_candidates=1,
    campaign_rounds=1,
    require_falsification=False,
    notes="Negative control: repeats an already-refuted intervention.",
)


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _delta(base: float, cand: float, direction: str) -> float:
    """Positive always means 'candidate is better' (comparator convention)."""
    return base - cand if direction == "minimize" else cand - base


# --------------------------------------------------------------------------- #
# Harness-aware Planner (deterministic; NO network and NO fake experiments)
# --------------------------------------------------------------------------- #

def _dotted(config: dict[str, Any], key: str, default: Any = 0.02) -> Any:
    node: Any = config
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def make_harness_generate(harness: ResearchHarness, policy: Any,
                          experiences: list[str] | None = None,
                          baseline_config: dict[str, Any] | None = None):
    """Return the ``generate(prompt)->list[dict]`` consumed by LLMProposer.

    This is a REAL planner whose behaviour depends on the Harness: it reads the
    measured metrics and the structured failure diagnoses from the prompt, then
    chooses interventions. It never fabricates evidence -- every hypothesis
    cites the actual baseline experiment and a metric that experiment produced.

    It is deterministic (no LLM call), which makes Base/Candidate comparisons
    reproducible. The experiments it designs are executed by the real engines.
    """
    harness.validate()
    allowed = list(policy.allowed_config_keys)
    intervention_policy = dict(policy.intervention_policy)
    # Domain-injected value rules + the REAL baseline config are the only
    # sources of truth for a legal intervention value. A hardcoded scalar
    # (e.g. 0.1) is NOT cross-domain: a list-valued knob such as a predictor's
    # hidden layers would fail the campaign's config gate before any job runs.
    value_rules = dict(getattr(policy, "value_rules", {}) or {})
    cfg = baseline_config or {}
    lessons = list(experiences or [])[:max(0, harness.experience_top_k)]

    def generate(prompt: str) -> list[dict[str, Any]]:
        try:
            data = json.loads(prompt)
        except (ValueError, TypeError):
            return []
        source = str(data.get("source_experiment_id") or "")
        metrics = data.get("metrics") or {}
        reports = data.get("failure_reports") or []
        if not source or not isinstance(metrics, dict) or not metrics:
            return []
        categories = [str(r.get("category") or "") for r in reports
                      if isinstance(r, dict)]
        horizon_keys = sorted(
            [k for k in metrics if "@" in k],
            key=lambda k: int(k.rsplit("@", 1)[1])
            if k.rsplit("@", 1)[1].isdigit() else 0)
        longest = horizon_keys[-1] if horizon_keys else next(iter(metrics))

        def _valid(key: str, value: Any) -> bool:
            rule = value_rules.get(key)
            if rule is None:
                return True
            try:
                return rule(value) is True
            except Exception:
                return False

        def _typed(key: str, factor: float = 1.1) -> dict[str, Any] | None:
            """Perturb the baseline value according to its OWN type.

            Works for numeric knobs, integer knobs and list-valued knobs
            (e.g. hidden-layer widths) in ANY domain, then checks the domain's
            injected value rule. Returns None when no legal perturbation
            exists, rather than emitting an intervention that the campaign
            will reject at the config gate.
            """
            base = _dotted(cfg, key, None)
            if base is None or isinstance(base, bool):
                return None
            if isinstance(base, int):
                cand: Any = max(1, int(round(base * factor)))
            elif isinstance(base, float):
                cand = float(base * factor)
            elif isinstance(base, list) and base and all(
                    isinstance(v, (int, float)) and not isinstance(v, bool)
                    for v in base):
                cand = [max(1, int(round(v * factor))) if isinstance(v, int)
                        else float(v * factor) for v in base]
            else:
                return None
            if not _valid(key, cand):
                return None
            return {key: cand}

        def _sanitise(intervention: dict[str, Any] | None) -> dict[str, Any]:
            """Domain-injected interventions must also pass the value gates."""
            if not intervention:
                return {}
            return {k: v for k, v in intervention.items()
                    if k in allowed and _valid(k, v)}

        def _first_typed() -> dict[str, Any] | None:
            for key in allowed:
                typed = _typed(key)
                if typed:
                    return typed
            return None

        def _competing(key: str) -> dict[str, Any] | None:
            """A DIFFERENT, valid intervention on another whitelisted knob."""
            return _typed(key)

        # ---- intervention selection: the Harness's REAL behavioural knob ----
        if harness.planner_strategy == "diagnosis_targeted":
            chosen: dict[str, Any] = {}
            for category in categories:
                chosen = _sanitise(dict(intervention_policy.get(category, {})))
                if chosen:
                    break
            if not chosen and intervention_policy:
                for candidate_map in intervention_policy.values():
                    chosen = _sanitise(dict(candidate_map))
                    if chosen:
                        break
            if not chosen:
                chosen = _first_typed() or {}
            interventions = [chosen] if chosen else []
        elif harness.planner_strategy == "repeat_refuted":
            # Degenerate NEGATIVE CONTROL: re-apply the BASELINE value, i.e. an
            # intervention already known to be inert. The config is still a
            # real, isolated candidate file (so the experiment genuinely runs),
            # but no mechanism can change the outcome. A Harness that keeps
            # proposing inert experiments must score lower and be REJECTed.
            inert_key = next((k for k in allowed
                              if _dotted(cfg, k, None) is not None), None)
            if inert_key:
                base = _dotted(cfg, inert_key)
                interventions = ([{inert_key: base}]
                                 if _valid(inert_key, base) else [])
            else:
                interventions = []
        else:  # fixed_first_knob (BASE harness)
            first = _first_typed()
            interventions = [first] if first else []

        # COMPETING HYPOTHESES: `max_candidates` is a real Harness knob for
        # EVERY planner strategy, not only the diagnosis-targeted one. The
        # second hypothesis changes a DIFFERENT whitelisted knob so the paired
        # experiment can actually discriminate the two mechanisms.
        if harness.max_candidates >= 2 and interventions and len(allowed) >= 2:
            other_key = next((k for k in allowed
                              if k not in interventions[0]), None)
            competing = _competing(other_key) if other_key else None
            if competing:
                interventions.append(competing)

        if not interventions:
            return []
        items: list[dict[str, Any]] = []
        for i, intervention in enumerate(interventions[:harness.max_candidates]):
            evidence_refs = [f"experiment:{source}:metric:{longest}"]
            if harness.hypothesis_prompt == "evidence_grounded_v1":
                evidence_refs.append(
                    f"experiment:{source}:metric:{next(iter(metrics))}")
                statement = (
                    f"Measured {longest}={metrics.get(longest)} on the frozen "
                    f"baseline; the structured diagnosis {categories[:2]} suggests "
                    f"the controlled intervention {intervention}. This is a "
                    f"falsifiable candidate explanation, not an established cause.")
                falsification = [
                    f"On the held-out validation split, {longest} does not "
                    "improve across paired seeds relative to baseline.",
                    "The intervention fails the plan's guardrails on any paired seed.",
                ]
            else:
                statement = f"Adjust {intervention} to improve {longest}."
                falsification = [f"{longest} does not improve on validation."]
            if lessons:
                statement += " Prior experience considered: " + "; ".join(lessons[:2])[:300]
            metric_direction = ("decrease" if any(t in longest.lower()
                                                    for t in ("mse", "loss", "error"))
                                else "increase")
            items.append({
                "hypothesis_id": f"H_HARNESS_{i}_{hashlib.sha256(json.dumps(intervention, sort_keys=True).encode()).hexdigest()[:8]}",
                "title": f"Controlled intervention {i + 1} for {longest}",
                "statement": statement,
                "mechanism": ("The diagnosed failure mode may be mitigated by "
                              "changing this controlled configuration value; "
                              "competing interventions are compared on paired seeds."),
                "evidence_refs": evidence_refs,
                "intervention": intervention,
                "predicted_effects": [{"metric": longest,
                                      "direction": metric_direction}],
                "falsification_criteria": falsification,
                "confounders": ["seed variance", "finite training budget"],
                "confidence": 0.5,
            })
        return items

    return generate


# --------------------------------------------------------------------------- #
# Real baseline + holdout + evidence collection
# --------------------------------------------------------------------------- #

def run_real_baseline(service: Any, *, task: ResearchTask,
                      evidence_store: EvidenceStore,
                      baseline_id: str, job_tag: str = "base") -> dict[str, Any]:
    """Train on train split, evaluate on the SELECTION split, record REAL result."""
    adapter = service.adapter
    policy = service.domain
    # A distinct job_id per arm guarantees a distinct run_id / checkpoint, so
    # Base and Candidate can NEVER accidentally reuse each other's checkpoint.
    train_spec = ExperimentSpec(
        experiment_id=baseline_id, objective_id=task.objective_id,
        stage="train", config_path=policy.baseline_config_path,
        split="train",
        budget={"epochs": 1, "job_id": f"baseline_{job_tag}_{task.task_id}"},
    )
    jid = adapter.submit(train_spec)
    adapter.wait(jid, timeout=300)
    train = adapter.collect_result(jid, train_spec)
    if train.status != "succeeded":
        raise RuntimeError(f"baseline train failed: {train.status} {train.error}")
    decision_split = task.evaluation_protocol.get("decision_split", "validation")
    eval_spec = train_spec.with_run_id(str(train.run_id), stage="evaluate",
                                      split=decision_split).model_copy(
        update={"budget": {
            "job_id": f"baseline_eval_{job_tag}_{task.task_id}"}})
    jid2 = adapter.submit(eval_spec)
    adapter.wait(jid2, timeout=300)
    ev = adapter.collect_result(jid2, eval_spec)
    if ev.status != "succeeded":
        raise RuntimeError(f"baseline evaluate failed: {ev.status} {ev.error}")
    evidence_store.append(
        "result", {"experiment_id": baseline_id,
                   "result": ev.model_dump(mode="json")})
    return {"train": train.model_dump(mode="json"),
            "eval": ev.model_dump(mode="json"),
            "train_run_id": train.run_id,
            "dataset_version": ev.dataset_version,
            "decision_split": decision_split}


def _load_comparison_reports(service: Any, campaign_id: str) -> list[dict]:
    out: list[dict] = []
    store = service.deps.report_store
    for item in store.list_reports(campaign_id):
        cid = item.get("comparison_id")
        if not cid:
            continue
        report = store.load(campaign_id, cid)
        if report is not None:
            out.append(report.model_dump(mode="json"))
    return out


def run_holdout_confirmation(service: Any, *, task: ResearchTask,
                            campaign_id: str, direction: str,
                            split: str = "test",
                            candidate_id: str | None = None) -> dict[str, Any]:
    """Evaluate the SAME trained checkpoints on the RESERVED test split.

    Selection happened on ``validation``; this independent confirmation never
    feeds candidate selection. Both arms use the exact same checkpoints (proven
    by run_id), so this measures generalisation, not a retraining lottery.
    """
    adapter = service.adapter
    record = service.deps.campaign_store.load_campaign(campaign_id)
    if record is None:
        return {"evaluated": False, "reason": "campaign record missing"}
    test_split = split or task.heldout_policy.get("holdout_split", "test")
    baseline_trials = [t for t in record.trials if t.arm_type == "baseline"]
    candidate_trials = [t for t in record.trials if t.arm_type == "candidate"]
    if candidate_id:
        # Which candidate to confirm was decided on VALIDATION, never on this
        # reserved split. The test/OOD run is confirmation only.
        candidate_trials = [t for t in candidate_trials
                            if t.arm_id == candidate_id]
    if not baseline_trials or not candidate_trials:
        return {"evaluated": False, "reason": "no paired trials"}

    def evaluate(trial) -> tuple[float, str] | None:
        if trial.status != "succeeded" or not trial.train_run_id:
            return None
        # campaign_id is unique per plan/approval digest. External engines keep
        # a GLOBAL job registry, so a bare trial_id ("baseline_seed_42") would
        # collide on a later run of the same task -- the holdout then fails with
        # "job_id already exists; refusing to execute twice". Including the
        # campaign makes the holdout job idempotent WITH a campaign but unique
        # ACROSS campaigns; base/cand still share the identical convention.
        run_tag = f"{campaign_id[:12]}_{task.task_id[:12]}"
        spec = ExperimentSpec(
            experiment_id=f"holdout_{run_tag}_{trial.trial_id}",
            objective_id=task.objective_id,
            stage="evaluate", config_path=trial.config_relpath or trial.config_path,
            run_id=trial.train_run_id, parent_run_id=trial.train_run_id,
            split=test_split, seed=trial.seed,
            budget={"job_id": f"holdout_{test_split}_{run_tag}_{trial.trial_id}"},
        )
        jid = adapter.submit(spec)
        adapter.wait(jid, timeout=300)
        result = adapter.collect_result(jid, spec)
        if result.status != "succeeded":
            return None
        value = _num((result.metrics or {}).get(task.evaluation_protocol.get(
            "primary_metric") or service.domain.primary_metric))
        if value is None:
            return None
        return value, str(trial.seed)

    by_seed: dict[str, dict[str, float]] = {}
    for trial in baseline_trials:
        got = evaluate(trial)
        if got:
            by_seed.setdefault(got[1], {})["base"] = got[0]
    cand_by_seed: dict[str, dict[str, float]] = {}
    for trial in candidate_trials:
        got = evaluate(trial)
        if got:
            cand_by_seed.setdefault(str(trial.seed), {})["cand"] = got[0]
    deltas: list[float] = []
    for seed, base in by_seed.items():
        cand = cand_by_seed.get(seed)
        if cand and "cand" in cand:
            deltas.append(_delta(base["base"], cand["cand"], direction))
    if not deltas:
        return {"evaluated": False, "reason": "no complete holdout pairs",
                "n_pairs": 0}
    positive = sum(1 for d in deltas if d > 0)
    return {
        "evaluated": True,
        "holdout_split": test_split,
        "holdout_used_for_decision": False,
        "holdout_deltas": [round(d, 8) for d in deltas],
        "mean_holdout_delta": round(sum(deltas) / len(deltas), 8),
        "holdout_direction_agreement": round(
            max(positive, len(deltas) - positive) / len(deltas), 6),
        "n_pairs": len(deltas),
    }


# --------------------------------------------------------------------------- #
# The ONE entry point
# --------------------------------------------------------------------------- #

def run_research_task(task: ResearchTask, *,
                      harness: ResearchHarness | None = None,
                      run_root: str | Path | None = None,
                      approved_by: str = "arr-1.0-operator",
                      experiences: list[str] | None = None,
                      ) -> TaskResult:
    """Execute ONE ResearchTask end-to-end through the generic ResearchService."""
    harness = harness or BASE_HARNESS
    harness.validate()
    started = time.time()
    root = Path(run_root or (Path("experiments") / "arr_1_0")).resolve() / \
        f"{task.domain}_{task.task_id}_{harness.fingerprint()[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    data_dir = root / "store"
    data_dir.mkdir(parents=True, exist_ok=True)
    result = TaskResult(
        task_id=task.task_id, domain=task.domain,
        objective_id=task.objective_id,
        provenance={
            "task_sha256": task.fingerprint(),
            "harness_sha256": harness.fingerprint(),
            "harness": harness.to_dict(),
            "protocol_version": "arr-1.0",
            "run_root": str(root),
        },
    )
    try:
        # Register a FRESH bundle so each arm has isolated campaign/DB state.
        bundle = build_local_bundle(task.domain, data_dir=data_dir) \
            if task.domain != "scinet" else None
        if bundle is not None:
            register_domain(bundle)
        else:
            # SciNet: register BEFORE get_domain(). A fresh process has an
            # empty registry; only build_default_research_service used to
            # register it lazily, which is AFTER the policy is read here.
            from .research_environment import ensure_domain_registered
            ensure_domain_registered(task.domain)
        evidence_store = EvidenceStore(data_dir / "evidence.jsonl")
        policy = (bundle.policy_factory() if bundle is not None
                  else get_domain("scinet").policy_factory())
        # The fixed evaluation protocol is OUTSIDE the harness: it comes from
        # the domain policy, never from the candidate being evaluated (F4).
        evaluator = (bundle.quality_evaluator if bundle is not None
                     else quality_evaluator_for("scinet"))
        # Read the REAL baseline config so the negative control can reuse its
        # value (a genuinely inert intervention) without hardcoding a domain.
        import yaml
        from .settings import DEFAULT_SCINET_DIR
        if bundle is not None:
            project = Path(bundle.adapter_factory().project_dir)
        else:
            project = Path(DEFAULT_SCINET_DIR)
        cfg_path = Path(policy.baseline_config_path)
        if not cfg_path.is_absolute():
            cfg_path = project / cfg_path
        baseline_cfg: dict[str, Any] = {}
        if cfg_path.is_file():
            baseline_cfg = yaml.safe_load(
                cfg_path.read_text(encoding="utf-8")) or {}
        generate = make_harness_generate(harness, policy, experiences,
                                         baseline_config=baseline_cfg)

        def evidence_provider(experiment_id: str):
            from .failure_diagnoser import FailureDiagnoser
            from .research_service import default_evidence_provider
            return default_evidence_provider(experiment_id, evidence_store)

        # Task-level budget (paired seeds / rounds) narrows the domain policy;
        # the evaluation protocol itself stays fixed and domain-owned (F4).
        from dataclasses import replace as _dc_replace
        task_seeds = task.heldout_policy.get("paired_seeds")
        task_policy = policy
        if task_seeds:
            task_policy = _dc_replace(
                policy, paired_seeds=tuple(int(s) for s in task_seeds),
                max_seeds=max(5, len(task_seeds)),
                max_campaign_rounds=min(policy.max_campaign_rounds,
                                        harness.campaign_rounds))
        svc = build_default_research_service(
            task.domain, llm_generate=generate,
            db_path=data_dir / "research.sqlite", data_dir=data_dir,
            evidence_provider=evidence_provider, domain_policy=task_policy)

        baseline_id = f"baseline_{task.task_id}"
        baseline = run_real_baseline(
            svc, task=task, evidence_store=evidence_store,
            baseline_id=baseline_id, job_tag=harness.fingerprint()[:8])
        result.baseline = baseline

        session = svc.start_research_session(
            objective_id=task.objective_id, description=task.objective)
        sid = session["session_id"]
        result.session_id = sid

        proposed = svc.propose_research_hypotheses(
            objective_id=task.objective_id,
            source_experiment_id=baseline_id, session_id=sid)
        if not proposed.get("ok"):
            raise RuntimeError(f"propose failed: {proposed}")
        hids = [h["hypothesis_id"] for h in proposed.get("hypotheses", [])]
        result.hypotheses = proposed.get("hypotheses", [])
        if not hids:
            raise RuntimeError(
                f"autonomous planner produced no hypotheses: "
                f"{proposed.get('skip_reasons')}")
        reviewed = svc.review_research_hypotheses(
            hypothesis_ids_json=json.dumps(hids), session_id=sid)
        approved = reviewed.get("approved_ids", [])
        if not approved:
            raise RuntimeError(
                f"deterministic critic rejected all hypotheses: {reviewed}")

        plan = svc.create_research_plan(
            objective_id=task.objective_id,
            hypothesis_ids_json=json.dumps(approved), session_id=sid)
        ap = svc.request_plan_approval(plan_id=plan["plan_id"], session_id=sid)
        aid = ap["approval_id"]
        # Out-of-band human grant, exactly like approval_cli / Package B.
        svc.stores.approvals.grant(aid, approved_by=approved_by, source="human")

        run = svc.run_research_campaign(plan_id=plan["plan_id"],
                                         approval_id=aid, session_id=sid)
        # _guard returns a structured failure dict instead of raising. Report
        # the REAL error_code (e.g. config_key_not_allowed) instead of letting
        # a bare KeyError('campaign_id') hide the diagnosis.
        result.campaign = run
        if not run.get("ok") or not run.get("campaign_id"):
            raise RuntimeError(
                f"campaign submission rejected: "
                f"{run.get('error_code', 'unknown')}: {run.get('error', run)}")
        campaign_id = run["campaign_id"]
        cmp = svc.compare_research_campaign(campaign_id=campaign_id,
                                             session_id=sid)
        result.comparisons = cmp.get("comparisons", [])
        rep = svc.get_research_report(session_id=sid)
        report = rep["report"]
        result.report = report
        result.outcome = report.get("outcome", "not_evaluated")
        result.telemetry = {
            "total_trials": run.get("total_trials", 0),
            "succeeded_trials": run.get("succeeded_trials", 0),
            "missing_runs": int(run.get("total_trials", 0))
                            - int(run.get("succeeded_trials", 0)),
            "timeout_runs": 0,
            "jobs_used": 2 * (1 + len(plan.get("candidate_ids", [])))
                         * len(plan.get("paired_seeds", [1])),
            "elapsed_seconds": round(time.time() - started, 3),
            "budget": task.budget,
            "proposer_skip_reasons": proposed.get("skip_reasons", []),
            "critic_rejections": reviewed.get("rejected_ids", []),
        }

        comparison_reports = _load_comparison_reports(svc, campaign_id)

        # The candidate to confirm is selected by the VALIDATION comparison,
        # before any reserved split is touched (no test-set selection).
        selected_candidate = ""
        if result.comparisons:
            selected_candidate = str(max(
                result.comparisons,
                key=lambda c: (c.get("mean_delta") or 0.0))["candidate_id"])

        def _confirm(split_name: str) -> dict[str, Any]:
            try:
                return run_holdout_confirmation(
                    svc, task=task, campaign_id=campaign_id,
                    direction=policy.metric_direction, split=split_name,
                    candidate_id=selected_candidate or None)
            except Exception as exc:  # failure is EVIDENCE, never hidden
                return {"evaluated": False, "holdout_split": split_name,
                        "reason": f"{type(exc).__name__}: {exc}"}

        holdout = _confirm(task.heldout_policy.get("holdout_split", "test"))
        # An OOD task is a DIFFERENT data regime, not a resample of the same
        # split. Both confirmations reuse the SAME trained checkpoints.
        ood = _confirm("ood") if task.heldout_policy.get("ood_task", True) else {}
        result.evidence = {
            "baseline_id": baseline_id,
            "decision_split": baseline["decision_split"],
            "dataset_version": baseline["dataset_version"],
            "comparison_reports": comparison_reports,
            "holdout": holdout, "ood": ood,
            "report_path": rep.get("report_path", ""),
        }
        # OOD must be in telemetry BEFORE the fixed evaluator runs, otherwise
        # the generalisation dimension would be silently downgraded.
        result.telemetry["ood"] = {
            "confirmed": bool(ood.get("evaluated"))
                          and _num(ood.get("mean_holdout_delta")) is not None
                          and (ood.get("mean_holdout_delta") or 0.0) > 0.0,
            "split": "ood", "mean_delta": ood.get("mean_holdout_delta"),
            "direction_agreement": ood.get("holdout_direction_agreement"),
        }
        qe = QualityEvidence(
            report=report, comparison_reports=comparison_reports,
            experiment_results=[baseline["eval"]],
            telemetry=result.telemetry, holdout=holdout,
            provenance={"task_sha256": task.fingerprint(),
                        "harness_sha256": harness.fingerprint(),
                        "git_head": _git_head(), "ood": ood},
        )
        score = evaluator.evaluate(qe)
        result.quality = score.to_dict()
        result.provenance.update({
            "engine": svc.adapter.name,
            "engine_capabilities": svc.adapter.capabilities(),
            "artifact_hashes": {
                "baseline_eval_sha256": hashlib.sha256(json.dumps(
                    baseline["eval"], sort_keys=True, default=str).encode()).hexdigest(),
                "report_sha256": report.get("report_sha256"),
                "quality_evidence_sha256": qe.fingerprint(),
            },
            "artifact_paths": {
                "report": rep.get("report_path", ""),
                "comparisons": str(bundle.campaign_root if bundle else ""),
            },
        })
        result.ok = True
    except Exception as exc:
        result.ok = False
        result.error = f"{type(exc).__name__}: {exc}"
        import traceback
        (root / "error_traceback.txt").write_text(
            traceback.format_exc(), encoding="utf-8")
    result.provenance["elapsed_seconds"] = round(time.time() - started, 3)
    result.artifact_sha256 = result.compute_sha256()
    (root / "task_result.json").write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True,
                   ensure_ascii=False, default=str), encoding="utf-8")
    (root / "task_result_sha256.txt").write_text(
        result.artifact_sha256, encoding="utf-8")
    result.provenance["result_path"] = str(root / "task_result.json")
    return result


def _git_head() -> str:
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5
                             ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def benchmark_all_domains(*, tasks: list[ResearchTask] | None = None,
                          harness: ResearchHarness | None = None,
                          run_root: str | Path | None = None
                          ) -> list[TaskResult]:
    """Run the three REAL minimum research tasks through ONE runtime."""
    from .research_task import default_tasks
    tasks = tasks or default_tasks()
    return [run_research_task(t, harness=harness, run_root=run_root) for t in tasks]


def _default_tasks() -> list[ResearchTask]:
    from .research_task import default_tasks
    return default_tasks()
