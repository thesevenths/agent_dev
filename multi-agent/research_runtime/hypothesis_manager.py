"""P1-02: evidence-grounded hypothesis generation.

Boundary (deliberate)
---------------------
* ``RuleBasedProposer`` NEVER invents a root cause. It reads STRUCTURED
  evidence (``ExperimentResult.metrics`` + ``FailureReport``) and emits a
  hypothesis only when (a) a generic ``<metric>@<k>`` family really degrades
  with horizon and (b) the caller injected an intervention policy for that
  diagnosis. No policy => no hypothesis, with an explicit skip reason. This is
  the "admit insufficient information" behaviour, not an LLM guessing why a
  model is bad.
* ``LLMProposer`` is a pluggable adapter: any callable producing text/JSON.
  Its output is UNTRUSTED. A parse error, a schema error, a forged source or an
  unknown intervention key rejects the WHOLE batch, so no candidate experiment
  can ever be launched from a hallucinated config.
* ``HypothesisManager`` is the only entry point; ``HypothesisCritic`` (rules)
  is the final gate. An LLM advisor can add notes but cannot approve.

Nothing in this module names a specific engine, model, metric or config field.
The mapping from a diagnosis to a concrete config change is injected policy.
"""
from __future__ import annotations

import copy
import json
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

from pydantic import ValidationError

from .hypothesis_critic import HypothesisCritic
from .schemas import (
    CandidateExperiment,
    ExperimentPlan,
    ExperimentResult,
    FailureReport,
    Hypothesis,
    MetricPrediction,
)

# Generic metric-name semantics, not an engine's metric whitelist.
LOWER_IS_BETTER = ("mse", "rmse", "mae", "loss", "error", "err")
HIGHER_IS_BETTER = ("acc", "accuracy", "cos", "similarity", "score",
                    "r2", "f1", "precision", "recall", "auc")

HORIZON_DIAGNOSIS = "dynamics_error_accumulation"
HORIZON_RULE = "horizon_degradation"

_HORIZON_RE = re.compile(r"^(.+)@(\d+)$")
_NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9]+")
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def improvement_direction(metric_name: str) -> str | None:
    """Which direction counts as an IMPROVEMENT for this metric family.

    Unknown semantics return None: callers must not guess a direction.
    """
    low = metric_name.lower()
    if any(token in low for token in LOWER_IS_BETTER):
        return "decrease"
    if any(token in low for token in HIGHER_IS_BETTER):
        return "increase"
    return None


def degraded_families(metrics: Mapping[str, float], *,
                      factor: float = 1.5) -> list[tuple[str, str, str]]:
    """Which ``<metric>@<k>`` families actually degrade with horizon.

    Returns ``(family, shortest_metric_key, longest_metric_key)``. Direction is
    inferred from the metric name, so the same code covers world models,
    forecasters and backtests without hard-coding one engine's metrics.
    """
    groups: dict[str, list[tuple[int, str, float]]] = {}
    for key, value in metrics.items():
        match = _HORIZON_RE.match(str(key))
        if not match:
            continue
        groups.setdefault(match.group(1), []).append(
            (int(match.group(2)), str(key), float(value))
        )
    degraded: list[tuple[str, str, str]] = []
    for family, points in sorted(groups.items()):
        if len(points) < 2:
            continue
        points.sort()
        direction = improvement_direction(family)
        if direction is None:
            continue
        first, last = points[0][2], points[-1][2]
        worsened = ((first > 0 and last > first * factor)
                    if direction == "decrease"
                    else (first > 0 and last < first / factor))
        if worsened:
            degraded.append((family, points[0][1], points[-1][1]))
    return degraded


def _is_horizon_report(report: Any) -> bool:
    if isinstance(report, Mapping):
        evidence = report.get("evidence") or {}
        return (evidence.get("rule") == HORIZON_RULE
                or report.get("category") in (HORIZON_DIAGNOSIS, HORIZON_RULE))
    evidence = getattr(report, "evidence", None) or {}
    return (evidence.get("rule") == HORIZON_RULE
            or getattr(report, "category", None) in (HORIZON_DIAGNOSIS, HORIZON_RULE))


def _report_id(report: Any) -> str | None:
    if isinstance(report, Mapping):
        return report.get("failure_id")
    return getattr(report, "failure_id", None)


class RuleBasedProposer:
    """Generate hypotheses from structured degradation evidence ONLY.

    ``intervention_policy`` maps a diagnosis category to a config intervention,
    e.g. ``{"dynamics_error_accumulation": {"model.width": 64}}``. It is
    injected by the domain layer -- this module ships no engine-specific keys.
    """

    def __init__(self,
                 intervention_policy: Mapping[str, dict[str, Any]] | None = None):
        self.intervention_policy = copy.deepcopy(dict(intervention_policy or {}))
        self.skipped_reasons: list[str] = []

    def propose(self, result: ExperimentResult,
                failure_reports: Sequence[FailureReport | dict] | None,
                objective_id: str) -> list[Hypothesis]:
        self.skipped_reasons = []
        reports = [report for report in (failure_reports or [])
                   if _is_horizon_report(report)]
        if not reports:
            self.skipped_reasons.append(
                "no horizon-degradation diagnosis in structured evidence")
            return []

        families = degraded_families(result.metrics)
        if not families:
            self.skipped_reasons.append(
                "no metric family shows measurable horizon degradation")
            return []

        intervention = self.intervention_policy.get(HORIZON_DIAGNOSIS)
        if not intervention:
            self.skipped_reasons.append(
                "no intervention policy injected for this diagnosis; "
                "refusing to invent a root cause")
            return []

        proposals: list[Hypothesis] = []
        for family, _, longest in families:
            direction = improvement_direction(family)
            refs = [f"experiment:{result.experiment_id}:metric:{longest}"]
            for report in reports[:2]:
                failure_id = _report_id(report)
                if failure_id:
                    refs.append(f"failure:{failure_id}")
            slug = _NON_ALNUM_RE.sub("_", family).strip("_").upper()
            proposals.append(Hypothesis(
                hypothesis_id=f"H_HORIZON_{slug}",
                objective_id=objective_id,
                source_experiment_id=result.experiment_id,
                title=f"Horizon degradation in {family}",
                statement=(
                    f"Longer-horizon {family} degrades relative to the "
                    "shortest horizon; a targeted configuration change may "
                    "reduce that accumulation."
                ),
                mechanism=(
                    "The current configuration may under-fit the multi-step "
                    "dynamics that dominate longer rollouts. This is a "
                    "candidate explanation, not an established root cause."
                ),
                evidence_refs=refs,
                intervention=copy.deepcopy(intervention),
                predicted_effects=[
                    MetricPrediction(metric=longest, direction=direction)
                ],
                falsification_criteria=[
                    f"On the held-out validation split, {longest} does not "
                    "improve across the paired seeds relative to baseline."
                ],
                confounders=["training budget", "optimization instability",
                             "random seed variance"],
                confidence=0.5,
            ))
        return proposals


class LLMProposer:
    """Untrusted, pluggable hypothesis source. No vendor is hard-coded.

    ``generate`` is any callable: ``prompt(str) -> str | dict | list``. It may
    be Qwen, GPT, a local model or a test double. Nothing here imports an LLM.
    """

    def __init__(self, generate: Callable[[str], Any], *,
                 allowed_intervention_keys: Iterable[str] | None = None):
        self.generate = generate
        allowed = allowed_intervention_keys
        self.allowed = frozenset(allowed) if allowed is not None else None
        self.rejections: list[str] = []

    def _prompt(self, result: ExperimentResult,
                failure_reports: Sequence[FailureReport | dict] | None,
                objective_id: str) -> str:
        reports = [
            report.model_dump() if hasattr(report, "model_dump") else report
            for report in (failure_reports or [])
        ]
        return json.dumps({
            "objective_id": objective_id,
            "source_experiment_id": result.experiment_id,
            "metrics": result.metrics,
            "failure_reports": reports,
            "allowed_intervention_keys": (sorted(self.allowed)
                                         if self.allowed is not None else None),
            "instructions": (
                "Return JSON: a list of hypothesis objects with hypothesis_id, "
                "title, statement, mechanism, evidence_refs, intervention, "
                "predicted_effects (metric/direction), "
                "falsification_criteria and confidence. Evidence must "
                "reference the experiment above. Do not invent config keys."
            ),
        }, ensure_ascii=False, default=str)

    @staticmethod
    def _parse(raw: Any) -> list[dict] | None:
        if isinstance(raw, (list, dict)):
            data = raw
        elif isinstance(raw, str):
            text = raw.strip()
            fenced = _FENCE_RE.search(text)
            if fenced:
                text = fenced.group(1).strip()
            try:
                data = json.loads(text)
            except (ValueError, TypeError):
                return None
        else:
            return None
        if isinstance(data, Mapping):
            data = data.get("hypotheses")
        if not isinstance(data, list) or not data:
            return None
        if not all(isinstance(item, Mapping) for item in data):
            return None
        return [dict(item) for item in data]

    def propose(self, result: ExperimentResult,
                failure_reports: Sequence[FailureReport | dict] | None,
                objective_id: str) -> list[Hypothesis]:
        self.rejections = []
        try:
            raw = self.generate(
                self._prompt(result, failure_reports, objective_id)
            )
        except Exception as exc:  # an unavailable model must never break the loop
            self.rejections.append(f"generator failed: {type(exc).__name__}: {exc}")
            return []

        items = self._parse(raw)
        if items is None:
            self.rejections.append("invalid or empty JSON payload: whole batch rejected")
            return []

        proposals: list[Hypothesis] = []
        for item in items:
            # The evidence source is NOT up to the model: force the real ids.
            item["objective_id"] = objective_id
            item["source_experiment_id"] = result.experiment_id
            try:
                hypothesis = Hypothesis.model_validate(item)
            except ValidationError as exc:
                self.rejections.append(
                    f"invalid hypothesis payload: whole batch rejected "
                    f"({exc.error_count()} field error(s))"
                )
                return []
            if self.allowed is not None and any(
                key not in self.allowed for key in hypothesis.intervention
            ):
                self.rejections.append(
                    "unknown intervention key: whole batch rejected"
                )
                return []
            proposals.append(hypothesis)
        return proposals


class HypothesisManager:
    """Single entry point: propose -> validate -> deterministic review."""

    def __init__(self, proposer: Any | None = None,
                 critic: HypothesisCritic | None = None):
        self.proposer = proposer if proposer is not None else RuleBasedProposer()
        self.critic = critic if critic is not None else HypothesisCritic()
        self.rejections: list[dict[str, Any]] = []

    def propose(self, result: ExperimentResult,
                failure_reports: Sequence[FailureReport | dict] | None,
                objective_id: str) -> list[Hypothesis]:
        self.rejections = []
        try:
            raw = self.proposer.propose(result, failure_reports, objective_id)
        except Exception as exc:
            self.rejections.append({
                "stage": "propose",
                "reason": f"{type(exc).__name__}: {exc}",
            })
            return []

        valid: list[Hypothesis] = []
        for item in raw or []:
            if isinstance(item, Hypothesis):
                valid.append(item)
                continue
            try:
                valid.append(Hypothesis.model_validate(item))
            except (ValidationError, TypeError) as exc:
                self.rejections.append({
                    "stage": "propose", "reason": f"invalid proposal: {exc}",
                })
        return valid

    def review(self, hypotheses: Iterable[Hypothesis]) -> list[Hypothesis]:
        approved: list[Hypothesis] = []
        for hypothesis in hypotheses:
            ok, errors = self.critic.review(hypothesis)
            if ok:
                approved.append(
                    hypothesis.model_copy(update={"status": "approved"})
                )
            else:
                self.rejections.append({
                    "stage": "review",
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "errors": errors,
                })
        return approved


def build_experiment_plan(*, plan_id: str, objective_id: str,
                          baseline_experiment_id: str, dataset_version: str,
                          hypotheses: Sequence[Hypothesis],
                          primary_metric: str,
                          metric_direction: str = "minimize",
                          paired_seeds: Sequence[int] = (42, 43, 44),
                          guardrails: dict[str, dict[str, float]] | None = None,
                          max_parallel: int = 1) -> ExperimentPlan:
    """Pure planner: approved hypotheses -> one paired ExperimentPlan.

    P1-04 will consume this plan; it does NOT execute anything here. Requiring
    ``status == approved`` is what stops an unreviewed hypothesis from ever
    becoming a candidate experiment.
    """
    candidates = []
    for hypothesis in hypotheses:
        if hypothesis.status != "approved":
            raise ValueError(
                f"hypothesis {hypothesis.hypothesis_id} has status "
                f"{hypothesis.status!r}; only reviewed/approved hypotheses "
                "may become candidate experiments"
            )
        candidates.append(CandidateExperiment(
            candidate_id=f"cand_{hypothesis.hypothesis_id}",
            hypothesis_id=hypothesis.hypothesis_id,
            config_overrides=copy.deepcopy(hypothesis.intervention),
            seeds=list(paired_seeds),
            dataset_version=dataset_version,
        ))
    return ExperimentPlan(
        plan_id=plan_id,
        objective_id=objective_id,
        baseline_experiment_id=baseline_experiment_id,
        dataset_version=dataset_version,
        candidates=candidates,
        primary_metric=primary_metric,
        metric_direction=metric_direction,
        paired_seeds=list(paired_seeds),
        guardrails=guardrails or {},
        max_parallel=max_parallel,
    )
