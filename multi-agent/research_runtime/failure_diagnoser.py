"""Two-layer failure diagnosis: Runtime (L1) vs Scientific Model (L2).

The key insight from the integration design: an Agent step can be *completely
successful* (train.py ran, metrics were produced, a report was written) while
the research objective still FAILED (the model cannot predict). These are
different layers with different scoreboards and must not be conflated.

Rules here read STRUCTURED fields only (error_type / metrics / artifacts),
never console prose. The horizon-degradation rule is generic: it works on any
``<metric>@<k>`` family, with direction inferred from the metric name, so it
applies to world models, forecasters and backtests alike.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from .schemas import ExperimentResult, FailureReport

MetricFamily = tuple[str, list[tuple[int, float]]]


@dataclass(frozen=True)
class DiagnosticRule:
    name: str
    category: str
    layer: str
    severity: str
    hypothesis: str
    recommended_action: str
    match: Callable[[ExperimentResult], bool]


def _horizon_groups(result: ExperimentResult) -> dict[str, list[tuple[int, float]]]:
    groups: dict[str, list[tuple[int, float]]] = {}
    for key, value in result.metrics.items():
        match = re.match(r"^(.+)@(\d+)$", key)
        if match:
            groups.setdefault(match.group(1), []).append((int(match.group(2)), value))
    for points in groups.values():
        points.sort()
    return groups


def _degrades_over_horizon(result: ExperimentResult, factor: float = 1.5) -> bool:
    """Generic multi-step degradation check.

    'Lower is better' metrics (mse/loss/error/rmse) worsening by >factor, or
    'higher is better' metrics (cos/acc/score/r2) dropping by >factor, between
    the shortest and longest horizon, is evidence of error accumulation.
    """
    for name, points in _horizon_groups(result).items():
        if len(points) < 2:
            continue
        first, last = points[0][1], points[-1][1]
        low = name.lower()
        if any(t in low for t in ("mse", "loss", "error", "rmse")):
            if first > 0 and last > first * factor:
                return True
        elif any(t in low for t in ("cos", "acc", "score", "r2")):
            if first > 0 and last < first / factor:
                return True
    return False


def _is(result: ExperimentResult, *types: str) -> bool:
    return result.error_type in types


def _evidence(result: ExperimentResult) -> dict:
    return {
        "status": result.status,
        "error_type": result.error_type,
        "error": (result.error or "")[:1000],
        "exit_code": result.exit_code,
        "engine": result.engine,
        "stage": result.stage,
        "metric_keys": sorted(result.metrics),
        "artifact_count": len(result.artifacts),
        "data_quality_notes": result.data_quality_notes,
    }


def default_rules() -> list[DiagnosticRule]:
    return [
        DiagnosticRule(
            name="budget_exceeded", category="execution_budget", layer="runtime",
            severity="high", hypothesis="stage exceeded its time budget",
            recommended_action="increase budget.timeout_s or shrink the dataset/model",
            match=lambda r: _is(r, "timeout"),
        ),
        DiagnosticRule(
            name="pipeline_incomplete", category="missing_artifact", layer="runtime",
            severity="high", hypothesis="stage exited 0 but did not write its expected output",
            recommended_action="inspect the job log and verify the artefact contract",
            match=lambda r: _is(r, "missing_artifact"),
        ),
        DiagnosticRule(
            name="worker_lost", category="execution_infrastructure", layer="runtime",
            severity="high", hypothesis="worker crashed, was killed, or host restarted",
            recommended_action="check heartbeat/JobStore recovery, then resubmit once",
            match=lambda r: _is(r, "worker_interrupted", "worker_exception",
                                "empty_command"),
        ),
        DiagnosticRule(
            name="process_failure", category="execution_failure", layer="runtime",
            severity="high", hypothesis="the stage process returned a non-zero exit code",
            recommended_action="read log_path tail; fix config/environment before retrying",
            match=lambda r: r.status == "failed" and _is(r, "process_failed"),
        ),
        DiagnosticRule(
            name="result_unavailable", category="evidence_gap", layer="runtime",
            severity="high", hypothesis="no machine-readable result was produced",
            recommended_action="verify JobStore state and result_path; do not accept the run",
            match=lambda r: r.status == "incomplete" or _is(r, "result_unavailable"),
        ),
        DiagnosticRule(
            name="metric_quality", category="data_quality", layer="model",
            severity="medium", hypothesis="some metrics were non-finite or non-numeric",
            recommended_action="check for divergence / NaN loss before trusting the result",
            match=lambda r: bool(r.data_quality_notes),
        ),
        DiagnosticRule(
            name="no_metrics", category="evidence_gap", layer="model",
            severity="medium", hypothesis="evaluation succeeded but reported no metrics",
            recommended_action="confirm the evaluator wrote its metrics artefact",
            match=lambda r: r.succeeded and r.stage in ("evaluate", "analyze")
                            and not r.metrics,
        ),
        DiagnosticRule(
            name="horizon_degradation", category="dynamics_error_accumulation",
            layer="model", severity="medium",
            hypothesis="multi-step prediction error accumulates with horizon; "
                       "the latent dynamics model may be under-trained",
            recommended_action="run a controlled candidate with multi-step loss "
                               "weighting or longer rollout training, then compare "
                               "on the SAME frozen test set",
            match=_degrades_over_horizon,
        ),
    ]


class FailureDiagnoser:
    def __init__(self, rules: list[DiagnosticRule] | None = None):
        self.rules = list(rules) if rules is not None else default_rules()

    def diagnose_all(self, result: ExperimentResult, *,
                     layer: str | None = None) -> list[FailureReport]:
        reports = []
        for rule in self.rules:
            if layer and rule.layer != layer:
                continue
            try:
                matched = bool(rule.match(result))
            except Exception:
                matched = False
            if matched:
                reports.append(FailureReport(
                    experiment_id=result.experiment_id,
                    job_id=result.job_id,
                    layer=rule.layer,
                    category=rule.category,
                    evidence={"rule": rule.name, **_evidence(result)},
                    hypothesis=rule.hypothesis,
                    recommended_action=rule.recommended_action,
                    severity=rule.severity,
                ))
        return reports

    def diagnose(self, result: ExperimentResult, *,
                 layer: str | None = None) -> FailureReport | None:
        reports = self.diagnose_all(result, layer=layer)
        return reports[0] if reports else None

    @staticmethod
    def diagnose_runtime(*, experiment_id: str, job_id: str,
                         category: str, error: str,
                         action: str = "inspect the runtime trace and retry") -> FailureReport:
        """L1 entry point for Agent-side (non-model) failures."""
        return FailureReport(
            experiment_id=experiment_id, job_id=job_id, layer="runtime",
            category=category, evidence={"error": error[:1000]},
            hypothesis="the research workflow failed before a model could be evaluated",
            recommended_action=action, severity="high",
        )
