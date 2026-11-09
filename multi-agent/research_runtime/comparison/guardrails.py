"""P1-05: guardrail evaluation -- stop a candidate that wins one metric by
breaking others.

Supported rule forms (ALL metric names / thresholds come from the injected
ExperimentPlan.guardrails; this module knows no domain)::

    {"secondary_error": {"max": 0.05}}            absolute upper bound
    {"quality_score": {"min": 0.80}}             absolute lower bound
    {"auxiliary_error": {"max": 0.15,
                         "max_relative_regression_pct": 5.0,
                         "direction": "minimize"}} relative tolerance

Semantics that are deliberately strict
--------------------------------------
* Guardrails are checked PER PAIRED SEED. Any seed violating a rule is a hard
  violation; averaging could hide a regression on one seed.
* A declared metric MISSING or non-finite on either arm is NEVER a pass. It
  yields ``insufficient`` and caps the decision at inconclusive.
* A relative-regression rule MUST declare its own ``direction``. Guessing the
  direction from a metric name is forbidden; a missing/unknown direction makes
  the rule ``invalid_rule`` -> inconclusive.
* Unknown rule keys, empty rules and non-numeric thresholds are ``invalid_rule``.
* With NO guardrail rules the assessment is empty and passes vacuously -- that
  is explicit: the plan declared no protections, not a hidden default.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .schemas import (
    GuardrailAssessment,
    GuardrailResult,
    PairedObservation,
)

_ALLOWED_KEYS = frozenset({
    "min", "max", "max_relative_regression_pct", "direction",
})
_DIRECTIONS = frozenset({"minimize", "maximize"})


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


class GuardrailEvaluator:
    """Injectable evaluator; Comparator owns no guardrail semantics."""

    def evaluate(self, rules: Mapping[str, Any] | None,
                 pairs: Sequence[PairedObservation]) -> GuardrailAssessment:
        results = [self._evaluate_one(metric, rule, pairs)
                   for metric, rule in (rules or {}).items()]
        return GuardrailAssessment(results=results)

    # -------------------------------------------------------------- internal
    def _evaluate_one(self, metric: str, rule: Any,
                       pairs: Sequence[PairedObservation]) -> GuardrailResult:
        if not isinstance(rule, dict):
            return GuardrailResult(
                metric=metric, status="invalid_rule", rule={"raw": rule},
                detail="guardrail rule must be a mapping")
        unknown = sorted(set(rule) - _ALLOWED_KEYS)
        if unknown:
            return GuardrailResult(
                metric=metric, status="invalid_rule", rule=dict(rule),
                detail=f"unknown rule key(s): {unknown}")

        lower = _number(rule["min"]) if "min" in rule else None
        upper = _number(rule["max"]) if "max" in rule else None
        if ("min" in rule and lower is None) or ("max" in rule and upper is None):
            return GuardrailResult(
                metric=metric, status="invalid_rule", rule=dict(rule),
                detail="min/max must be finite numbers")
        if lower is not None and upper is not None and lower > upper:
            return GuardrailResult(
                metric=metric, status="invalid_rule", rule=dict(rule),
                detail="min must not exceed max")

        relative = rule.get("max_relative_regression_pct")
        relative_pct = _number(relative) if relative is not None else None
        if relative is not None and (relative_pct is None or relative_pct < 0.0):
            return GuardrailResult(
                metric=metric, status="invalid_rule", rule=dict(rule),
                detail="max_relative_regression_pct must be a finite number >= 0")
        direction = rule.get("direction")
        if relative is not None and direction not in _DIRECTIONS:
            return GuardrailResult(
                metric=metric, status="invalid_rule", rule=dict(rule),
                detail=("a relative-regression rule must declare direction="
                       "'minimize' or 'maximize'; refusing to guess"))
        if direction is not None and direction not in _DIRECTIONS:
            return GuardrailResult(
                metric=metric, status="invalid_rule", rule=dict(rule),
                detail=f"unknown direction {direction!r}")
        if lower is None and upper is None and relative is None:
            return GuardrailResult(
                metric=metric, status="invalid_rule", rule=dict(rule),
                detail="guardrail rule declares no bound")

        if not pairs:
            return GuardrailResult(
                metric=metric, status="insufficient", rule=dict(rule),
                detail="no valid paired seeds to evaluate this guardrail")

        per_seed: list[dict[str, Any]] = []
        worst = "passed"
        for pair in pairs:
            b = pair.baseline_guardrail_metrics.get(metric)
            c = pair.candidate_guardrail_metrics.get(metric)
            b_num, c_num = _number(b), _number(c)
            if b_num is None or c_num is None:
                per_seed.append({"seed": pair.seed, "baseline": b,
                                 "candidate": c, "status": "insufficient"})
                worst = "insufficient"
                continue
            checks: list[str] = []
            status = "passed"
            if lower is not None and c_num < lower:
                checks.append(f"candidate {c_num} < min {lower}")
                status = "violated"
            if upper is not None and c_num > upper:
                checks.append(f"candidate {c_num} > max {upper}")
                status = "violated"
            if relative_pct is not None:
                if b_num == 0.0:
                    checks.append("relative regression undefined at baseline=0")
                    status = "insufficient" if status != "violated" else status
                else:
                    regression_pct = ((c_num - b_num) / abs(b_num) * 100.0
                                     if direction == "minimize"
                                     else (b_num - c_num) / abs(b_num) * 100.0)
                    if regression_pct > relative_pct:
                        checks.append(
                            f"regression {regression_pct:.4f}% > allowed "
                            f"{relative_pct:.4f}%")
                        status = "violated"
            per_seed.append({"seed": pair.seed, "baseline": b_num,
                             "candidate": c_num, "status": status,
                             "checks": checks})
            if status == "violated":
                worst = "violated"
            elif status == "insufficient" and worst != "violated":
                worst = "insufficient"

        detail = ("all paired seeds passed" if worst == "passed"
                  else f"per-seed verdict = {worst}")
        return GuardrailResult(metric=metric, status=worst,  # type: ignore[arg-type]
                              rule=dict(rule), detail=detail,
                              per_seed=per_seed)
