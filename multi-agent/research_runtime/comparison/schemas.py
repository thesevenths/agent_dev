"""P1-05: comparison-layer protocol, kept SEPARATE from the campaign layer.

The campaign layer records experiment FACTS. This module defines the objects
used to EVALUATE those facts: paired observations, missing pairs, guardrail
verdicts, the injected decision policy, and the one report subclass allowed
to describe a comparison with zero valid pairs.

Nothing here names a metric, a model, a dataset or an engine. Metric names and
guardrail thresholds arrive in ``ExperimentPlan``; the minimum seed count and
the minimum practical effect arrive in ``ComparisonPolicy`` (both injected).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from ..schemas import ComparisonReport

Decision = Literal["accepted_for_confirmation", "rejected", "inconclusive"]

# Changing the decision algorithm MUST change this string. It is part of the
# deterministic comparison_id, so a new algorithm version can never overwrite
# (or be mistaken for) a report produced by an older algorithm.
COMPARISON_ALGORITHM_VERSION = "1.0.0"


class PairedObservation(BaseModel):
    """One seed with BOTH arms present, valid and traceable."""
    seed: int
    baseline_value: float
    candidate_value: float
    delta: float                      # convention: positive = improvement
    baseline_trial_id: str
    candidate_trial_id: str
    baseline_train_job_id: str | None = None
    baseline_evaluation_job_id: str | None = None
    baseline_train_run_id: str | None = None
    candidate_train_job_id: str | None = None
    candidate_evaluation_job_id: str | None = None
    candidate_train_run_id: str | None = None
    baseline_config_sha256: str = ""
    candidate_config_sha256: str = ""
    dataset_version: str = ""
    evaluation_split: str = ""
    # Guardrail metrics are copied verbatim so guardrail evaluation never has
    # to reach back into a mutable campaign record.
    baseline_guardrail_metrics: dict[str, float] = Field(default_factory=dict)
    candidate_guardrail_metrics: dict[str, float] = Field(default_factory=dict)


class MissingPair(BaseModel):
    """A planned seed that cannot be compared -- with an explicit reason."""
    seed: int
    arm: Literal["baseline", "candidate", "both"]
    reason: str


class GuardrailResult(BaseModel):
    """Verdict for ONE guardrail metric across ALL valid paired seeds."""
    metric: str
    status: Literal["passed", "violated", "insufficient", "invalid_rule"]
    rule: dict[str, Any] = Field(default_factory=dict)
    detail: str = ""
    per_seed: list[dict[str, Any]] = Field(default_factory=list)


class GuardrailAssessment(BaseModel):
    """Aggregate of every guardrail rule, used by the decision rules."""
    results: list[GuardrailResult] = Field(default_factory=list)

    @property
    def has_hard_violation(self) -> bool:
        return any(r.status == "violated" for r in self.results)

    @property
    def insufficient(self) -> bool:
        return any(r.status in ("insufficient", "invalid_rule")
                   for r in self.results)

    @property
    def all_passed(self) -> bool:
        return all(r.status == "passed" for r in self.results)

    @property
    def violations(self) -> list[str]:
        return [f"{r.metric}: {r.status}"
                + (f" ({r.detail})" if r.detail else "")
                for r in self.results if r.status != "passed"]


class PairedComparisonReport(ComparisonReport):
    """Backward-compatible extension able to describe ZERO valid pairs.

    The P1-01 base class insists on non-empty arrays. A campaign in which every
    paired seed failed still deserves an auditable report -- but it must never
    be able to claim ``accepted_for_confirmation``.
    """
    @model_validator(mode="after")
    def validate_pairs(self) -> "PairedComparisonReport":
        count = len(self.baseline_values)
        if (len(self.candidate_values) != count
                or len(self.paired_deltas) != count):
            raise ValueError(
                "baseline_values / candidate_values / paired_deltas must be "
                "of equal length"
            )
        if count == 0 and self.decision != "inconclusive":
            raise ValueError(
                "a comparison with zero valid pairs must be inconclusive"
            )
        for value in (self.mean_delta, self.median_delta):
            if not math.isfinite(float(value)):
                raise ValueError("mean_delta / median_delta must be finite")
        return self


@dataclass(frozen=True)
class ComparisonPolicy:
    """Injected DECISION policy. No metric name or domain threshold here.

    ``min_paired_seeds`` is an engineering gate, not a statistical law: with
    fewer paired seeds the report still shows the numbers, but the decision is
    capped at ``inconclusive``. ``practical_effect_threshold`` is in the
    primary metric's own units (absolute) or a fraction of the baseline mean
    (relative); 0.0 means "require a statistically significant improvement",
    which is the honest default until a domain expert supplies a real minimum
    detectable effect.
    """
    min_paired_seeds: int = 5
    familywise_alpha: float = 0.05
    practical_effect_threshold: float = 0.0
    practical_effect_mode: Literal["absolute", "relative"] = "absolute"
    multiple_comparison_method: Literal["holm", "none"] = "holm"

    def __post_init__(self) -> None:
        if self.min_paired_seeds < 2:
            raise ValueError("min_paired_seeds must be >= 2")
        if not (0.0 < self.familywise_alpha < 1.0):
            raise ValueError("familywise_alpha must be in (0, 1)")
        if self.practical_effect_threshold < 0.0:
            raise ValueError("practical_effect_threshold must be >= 0")
        if self.practical_effect_mode not in ("absolute", "relative"):
            raise ValueError("practical_effect_mode must be absolute/relative")
        if self.multiple_comparison_method not in ("holm", "none"):
            raise ValueError("multiple_comparison_method must be holm/none")

    def snapshot(self) -> dict[str, Any]:
        return {
            "min_paired_seeds": self.min_paired_seeds,
            "familywise_alpha": self.familywise_alpha,
            "practical_effect_threshold": self.practical_effect_threshold,
            "practical_effect_mode": self.practical_effect_mode,
            "multiple_comparison_method": self.multiple_comparison_method,
            "comparison_algorithm_version": COMPARISON_ALGORITHM_VERSION,
        }

    def effective_threshold(self, baseline_values: list[float]) -> tuple[float, str]:
        """Resolve the practical-effect threshold for one comparison."""
        if self.practical_effect_mode == "absolute":
            return self.practical_effect_threshold, "absolute"
        if baseline_values and sum(baseline_values) / len(baseline_values) > 0:
            mean = sum(baseline_values) / len(baseline_values)
            return self.practical_effect_threshold * mean, "relative_to_baseline_mean"
        return 0.0, "relative_undefined_nonpositive_baseline"
