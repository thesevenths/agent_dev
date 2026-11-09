"""A2/A3: INJECTED domain + approval policy (no engine names in this module).

The generic service must not know what a learning rate, a pendulum or a Pusher
is. A domain supplies config whitelists, value rules, the intervention policy
used by the proposer, dataset identity, metric protocol and the budget. A
robotics or finance domain replaces ONE factory, not the runtime.

Boundary note: this file is policy DATA and pure predicates. It never executes a
campaign, never stores state and never calls an LLM.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .candidate_config import DEFAULT_FORBIDDEN_ROOTS
from .schemas import CONFIG_KEY_RE, RESERVED_FINAL_SPLITS

ValueRule = Callable[[Any], "bool | str"]


@dataclass(frozen=True)
class ApprovalPolicy:
    """Which approvals may authorise an execution (server-side, not LLM input)."""
    allowed_sources: tuple[str, ...] = ("human",)
    required_scope: str = "run_campaign"
    resume_scope: str = "resume_campaign"
    require_approved_by: bool = True


@dataclass(frozen=True)
class ResearchDomainPolicy:
    domain: str
    baseline_config_path: str
    dataset_version: str
    allowed_dataset_versions: tuple[str, ...]
    allowed_config_keys: tuple[str, ...]
    value_rules: Mapping[str, ValueRule] = field(default_factory=dict)
    forbidden_roots: frozenset[str] = DEFAULT_FORBIDDEN_ROOTS
    intervention_policy: Mapping[str, dict[str, Any]] = field(default_factory=dict)
    primary_metric: str = ""
    metric_direction: str = "minimize"
    guardrails: dict[str, dict[str, Any]] = field(default_factory=dict)
    paired_seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
    evaluation_split: str = "validation"
    epochs: int = 1
    max_candidates: int = 2
    max_seeds: int = 5
    max_jobs: int = 30
    max_total_seconds: float = 3600.0
    job_timeout_s: float = 300.0
    max_campaign_rounds: int = 2
    campaign_root: str = "outputs/research_campaigns"
    approval_policy: ApprovalPolicy = field(default_factory=ApprovalPolicy)
    reserved_final_splits: frozenset[str] = RESERVED_FINAL_SPLITS
    report_dir: str = ""

    @staticmethod
    def expected_jobs(n_candidates: int, n_seeds: int) -> int:
        """Every trial is exactly one train + one evaluation job."""
        return (1 + n_candidates) * n_seeds * 2

    def override_error(self, key: Any, value: Any) -> str | None:
        """One config override's preflight verdict. None means allowed."""
        if not isinstance(key, str) or not CONFIG_KEY_RE.match(key):
            return f"invalid dotted config path {key!r}"
        root = key.split(".")[0]
        if root in self.forbidden_roots:
            return (f"{key!r} is under protected root {root!r}; data, outputs, "
                    "device, experiment identity and the RNG root are immutable")
        if key not in self.allowed_config_keys:
            return f"{key!r} is outside the injected config whitelist"
        if isinstance(value, bool):
            return f"{key!r}: bool is not a controlled numeric/structured value"
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            return f"{key!r}: value must be finite"
        rule = self.value_rules.get(key)
        if rule is not None:
            verdict = rule(value)
            if verdict is False:
                return f"{key}={value!r} violates its injected value rule"
            if isinstance(verdict, str) and verdict:
                return f"{key}={value!r} violates its injected value rule: {verdict}"
        return None
