"""P1-04: injected execution policy -- the ONLY place domain rules may live.

The generic campaign runtime must not know about any domain's config field
names, metric names or hardware limits. Those decisions are injected here by
the domain wiring (for this repository: ``campaign/scinet.py``). Policy is a
deployment constraint, not a protocol.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..schemas import RESERVED_FINAL_SPLITS

ValueRule = Callable[[Any], "bool | str"]


@dataclass(frozen=True)
class CampaignPolicy:
    # Config safety (shared with CandidateConfigBuilder, injected by the caller)
    allowed_config_keys: tuple[str, ...] = ()
    value_rules: Mapping[str, ValueRule] = field(default_factory=dict)
    config_seed_key: str = "seed"

    # Execution shape
    epochs: int = 1
    max_parallel: int = 1
    max_candidates: int = 2
    max_seeds: int = 5
    max_jobs: int = 30
    timeout_per_job_s: float = 3600.0
    continue_on_trial_failure: bool = True

    # Engine-agnostic stage/split vocabulary
    train_stage: str = "train"
    evaluate_stage: str = "evaluate"
    train_split: str = "train"
    evaluation_split: str = "validation"
    reserved_final_splits: frozenset[str] = RESERVED_FINAL_SPLITS

    # What a successful validation result must contain
    required_metrics: tuple[str, ...] = ()
    required_artifacts: tuple[str, ...] = ()

    def snapshot(self) -> dict[str, Any]:
        """A JSON-safe, auditable record of the policy actually used."""
        return {
            "allowed_config_keys": sorted(self.allowed_config_keys),
            "config_seed_key": self.config_seed_key,
            "epochs": self.epochs,
            "max_parallel": self.max_parallel,
            "max_candidates": self.max_candidates,
            "max_seeds": self.max_seeds,
            "max_jobs": self.max_jobs,
            "timeout_per_job_s": self.timeout_per_job_s,
            "continue_on_trial_failure": self.continue_on_trial_failure,
            "train_stage": self.train_stage,
            "evaluate_stage": self.evaluate_stage,
            "train_split": self.train_split,
            "evaluation_split": self.evaluation_split,
            "reserved_final_splits": sorted(self.reserved_final_splits),
            "required_metrics": list(self.required_metrics),
            "required_artifacts": list(self.required_artifacts),
            # Rules are callables: record their names, never their code.
            "value_rule_keys": sorted(str(k) for k in self.value_rules),
        }

    def expected_jobs(self, n_candidates: int, n_seeds: int) -> int:
        """Every trial is exactly one train + one validation job."""
        return (1 + n_candidates) * n_seeds * 2
