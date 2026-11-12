"""DOMAIN WIRING for SciNet++ Pendulum research (P1-06).

This is the ONE place allowed to name the engine's configs, dataset and metric
names for the autonomous research loop. P2 adds ``domains/robotics/`` with the
same shape; no generic module needs to change.

Thresholds here are declared BEFORE any experiment is run. Nothing in this file
is tuned from observed results.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from ..campaign.scinet import (
    SCINET_ALLOWED_CONFIG_KEYS,
    SCINET_VALUE_RULES,
)
from ..comparison.scinet import SCINET_DEFAULT_GUARDRAILS
from ..research_policy import ApprovalPolicy, ResearchDomainPolicy
from ..settings import DATA_DIR, DEFAULT_SCINET_DIR, DEFAULT_CONFIG

PAIRED_SEEDS = (42, 43, 44, 45, 46)

# The rule-based proposer needs an intervention policy injected for a
# diagnosis; this is a DOMAIN decision, not a runtime constant. It only fires
# when structured evidence actually shows horizon degradation.
SCINET_INTERVENTION_POLICY = {
    "dynamics_error_accumulation": {"train.lr": 0.0008},
}


def read_frozen_dataset_version(project_dir: str | Path | None = None) -> str:
    """Read the real content-addressed version from the frozen manifest."""
    root = Path(project_dir or DEFAULT_SCINET_DIR)
    path = root / "datasets" / "pendulum_smoke_v1" / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"frozen dataset manifest missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    version = manifest.get("dataset_version")
    if not version:
        raise ValueError(f"manifest {path} has no dataset_version")
    return str(version)


#: Single source of truth for this domain's evaluation metric and its optimization
#: direction. Consumed by the dual-layer RRSI (Package F/G): a candidate that
#: LOWERS ``latent_mse@1`` is an improvement (minimize). SciNet++ exposes no
#: ALLOWED_KEYS, so it produces no DOMAIN_MODEL candidates -- its loop runs only
#: the harness layer -- but the metric definition still lives here for reuse.
PRIMARY_METRIC = "latent_mse@1"
METRIC_DIRECTION = "minimize"


def scinet_research_policy(*, project_dir: str | Path | None = None,
                           epochs: int = 1,
                           paired_seeds: tuple[int, ...] = PAIRED_SEEDS,
                           max_campaign_rounds: int = 2,
                           report_dir: str | Path | None = None,
                           **overrides: Any) -> ResearchDomainPolicy:
    kwargs: dict[str, Any] = dict(
        domain="pendulum_world_model",
        baseline_config_path=DEFAULT_CONFIG,
        dataset_version=read_frozen_dataset_version(project_dir),
        allowed_dataset_versions=(read_frozen_dataset_version(project_dir),),
        allowed_config_keys=SCINET_ALLOWED_CONFIG_KEYS,
        value_rules=SCINET_VALUE_RULES,
        intervention_policy=copy.deepcopy(SCINET_INTERVENTION_POLICY),
        primary_metric="latent_mse@1",
        metric_direction="minimize",
        guardrails=copy.deepcopy(SCINET_DEFAULT_GUARDRAILS),
        paired_seeds=tuple(paired_seeds),
        evaluation_split="validation",
        epochs=epochs,
        max_candidates=2,
        max_seeds=5,
        max_jobs=30,
        max_total_seconds=3600.0,
        job_timeout_s=300.0,
        max_campaign_rounds=max_campaign_rounds,
        campaign_root="outputs/research_campaigns",
        report_dir=str(report_dir or (DATA_DIR / "reports")),
        approval_policy=ApprovalPolicy(allowed_sources=("human",)),
    )
    kwargs.update(overrides)
    return ResearchDomainPolicy(**kwargs)
