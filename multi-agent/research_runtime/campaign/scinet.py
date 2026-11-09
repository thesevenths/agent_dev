"""DOMAIN WIRING for SciNet++ -- the ONLY module allowed to name engine fields.

Everything under ``campaign/`` except this file is a generic research runtime:
it knows about plans, trials, jobs and evidence, but not about a world model,
a latent dimension or a learning rate. Deployment choices live here so a
robotics simulator or a finance backtester can supply a different module
without touching the runner.
"""
from __future__ import annotations

from typing import Any

from ..adapters import SciNetAdapter
from ..candidate_config import CandidateConfigBuilder
from ..experiment_manager import ExperimentManager
from ..settings import DATA_DIR
from .campaign_store import CampaignStore
from .policy import CampaignPolicy, ValueRule
from .runner import CampaignRunner

# Exactly the fields the engine's current smoke config actually reads.
SCINET_ALLOWED_CONFIG_KEYS = (
    "model.predictor_hidden",
    "model.latent_dim",
    "train.lr",
    "train.weight_decay",
    "jepa.pred_weight",
    "jepa.ema_momentum",
)


def _finite_number_between(low: float, high: float) -> ValueRule:
    def rule(value: Any) -> bool | str:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "expected a finite number"
        number = float(value)
        if not (low < number <= high):
            return f"expected a value in ({low}, {high}]"
        return True
    return rule


def _positive_int(value: Any) -> bool | str:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return "expected a positive integer"
    return True


def _hidden_layers(value: Any) -> bool | str:
    if not isinstance(value, list) or not value:
        return "expected a non-empty list of layer widths"
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in value):
        return "all layer widths must be positive integers"
    return True


SCINET_VALUE_RULES: dict[str, ValueRule] = {
    "model.predictor_hidden": _hidden_layers,
    "model.latent_dim": _positive_int,
    "train.lr": _finite_number_between(0.0, 1.0),
    "train.weight_decay": _finite_number_between(0.0, 1.0),
    "jepa.pred_weight": _finite_number_between(0.0, 100.0),
    "jepa.ema_momentum": _finite_number_between(0.0, 1.0),
}


def scinet_campaign_policy(*, epochs: int = 1,
                           required_metrics: tuple[str, ...] = (),
                           required_artifacts: tuple[str, ...] = (),
                           **overrides: Any) -> CampaignPolicy:
    """Serial deployment: one seed at a time, validation for selection."""
    kwargs: dict[str, Any] = dict(
        allowed_config_keys=SCINET_ALLOWED_CONFIG_KEYS,
        value_rules=SCINET_VALUE_RULES,
        epochs=epochs,
        max_parallel=1,
        max_candidates=2,
        max_seeds=5,
        max_jobs=30,
        timeout_per_job_s=600.0,
        evaluation_split="validation",
        required_metrics=required_metrics,
        required_artifacts=required_artifacts,
    )
    kwargs.update(overrides)
    return CampaignPolicy(**kwargs)


def build_scinet_campaign_runner(*, db_path: str | None = None,
                                 policy: CampaignPolicy | None = None,
                                 adapter: Any = None,
                                 manager: ExperimentManager | None = None,
                                 config_builder: CandidateConfigBuilder | None = None,
                                 store: CampaignStore | None = None
                                 ) -> CampaignRunner:
    """Convenience factory for real E2E runs. Tests inject their own fakes."""
    adapter = adapter or SciNetAdapter()
    manager = manager or ExperimentManager(adapter)
    project_dir = getattr(adapter, "project_dir", None)
    builder = config_builder or CandidateConfigBuilder(
        project_dir, allowed_keys=SCINET_ALLOWED_CONFIG_KEYS,
        value_rules=SCINET_VALUE_RULES,
        campaign_root="outputs/research_campaigns",
    )
    store = store or CampaignStore(db_path or (DATA_DIR / "campaigns.sqlite"))
    return CampaignRunner(manager, builder, store,
                          policy or scinet_campaign_policy())
