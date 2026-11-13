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

#: Mirror embodied/quant: the dual-layer RRSI (Package F/G) reads ALLOWED_KEYS /
#: VALUE_RULES from the domain module, so ``generate_domain_candidates`` can scope
#: real SciNet++ knobs (no DOMAIN_MODEL candidates are produced if the dominant
#: failure maps to nothing whitelisted). Re-exported from campaign.scinet so the
#: single source of truth stays in one place.
ALLOWED_KEYS = SCINET_ALLOWED_CONFIG_KEYS
VALUE_RULES = SCINET_VALUE_RULES

#: The objective the autonomous loop drives through the real SciNet++ engine.
SCINET_OBJECTIVE_ID = "obj_pendulum_multistep"

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


# --------------------------------------------------------------------------- #
# Real engine runner for the dual-layer RRSI loop (M2.6)
# --------------------------------------------------------------------------- #

def scinet_compute(config: dict[str, Any], spec: dict[str, Any],
                   job_dir: Any, *, config_path: str | Path) -> dict[str, Any]:
    """REAL SciNet++ engine runner for the dual-layer loop.

    Drives the genuine SciNet++ engine through ``SciNetAdapter`` (subprocess
    JSON-RPC over ``research_api.cli``), mirroring the contract of
    ``embodied_compute`` / ``quant_compute`` so ``_build_real_domain_runner``
    treats every domain uniformly: ``runner(art) -> float`` reads the returned
    ``latent_mse@1`` from ``metrics``.

    Fail-closed by design: if the engine is unreachable (SCINET_DIR /
    SCINET_PYTHON unset, or ``research_api/cli.py`` absent) the adapter raises and
    we return ``status="failed"`` -- the loop records NaN and the candidate is
    INCONCLUSIVE. We NEVER fabricate a metric or a checkpoint.
    """
    from ..adapters import SciNetAdapter, AdapterError
    from ..settings import ConfigurationError, WAIT_TIMEOUT_S
    from ..schemas import ExperimentSpec

    stage = spec.get("stage")
    if stage not in ("train", "evaluate"):
        return {"status": "failed", "error_type": "unsupported_stage",
                "error": f"unsupported stage {stage!r}"}
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    seed = int(spec.get("seed") or int(config.get("seed", 42) or 42))
    experiment_id = f"arr_loop_{job_dir.name}"
    epochs = int((config.get("train") or {}).get("epochs", 1) or 1)

    try:
        adapter = SciNetAdapter()
    except (AdapterError, ConfigurationError) as exc:
        return {"status": "failed", "error_type": "engine_unavailable",
                "error": f"SciNet++ engine unreachable: {exc}"}

    if stage == "train":
        train_spec = ExperimentSpec(
            experiment_id=experiment_id, objective_id=SCINET_OBJECTIVE_ID,
            stage="train", config_path=str(config_path), split="train",
            seed=seed, budget={"epochs": epochs})
        try:
            jid = adapter.submit(train_spec)
            adapter.wait(jid, timeout=WAIT_TIMEOUT_S)
            res = adapter.collect_result(jid, train_spec)
        except (AdapterError, ConfigurationError) as exc:
            return {"status": "failed", "error_type": "train_error",
                    "error": f"SciNet++ train failed: {exc}"}
        if res.status != "succeeded" or not res.run_id:
            return {"status": "failed", "error_type": "train_failed",
                    "error": res.error or f"train status={res.status}"}
        (job_dir / "_scinet_run_id.txt").write_text(
            str(res.run_id), encoding="utf-8")
        ckpt = {"run_id": res.run_id, "dataset_version": res.dataset_version,
                "config_path": str(config_path), "seed": seed}
        (job_dir / "checkpoint.json").write_text(
            json.dumps(ckpt, sort_keys=True), encoding="utf-8")
        return {"status": "succeeded", "metrics": {}, "checkpoint": ckpt,
                "artifacts": ["checkpoint.json", "_scinet_run_id.txt"]}

    # ---- evaluate: reuse the trained checkpoint via its engine run_id ----
    rid_path = job_dir / "_scinet_run_id.txt"
    if not rid_path.is_file():
        return {"status": "failed", "error_type": "missing_run_id",
                "error": "no train run_id; evaluate cannot find the checkpoint"}
    run_id = rid_path.read_text(encoding="utf-8").strip()
    split = spec.get("split") or "validation"
    eval_spec = ExperimentSpec(
        experiment_id=experiment_id, objective_id=SCINET_OBJECTIVE_ID,
        stage="evaluate", config_path=str(config_path), run_id=run_id,
        split=split, seed=seed, budget={"epochs": epochs})
    try:
        jid = adapter.submit(eval_spec)
        adapter.wait(jid, timeout=WAIT_TIMEOUT_S)
        res = adapter.collect_result(jid, eval_spec)
    except (AdapterError, ConfigurationError) as exc:
        return {"status": "failed", "error_type": "eval_error",
                "error": f"SciNet++ evaluate failed: {exc}"}
    if res.status != "succeeded":
        return {"status": "failed", "error_type": "eval_failed",
                "error": res.error or f"evaluate status={res.status}"}
    return {"status": "succeeded", "metrics": dict(res.metrics),
            "checkpoint": {"run_id": run_id},
            "artifacts": ["backtest_metrics.json"]}

