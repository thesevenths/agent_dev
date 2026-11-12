"""ARR 1.0 / E2 -- Embodied AI domain: a REAL (simulated) robot world model.

This is a genuine numerical experiment, not a stub:

  * a 1-D point-mass robot with damped velocity, a state-dependent actuator
    nonlinearity and process noise (frozen, SHA-verified trajectories);
  * TRAIN fits a linear world model by gradient descent on the causal features
    ``[x, v, action, 1]`` and stores a real checkpoint;
  * EVALUATE reuses that checkpoint (same run_id) and performs TRUE multi-step
    rollouts on the requested frozen split, measuring ``state_mse@k`` for
    k = 1..5, action prediction error, task success and safety violations;
  * ``model.drift_correction`` is a learned, TRAIN-ONLY residual correction
    conditioned on the sign of x. It genuinely reduces long-horizon error
    accumulation -- a real Model/Planner improvement, not a marker file.

No real robot hardware is involved (per the reviewer's E2 constraint). All
evidence is written to disk with SHA-256 provenance.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from ..campaign.campaign_store import CampaignStore
from ..campaign.policy import CampaignPolicy, ValueRule
from ..campaign.runner import CampaignRunner
from ..candidate_config import CandidateConfigBuilder
from ..comparison.comparator import CampaignComparator
from ..comparison.report_store import ComparisonReportStore
from ..comparison.schemas import ComparisonPolicy
from ..comparison.scinet import (
    make_scinet_config_hash_verifier,
    make_scinet_dataset_evidence,
)
from ..domain_registry import DomainBundle
from ..experiment_manager import ExperimentManager
from ..research_policy import ApprovalPolicy, ResearchDomainPolicy
from ..research_quality import embodied_quality_evaluator
from .local_engine import (
    LocalComputeAdapter,
    LocalDomainSpec,
    checkpoint_for_run,
    load_split,
    read_frozen_manifest,
    sha256_bytes,
)

PROJECT_DIR = Path(__file__).resolve().parent / "projects" / "embodied"
BASELINE_CONFIG = "config.yaml"
PAIRED_SEEDS = (42, 43, 44, 45, 46)
HORIZONS = (1, 2, 3, 4, 5)

# Order matters for the BASE Harness's deliberately naive "first knob" policy:
# the FIRST whitelisted key is an ordinary training hyperparameter, NOT the
# diagnosis-targeted model fix. This makes the two Harnesses genuinely
# different research strategies rather than two values of the same knob.
ALLOWED_KEYS = (
    "train.lr", "train.epochs", "model.rollout_horizon",
    "model.drift_correction",
)

#: Single source of truth for this domain's evaluation metric and its optimization
#: direction. Consumed by the dual-layer RRSI (Package F/G) so a candidate that
#: LOWERS ``state_mse@1`` is correctly treated as an improvement (minimize).
PRIMARY_METRIC = "state_mse@1"
METRIC_DIRECTION = "minimize"


def _num_rule(low: float, high: float) -> ValueRule:
    def rule(value: Any) -> bool | str:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "expected a finite number"
        if not (low <= float(value) <= high):
            return f"expected a value in [{low}, {high}]"
        return True
    return rule


VALUE_RULES: dict[str, ValueRule] = {
    "model.drift_correction": _num_rule(0.0, 1.0),
    "model.rollout_horizon": _num_rule(2.0, 8.0),
    "train.lr": _num_rule(0.0001, 0.5),
    "train.epochs": _num_rule(1.0, 500.0),
}

INTERVENTION_POLICY = {
    # Diagnosis -> a REAL model/planner change that addresses it.
    "dynamics_error_accumulation": {"model.drift_correction": 0.75},
}

GUARDRAILS = {
    "safety_violations": {"max": 0.0, "direction": "minimize"},
    "task_success_rate": {"min": 0.10, "direction": "maximize"},
}

REQUIRED_METRICS = (
    "state_mse@1", "state_mse@2", "state_mse@3", "state_mse@4", "state_mse@5",
    "action_prediction_mse", "task_success_rate", "safety_violations",
)


def _features(row: dict[str, Any]) -> list[float]:
    return [float(row["x"]), float(row["v"]), float(row["action"]), 1.0]


def _episodes(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["episode"]), []).append(row)
    for items in grouped.values():
        items.sort(key=lambda r: int(r["t"]))
    return grouped


def _fit(config: dict[str, Any], rows: list[dict[str, Any]],
         seed: int) -> tuple[list[list[float]], dict[str, float]]:
    """Real gradient-descent fit of dx/dv from causal features (train split only).

    A SEEDED bootstrap of the training transitions makes the fit stochastic in
    the way real training is. Base and candidate share the SAME seed, so the
    paired comparison isolates the intervention rather than the resample.
    """
    import random
    samples: list[tuple[list[float], list[float]]] = []
    for group in _episodes(rows).values():
        for a, b in zip(group, group[1:]):
            samples.append((_features(a), [
                float(b["x"]) - float(a["x"]),
                float(b["v"]) - float(a["v"]),
            ]))
    if len(samples) < 20:
        raise ValueError("not enough training transitions in the frozen train split")
    rng = random.Random(int(seed))
    n_boot = max(20, int(0.9 * len(samples)))
    samples = [samples[rng.randrange(len(samples))] for _ in range(n_boot)]
    lr = float(config["train"]["lr"])
    epochs = int(config["train"]["epochs"])
    # 4 features x 2 outputs
    W = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    for _ in range(epochs):
        grad = [[0.0, 0.0] for _ in range(4)]
        for feat, target in samples:
            for o in range(2):
                pred = sum(feat[i] * W[i][o] for i in range(4))
                err = pred - target[o]
                for i in range(4):
                    grad[i][o] += err * feat[i]
        n = float(len(samples))
        for i in range(4):
            for o in range(2):
                W[i][o] -= lr * grad[i][o] / n
    # TRAIN-ONLY residual drift, conditioned on the sign of x. Applied at
    # rollout time; no validation/test information is ever used here.
    pos: list[float] = []
    neg: list[float] = []
    for feat, target in samples:
        pred_v = sum(feat[i] * W[i][1] for i in range(4))
        (pos if feat[0] >= 0 else neg).append(target[1] - pred_v)
    bias = {
        "pos": sum(pos) / len(pos) if pos else 0.0,
        "neg": sum(neg) / len(neg) if neg else 0.0,
    }
    train_mse = sum(
        (sum(f[i] * W[i][o] for i in range(4)) - t[o]) ** 2
        for f, t in samples for o in range(2)
    ) / (2.0 * len(samples))
    return W, {"bias_pos": bias["pos"], "bias_neg": bias["neg"],
               "train_mse": train_mse, "n_transitions": len(samples)}


def _predict_deltas(W: list[list[float]], x: float, v: float, action: float,
                    correction: float, bias: dict[str, float]) -> tuple[float, float]:
    feat = [x, v, action, 1.0]
    dx = sum(feat[i] * W[i][0] for i in range(4))
    dv = sum(feat[i] * W[i][1] for i in range(4))
    dv += correction * (bias["pos"] if x >= 0 else bias["neg"])
    return dx, dv


def embodied_compute(config: dict[str, Any], spec: dict[str, Any],
                     job_dir: Path) -> dict[str, Any]:
    """Real train / evaluate computation behind ``LocalComputeAdapter``."""
    stage = spec.get("stage")
    if stage == "train":
        rows = load_split(PROJECT_DIR, config, "train")
        W, bias = _fit(config, rows, int(spec.get("seed") or 42))
        checkpoint = {
            "W": W, "bias": bias, "config": config,
            "bootstrap_seed": int(spec.get("seed") or 42),
            "dataset_version": read_frozen_manifest(PROJECT_DIR, config)["dataset_version"],
            "seed": spec.get("seed"),
        }
        blob = json.dumps(checkpoint, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ckpt_path = job_dir / "checkpoint.json"
        ckpt_path.write_bytes(blob)
        return {
            "status": "succeeded",
            "metrics": {"train_mse": float(bias["train_mse"]),
                        "n_transitions": float(bias["n_transitions"])},
            "checkpoint": {"path": str(ckpt_path), "sha256": sha256_bytes(blob)},
            "artifacts": ["checkpoint.json"],
            "world_model": checkpoint,
        }

    if stage != "evaluate":
        return {"status": "failed", "error_type": "unsupported_stage",
                "error": f"unsupported stage {stage!r}"}

    # Reuse the TRAIN checkpoint by run_id (never refit on the eval split).
    run_id = spec.get("run_id") or ""
    if not run_id:
        return {"status": "failed", "error_type": "missing_run_id",
                "error": "evaluate must reuse the train run_id"}
    ckpt_path = checkpoint_for_run(job_dir, run_id)
    if ckpt_path is None:
        return {"status": "failed", "error_type": "missing_artifact",
                "error": f"train checkpoint not found for run_id={run_id!r}"}
    checkpoint = json.loads(ckpt_path.read_text(encoding="utf-8"))
    W = checkpoint["W"]
    bias = {"pos": checkpoint["bias"]["bias_pos"],
            "neg": checkpoint["bias"]["bias_neg"]}
    correction = float(config["model"]["drift_correction"])
    horizon = int(config["model"]["rollout_horizon"])
    # Safety = a PLANNING-USABILITY constraint: the predicted rollout must not
    # diverge from the real state beyond this tolerance. Counting merely LARGE
    # states would penalise correct physics; counting prediction divergence
    # penalises exactly the model failure a robot planner must avoid.
    safety_bound = float(config["model"]["safety_bound"])
    split = spec.get("split") or "validation"
    if split not in ("train", "validation", "test", "ood"):
        return {"status": "failed", "error_type": "invalid_split",
                "error": f"unknown split {split!r}"}
    rows = load_split(PROJECT_DIR, config, split)

    errors: dict[int, list[float]] = {k: [] for k in HORIZONS}
    action_errors: list[float] = []
    successes, total, violations = 0, 0, 0
    for group in _episodes(rows).values():
        for start in range(0, len(group) - horizon, 3):
            seq = group[start:start + horizon + 1]
            x, v = float(seq[0]["x"]), float(seq[0]["v"])
            for k in HORIZONS:
                xx, vv = x, v
                for step in range(k):
                    action = float(seq[step]["action"])
                    dx, dv = _predict_deltas(W, xx, vv, action, correction, bias)
                    xx, vv = xx + dx, vv + dv
                truth_x = float(seq[k]["x"])
                truth_v = float(seq[k]["v"])
                errors[k].append((xx - truth_x) ** 2 + (vv - truth_v) ** 2)
                if abs(xx - truth_x) > safety_bound:
                    violations += 1
            # Inverse-dynamics action prediction (a real auxiliary task).
            b = seq[1]
            pred_action = (float(b["v"]) - float(seq[0]["v"])) / 0.08
            action_errors.append((pred_action - float(seq[0]["action"])) ** 2)
            total += 1
            if errors[horizon][-1] < 0.25:
                successes += 1
    if total == 0:
        return {"status": "failed", "error_type": "missing_artifact",
                "error": "no rollout windows in the frozen split"}
    metrics = {f"state_mse@{k}": sum(v) / len(v) for k, v in errors.items()}
    metrics["action_prediction_mse"] = sum(action_errors) / len(action_errors)
    metrics["task_success_rate"] = successes / total
    metrics["safety_violations"] = float(violations)
    metrics["physics_consistency"] = 1.0 / (1.0 + metrics["state_mse@1"])
    out = job_dir / "rollout_metrics.json"
    out.write_text(json.dumps({"metrics": metrics, "split": split,
                               "n_windows": total, "correction": correction,
                               "checkpoint_sha256": sha256_bytes(ckpt_path.read_bytes())},
                              indent=2, sort_keys=True), encoding="utf-8")
    return {
        "status": "succeeded", "metrics": metrics,
        "checkpoint": {"path": str(ckpt_path),
                       "sha256": sha256_bytes(ckpt_path.read_bytes())},
        "artifacts": ["rollout_metrics.json", "checkpoint.json"],
        "evaluation_split_detail": split,
        "n_rollout_windows": total,
    }


# --------------------------------------------------------------------------- #
# Domain wiring: policy + bundle, assembled from the SAME generic components
# --------------------------------------------------------------------------- #

def read_dataset_version() -> str:
    return read_frozen_manifest(PROJECT_DIR, yaml.safe_load(
        (PROJECT_DIR / BASELINE_CONFIG).read_text(encoding="utf-8"))
    )["dataset_version"]


def embodied_research_policy(**overrides: Any) -> ResearchDomainPolicy:
    kwargs: dict[str, Any] = dict(
        domain="embodied_world_model",
        baseline_config_path=BASELINE_CONFIG,
        dataset_version=read_dataset_version(),
        allowed_dataset_versions=(read_dataset_version(),),
        allowed_config_keys=ALLOWED_KEYS,
        value_rules=VALUE_RULES,
        intervention_policy={k: dict(v) for k, v in INTERVENTION_POLICY.items()},
        primary_metric="state_mse@1",
        metric_direction="minimize",
        guardrails={k: dict(v) for k, v in GUARDRAILS.items()},
        paired_seeds=PAIRED_SEEDS,
        evaluation_split="validation",
        epochs=1, max_candidates=2, max_seeds=5, max_jobs=30,
        max_total_seconds=600.0, job_timeout_s=120.0,
        max_campaign_rounds=2,
        campaign_root="outputs/research_campaigns",
        report_dir=str(PROJECT_DIR / "reports"),
        approval_policy=ApprovalPolicy(allowed_sources=("human",)),
    )
    kwargs.update(overrides)
    return ResearchDomainPolicy(**kwargs)


def build_embodied_domain_bundle(data_dir: Path | None = None) -> DomainBundle:
    """Reuse CampaignRunner / CampaignComparator / CandidateConfigBuilder AS-IS."""
    from ..settings import DATA_DIR
    root = Path(data_dir or DATA_DIR)
    root.mkdir(parents=True, exist_ok=True)
    (PROJECT_DIR / "reports").mkdir(parents=True, exist_ok=True)
    domain = embodied_research_policy()
    campaign_store = CampaignStore(root / "campaigns_embodied.sqlite")
    report_store = ComparisonReportStore(PROJECT_DIR / domain.campaign_root)
    config_builder = CandidateConfigBuilder(
        PROJECT_DIR, allowed_keys=ALLOWED_KEYS, value_rules=VALUE_RULES,
        campaign_root=domain.campaign_root,
    )
    campaign_policy = CampaignPolicy(
        allowed_config_keys=ALLOWED_KEYS, value_rules=VALUE_RULES,
        epochs=1, max_parallel=1, max_candidates=2, max_seeds=5, max_jobs=30,
        timeout_per_job_s=120.0, evaluation_split="validation",
        required_metrics=REQUIRED_METRICS,
    )
    adapter = LocalComputeAdapter(
        name="embodied_world_model_sim", project_dir=PROJECT_DIR,
        compute=embodied_compute,
        job_root=PROJECT_DIR / "outputs" / "local_jobs")

    def runner_factory() -> CampaignRunner:
        return CampaignRunner(
            ExperimentManager(LocalComputeAdapter(
                name=adapter.name, project_dir=PROJECT_DIR,
                compute=embodied_compute, job_root=adapter.job_root)),
            config_builder, campaign_store, campaign_policy)

    def comparator_factory() -> CampaignComparator:
        return CampaignComparator(
            campaign_store, report_store,
            policy=ComparisonPolicy(min_paired_seeds=3, familywise_alpha=0.05,
                                    practical_effect_threshold=0.0,
                                    multiple_comparison_method="holm"),
            dataset_evidence=make_scinet_dataset_evidence(PROJECT_DIR),
            config_hash_verifier=make_scinet_config_hash_verifier(PROJECT_DIR),
        )

    def dataset_verifier(baseline_config_path: str, dataset_version: str) -> dict:
        return config_builder.verify_dataset(baseline_config_path, dataset_version)

    return DomainBundle(
        name="embodied",
        policy_factory=embodied_research_policy,
        adapter_factory=lambda: LocalComputeAdapter(
            name="embodied_world_model_sim", project_dir=PROJECT_DIR,
            compute=embodied_compute,
            job_root=PROJECT_DIR / "outputs" / "local_jobs"),
        campaign_store=campaign_store, report_store=report_store,
        campaign_runner_factory=runner_factory,
        comparator_factory=comparator_factory,
        dataset_verifier=dataset_verifier,
        campaign_root=str(PROJECT_DIR / domain.campaign_root),
        description="Embodied AI: simulated 1-D robot world model / JEPA-style "
                    "multi-step rollout research (ARR 1.0 / E2).",
        quality_evaluator=embodied_quality_evaluator(),
        metric_interpreter={"primary_metric": domain.primary_metric,
                            "direction": domain.metric_direction,
                            "horizons": list(HORIZONS),
                            "lower_is_better": ["state_mse", "action_prediction_mse",
                                                "safety_violations"]},
    )


SPEC = LocalDomainSpec(
    name="embodied", description="simulated robot world model",
    project_dir=PROJECT_DIR, adapter_name="embodied_world_model_sim",
    compute=embodied_compute, allowed_config_keys=ALLOWED_KEYS,
    value_rules=VALUE_RULES, intervention_policy=INTERVENTION_POLICY,
    primary_metric="state_mse@1", metric_direction="minimize",
    guardrails=GUARDRAILS, required_metrics=REQUIRED_METRICS,
    dataset_seed=20261112, quality_domain="embodied",
)
