"""ARR-05 benchmark: fresh identifiable task variants (NOT ARR-04 task IDs).

5 root causes, distinct signatures, registered discriminative experiments.
Task variants use v5+ indices and arr05 prefix so they never collide with
ARR-04 formal evaluation tasks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .core import assert_no_leak, BENCHMARK_VERSION

CAUSES = (
    "insufficient_model_capacity",
    "insufficient_state_action_coverage",
    "dynamics_distribution_shift",
    "optimization_instability",
    "action_conditioning_failure",
)

TEST_CFG = dict(n_episodes=8, max_steps=30, target_mode="mixed",
                action_mode="random", action_scale=1.0)
HELDOUT_CFG = dict(n_episodes=8, max_steps=30, target_mode="far",
                   action_mode="random", action_scale=1.0)
OOD_CFG = dict(n_episodes=8, max_steps=30, target_mode="far",
               action_mode="biased", action_scale=0.5)


@dataclass(frozen=True)
class DiagnosticExperiment:
    experiment_id: str
    description: str
    discriminates: tuple[str, ...]
    overrides: dict[str, dict[str, Any]]


DIAGNOSTIC_EXPERIMENTS = (
    DiagnosticExperiment("EXP_CAP", "raise predictor capacity to [64,64]",
                         ("insufficient_model_capacity",), {"model": {"hidden": [64, 64]}}),
    DiagnosticExperiment("EXP_COV", "increase trajectory coverage n_episodes",
                         ("insufficient_state_action_coverage",), {"train": {"n_episodes": 20}}),
    DiagnosticExperiment("EXP_DIST", "restore random action distribution",
                         ("dynamics_distribution_shift",),
                         {"train": {"action_mode": "random", "action_scale": 1.0}}),
    DiagnosticExperiment("EXP_OPT", "stabilise learning rate to 0.001",
                         ("optimization_instability",), {"model": {"lr": 0.001}}),
    DiagnosticExperiment("EXP_ACT", "restore action conditioning",
                         ("action_conditioning_failure",), {"train": {"action_mode": "random"}}),
    DiagnosticExperiment("EXP_PLACEBO", "no-op control", (), {}),
)
EXP_BY_ID = {e.experiment_id: e for e in DIAGNOSTIC_EXPERIMENTS}


@dataclass(frozen=True)
class Task:
    task_id: str
    cause: str
    split: str        # calibration / heldout / ood
    variant: int
    failure_train: dict[str, Any]
    failure_model: dict[str, Any]
    test_cfg: dict[str, Any]
    expected_signature: dict[str, str]

    def sealed_truth(self) -> dict[str, Any]:
        return {"cause": self.cause, "split": self.split}

    def blind_view(self) -> dict[str, Any]:
        opaque = "task_" + __import__("hashlib").sha256(
            self.task_id.encode()).hexdigest()[:12]
        view = {"task_id": opaque, "split": self.split,
                "objective": "Identify the dominant failure mechanism from normalized "
                             "evidence; you may run ONE discriminative experiment.",
                "available_experiments": [e.experiment_id for e in DIAGNOSTIC_EXPERIMENTS],
                "benchmark_version": BENCHMARK_VERSION}
        assert_no_leak(view)
        return view


def _task(cause: str, split: str, variant: int) -> Task:
    # Variant-dependent but mechanism-preserving perturbations (fresh task IDs).
    rng_seed = 1000 + variant * 7
    base_train = dict(n_episodes=8 + (variant % 3), max_steps=30,
                      target_mode="mixed", action_mode="random", action_scale=1.0)
    base_model = dict(hidden=[32, 32], lr=0.002)
    sig: dict[str, str] = {}
    if cause == "insufficient_model_capacity":
        base_model["hidden"] = [4] if variant % 2 == 0 else [6]
        sig = {"train_loss": "high", "heldout_gap": "low", "lr_scale": "normal"}
    elif cause == "insufficient_state_action_coverage":
        base_train["n_episodes"] = 2 + (variant % 2)
        sig = {"train_loss": "low", "heldout_gap": "high", "lr_scale": "normal"}
    elif cause == "dynamics_distribution_shift":
        base_train["action_mode"] = "biased" if variant % 2 == 0 else "zero"
        if base_train["action_mode"] == "zero":
            sig = {"train_loss": "moderate", "heldout_gap": "high", "action_signature": "zero"}
        else:
            sig = {"train_loss": "moderate", "heldout_gap": "high", "action_signature": "biased"}
    elif cause == "optimization_instability":
        base_model["lr"] = 0.3 if variant % 2 == 0 else 0.2
        sig = {"train_loss": "very_high", "heldout_gap": "moderate", "lr_scale": "very_high"}
    elif cause == "action_conditioning_failure":
        base_train["action_mode"] = "zero"
        base_train["action_scale"] = 0.3 + 0.1 * (variant % 3)
        sig = {"train_loss": "moderate", "heldout_gap": "high", "action_signature": "zero"}
    else:
        raise ValueError(cause)
    cfg = {"calibration": TEST_CFG, "heldout": HELDOUT_CFG, "ood": OOD_CFG}[split]
    return Task(f"arr05::{cause}::{split}::v{variant}", cause, split, variant,
                base_train, base_model, dict(cfg), sig)


def build_tasks(split: str | None = None, n_variants: int = 26) -> list[Task]:
    """Fresh variants; ARR-04 used v0-v3 under different task-id scheme."""
    out = []
    for cause in CAUSES:
        for v in range(5, 5 + n_variants):  # v5..v30 -> never ARR-04 ids
            for sp in (("calibration",) if split == "calibration" else
                       ("heldout",) if split == "heldout" else
                       ("ood",) if split == "ood" else
                       ("calibration", "heldout", "ood")):
                out.append(_task(cause, sp, v))
    return out


def agent_config_view(task: Task) -> dict[str, Any]:
    return {"model.hidden_width": len(task.failure_model.get("hidden", [])),
            "model.hidden_values": task.failure_model.get("hidden"),
            "model.lr": task.failure_model.get("lr"),
            "data.n_episodes": task.failure_train.get("n_episodes"),
            "data.action_mode": task.failure_train.get("action_mode"),
            "data.action_scale": task.failure_train.get("action_scale"),
            "evaluation_protocol": task.split}


__all__ = ["CAUSES", "TEST_CFG", "HELDOUT_CFG", "OOD_CFG", "DiagnosticExperiment",
           "DIAGNOSTIC_EXPERIMENTS", "EXP_BY_ID", "Task", "build_tasks",
           "agent_config_view", "BENCHMARK_VERSION"]
