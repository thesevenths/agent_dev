"""ARR-04 benchmark: identifiable failure mechanisms + discriminative experiments.

Repairs ARR-03's E01/E03 information-theoretic ambiguity: each cause has a
DISTINCT observable signature (train-vs-heldout gap, lr scale, action-mode
signature, capacity signal), and each has a registered discriminative
experiment whose real-engine result reduces hypothesis entropy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .core import assert_no_leak

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


@dataclass(frozen=True)
class DiagnosticExperiment:
    experiment_id: str
    description: str
    discriminates: tuple[str, ...]
    overrides: dict[str, dict[str, Any]]


DIAGNOSTIC_EXPERIMENTS = (
    DiagnosticExperiment("EXP_CAP", "raise capacity hidden to [64,64]",
                         ("insufficient_model_capacity",),
                         {"model": {"hidden": [64, 64]}}),
    DiagnosticExperiment("EXP_COV", "raise trajectory coverage n_episodes to 20",
                         ("insufficient_state_action_coverage",),
                         {"train": {"n_episodes": 20}}),
    DiagnosticExperiment("EXP_DIST", "restore full random action distribution",
                         ("dynamics_distribution_shift",),
                         {"train": {"action_mode": "random", "action_scale": 1.0}}),
    DiagnosticExperiment("EXP_OPT", "stabilise optimisation lr to 0.001",
                         ("optimization_instability",),
                         {"model": {"lr": 0.001}}),
    DiagnosticExperiment("EXP_ACT", "restore action conditioning (random action)",
                         ("action_conditioning_failure",),
                         {"train": {"action_mode": "random"}}),
    DiagnosticExperiment("EXP_PLACEBO", "no-op control (identical config, new seed tag)",
                         (), {}),
)
EXP_BY_ID = {e.experiment_id: e for e in DIAGNOSTIC_EXPERIMENTS}


@dataclass(frozen=True)
class Task:
    task_id: str
    cause: str
    split: str
    failure_train: dict[str, Any]
    failure_model: dict[str, Any]
    test_cfg: dict[str, Any]
    expected_signature: dict[str, str]

    def sealed_truth(self) -> dict[str, Any]:
        return {"cause": self.cause, "split": self.split}

    def blind_view(self) -> dict[str, Any]:
        # Opaque id: the raw task_id encodes the cause -> never expose it.
        opaque = "task_" + __import__("hashlib").sha256(
            self.task_id.encode()).hexdigest()[:12]
        view = {
            "task_id": opaque, "split": self.split,
            "objective": ("Identify the dominant failure mechanism from normalized "
                          "evidence; you may run ONE discriminative experiment "
                          "before final diagnosis."),
            "available_experiments": [e.experiment_id for e in DIAGNOSTIC_EXPERIMENTS],
            "budget": {"max_diagnostic_experiments": 1, "max_epochs": 80},
        }
        assert_no_leak(view)
        return view


def _task(cause: str, split: str, idx: int) -> Task:
    base_train = dict(n_episodes=8, max_steps=30, target_mode="mixed",
                      action_mode="random", action_scale=1.0)
    base_model = dict(hidden=[32, 32], lr=0.002)
    sig: dict[str, str] = {}
    if cause == "insufficient_model_capacity":
        base_model["hidden"] = [4]
        sig = {"train_loss": "high", "heldout_gap": "low", "lr_scale": "normal"}
    elif cause == "insufficient_state_action_coverage":
        base_train["n_episodes"] = 2
        sig = {"train_loss": "low", "heldout_gap": "high", "lr_scale": "normal"}
    elif cause == "dynamics_distribution_shift":
        base_train["action_mode"] = "biased"
        sig = {"train_loss": "moderate", "heldout_gap": "high", "lr_scale": "normal",
               "action_signature": "biased"}
    elif cause == "optimization_instability":
        base_model["lr"] = 0.3
        sig = {"train_loss": "very_high", "heldout_gap": "moderate",
               "lr_scale": "very_high"}
    elif cause == "action_conditioning_failure":
        base_train["action_mode"] = "zero"
        sig = {"train_loss": "moderate", "heldout_gap": "high",
               "action_signature": "zero"}
    else:
        raise ValueError(cause)
    test = HELDOUT_CFG if split == "ood" else TEST_CFG
    return Task(f"{cause}__{split}__v{idx}", cause, split,
                base_train, base_model, dict(test), sig)


def build_tasks() -> list[Task]:
    tasks: list[Task] = []
    splits = ("train", "dev", "heldout", "ood")
    for cause in CAUSES:
        for i, split in enumerate(splits):
            tasks.append(_task(cause, split, i))
    return tasks


def tasks_for_split(split: str) -> list[Task]:
    return [t for t in build_tasks() if t.split == split]


def agent_config_view(task: Task) -> dict[str, Any]:
    """Neutral config view; includes real observed config values as evidence."""
    return {
        "model.hidden_width": len(task.failure_model.get("hidden", [])),
        "model.hidden_values": task.failure_model.get("hidden"),
        "model.lr": task.failure_model.get("lr"),
        "data.n_episodes": task.failure_train.get("n_episodes"),
        "data.action_mode": task.failure_train.get("action_mode"),
        "data.action_scale": task.failure_train.get("action_scale"),
        "evaluation_protocol": "held-out Reacher target distribution",
    }


__all__ = ["CAUSES", "TEST_CFG", "HELDOUT_CFG", "DiagnosticExperiment",
           "DIAGNOSTIC_EXPERIMENTS", "EXP_BY_ID", "Task", "build_tasks",
           "tasks_for_split", "agent_config_view"]
