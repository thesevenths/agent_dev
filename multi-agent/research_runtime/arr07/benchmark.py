"""ARR-07 benchmark: task families with BLIND vs LEAKY observation views.

BLIND view exposes ONLY experimental measurements (MSE/loss curves from
executed experiments). LEAKY view also exposes the config that encodes the
cause. The whole point of ARR-07 is to prove the loop works in BLIND mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import core

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
class Exp:
    id: str
    name: str
    discriminates: tuple[str, ...]
    overrides: dict[str, dict[str, Any]]


EXPERIMENTS = (
    Exp("EXP_CAP", "increase model capacity", ("insufficient_model_capacity",),
        {"model": {"hidden": [64, 64]}}),
    Exp("EXP_COV", "increase trajectory coverage", ("insufficient_state_action_coverage",),
        {"train": {"n_episodes": 20}}),
    Exp("EXP_DIST", "restore random action distribution", ("dynamics_distribution_shift",),
        {"train": {"action_mode": "random", "action_scale": 1.0}}),
    Exp("EXP_OPT", "stabilise learning rate", ("optimization_instability",),
        {"model": {"lr": 0.001}}),
    Exp("EXP_ACT", "restore action conditioning", ("action_conditioning_failure",),
        {"train": {"action_mode": "random"}}),
)
EXP_BY_ID = {e.id: e for e in EXPERIMENTS}


@dataclass(frozen=True)
class Task:
    task_id: str
    cause: str
    family: str
    variant: int
    train: dict[str, Any]
    model: dict[str, Any]
    test: dict[str, Any]

    def blind_observation(self, base_metrics: dict[str, float]) -> dict[str, Any]:
        """ONLY experimental measurements. No config, no cause, no setup hints."""
        v = {"task_ref": core.sha256_json(self.task_id)[:12],
             "family": self.family,
             "measurement": {k: float(x) for k, x in base_metrics.items()
                             if isinstance(x, (int, float))},
             "available_experiments": [{"id": e.id, "name": e.name}
                                        for e in EXPERIMENTS]}
        core.assert_no_leak(v)
        return v

    def leaky_observation(self, base_metrics: dict[str, float]) -> dict[str, Any]:
        """ARR-05/06 style: config INCLUDED (this is the suspected leakage)."""
        v = self.blind_observation(base_metrics)
        v["configuration"] = {"model": dict(self.model), "data": dict(self.train)}
        return v


def _task(cause: str, family: str, v: int) -> Task:
    tr = dict(n_episodes=8 + (v % 3), max_steps=30, target_mode="mixed",
              action_mode="random", action_scale=1.0)
    mo = dict(hidden=[32, 32], lr=0.002)
    if cause == "insufficient_model_capacity":
        mo["hidden"] = [4] if v % 2 == 0 else [6]
    elif cause == "insufficient_state_action_coverage":
        tr["n_episodes"] = 2 + (v % 2)
    elif cause == "dynamics_distribution_shift":
        tr["action_mode"] = "biased" if v % 2 == 0 else "zero"
    elif cause == "optimization_instability":
        mo["lr"] = 0.3 if v % 2 == 0 else 0.25
    elif cause == "action_conditioning_failure":
        tr["action_mode"] = "zero"
        tr["action_scale"] = 0.3 + 0.1 * (v % 3)
    test = HELDOUT_CFG if family in ("heldout", "replication") else TEST_CFG
    return Task(f"arr07::{cause}::{family}::v{v}", cause, family, v, tr, mo, dict(test))


def build_tasks(family: str | None = None, variants=range(30, 35)) -> list[Task]:
    fams = (family,) if family else ("train", "heldout", "replication")
    return [_task(c, f, v) for c in CAUSES for v in variants for f in fams]


def safe_id(tid: str) -> str:
    return tid.replace("::", "_")


__all__ = ["CAUSES", "TEST_CFG", "HELDOUT_CFG", "Exp", "EXPERIMENTS",
           "EXP_BY_ID", "Task", "build_tasks", "safe_id"]
