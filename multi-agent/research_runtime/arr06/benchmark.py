"""ARR-06 benchmark: independent task families (not ARR-05 variants)."""
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
class Exp:
    id: str
    discriminates: tuple[str, ...]
    overrides: dict[str, dict[str, Any]]


EXPERIMENTS = (
    Exp("EXP_CAP", ("insufficient_model_capacity",), {"model": {"hidden": [64, 64]}}),
    Exp("EXP_COV", ("insufficient_state_action_coverage",), {"train": {"n_episodes": 20}}),
    Exp("EXP_DIST", ("dynamics_distribution_shift",),
        {"train": {"action_mode": "random", "action_scale": 1.0}}),
    Exp("EXP_OPT", ("optimization_instability",), {"model": {"lr": 0.001}}),
    Exp("EXP_ACT", ("action_conditioning_failure",), {"train": {"action_mode": "random"}}),
    Exp("EXP_PLACEBO", (), {}),
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

    def blind_view(self) -> dict[str, Any]:
        opaque = "task_" + __import__("hashlib").sha256(self.task_id.encode()).hexdigest()[:12]
        v = {"task_id": opaque, "family": self.family,
             "experiments": [e.id for e in EXPERIMENTS],
             "benchmark_version": BENCHMARK_VERSION}
        assert_no_leak(v)
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
    test = OOD_CFG if family == "ood" else HELDOUT_CFG if family in ("heldout", "replication") else TEST_CFG
    return Task(f"arr06::{cause}::{family}::v{v}", cause, family, v, tr, mo, dict(test))


def build_tasks(family: str | None = None, variants=range(20, 28)) -> list[Task]:
    fams = (family,) if family else ("train", "heldout", "replication", "ood")
    return [_task(c, f, v) for c in CAUSES for v in variants for f in fams]


def safe_id(task_id: str) -> str:
    return task_id.replace("::", "_")


__all__ = ["CAUSES", "TEST_CFG", "HELDOUT_CFG", "OOD_CFG", "Exp", "EXPERIMENTS",
           "EXP_BY_ID", "Task", "build_tasks", "safe_id"]
