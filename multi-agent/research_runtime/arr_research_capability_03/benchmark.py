"""ARR-03 unified benchmark definitions.

M2 state-based track: the 10 validated independent regimes from ARR_EMB_RSI_02
(reused, not redefined). M3 Reacher track: 6 regimes whose Oracle sensitivity
was validated on the real MuJoCo simulator (see M3 scouting evidence).

Ground truth is sealed; agent-facing views never carry failure identity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .integrity import assert_no_ground_truth

CAUSES = (
    "insufficient_model_capacity",
    "insufficient_state_action_coverage",
    "dynamics_distribution_shift",
    "optimization_instability",
    "action_conditioning_failure",
)

M2_EXCLUDED = {
    "newton::E02": "Oracle inversion in real engine (4 samples beat 64)",
    "double_pendulum::E02": "Oracle inversion in real engine",
}


def m2_state_regimes() -> list[dict[str, Any]]:
    from research_runtime.arr_emb_rsi_02.benchmark import build_regimes
    return [r.to_dict() for r in build_regimes()
            if r.regime_id not in M2_EXCLUDED]


TEST_CFG = dict(n_episodes=6, max_steps=30, target_mode="mixed",
                action_mode="random", action_scale=1.0)


@dataclass(frozen=True)
class ReacherRegime:
    regime_id: str
    cause: str
    oracle_sensitivity: float
    failure_train: dict[str, Any]
    oracle_train: dict[str, Any]
    failure_model: dict[str, Any]
    oracle_model: dict[str, Any]
    test_cfg: dict[str, Any] = field(default_factory=lambda: dict(TEST_CFG))

    def sealed_truth(self) -> dict[str, Any]:
        return {"cause": self.cause,
                "oracle_sensitivity": self.oracle_sensitivity}

    def blind_view(self) -> dict[str, Any]:
        view = {
            "regime_id": "regime_" + self.regime_id.split("::")[-1],
            "objective": ("Diagnose the dominant failure of this action-conditioned "
                          "dynamics model from normalized evidence and propose one "
                          "registered intervention."),
            "budget": {"max_epochs": 160},
        }
        assert_no_ground_truth(view)
        return view


REACHER_REGIMES: tuple[ReacherRegime, ...] = (
    ReacherRegime(
        "reacher::E01", "insufficient_model_capacity", 0.9095,
        dict(n_episodes=8, max_steps=30, target_mode="mixed",
             action_mode="random", action_scale=1.0),
        dict(n_episodes=8, max_steps=30, target_mode="mixed",
             action_mode="random", action_scale=1.0),
        dict(hidden=[4], lr=0.002), dict(hidden=[64, 64], lr=0.002)),
    ReacherRegime(
        "reacher::E02", "insufficient_state_action_coverage", 0.9409,
        dict(n_episodes=2, max_steps=30, target_mode="mixed",
             action_mode="random", action_scale=1.0),
        dict(n_episodes=16, max_steps=30, target_mode="mixed",
             action_mode="random", action_scale=1.0),
        dict(hidden=[32, 32], lr=0.002), dict(hidden=[32, 32], lr=0.002)),
    ReacherRegime(
        "reacher::E03", "dynamics_distribution_shift", 0.8570,
        dict(n_episodes=8, max_steps=30, target_mode="mixed",
             action_mode="biased", action_scale=1.0),
        dict(n_episodes=8, max_steps=30, target_mode="mixed",
             action_mode="random", action_scale=1.0),
        dict(hidden=[32, 32], lr=0.002), dict(hidden=[32, 32], lr=0.002)),
    ReacherRegime(
        "reacher::E04", "dynamics_distribution_shift", 0.7347,
        dict(n_episodes=8, max_steps=30, target_mode="mixed",
             action_mode="random", action_scale=0.2),
        dict(n_episodes=8, max_steps=30, target_mode="mixed",
             action_mode="random", action_scale=1.0),
        dict(hidden=[32, 32], lr=0.002), dict(hidden=[32, 32], lr=0.002)),
    ReacherRegime(
        "reacher::E05", "optimization_instability", 0.8954,
        dict(n_episodes=8, max_steps=30, target_mode="mixed",
             action_mode="random", action_scale=1.0),
        dict(n_episodes=8, max_steps=30, target_mode="mixed",
             action_mode="random", action_scale=1.0),
        dict(hidden=[32, 32], lr=0.3), dict(hidden=[32, 32], lr=0.001)),
    ReacherRegime(
        "reacher::E06", "action_conditioning_failure", 0.9440,
        dict(n_episodes=8, max_steps=30, target_mode="mixed",
             action_mode="zero", action_scale=1.0),
        dict(n_episodes=8, max_steps=30, target_mode="mixed",
             action_mode="random", action_scale=1.0),
        dict(hidden=[32, 32], lr=0.002), dict(hidden=[32, 32], lr=0.002)),
)


def reacher_regimes() -> list[ReacherRegime]:
    return list(REACHER_REGIMES)


INTERVENTION_REGISTRY: dict[str, dict[str, Any]] = {
    "insufficient_model_capacity": {
        "id": "IV-CAP-03", "parameter": "model.hidden",
        "baseline": [4], "candidate": [64, 64],
        "expected_metric": "one_step_mse", "expected_direction": "decrease",
        "minimum_effect": 0.05, "budget": {"epochs": 160},
        "mechanism": "increase predictor capacity to represent the dynamics map"},
    "insufficient_state_action_coverage": {
        "id": "IV-COV-03", "parameter": "data.n_episodes",
        "baseline": 2, "candidate": 16,
        "expected_metric": "one_step_mse", "expected_direction": "decrease",
        "minimum_effect": 0.05, "budget": {"epochs": 160},
        "mechanism": "increase state-action trajectory coverage"},
    "dynamics_distribution_shift": {
        "id": "IV-SHIFT-03", "parameter": "data.action_distribution",
        "baseline": "biased/narrow", "candidate": "random/full",
        "expected_metric": "one_step_mse", "expected_direction": "decrease",
        "minimum_effect": 0.05, "budget": {"epochs": 160},
        "mechanism": "match training action distribution to evaluation regime"},
    "optimization_instability": {
        "id": "IV-OPT-03", "parameter": "model.lr",
        "baseline": 0.3, "candidate": 0.001,
        "expected_metric": "one_step_mse", "expected_direction": "decrease",
        "minimum_effect": 0.05, "budget": {"epochs": 160},
        "mechanism": "reduce learning rate to stabilise gradient descent"},
    "action_conditioning_failure": {
        "id": "IV-ACT-03", "parameter": "data.action_mode",
        "baseline": "zero", "candidate": "random",
        "expected_metric": "one_step_mse", "expected_direction": "decrease",
        "minimum_effect": 0.05, "budget": {"epochs": 160},
        "mechanism": "restore action signal so the model learns a_t->s_t+1"},
}

AGENT_ACTION_KEYS = (
    "model.hidden", "data.n_episodes", "model.lr",
    "data.action_distribution", "data.action_mode",
)


def intervention_to_reacher_config(
        cause: str, base_failure_train: dict[str, Any],
        base_failure_model: dict[str, Any]) -> dict[str, Any]:
    train = dict(base_failure_train)
    model = dict(base_failure_model)
    if cause == "insufficient_model_capacity":
        model["hidden"] = [64, 64]
    elif cause == "insufficient_state_action_coverage":
        train["n_episodes"] = 16
    elif cause == "dynamics_distribution_shift":
        train["action_mode"] = "random"
        train["action_scale"] = 1.0
    elif cause == "optimization_instability":
        model["lr"] = 0.001
    elif cause == "action_conditioning_failure":
        train["action_mode"] = "random"
    else:
        raise ValueError(f"unknown cause {cause!r}")
    return {"train": train, "model": model}


__all__ = ["CAUSES", "M2_EXCLUDED", "m2_state_regimes", "TEST_CFG",
           "ReacherRegime", "REACHER_REGIMES", "reacher_regimes",
           "INTERVENTION_REGISTRY", "AGENT_ACTION_KEYS",
           "intervention_to_reacher_config"]
