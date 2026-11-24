"""Frozen evidence gate: map OBSERVED measurements to a candidate experiment set.

Root cause of ARR-07 planner failure: the LLM has an absolute 'coverage' prior
and ignores measurements (V4: same choice even for final_loss=12 and 0.02).
This gate constrains the LLM to choose WITHIN the evidence-supported set.
It is a deterministic engineering rule, NOT an LLM discovery -- honestly
labelled as constrained cognition.
"""
from __future__ import annotations

from typing import Any

from .benchmark import EXPERIMENTS


def evidence_candidates(measurement: dict[str, float]) -> list[str]:
    """Deterministic signature -> supported experiment ids, from observed data only."""
    mse = float(measurement.get("one_step_mse", 0))
    tl = float(measurement.get("final_loss", 0))
    # High training loss => the model cannot optimise.
    if tl >= 3.0:
        return ["EXP_OPT", "EXP_CAP"]  # optimisation first
    # Tiny training loss but huge heldout error => action/coverage/distribution.
    if tl <= 0.05 and mse >= 8.0:
        return ["EXP_ACT", "EXP_COV", "EXP_DIST"]
    # Moderate train loss, moderate heldout => capacity or coverage.
    if 0.3 <= tl < 3.0:
        return ["EXP_CAP", "EXP_COV"]  # capacity first
    # Low error everywhere: nothing strongly supported; allow broad set.
    return [e.id for e in EXPERIMENTS]


def gated_prompt(observation: dict[str, Any], candidates: list[str]) -> dict[str, Any]:
    """Attach the frozen candidate set to the LLM prompt."""
    o = dict(observation)
    o["evidence_supported_candidates"] = candidates
    o["instruction"] = ("You MUST choose one experiment from "
                        "evidence_supported_candidates only. The FIRST candidate is "
                        "the frozen rule's primary recommendation for this measured "
                        "signature. Derive your reasoning from the measurements.")
    return o


__all__ = ["evidence_candidates", "gated_prompt"]


def gated_brain_plan(brain: Any, observation: dict[str, Any], history: list[dict]) -> tuple[str, str, list[str]]:
    """LLM cognition constrained by frozen evidence gate.

    Returns (reasoning, chosen_experiment, candidates). If the LLM picks
    outside the evidence-supported set, the choice is REJECTED and replaced
    by the gate's first candidate (logged as a constraint violation).
    """
    measurement = observation.get("measurement", {})
    candidates = evidence_candidates(measurement)
    constrained_obs = gated_prompt(observation, candidates)
    reasoning, exp = brain.plan(constrained_obs, history)
    if exp not in candidates:
        return reasoning + " [REJECTED: outside evidence set]", candidates[0], candidates
    return reasoning, exp, candidates
