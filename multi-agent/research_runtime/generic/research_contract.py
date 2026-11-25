"""ARR-07 generic harness hardening: frozen criteria, structural sanity, search memory.

ARCHITECTURE PRINCIPLE:
- EXECUTABLE constraints & JUDGEMENTS live in code: frozen, reproducible,
  cannot be silently changed by the LLM or by prompt wording.
- RATIONALE/context goes in the PROMPT so the LLM understands WHY.
- The LLM still DECIDES finish; the harness computes the criterion and feeds
  the RESULT into that decision. Decision right stays with the LLM.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MetricGoal:
    metric: str
    direction: str
    min_relative_gain: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"metric": self.metric, "direction": self.direction,
                "min_relative_gain": self.min_relative_gain}


@dataclass(frozen=True)
class ResearchContract:
    """FROZEN before campaign. Quantified, multi-objective, auditable."""
    goals: tuple[MetricGoal, ...]
    budget: int
    max_params_per_train_sample: float = 8.0

    def to_dict(self) -> dict[str, Any]:
        return {"goals": [g.to_dict() for g in self.goals], "budget": self.budget,
                "max_params_per_train_sample": self.max_params_per_train_sample}

    def evaluate(self, baseline: dict[str, float],
                 latest: dict[str, float]) -> dict[str, Any]:
        per_goal = []
        for g in self.goals:
            b, v = baseline.get(g.metric), latest.get(g.metric)
            if b is None or v is None:
                per_goal.append({"metric": g.metric, "met": False,
                                 "reason": "missing measurement"})
                continue
            if g.direction == "min":
                gain = (b - v) / abs(b) if abs(b) > 1e-12 else 0.0
            else:
                gain = (v - b) / abs(b) if abs(b) > 1e-12 else 0.0
            per_goal.append({"metric": g.metric, "direction": g.direction,
                             "relative_gain": round(gain, 6),
                             "required": g.min_relative_gain,
                             "met": gain >= g.min_relative_gain})
        met = all(x.get("met", False) for x in per_goal)
        return {"contract_met": met, "per_goal": per_goal,
                "rule": "ALL goals must meet min_relative_gain (frozen Pareto)"}


def sanity_check(overrides: dict[str, Any], *, n_train_samples: int,
                 ratio_limit: float = 8.0) -> dict[str, Any]:
    """Reject scientifically absurd structures given a tiny training set."""
    hidden = overrides.get("model.predictor_hidden") or []
    latent = overrides.get("model.latent_dim") or 0
    params = sum(hidden) + (latent * hidden[0] if hidden and latent else 0)
    max_params = int(n_train_samples * ratio_limit)
    ok = params <= max_params
    return {"ok": ok, "estimated_params": params, "max_allowed": max_params,
            "n_train_samples": n_train_samples,
            "reason": "" if ok else f"structure ~{params} params exceeds "
                                    f"{max_params} allowed for {n_train_samples} samples"}


class SearchMemory:
    """Parameter -> measured result table (distinct from claim/evidence memory)."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.trials: list[dict[str, Any]] = []
        if self.path and self.path.is_file():
            self.trials = json.loads(self.path.read_text(encoding="utf-8"))

    def record(self, *, iteration: int, overrides: dict[str, Any],
               metrics: dict[str, float], verdict: dict[str, Any],
               engine_ok: bool) -> None:
        key = json.dumps(overrides, sort_keys=True)
        self.trials.append({"iteration": iteration, "param_key": key,
                            "overrides": overrides, "metrics": metrics,
                            "contract_met": verdict.get("contract_met", False),
                            "engine_ok": engine_ok})
        self.save()

    def seen(self, overrides: dict[str, Any]) -> bool:
        key = json.dumps(overrides, sort_keys=True)
        return any(t["param_key"] == key for t in self.trials)

    def direction_hint(self, *, metric: str) -> dict[str, Any]:
        gains = []
        for t in self.trials:
            if t["engine_ok"]:
                v = t["metrics"].get(metric)
                if v is not None:
                    gains.append((t["overrides"], v))
        if len(gains) < 2:
            return {"n": len(gains), "hint": "insufficient valid trials"}
        best = min(gains, key=lambda x: x[1])
        return {"n": len(gains), "best_overrides": best[0],
                "best_value": best[1],
                "hint": "highest-performing measured configuration so far"}

    def to_dict(self) -> dict[str, Any]:
        return {"n_trials": len(self.trials), "trials": self.trials}

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.trials, indent=2,
                                            sort_keys=True, ensure_ascii=False),
                                 encoding="utf-8")


def paired_seed_verdict(baseline_pairs: list[tuple[float, float]],
                        arm_pairs: list[tuple[float, float]],
                        *, direction: str = "min") -> dict[str, Any]:
    if len(baseline_pairs) != len(arm_pairs) or not baseline_pairs:
        return {"ok": False, "reason": "unequal or empty seed pairs"}
    gains = []
    for (b, _), (a, _) in zip(baseline_pairs, arm_pairs):
        denom = abs(b) if abs(b) > 1e-12 else 1.0
        g = (b - a) / denom if direction == "min" else (a - b) / denom
        gains.append(g)
    n = len(gains)
    mean = sum(gains) / n
    var = sum((x - mean) ** 2 for x in gains) / max(1, n - 1)
    se = math.sqrt(var / n) if n > 1 else 0.0
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    return {"ok": True, "n_seeds": n, "mean_gain": round(mean, 6),
            "ci95": [round(lo, 6), round(hi, 6)],
            "significantly_positive": lo > 0,
            "note": "normal-approx CI; use t critical for n<30 in production"}


__all__ = ["MetricGoal", "ResearchContract", "sanity_check", "SearchMemory",
           "paired_seed_verdict"]
