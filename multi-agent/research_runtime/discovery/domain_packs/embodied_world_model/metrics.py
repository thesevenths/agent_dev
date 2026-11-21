"""M7 Domain Pack (embodied): metric extraction and paired comparisons."""
from __future__ import annotations

import math
from typing import Any

METRIC_SPECS: dict[str, dict[str, Any]] = {
    "state_mse@1": {"direction": "minimize", "available": True},
    "state_mse@5": {"direction": "minimize", "available": True},
    "ood_prediction_error": {"direction": "minimize", "available": True,
                             "mapping": "state_mse@1", "split": "ood"},
    "long_horizon_stability": {"direction": "maximize", "available": True,
                               "mapping": "physics_consistency", "split": "ood"},
    "planning_performance": {"direction": "maximize", "available": True,
                             "mapping": "task_success_rate"},
    "calibration": {"direction": "maximize", "available": False},
    "compute_cost": {"direction": "minimize", "available": True,
                     "mapping": "elapsed_seconds", "source": "summary"},
}


def resolve_metric_name(metric: str) -> str:
    spec = METRIC_SPECS.get(metric, {})
    return str(spec.get("mapping", metric))


def direction(metric: str) -> str:
    return str(METRIC_SPECS.get(metric, {}).get("direction", "minimize"))


def extract(result: dict[str, Any], metric: str) -> float | None:
    name = resolve_metric_name(metric)
    spec = METRIC_SPECS.get(metric, {})
    if spec.get("available") is False:
        return None
    if spec.get("source") == "summary":
        value = result.get("elapsed_seconds")
    else:
        value = (result.get("metrics") or {}).get(name)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def compare(baseline: dict[str, Any], treatment: dict[str, Any],
            metric: str, mme: float) -> dict[str, Any]:
    b, t = extract(baseline, metric), extract(treatment, metric)
    if b is None or t is None:
        return {"ok": False, "reason": "metric_missing", "metric": metric}
    delta = t - b
    favorable = delta < 0 if direction(metric) == "minimize" else delta > 0
    magnitude = -delta if direction(metric) == "minimize" else delta
    return {"ok": True, "metric": metric, "baseline": b, "treatment": t,
            "delta": round(delta, 6),
            "magnitude": round(magnitude, 6),
            "favorable": favorable,
            "exceeds_mme": magnitude >= float(mme),
            "direction": direction(metric)}


__all__ = ["METRIC_SPECS", "resolve_metric_name", "direction", "extract",
           "compare"]
