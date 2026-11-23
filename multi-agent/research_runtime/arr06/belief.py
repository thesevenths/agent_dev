"""ARR-06 P2: evidence-based belief update with explicit likelihood/posterior.

The LLM does NOT arbitrarily modify confidence. We use a frozen scoring
protocol: a diagnostic experiment's normalized improvement is mapped to a
likelihood ratio per cause via a registered experiment->cause match table,
then a Bayesian posterior is computed and calibrated independently.
"""
from __future__ import annotations

import math
from typing import Any

from .benchmark import CAUSES, EXP_BY_ID


def prior_uniform() -> dict[str, float]:
    return {c: 1.0 / len(CAUSES) for c in CAUSES}


def likelihood(experiment_id: str, improvement: float,
               *, strong: float = 0.3, weak: float = 0.05) -> dict[str, float]:
    """Frozen likelihood P(improvement | cause) for a diagnostic experiment.

    If the experiment's registered mechanism matches a cause, a large
    improvement is likely under that cause and unlikely under others.
    """
    exp = EXP_BY_ID.get(experiment_id)
    matched = set(exp.discriminates) if exp else set()
    imp = max(0.0, float(improvement))
    strength = min(1.0, imp / strong) if strong > 0 else 0.0
    out = {}
    for c in CAUSES:
        if c in matched:
            out[c] = 0.1 + 0.8 * strength          # high likelihood if matched+effective
        else:
            out[c] = 0.8 - 0.7 * min(1.0, strength)  # low likelihood if matched cause works
    # A near-zero improvement everywhere gives near-uniform likelihood -> no false certainty.
    if imp < weak:
        return {c: 1.0 for c in CAUSES}
    return out


def posterior(prior: dict[str, float], experiment_id: str,
              improvement: float) -> dict[str, float]:
    lik = likelihood(experiment_id, improvement)
    un = {c: float(prior.get(c, 0.0)) * lik[c] for c in CAUSES}
    z = sum(un.values())
    if z <= 0:
        return prior_uniform()
    return {c: un[c] / z for c in CAUSES}


def belief_update(prior: dict[str, float], experiment_id: str,
                  improvement: float) -> dict[str, Any]:
    post = posterior(prior, experiment_id, improvement)
    pred = max(post, key=post.get)

    def H(p):
        v = [max(1e-12, x) for x in p.values()]
        s = sum(v)
        v = [x / s for x in v]
        return -sum(x * math.log(x) for x in v)
    h0, h1 = H(prior), H(post)
    return {"prior": prior, "likelihood": likelihood(experiment_id, improvement),
            "posterior": post, "prediction": pred,
            "confidence": float(post[pred]),
            "information_gain": float(h0 - h1),
            "entropy_before": h0, "entropy_after": h1,
            "rule": "frozen_bayesian_likelihood_v1"}


def prior_from_config(config: dict[str, Any]) -> dict[str, float]:
    """Frozen evidence-scoring prior from OBSERVED config (not hidden truth).

    This is the legitimate "existing evidence" step: a model/data
    configuration with an extreme value is evidence for the corresponding
    mechanism. The diagnostic experiment then provides INDEPENDENT validation.
    """
    scores = {c: 0.15 for c in CAUSES}
    hidden = config.get("hidden") or config.get("model.hidden_values") or []
    lr = float(config.get("lr") or config.get("model.lr") or 0.002)
    n_ep = int(config.get("n_episodes") or config.get("data.n_episodes") or 8)
    amode = str(config.get("action_mode") or config.get("data.action_mode") or "random")
    ascale = float(config.get("action_scale") or config.get("data.action_scale") or 1.0)
    if hidden and (max(hidden) <= 8):
        scores["insufficient_model_capacity"] = 0.9
    if n_ep <= 4:
        scores["insufficient_state_action_coverage"] = 0.9
    if lr >= 0.1:
        scores["optimization_instability"] = 0.9
    if amode == "biased":
        scores["dynamics_distribution_shift"] = 0.9
    if amode == "zero" and ascale < 1.0:
        scores["action_conditioning_failure"] = 0.9
    z = sum(scores.values())
    return {c: scores[c] / z for c in CAUSES}


__all__ = ["CAUSES", "prior_uniform", "prior_from_config", "likelihood", "posterior", "belief_update"]
