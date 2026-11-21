"""M6.1 -- REAL Hypothesis Critic strategies injected into the production chain.

The M6 rule-level ``hypothesis_critic`` component NEVER touched the production
``HypothesisCritic`` used by ``ResearchService._review``. This module closes
that gap WITHOUT rewriting the service: it builds a
``hypothesis_manager_factory`` that the existing ``ResearchDependencies``
already accepts, so Base and Candidate run through the SAME service, SAME
planner, SAME campaign engine and SAME deterministic Critic class -- the ONLY
difference is the injected Critic policy.

Base policy     : production default (strict evidence grounding, structural
                  falsification presence).
Candidate policy: additionally requires that every PREDICTED METRIC is named
                  in at least one falsification criterion, so the proposed
                  experiment can actually distinguish support from refutation.
                  This is a genuine research-method rule, evaluated on REAL
                  hypotheses generated from REAL baseline evidence.

No score is predetermined. Both policies receive the same real hypotheses; a
verdict is derived from the hypothesis content + real evidence resolver only.
"""
from __future__ import annotations

import re
from typing import Any, Callable

from ..hypothesis_critic import HypothesisCritic
from ..hypothesis_manager import HypothesisManager

CRITIC_VERSION = "m6_1_critic_strategies_v1"


class _FalsifiabilityGroundedCritic(HypothesisCritic):
    """Production Critic subclass with an added, real falsifiability rule."""

    def review(self, hypothesis: Any) -> tuple[bool, list[str]]:
        passed, errors = super().review(hypothesis)
        # Only add the method rule when the structural review passed; still
        # evaluate on the REAL generated hypothesis, never a synthetic probe.
        if passed:
            predicted = {
                str(getattr(p, "metric", "") or "").strip()
                for p in (getattr(hypothesis, "predicted_effects", None) or [])
            }
            predicted = {m for m in predicted if m}
            falsification = " ".join(
                str(x) for x in (getattr(hypothesis,
                                         "falsification_criteria", None) or []))
            tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_@]*", falsification))
            uncovered = sorted(m for m in predicted
                               if m not in falsification
                               and not (set(re.findall(
                                   r"[A-Za-z_][A-Za-z0-9_@]*", m)) & tokens))
            if uncovered:
                errors.append("falsification_does_not_cover_predicted_metrics:"
                              + ",".join(uncovered))
                passed = False
        return passed, errors


def build_critic_manager_factory(
        *, strategy: str, evidence_resolver: Callable[[str], Any] | None,
        allowed_metric_patterns: list[str] | None = None,
        config: dict[str, Any] | None = None,
        proposer: Any = None
        ) -> Callable[[], HypothesisManager]:
    """Return a factory the production ResearchService can call per review.

    ``strategy`` is one of ``base`` / ``falsifiability_grounded``. Both use the
    REAL evidence resolver, so evidence claims are grounded in actual recorded
    experiment results in BOTH arms -- only the method rule differs.
    """
    cfg = dict(config or {})
    # Guard against a candidate silently weakening evidence grounding: the
    # strict gate is frozen OUTSIDE the candidate (F4-style boundary).
    strict = True

    def factory() -> HypothesisManager:
        if strategy == "base":
            critic = HypothesisCritic(
                allowed_metric_patterns=allowed_metric_patterns,
                evidence_resolver=evidence_resolver, strict=strict)
        elif strategy == "falsifiability_grounded":
            critic = _FalsifiabilityGroundedCritic(
                allowed_metric_patterns=allowed_metric_patterns,
                evidence_resolver=evidence_resolver, strict=strict)
            if cfg.get("min_falsification_criteria", 0):
                critic._m61_min_criteria = int(cfg["min_falsification_criteria"])
        else:
            raise ValueError(f"unknown critic strategy: {strategy!r}")
        # Use the SAME real proposer the production service would use. Passing
        # None would silently fall back to RuleBasedProposer and drop the
        # harness-controlled planner -- that would invalidate the comparison.
        return HypothesisManager(proposer=proposer, critic=critic)

    return factory


def strategy_from_bundle(component_config: dict[str, Any]) -> str:
    """Map the frozen Harness Candidate component config to a real strategy."""
    if component_config.get("require_falsifying_observation") is True:
        return "falsifiability_grounded"
    return "base"


__all__ = ["CRITIC_VERSION", "build_critic_manager_factory",
           "strategy_from_bundle", "_FalsifiabilityGroundedCritic"]
