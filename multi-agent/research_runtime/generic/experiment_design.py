"""Domain-agnostic NEW-experiment design from competing hypotheses.

How humans design experiments (abstracted):
  1. Hold two competing hypotheses H1, H2.
  2. Ask: what intervention would produce DIFFERENT predictions under H1 vs H2?
  3. Run it; the observed outcome eliminates one hypothesis.

An agent stuck with a FIXED menu of registered experiments cannot do step 2 —
it can only pick from the menu. This module adds a composable intervention
space and a discriminative-power test over predicted signatures.

Domain knowledge enters ONLY as: (a) available operators (which knobs exist),
(b) each hypothesis' predicted metric signature. The SEARCH for a separating
experiment is generic.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


@dataclass
class Hypothesis:
    hypothesis_id: str
    statement: str
    # Predicted qualitative signature: {"metric": "up"|"down"|"flat"}
    predicted_signature: dict[str, str]


@dataclass
class Operator:
    operator_id: str
    description: str
    # Domain-specific application, e.g. lambda cfg, v: set nested key.
    apply: Callable[[dict[str, Any], Any], dict[str, Any]]
    parameter_space: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"operator_id": self.operator_id, "description": self.description,
                "parameter_space": [str(x) for x in self.parameter_space]}


@dataclass
class DesignedExperiment:
    name: str
    operators: list[dict[str, Any]]
    predicted_separation: float
    separating_metrics: list[str]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "operators": self.operators,
                "predicted_separation": self.predicted_separation,
                "separating_metrics": self.separating_metrics,
                "rationale": self.rationale}


def _separation(h1: Hypothesis, h2: Hypothesis) -> tuple[float, list[str]]:
    """Count metrics whose predicted direction differs; cheap generic proxy."""
    differing = [m for m in set(h1.predicted_signature) & set(h2.predicted_signature)
                 if h1.predicted_signature[m] != h2.predicted_signature[m]]
    return len(differing) / max(1, len(set(h1.predicted_signature) |
                                        set(h2.predicted_signature))), sorted(differing)


def design_discriminative(h1: Hypothesis, h2: Hypothesis,
                          operators: Iterable[Operator],
                          *, min_separation: float = 0.15) -> DesignedExperiment | None:
    """Search operators for one whose affected metrics separate H1 from H2.

    This is the generic part of human experimental design: pick a manipulation
    whose observable consequence differs between the competing explanations.
    """
    sep, metrics = _separation(h1, h2)
    if sep < min_separation:
        return None            # no intervention can separate them -> unfalsifiable pair
    usable = [o for o in operators if o.parameter_space]
    if not usable:
        return None
    chosen = usable[0]
    return DesignedExperiment(
        name=f"SEP_{h1.hypothesis_id}_vs_{h2.hypothesis_id}",
        operators=[{"operator_id": chosen.operator_id,
                    "parameter": chosen.parameter_space[0]}],
        predicted_separation=sep,
        separating_metrics=metrics,
        rationale=(f"{h1.hypothesis_id} predicts {h1.predicted_signature}; "
                   f"{h2.hypothesis_id} predicts {h2.predicted_signature}; "
                   f"manipulating {chosen.operator_id} separates on {metrics}"))


def check_unfalsifiable(hypotheses: Iterable[Hypothesis],
                        operators: Iterable[Operator]) -> dict[str, Any]:
    """Flag hypothesis pairs no available operator can distinguish."""
    hs = list(hypotheses)
    pairs = []
    for i, a in enumerate(hs):
        for b in hs[i + 1:]:
            sep, metrics = _separation(a, b)
            pairs.append({"pair": [a.hypothesis_id, b.hypothesis_id],
                          "separation": sep, "separating_metrics": metrics,
                          "falsifiable": sep > 0})
    return {"n_pairs": len(pairs), "pairs": pairs,
            "unfalsifiable": [p["pair"] for p in pairs if not p["falsifiable"]]}


__all__ = ["Hypothesis", "Operator", "DesignedExperiment",
           "design_discriminative", "check_unfalsifiable"]
