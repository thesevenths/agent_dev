"""M5-02 -- evolution experiment contract and promotion gate."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DOMAIN_MODEL_LAYER = "DOMAIN_MODEL"
RESEARCH_HARNESS_LAYER = "RESEARCH_HARNESS"
LAYERS = frozenset({DOMAIN_MODEL_LAYER, RESEARCH_HARNESS_LAYER})


class LayerMixError(RuntimeError):
    pass


@dataclass
class EvolutionExperiment:
    experiment_id: str
    layer: str
    candidate: dict[str, Any]
    parent_sha: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    decision: str = "INCONCLUSIVE"
    gate_passed: bool = False
    holdout_used_for_selection: bool = False

    def __post_init__(self) -> None:
        if self.layer not in LAYERS:
            raise LayerMixError(f"unknown evolution layer {self.layer!r}")
        if self.holdout_used_for_selection:
            raise LayerMixError(
                "held-out data must never be used for candidate selection")

    def to_record(self, *, candidate_sha: str,
                  status: str | None = None) -> dict[str, Any]:
        return {"experiment_id": self.experiment_id, "layer": self.layer,
                "candidate_id": str(self.candidate.get("candidate_id", "")),
                "candidate_sha": candidate_sha,
                "parent_sha": self.parent_sha,
                "status": status or self.decision,
                "gate_passed": self.gate_passed,
                "evidence": self.evidence}


def gate_allows_promotion(decision: str, *, holdout_confirmed: bool,
                          frozen_test_used: bool = False,
                          training_only: bool = False) -> bool:
    """Never lower the promotion bar: ACCEPT + independent holdout only."""
    return (str(decision).strip().upper() == "ACCEPTED"
            and bool(holdout_confirmed)
            and not frozen_test_used
            and not training_only)


def assert_single_layer(layer: str, changed_layers: set[str]) -> None:
    if layer not in LAYERS:
        raise LayerMixError(f"unknown layer {layer!r}")
    extra = set(changed_layers) - {layer}
    if extra:
        raise LayerMixError(
            f"domain model and harness may not change together: layer={layer}, "
            f"also_changed={sorted(extra)}")


__all__ = [
    "DOMAIN_MODEL_LAYER", "RESEARCH_HARNESS_LAYER", "LAYERS",
    "LayerMixError", "EvolutionExperiment", "gate_allows_promotion",
    "assert_single_layer",
]
