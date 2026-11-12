"""ARR 1.1 / F1 -- harness evolution layer (REUSE, do not rewrite).

This is a thin facade over the existing, proven
``adaptive_runtime.harness_evolution`` module. Package F explicitly forbids
rewriting the existing harness evolution; this module merely exposes it under a
uniform layer name so the dual-layer controller can drive both layers through
one interface.
"""
from __future__ import annotations

from typing import Any

from adaptive_runtime import harness_evolution as _he

#: The canonical layer tag for Harness evolution.
HARNESS_LAYER = "research_harness"

__all__ = [
    "HARNESS_LAYER",
    "evolve", "evaluate_candidate", "generate_harness_candidates",
    "promote_harness", "degenerate_control_candidate", "HarnessCandidate",
    "EvolutionRecord", "apply_harness_patch",
]

# Re-export the existing, battle-tested harness machinery unchanged.
evolve = _he.evolve
evaluate_candidate = _he.evaluate_candidate
generate_harness_candidates = _he.generate_harness_candidates
promote_harness = _he.promote_harness
degenerate_control_candidate = _he.degenerate_control_candidate
apply_harness_patch = _he.apply_harness_patch
HarnessCandidate = _he.HarnessCandidate
EvolutionRecord = _he.EvolutionRecord


def layer_name() -> str:
    return HARNESS_LAYER
