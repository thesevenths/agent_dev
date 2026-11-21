"""ARR 1.1 / F1 -- dual-layer RRSI: ONE unified candidate protocol.

Both evolution layers (``DOMAIN_MODEL`` and ``RESEARCH_HARNESS``) share a single
candidate LIFECYCLE and a single NAME-SPACED ledger, so a domain knob can never
be mistaken for a harness improvement (or vice versa). This package only ADDS
the lifecycle/registry abstraction and a thin coordination controller; it does
NOT rewrite the existing harness evolution or the existing per-domain config
safety boundary (``CandidateConfigBuilder``).
"""
from __future__ import annotations

from .candidate import (
    CandidateStatus, DomainModelCandidate, CandidateType,
)
from .candidate_registry import CandidateRegistry, RegistryEntry
from .harness_evolver import HARNESS_LAYER
from .domain_evolver import (
    DOMAIN_LAYER, generate_domain_candidates, materialize_domain_candidate,
)
from .evolution_controller import DualLayerController, run_dual_layer

__all__ = [
    "CandidateStatus", "DomainModelCandidate", "CandidateType",
    "CandidateRegistry", "RegistryEntry",
    "HARNESS_LAYER", "DOMAIN_LAYER", "generate_domain_candidates",
    "materialize_domain_candidate", "DualLayerController", "run_dual_layer",
]
