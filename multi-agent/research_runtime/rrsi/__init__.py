"""M5-02 -- thin dual-layer RRSI skeleton (domain model vs research harness).

Both layers reuse the existing evolution machinery. This package only adds
strict layer isolation, durable version/experiment records and a coordinator;
it does NOT implement Meta-RL or modify production code automatically.
"""
from .coordinator import RRSICoordinator
from .domain_model_evolver import DOMAIN_MODEL_LAYER
from .evolution_experiment import EvolutionExperiment, gate_allows_promotion
from .evolution_registry import EvolutionRegistry
from .harness_evolver import RESEARCH_HARNESS_LAYER
from .holdout_evidence import (
    verify_holdout_evidence, gate_allows_promotion_verified,
)

__all__ = [
    "RRSICoordinator", "EvolutionRegistry", "EvolutionExperiment",
    "gate_allows_promotion", "DOMAIN_MODEL_LAYER", "RESEARCH_HARNESS_LAYER",
    "verify_holdout_evidence", "gate_allows_promotion_verified",
]
