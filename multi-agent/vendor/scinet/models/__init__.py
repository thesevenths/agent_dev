"""
models
======
SciNet++ (2026) model package.

Exposes the V-JEPA world model (online encoder + latent predictor + EMA target
encoder), the standalone encoder / predictor building blocks, and the physics
probe used for SciNet-style concept read-out.
"""

from .encoder import Encoder, build_mlp
from .predictor import Predictor
from .world_model import WorldModel
from .probe import Probe

__all__ = [
    "Encoder",
    "build_mlp",
    "Predictor",
    "WorldModel",
    "Probe",
]
