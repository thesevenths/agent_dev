"""Engine adapters: the ONLY place that knows how to talk to a model backend."""

from .base import ModelAdapter
from .scinet_adapter import AdapterError, SciNetAdapter

__all__ = ["ModelAdapter", "SciNetAdapter", "AdapterError"]
