"""ARR-05 knowledge compatibility layer: INTERVENTION_POOL / resolve_intervention.

Track M2/M3 import from this module; the full cause-specific memory with
promotion ladder and applicability gate lives in memory.py.
"""
from .memory import (INTERVENTION_POOL, MemoryItem, applicability_gate,
                     build_memory_items, memory_digest, resolve_intervention)

__all__ = ["INTERVENTION_POOL", "MemoryItem", "resolve_intervention",
           "build_memory_items", "applicability_gate", "memory_digest"]
