"""M7 -- generic Domain Pack contract.

The generic discovery layer only knows this protocol. Domain-specific dataset
layouts, config knobs, engine calls, metrics and safety rules stay inside the
pack, so adding a new science domain never modifies discovery modules.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DomainPack(Protocol):
    name: str

    def literature_input(self, *, question: str, concepts: list[str],
                         target_claims: list[str]) -> Any: ...

    def dataset_identity(self) -> dict[str, Any]: ...

    def make_configs(self, out_dir: str | Path) -> dict[str, str]: ...

    def run_experiment(self, *, experiment_id: str, arms: list[str],
                       seeds: list[int], splits: list[str],
                       out_dir: str | Path) -> dict[str, Any]: ...

    def evaluate(self, *, results: dict[str, Any], primary_metric: str,
                 mme: float, direction: str) -> dict[str, Any]: ...

    def safety_check(self, *, results: dict[str, Any]) -> dict[str, Any]: ...


def load_pack(name: str) -> DomainPack:
    if name == "embodied_world_model":
        from .embodied_world_model import EmbodiedWorldModelPack
        return EmbodiedWorldModelPack()
    raise KeyError(f"unknown M7 domain pack: {name}")
