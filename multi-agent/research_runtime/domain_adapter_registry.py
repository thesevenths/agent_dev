"""M5-01 -- domain adapter contract and registry (THIN, engine-agnostic).

The scheduler never imports a domain metric, dataset or config key. A domain
declares its capabilities and supplies the seven contract callbacks below;
routing only compares capability sets. Domain-specific facts stay inside the
adapter module that owns them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from .discovery_context_provider import DiscoveryContextProvider

CAPABILITY_RESEARCH_CAMPAIGN = "research_campaign"
CAPABILITY_PAIRED_SEEDS = "paired_seeds"
CAPABILITY_SIMULATION = "simulation"
CAPABILITY_BACKTEST = "backtest"
CAPABILITY_VALIDATION_SPLIT = "validation_split"
CAPABILITY_REAL_ENGINE = "real_engine"
CAPABILITY_SYNTHETIC_DATA = "synthetic_data"


@dataclass(frozen=True)
class DomainAdapterContract:
    domain_id: str
    capabilities: tuple[str, ...]
    preflight: Callable[[], dict[str, Any]]
    build_snapshot: Callable[[str, list[str] | None], Any]
    run_campaign: Callable[[Any, str, str], dict[str, Any]]
    collect_result: Callable[[Any, str], dict[str, Any]]
    resource_estimate: Callable[[], dict[str, Any]]
    health_check: Callable[[], dict[str, Any]]
    description: str = ""

    def supports(self, required: list[str] | tuple[str, ...]) -> bool:
        return set(required or ()).issubset(set(self.capabilities))

    def to_dict(self) -> dict[str, Any]:
        return {"domain_id": self.domain_id,
                "capabilities": sorted(self.capabilities),
                "description": self.description,
                "resource_estimate": self.resource_estimate()}


class DomainAdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, DomainAdapterContract] = {}

    def register(self, adapter: DomainAdapterContract) -> None:
        if not adapter.domain_id:
            raise ValueError("domain_id is required")
        if adapter.domain_id in self._adapters:
            raise ValueError(f"domain adapter already registered: {adapter.domain_id}")
        self._adapters[adapter.domain_id] = adapter

    def get(self, domain_id: str) -> DomainAdapterContract:
        try:
            return self._adapters[domain_id]
        except KeyError as exc:
            raise KeyError(
                f"unknown domain {domain_id!r}; registered="
                f"{sorted(self._adapters)}") from exc

    def contains(self, domain_id: str) -> bool:
        return domain_id in self._adapters

    def names(self) -> list[str]:
        return sorted(self._adapters)

    def all(self) -> list[DomainAdapterContract]:
        return [self._adapters[n] for n in self.names()]

    def route(self, required: list[str] | tuple[str, ...],
              *, domain_hint: str | None = None) -> DomainAdapterContract:
        candidates = ([self.get(domain_hint)] if domain_hint
                      else self.all())
        matched = [a for a in candidates if a.supports(required)]
        if not matched:
            raise ValueError(
                f"no domain supports capabilities={sorted(required or ())} "
                f"(hint={domain_hint!r}, registered={self.names()})")
        return matched[0]


def _snapshot_builder(service_holder: dict[str, Any]):
    """Shared, read-only snapshot builder; never embeds domain-specific logic."""
    def build(run_id: str, objective_ids: list[str] | None = None) -> Any:
        service = service_holder.get("service")
        if service is None:
            raise RuntimeError("no service bound to this adapter")
        return DiscoveryContextProvider(stores=service.stores).build_snapshot(
            run_id, objective_ids=objective_ids)
    return build


def _campaign_runner(service_holder: dict[str, Any]):
    def run(service: Any, plan_id: str, approval_id: str,
            session_id: str | None = None) -> dict[str, Any]:
        # The adapter NEVER grants approval; the service/Director gate does.
        kwargs = {"plan_id": plan_id, "approval_id": approval_id}
        if session_id:
            kwargs["session_id"] = session_id
        return service.run_research_campaign(**kwargs)
    return run


def _result_collector():
    def collect(service: Any, session_id: str) -> dict[str, Any]:
        return service.get_research_report(session_id=session_id)
    return collect


def _bind(domain_id: str, holder: dict[str, Any], *,
          capabilities: tuple[str, ...], preflight: Callable[[], dict],
          resource_estimate: Callable[[], dict], description: str) -> DomainAdapterContract:
    return DomainAdapterContract(
        domain_id=domain_id, capabilities=capabilities,
        preflight=preflight, build_snapshot=_snapshot_builder(holder),
        run_campaign=_campaign_runner(holder),
        collect_result=_result_collector(),
        resource_estimate=resource_estimate,
        health_check=lambda: {"domain_id": domain_id,
                              "preflight": preflight(),
                              "healthy": bool(preflight().get("ready"))},
        description=description)


def build_default_adapters(*, service_holders: dict[str, dict[str, Any]] | None = None
                           ) -> DomainAdapterRegistry:
    """Register scinet / embodied / quant using their REAL domain modules.

    ``service_holders`` maps domain_id -> {"service": ResearchService}; the
    driver binds each service after constructing it with the existing generic
    service builder. No engine is imported until preflight/health_check.
    """
    holders = dict(service_holders or {})
    registry = DomainAdapterRegistry()

    def holder(name: str) -> dict[str, Any]:
        return holders.setdefault(name, {})

    # --- scinet: real external SciNet++ engine ---------------------------
    def scinet_preflight() -> dict[str, Any]:
        from .settings import resolve_engine
        try:
            project, python = resolve_engine()
            return {"ready": True, "project_dir": str(project),
                    "python": str(python), "engine": "scinet++"}
        except Exception as exc:  # noqa: BLE001 - fail closed
            return {"ready": False, "error": f"{type(exc).__name__}: {exc}"}

    registry.register(_bind(
        "scinet", holder("scinet"),
        capabilities=(CAPABILITY_RESEARCH_CAMPAIGN, CAPABILITY_REAL_ENGINE,
                      CAPABILITY_PAIRED_SEEDS, CAPABILITY_VALIDATION_SPLIT),
        preflight=scinet_preflight,
        resource_estimate=lambda: {"max_parallel": 1,
                                   "requires_external_engine": True},
        description="AI4S / scientific discovery via the real SciNet++ engine"))

    # --- embodied: real local robot world-model simulator ----------------
    def embodied_preflight() -> dict[str, Any]:
        from .domains.embodied import (
            BASELINE_CONFIG, PROJECT_DIR, read_dataset_version,
        )
        from .domains.local_engine import read_frozen_manifest
        cfg = yaml.safe_load(
            (PROJECT_DIR / BASELINE_CONFIG).read_text(encoding="utf-8"))
        frozen = (cfg.get("data") or {}).get("frozen_dir", "")
        manifest_path = PROJECT_DIR / frozen / "manifest.json"
        read_frozen_manifest(PROJECT_DIR, cfg)   # verifies split SHAs
        return {"ready": manifest_path.is_file(),
                "engine": "robot_world_model_sim",
                "dataset_version": read_dataset_version(),
                "manifest": str(manifest_path),
                "synthetic_data": False}

    registry.register(_bind(
        "embodied", holder("embodied"),
        capabilities=(CAPABILITY_RESEARCH_CAMPAIGN, CAPABILITY_SIMULATION,
                      CAPABILITY_PAIRED_SEEDS, CAPABILITY_VALIDATION_SPLIT),
        preflight=embodied_preflight,
        resource_estimate=lambda: {"max_parallel": 1,
                                   "requires_external_engine": False},
        description="Robot world-model research via the real local simulator"))

    # --- quant: real local backtest engine, explicit data provenance -----
    def quant_preflight() -> dict[str, Any]:
        from .domains.quant import (
            BASELINE_CONFIG, PROJECT_DIR, read_dataset_version,
        )
        from .domains.local_engine import read_frozen_manifest
        cfg = yaml.safe_load(
            (PROJECT_DIR / BASELINE_CONFIG).read_text(encoding="utf-8"))
        frozen = (cfg.get("data") or {}).get("frozen_dir", "")
        manifest_path = PROJECT_DIR / frozen / "manifest.json"
        read_frozen_manifest(PROJECT_DIR, cfg)   # verifies split SHAs
        return {"ready": manifest_path.is_file(),
                "engine": "quant_backtest_engine",
                "dataset_version": read_dataset_version(),
                "manifest": str(manifest_path),
                # Explicitly synthetic; never presented as live market data.
                "synthetic_data": True,
                "provenance": "generated deterministic price CSV + manifest SHA"}

    registry.register(_bind(
        "quant", holder("quant"),
        capabilities=(CAPABILITY_RESEARCH_CAMPAIGN, CAPABILITY_BACKTEST,
                      CAPABILITY_PAIRED_SEEDS, CAPABILITY_VALIDATION_SPLIT,
                      CAPABILITY_SYNTHETIC_DATA),
        preflight=quant_preflight,
        resource_estimate=lambda: {"max_parallel": 1,
                                   "requires_external_engine": False},
        description="Quant research via a real local backtest; synthetic data flagged"))

    return registry


__all__ = [
    "DomainAdapterContract", "DomainAdapterRegistry", "build_default_adapters",
    "CAPABILITY_RESEARCH_CAMPAIGN", "CAPABILITY_PAIRED_SEEDS",
    "CAPABILITY_SIMULATION", "CAPABILITY_BACKTEST",
    "CAPABILITY_VALIDATION_SPLIT", "CAPABILITY_REAL_ENGINE",
    "CAPABILITY_SYNTHETIC_DATA",
]
