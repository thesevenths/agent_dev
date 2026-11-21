"""M5 Integrated -- public entry point for the Autonomous Research Runtime.

Usage (real engine, external human approval)::

    from research_runtime.autonomous_runtime import AutonomousRuntime
    rt = AutonomousRuntime.from_services(
        config=RuntimeConfig(run_id="..."),
        services={...}, adapters=..., databases=...,
        approver=external_human_approver,   # NEVER auto-granted
        dataset_shas={...}, out_dir="...")
    result = rt.run()

The runtime does NOT grant approvals, does NOT fabricate metrics, and does NOT
mark a run COMPLETED when any domain fails. It is engine-agnostic: real engines
are injected by the caller (production) or replaced by contract doubles in
engine-free tests.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .runtime_coordinator import (
    RuntimeCoordinator, RuntimeConfig, RuntimeApprovalRequired,
    RuntimeRecoveryRequired,
)
from .runtime_state import RuntimeState


class AutonomousRuntime:
    def __init__(self, *, coordinator: RuntimeCoordinator):
        self.coordinator = coordinator

    @classmethod
    def from_services(cls, *, config: RuntimeConfig,
                      state_db: Any, scheduler: Any, discovery: Any,
                      rrsi: Any | None, dataset_shas: dict[str, str],
                      approver: Callable[[str, str], None] | None,
                      out_dir: str | Path,
                      objective_specs: dict[str, dict[str, Any]] | None = None
                      ) -> "AutonomousRuntime":
        state = RuntimeState(db=state_db)
        coordinator = RuntimeCoordinator(
            config=config, state=state, scheduler=scheduler,
            discovery=discovery, rrsi=rrsi, dataset_shas=dataset_shas,
            approver=approver, out_dir=out_dir,
            objective_specs=objective_specs)
        return cls(coordinator=coordinator)

    def start(self) -> dict[str, Any]:
        return self.coordinator.start()

    def recover_or_raise(self) -> dict[str, Any]:
        return self.coordinator.recover_or_raise()

    def run_round(self, round_no: int, **kwargs: Any) -> dict[str, Any]:
        return self.coordinator.run_round(round_no, **kwargs)

    def cross_domain_discovery(self, **kwargs: Any) -> dict[str, Any]:
        return self.coordinator.cross_domain_discovery(**kwargs)

    def run_rrsi(self, **kwargs: Any) -> dict[str, Any]:
        return self.coordinator.run_rrsi(**kwargs)

    def finalize(self, **kwargs: Any) -> dict[str, Any]:
        return self.coordinator.finalize(**kwargs)

    def status(self) -> dict[str, Any]:
        return self.coordinator.status()

    def request_stop(self, reason: str = "human stop") -> None:
        self.coordinator.request_stop(reason)


__all__ = [
    "AutonomousRuntime", "RuntimeConfig", "RuntimeApprovalRequired",
    "RuntimeRecoveryRequired",
]
