"""Abstract engine contract.

Future backends (MuJoCo / PyTorch / finance backtester / any external service)
implement these three methods. Research Runtime never imports their internals.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod

from ..schemas import ExperimentResult, ExperimentSpec, TERMINAL_STATUSES


class ModelAdapter(ABC):
    name: str = "abstract"
    stages: tuple[str, ...] = ()

    @abstractmethod
    def submit(self, spec: ExperimentSpec) -> str:
        """Submit a job ASYNCHRONOUSLY and return its job_id immediately."""

    @abstractmethod
    def get_status(self, job_id: str) -> str:
        """Return a status string; must work after the caller restarts."""

    @abstractmethod
    def collect_result(self, job_id: str,
                       spec: ExperimentSpec | None = None) -> ExperimentResult:
        """Return a COMPLETE ExperimentResult, including for failed jobs."""

    def wait(self, job_id: str, timeout: float = 3600.0,
             poll_interval: float = 1.0) -> str:
        """Default polling implementation; adapters may override with a
        server-side wait to avoid spawning a process per poll."""
        deadline = time.monotonic() + timeout
        status = self.get_status(job_id)
        while status not in TERMINAL_STATUSES and time.monotonic() < deadline:
            time.sleep(poll_interval)
            status = self.get_status(job_id)
        return status

    def capabilities(self) -> dict:
        return {"name": self.name, "stages": list(self.stages)}

    def health_check(self) -> dict:
        return {"ok": True, **self.capabilities()}
