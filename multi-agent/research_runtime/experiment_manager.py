"""Unified experiment lifecycle: submit -> status -> wait -> collect -> evidence."""
from __future__ import annotations

from typing import Any

from .adapters.base import ModelAdapter
from .evidence_store import EvidenceStore
from .schemas import ExperimentResult, ExperimentSpec
from .settings import WAIT_TIMEOUT_S


class ExperimentManager:
    """Thin, adapter-injected coordinator. No queue framework in P0.

    The adapter is responsible for TRUE async submission; this layer only
    keeps the spec<->job association and writes the audit trail.
    """

    def __init__(self, adapter: ModelAdapter, *,
                 evidence_store: EvidenceStore | None = None,
                 wait_timeout: float | None = None):
        self.adapter = adapter
        self.evidence = evidence_store or EvidenceStore()
        self.wait_timeout = wait_timeout if wait_timeout is not None else WAIT_TIMEOUT_S
        self._specs: dict[str, ExperimentSpec] = {}

    def submit(self, spec: ExperimentSpec) -> dict[str, Any]:
        job_id = self.adapter.submit(spec)
        self._specs[job_id] = spec
        record = {
            "experiment_id": spec.experiment_id,
            "objective_id": spec.objective_id,
            "job_id": job_id,
            "status": "queued",
            "engine": spec.engine,
            "stage": spec.stage,
            "run_id": spec.run_id,
        }
        self.evidence.append("submitted", {**record, "spec": spec.model_dump()})
        return record

    def get_status(self, job_id: str) -> str:
        return self.adapter.get_status(job_id)

    def wait(self, job_id: str, timeout: float | None = None) -> str:
        return self.adapter.wait(job_id, timeout=timeout or self.wait_timeout)

    def collect_result(self, job_id: str,
                       spec: ExperimentSpec | None = None) -> ExperimentResult:
        effective = spec or self._specs.get(job_id)
        result = self.adapter.collect_result(job_id, effective)
        self.evidence.append("result", {
            "experiment_id": result.experiment_id,
            "job_id": job_id,
            "status": result.status,
            "result": result.model_dump(exclude={"raw"}),
        })
        return result

    def run(self, spec: ExperimentSpec, *,
            timeout: float | None = None) -> tuple[str, ExperimentResult]:
        """Convenience: submit + wait + collect. Returns (job_id, result)."""
        record = self.submit(spec)
        job_id = str(record["job_id"])
        self.wait(job_id, timeout=timeout)
        return job_id, self.collect_result(job_id, spec)

    def get_spec(self, job_id: str) -> ExperimentSpec | None:
        return self._specs.get(job_id)

    @property
    def jobs(self) -> dict[str, ExperimentSpec]:
        return dict(self._specs)
