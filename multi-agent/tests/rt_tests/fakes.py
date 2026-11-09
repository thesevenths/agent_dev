"""In-memory ModelAdapter for fast tests (no GPU, no subprocess)."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from research_runtime.adapters.base import ModelAdapter
from research_runtime.schemas import ExperimentResult, ExperimentSpec


class FakeAdapter(ModelAdapter):
    name = "fake"
    stages = ("train", "evaluate", "analyze")

    def __init__(self, *, fail_stage: str | None = None,
                 metrics: dict[str, float] | None = None):
        self.specs: dict[str, ExperimentSpec] = {}
        self.submits: list[tuple[str, ExperimentSpec]] = []
        self.counter = 0
        self.fail_stage = fail_stage
        self.metrics = metrics if metrics is not None else {
            "latent_mse@1": 0.02,
            "latent_mse@3": 0.05,
            "cosine@1": 0.91,
            "cosine@3": 0.88,
        }

    def submit(self, spec: ExperimentSpec) -> str:
        self.counter += 1
        job_id = f"job{self.counter}"
        self.specs[job_id] = spec
        self.submits.append((job_id, spec))
        return job_id

    def get_status(self, job_id: str) -> str:
        return "succeeded" if job_id in self.specs else "incomplete"

    def probe_job(self, job_id: str) -> str | None:
        """None = never submitted (safe to submit); never a transport guess."""
        return "succeeded" if job_id in self.specs else None

    def collect_result(self, job_id: str,
                       spec: ExperimentSpec | None = None) -> ExperimentResult:
        spec = spec or self.specs[job_id]
        base = dict(experiment_id=spec.experiment_id, job_id=job_id,
                    engine=self.name, stage=spec.stage, run_id="run1",
                    dataset_version="dataset-v1")
        if self.fail_stage and spec.stage == self.fail_stage:
            return ExperimentResult(
                **base, status="failed", error_type="process_failed",
                error="simulated stage failure",
            )
        if spec.stage == "train":
            return ExperimentResult(
                **base, status="succeeded",
                checkpoint={"path": f"/tmp/{job_id}/checkpoint.pt",
                            "sha256": "deadbeef", "bytes": 123},
                artifacts=["checkpoint.pt", "train.log"],
            )
        return ExperimentResult(
            **base, status="succeeded", metrics=dict(self.metrics),
            artifacts=["horizon_metrics.npy", "evaluate.log"],
        )
