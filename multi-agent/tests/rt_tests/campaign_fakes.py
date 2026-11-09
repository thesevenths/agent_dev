"""Shared P1-04 test infrastructure: a real (in-memory) campaign stack.

Only the ENGINE is faked. Everything else -- SQLite state, event log, config
builder, hash checks, scheduler, trial runner, manifest -- is the real code.
That is what makes the crash-recovery tests meaningful: after ``restart()``
a brand-new ExperimentManager/Runner is built against the SAME adapter job
registry and the SAME SQLite file, with no in-memory state carried over.
"""
from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from research_runtime.adapters.base import ModelAdapter
from research_runtime.campaign import (
    CampaignPolicy,
    CampaignRunSpec,
    CampaignRunner,
    CampaignStore,
)
from research_runtime.candidate_config import CandidateConfigBuilder
from research_runtime.experiment_manager import ExperimentManager
from research_runtime.schemas import (
    CandidateExperiment,
    ExperimentPlan,
    ExperimentResult,
    ExperimentSpec,
)

DATASET_VERSION = "ds-v1"
PRIMARY_METRIC = "mse@1"
BASE_YAML = """experiment: campaign_smoke
seed: 42
device: cpu
output_dir: outputs
data:
  generator: dummy
  dataset_seed: 7
  frozen_dir: datasets/frozen_v1
  train_samples: 8
model:
  predictor_hidden: [32, 32]
  latent_dim: 4
train:
  lr: 0.001
  weight_decay: 0.0001
jepa:
  pred_weight: 1.0
  ema_momentum: 0.996
"""
ALLOWED_KEYS = ("model.latent_dim", "model.predictor_hidden", "train.lr")


class NoEvidence:
    def append(self, *args: Any, **kwargs: Any) -> None:
        pass


class MockAdapter(ModelAdapter):
    """Deterministic engine. Its job registry survives the simulated restart."""

    name = "mock"
    stages = ("train", "evaluate")

    def __init__(self, **opts: Any):
        self.opts = opts
        self.jobs: dict[str, ExperimentSpec] = {}
        self.submits: list[str] = []
        self.wait_calls: list[tuple[str, float]] = []
        self.collect_calls: list[tuple[str, ExperimentSpec | None]] = []

    # -- engine contract -------------------------------------------------
    def probe_job(self, job_id: str) -> str | None:
        if self.opts.get("probe_error"):
            raise RuntimeError("engine unreachable (simulated)")
        return "succeeded" if job_id in self.jobs else None

    def get_status(self, job_id: str) -> str:
        return self.probe_job(job_id) or "incomplete"

    def submit(self, spec: ExperimentSpec) -> str:
        if self.opts.get("submit_error"):
            raise RuntimeError("engine submit rejected (simulated)")
        job_id = str(spec.budget["job_id"])
        if job_id in self.jobs:
            raise RuntimeError(f"duplicate job_id {job_id}")
        self.jobs[job_id] = spec
        self.submits.append(job_id)
        return job_id

    def wait(self, job_id: str, timeout: float = 3600.0,
             poll_interval: float = 1.0) -> str:
        self.wait_calls.append((job_id, timeout))
        sequence = self.opts.get("wait_status_seq")
        if sequence:
            return str(sequence.pop(0))
        return str(self.opts.get("wait_status") or "succeeded")

    def collect_result(self, job_id: str,
                       spec: ExperimentSpec | None = None) -> ExperimentResult:
        self.collect_calls.append((job_id, spec))
        spec = spec or self.jobs[job_id]
        if self.opts.get("collect_error"):
            raise RuntimeError("result unavailable (simulated)")
        common: dict[str, Any] = dict(
            experiment_id=spec.experiment_id, job_id=job_id,
            status="succeeded", engine=self.name, stage=spec.stage,
            dataset_version=self.opts.get("dataset_version")
            or spec.dataset_version,
        )
        if spec.stage == "train":
            if (self.opts.get("train_status")
                    or spec.experiment_id in (self.opts.get("fail_experiments") or ())):
                common["status"] = self.opts.get("train_status") or "failed"
                common["error_type"] = "process_failed"
                common["error"] = "simulated train failure"
                return ExperimentResult(**common)
            checkpoint = None if self.opts.get("missing_checkpoint") else {
                "path": f"/tmp/{job_id}/checkpoint.pt", "sha256": "abc"}
            run_id = None if self.opts.get("missing_run_id") else f"run_{job_id[:8]}"
            return ExperimentResult(**common, run_id=run_id,
                                    checkpoint=checkpoint,
                                    artifacts=["checkpoint.pt"])
        if self.opts.get("eval_status"):
            common["status"] = self.opts["eval_status"]
            common["error_type"] = "process_failed"
            common["error"] = "simulated evaluation failure"
            return ExperimentResult(**common)
        metrics_opt = self.opts.get("metrics")
        metrics = dict(metrics_opt if metrics_opt is not None
                       else {PRIMARY_METRIC: 0.02})
        return ExperimentResult(
            **common,
            run_id=self.opts.get("actual_run_id") or spec.run_id,
            evaluation_split=self.opts.get("actual_split") or spec.split,
            metrics=metrics,
            artifacts=["horizon_metrics.npy", "evaluate.log"],
        )

    def health_check(self) -> dict:
        if self.opts.get("unhealthy"):
            return {"ok": False}
        return {"ok": True, "stages": list(self.stages)}


class CampaignTestBase(unittest.TestCase):
    """Fresh on-disk project + frozen dataset + real SQLite store per test."""

    def setUp(self) -> None:
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "configs").mkdir()
        self.base_config = self.root / "configs" / "base.yaml"
        self.base_config.write_text(BASE_YAML, encoding="utf-8")

        frozen = self.root / "datasets" / "frozen_v1"
        frozen.mkdir(parents=True)
        splits = {}
        for name in ("train", "validation", "adapt", "test"):
            payload = f"frozen-split-{name}".encode()
            (frozen / f"{name}.npz").write_bytes(payload)
            splits[name] = {
                "file": f"{name}.npz",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "samples": 8,
            }
        self.manifest = frozen / "manifest.json"
        self.manifest.write_text(json.dumps({
            "schema_version": 1,
            "dataset_version": DATASET_VERSION,
            "splits": splits,
        }, indent=2), encoding="utf-8")
        self.frozen = frozen
        self.db_path = self.root / "campaigns.sqlite"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    # -- builders --------------------------------------------------------
    def adapter(self, **opts: Any) -> MockAdapter:
        return MockAdapter(**opts)

    def builder(self, *, allowed_keys=ALLOWED_KEYS) -> CandidateConfigBuilder:
        return CandidateConfigBuilder(
            self.root, allowed_keys=allowed_keys,
            campaign_root="outputs/research_campaigns")

    def store(self) -> CampaignStore:
        return CampaignStore(self.db_path)

    def policy(self, **overrides: Any) -> CampaignPolicy:
        kwargs = dict(
            allowed_config_keys=ALLOWED_KEYS,
            epochs=1, max_parallel=1, max_candidates=2, max_seeds=5,
            max_jobs=30, timeout_per_job_s=30,
            evaluation_split="validation",
        )
        kwargs.update(overrides)
        return CampaignPolicy(**kwargs)

    def manager(self, adapter: MockAdapter) -> ExperimentManager:
        return ExperimentManager(adapter, evidence_store=NoEvidence())

    def runner(self, adapter: MockAdapter | None = None, *,
               policy: CampaignPolicy | None = None,
               store: CampaignStore | None = None,
               **adapter_opts: Any) -> tuple[CampaignRunner, MockAdapter, CampaignStore]:
        adapter = adapter or self.adapter(**adapter_opts)
        store = store or self.store()
        runner = CampaignRunner(self.manager(adapter), self.builder(),
                                store, policy or self.policy())
        return runner, adapter, store

    def plan(self, *, seeds=(42,), n_candidates: int = 1,
             overrides: dict[str, Any] | None = None,
             primary_metric: str = PRIMARY_METRIC) -> ExperimentPlan:
        candidates = []
        for index in range(n_candidates):
            candidates.append(CandidateExperiment(
                candidate_id=f"cand_{index}",
                hypothesis_id=f"H{index}",
                config_overrides=dict(overrides or {"model.latent_dim": 8}),
                seeds=list(seeds),
                dataset_version=DATASET_VERSION,
            ))
        return ExperimentPlan(
            plan_id="plan_001", objective_id="obj_001",
            baseline_experiment_id="baseline_001",
            dataset_version=DATASET_VERSION,
            candidates=candidates,
            primary_metric=primary_metric,
            paired_seeds=list(seeds),
        )

    def run_spec(self, plan: ExperimentPlan | None = None, **overrides: Any
                 ) -> CampaignRunSpec:
        plan = plan or self.plan()
        kwargs: dict[str, Any] = dict(
            campaign_id="campaign_001",
            objective_id=plan.objective_id,
            plan_id=plan.plan_id,
            baseline_config_path="configs/base.yaml",
            dataset_version=plan.dataset_version,
            evaluation_split="validation",
            job_timeout_s=30,
            max_jobs=100,
            approved_hypothesis_ids=[c.hypothesis_id for c in plan.candidates],
        )
        kwargs.update(overrides)
        return CampaignRunSpec(**kwargs)

    def campaign_dir(self, campaign_id: str = "campaign_001") -> Path:
        return self.root / "outputs" / "research_campaigns" / campaign_id
