"""ARR 1.0 / E4 -- a REAL, engine-agnostic local execution substrate.

This module is deliberately NOT a FakeRunner. It implements the existing
``ModelAdapter`` contract with genuine numerical computation: it reads an
isolated, hash-verified YAML config, loads a frozen dataset split, performs a
real fit / rollout / backtest, writes a real checkpoint plus ``result.json``,
and returns a complete ``ExperimentResult`` (success OR failure).

Why a local engine at all? The generic ``CampaignRunner``, ``TrialRunner``,
``CandidateConfigBuilder``, ``CampaignComparator`` and ``ResearchService`` are
already engine-agnostic. A new domain needs only:

  1. a project dir with ``config.yaml`` + a frozen ``manifest.json`` dataset;
  2. a ``compute(config, spec, job_dir) -> payload`` function that does the
     actual science;
  3. a ``LocalDomainSpec`` describing whitelists / metrics / guardrails.

No generic layer is copied or forked. External engines (LeRobot / Isaac /
a real exchange feed) can replace ``compute`` without touching the runtime.

Data provenance: the adapter verifies the frozen manifest's per-split SHA-256
digests ON DISK before each job. A shared dataset_version string is never
accepted as evidence (same rule as CandidateConfigBuilder.verify_dataset).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from ..adapters.base import ModelAdapter
from ..candidate_config import sha256_file
from ..schemas import ExperimentResult, ExperimentSpec

ComputeFn = Callable[[dict, dict, Path], dict]


def _now() -> str:
    import datetime
    return datetime.datetime.now().isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_frozen_manifest(project_dir: str | Path, config: Mapping[str, Any]) -> dict:
    """Verify dataset identity ON DISK; raise on any tampering."""
    frozen = (config.get("data") or {}).get("frozen_dir")
    if not frozen:
        raise ValueError("config has no data.frozen_dir")
    root = (Path(project_dir) / frozen).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, info in (manifest.get("splits") or {}).items():
        path = root / str(info["file"])
        if not path.is_file():
            raise FileNotFoundError(f"frozen split {name!r} missing: {path}")
        if sha256_file(path) != info.get("sha256"):
            raise ValueError(f"frozen split {name!r} failed its SHA-256 check")
    return manifest


def load_split(project_dir: str | Path, config: Mapping[str, Any],
               split: str) -> list[dict[str, Any]]:
    """Load ONE frozen split as real rows, with an on-disk hash check."""
    manifest = read_frozen_manifest(project_dir, config)
    info = (manifest.get("splits") or {}).get(split)
    if not info:
        raise ValueError(f"split {split!r} is not in the frozen manifest")
    path = (Path(project_dir) / (config["data"]["frozen_dir"])
            / str(info["file"])).resolve()
    import csv
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows


def record_checkpoint(job_dir: Path, *, run_id: str,
                      checkpoint_path: Path) -> None:
    """O(1) run_id -> checkpoint index (no glob over thousands of job dirs)."""
    index_dir = job_dir.parent / "_runs"
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / f"{run_id}.json").write_text(
        json.dumps({"run_id": run_id,
                    "checkpoint": str(checkpoint_path),
                    "train_job_dir": str(job_dir)},
                   indent=2, sort_keys=True), encoding="utf-8")


def checkpoint_for_run(job_dir: Path, run_id: str) -> Path | None:
    """Resolve a trained checkpoint by run_id WITHOUT scanning the job root.

    Falls back to a bounded scan only for legacy job dirs written before the
    index existed, so old evidence remains readable.
    """
    index = job_dir.parent / "_runs" / f"{run_id}.json"
    if index.is_file():
        try:
            path = Path(json.loads(index.read_text(encoding="utf-8"))["checkpoint"])
            if path.is_file():
                return path
        except (ValueError, OSError, KeyError):
            pass
    for candidate in sorted(job_dir.parent.glob("*/result.json")):
        try:
            raw = json.loads(candidate.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if raw.get("run_id") == run_id and raw.get("stage") == "train":
            path = candidate.parent / "checkpoint.json"
            if path.is_file():
                return path
    return None


def _coerce(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            out[key] = value
    return out


class LocalComputeAdapter(ModelAdapter):
    """A real ``ModelAdapter`` whose jobs are actual local computations.

    ``submit`` executes the computation synchronously and persists the complete
    result BEFORE returning, so ``wait``/``get_status``/``collect_result`` are
    restart-safe (status comes from the on-disk job record, not process memory).
    The deterministic ``job_id`` supplied by the campaign runner is honoured.
    """

    def __init__(self, *, name: str, project_dir: str | Path,
                 compute: ComputeFn, stages: tuple[str, ...] = ("train", "evaluate"),
                 job_root: str | Path | None = None) -> None:
        self.name = name
        self.stages = stages
        self.project_dir = Path(project_dir).resolve()
        if not self.project_dir.is_dir():
            raise FileNotFoundError(f"local engine project dir missing: {self.project_dir}")
        self._compute = compute
        self.job_root = Path(job_root or (self.project_dir / "outputs" / "local_jobs")).resolve()
        self.job_root.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------- helpers
    def _job_dir(self, job_id: str) -> Path:
        return self.job_root / job_id

    @staticmethod
    def _run_id(spec: ExperimentSpec) -> str:
        if spec.run_id:
            return str(spec.run_id)
        return "run_" + hashlib.sha256(
            str(spec.budget.get("job_id") or spec.experiment_id).encode("utf-8")
        ).hexdigest()[:16]

    def _config(self, spec: ExperimentSpec) -> dict[str, Any]:
        path = Path(spec.config_path)
        if not path.is_absolute():
            path = self.project_dir / path
        if not path.is_file():
            raise FileNotFoundError(f"config not found: {path}")
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("baseline config must be a YAML mapping")
        return config

    # ------------------------------------------------------------ ModelAdapter
    def submit(self, spec: ExperimentSpec) -> str:
        job_id = str(spec.budget.get("job_id") or
                     "job_" + hashlib.sha256(
                         json.dumps(spec.model_dump(mode="json"), sort_keys=True,
                                    default=str).encode("utf-8")).hexdigest()[:16])
        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "spec.json").write_text(
            json.dumps(spec.model_dump(mode="json"), indent=2, sort_keys=True,
                       ensure_ascii=False, default=str), encoding="utf-8")
        started = time.time()
        try:
            config = self._config(spec)
            manifest = read_frozen_manifest(self.project_dir, config)
            config_path = Path(spec.config_path)
            if not config_path.is_absolute():
                config_path = self.project_dir / config_path
            payload = self._compute(config, spec.model_dump(mode="json"), job_dir)
            if not isinstance(payload, dict):
                raise ValueError("compute() must return a result payload dict")
            split = (spec.split if spec.stage == "evaluate"
                     else str((config.get("data") or {}).get("train_split", "train")))
            result = {
                "job_id": job_id,
                "experiment_id": spec.experiment_id,
                "objective_id": spec.objective_id,
                "status": str(payload.get("status") or "succeeded"),
                "engine": self.name,
                "stage": spec.stage,
                "metrics": dict(payload.get("metrics") or {}),
                "artifacts": list(payload.get("artifacts") or []),
                "checkpoint": payload.get("checkpoint"),
                "run_id": self._run_id(spec),
                "dataset_version": manifest.get("dataset_version"),
                "evaluation_split": split if spec.stage == "evaluate" else None,
                "config_path": str(config_path),
                "config_sha256": sha256_file(config_path),
                "manifest_sha256": sha256_file(
                    (self.project_dir / config["data"]["frozen_dir"] / "manifest.json")),
                "dataset_split_hashes": {
                    k: v.get("sha256") for k, v in
                    (manifest.get("splits") or {}).items()
                },
                "error": payload.get("error"),
                "error_type": payload.get("error_type"),
                "exit_code": 0,
                "started_at": _now(),
                "finished_at": _now(),
                "elapsed_seconds": round(time.time() - started, 6),
                "raw": {k: v for k, v in payload.items()
                        if k not in ("status", "metrics", "artifacts",
                                     "checkpoint", "error", "error_type")},
            }
        except Exception as exc:  # a failed run is still COMPLETE evidence
            result = {
                "job_id": job_id, "experiment_id": spec.experiment_id,
                "objective_id": spec.objective_id, "status": "failed",
                "engine": self.name, "stage": spec.stage, "metrics": {},
                "artifacts": [], "checkpoint": None, "run_id": self._run_id(spec),
                "dataset_version": spec.dataset_version,
                "evaluation_split": spec.split if spec.stage == "evaluate" else None,
                "error": f"{type(exc).__name__}: {exc}",
                "error_type": "process_failed", "exit_code": 1,
                "started_at": _now(), "finished_at": _now(),
                "elapsed_seconds": round(time.time() - started, 6),
                "traceback": traceback.format_exc()[-2000:],
            }
        (job_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False,
                       default=str), encoding="utf-8")
        (job_dir / "result_sha256.txt").write_text(
            sha256_file(job_dir / "result.json"), encoding="utf-8")
        ckpt = (result.get("checkpoint") or {}).get("path")
        if spec.stage == "train" and result.get("status") == "succeeded" and ckpt:
            record_checkpoint(job_dir, run_id=str(result.get("run_id") or ""),
                              checkpoint_path=Path(ckpt))
        return job_id

    def get_status(self, job_id: str) -> str:
        path = self._job_dir(job_id) / "result.json"
        if not path.is_file():
            return "incomplete"
        try:
            return str(json.loads(path.read_text(encoding="utf-8")).get("status")
                       or "incomplete")
        except (ValueError, OSError):
            return "incomplete"

    def probe_job(self, job_id: str) -> str | None:
        if not (self._job_dir(job_id) / "result.json").is_file():
            return None
        return self.get_status(job_id)

    def wait(self, job_id: str, timeout: float = 3600.0,
             poll_interval: float = 1.0) -> str:
        return self.get_status(job_id)

    def collect_result(self, job_id: str,
                       spec: ExperimentSpec | None = None) -> ExperimentResult:
        path = self._job_dir(job_id) / "result.json"
        if not path.is_file():
            return ExperimentResult.incomplete(
                experiment_id=spec.experiment_id if spec else "unknown",
                job_id=job_id, engine=self.name,
                error=f"no result.json for {job_id}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["result_path"] = str(path)
        raw["log_path"] = str(self._job_dir(job_id) / "result.json")
        return ExperimentResult.from_cli_result(
            raw, experiment_id=spec.experiment_id if spec else None)

    def capabilities(self) -> dict[str, Any]:
        return {"name": self.name, "stages": list(self.stages),
                "project_dir": str(self.project_dir),
                "transport": "local/in-process-compute",
                "job_root": str(self.job_root)}

    def health_check(self) -> dict[str, Any]:
        return {"ok": self.job_root.is_dir(), **self.capabilities()}


@dataclass(frozen=True)
class LocalDomainSpec:
    """Everything a local domain must declare. No generic layer is modified."""

    name: str
    description: str
    project_dir: Path
    adapter_name: str
    compute: ComputeFn
    allowed_config_keys: tuple[str, ...]
    value_rules: dict[str, Any]
    intervention_policy: dict[str, dict[str, Any]]
    primary_metric: str
    metric_direction: str
    guardrails: dict[str, dict[str, Any]]
    required_metrics: tuple[str, ...]
    dataset_seed: int
    quality_domain: str
    paired_seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
    baseline_config: str = "config.yaml"
    description_extra: str = ""
