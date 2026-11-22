"""ARR_EMB_RSI_01 -- real SciNet++ world-model adapter.

No FakeRunner / MockEngine. Every experiment job is submitted to the existing
``SciNetAdapter`` and executed by the real SciNet++ CLI.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..adapters import SciNetAdapter
from ..schemas import ExperimentSpec, ExperimentResult
from ..settings import DEFAULT_SCINET_DIR, WAIT_TIMEOUT_S


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _dotted_set(cfg: dict[str, Any], key: str, value: Any) -> None:
    node = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"config path {key!r} is not a mapping")
    node[parts[-1]] = value


def _dotted_get(cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


@dataclass
class WorldModelRun:
    arm: str
    failure_id: str
    seed: int
    config_path: str
    config_sha256: str
    applied_config: dict[str, Any]
    experiment_id: str
    run_id: str = ""
    train: dict[str, Any] = field(default_factory=dict)
    evaluate: dict[str, Any] = field(default_factory=dict)
    ood_evaluate: dict[str, Any] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    artifact_hashes: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    ood_metrics: dict[str, float] = field(default_factory=dict)
    infrastructure_failures: list[dict[str, Any]] = field(default_factory=list)
    budget: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""
    ok: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WorldModelAdapter:
    """Thin, auditable wrapper around the real SciNet++ adapter."""

    def __init__(self, *, run_root: str | Path,
                 base_config: str = "configs/pendulum_smoke.yaml",
                 ood_config: str = "configs/pendulum_smoke_ood.yaml",
                 project_dir: str | Path | None = None,
                 cli_timeout: float = 180.0,
                 max_jobs: int = 4,
                 max_seconds: float = 900.0):
        self.run_root = Path(run_root).resolve()
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.config_root = self.run_root / "configs"
        self.config_root.mkdir(parents=True, exist_ok=True)
        self.project = Path(project_dir or DEFAULT_SCINET_DIR)
        self.base_config = base_config
        self.ood_config = ood_config
        self.adapter = SciNetAdapter(
            project_dir=str(self.project), cli_timeout=cli_timeout)
        self.max_jobs = int(max_jobs)
        self.max_seconds = float(max_seconds)
        self._jobs = 0
        self._started = time.time()

    def _load_base(self) -> dict[str, Any]:
        path = self.project / self.base_config
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def build_config(self, *, run: WorldModelRun,
                     overrides: dict[str, Any],
                     config_kind: str = "main",
                     source_config: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
        cfg = dict(source_config or self._load_base())
        if config_kind == "ood" and source_config is None:
            ood = yaml.safe_load((self.project / self.ood_config).read_text(
                encoding="utf-8")) or {}
            cfg["data"] = dict(ood.get("data") or cfg.get("data") or {})
        for key, value in overrides.items():
            _dotted_set(cfg, key, value)
        path = self.config_root / (
            f"{run.failure_id}__{run.arm}__s{run.seed}__{config_kind}.yaml")
        path.write_text(yaml.safe_dump(cfg, sort_keys=True, allow_unicode=True),
                        encoding="utf-8")
        return str(path), cfg

    def _check_budget(self) -> None:
        if self._jobs >= self.max_jobs:
            raise RuntimeError(f"job budget exceeded: {self._jobs}/{self.max_jobs}")
        if time.time() - self._started > self.max_seconds:
            raise TimeoutError("wall-clock research budget exceeded")

    def _run_stage(self, *, spec: ExperimentSpec) -> tuple[dict[str, Any], ExperimentResult]:
        self._check_budget()
        self._jobs += 1
        job_id = self.adapter.submit(spec)
        status = self.adapter.wait(job_id, timeout=WAIT_TIMEOUT_S)
        result = self.adapter.collect_result(job_id, spec)
        payload = json.loads(result.model_dump_json())
        payload["job_id"] = job_id
        payload["wait_status"] = status
        return payload, result

    def _find_run_artifacts(self, *, experiment_id: str,
                            run_id: str) -> dict[str, str]:
        if not run_id:
            return {}
        candidates = list(self.project.glob(
            f"outputs/*/{run_id}/result_*.json"))
        out: dict[str, str] = {}
        if candidates:
            out["result_json"] = str(candidates[0])
        checkpoints = list(self.project.glob(
            f"outputs/*/{run_id}/checkpoint.pt"))
        if checkpoints:
            out["checkpoint.pt"] = str(checkpoints[0])
        return out

    @staticmethod
    def _experiment_family(experiment_id: str) -> str:
        return experiment_id.split("__")[0] if "__" in experiment_id else ""

    def run_arm(self, *, arm: str, failure_id: str, seed: int,
                overrides: dict[str, Any],
                ood_overrides: dict[str, Any] | None = None,
                run_suffix: str = "") -> WorldModelRun:
        rid = (f"{failure_id}__{arm}__s{seed}__"
               f"{run_suffix or hashlib.sha256(json.dumps(overrides, sort_keys=True).encode()).hexdigest()[:8]}")
        config_path, applied = self.build_config(
            run=WorldModelRun(arm=arm, failure_id=failure_id, seed=seed,
                              config_path="", config_sha256="",
                              applied_config={}, experiment_id=rid),
            overrides=overrides)
        config_sha = sha256_file(config_path)
        run = WorldModelRun(
            arm=arm, failure_id=failure_id, seed=seed,
            config_path=config_path, config_sha256=config_sha,
            applied_config=applied, experiment_id=rid,
            budget={"max_jobs": self.max_jobs, "max_seconds": self.max_seconds,
                    "seed": seed})
        try:
            train_spec = ExperimentSpec(
                experiment_id=rid, objective_id=f"obj_{failure_id}",
                stage="train", config_path=config_path, seed=seed,
                budget={"epochs": int(_dotted_get(applied, "train.epochs", 1))})
            train_payload, _ = self._run_stage(spec=train_spec)
            run.train = train_payload
            if train_payload.get("status") != "succeeded" or not train_payload.get("run_id"):
                raise RuntimeError(f"train failed: {train_payload.get('error')}")
            run.run_id = str(train_payload["run_id"])

            eval_spec = ExperimentSpec(
                experiment_id=rid, objective_id=f"obj_{failure_id}",
                stage="evaluate", config_path=config_path, run_id=run.run_id,
                split="validation", seed=seed, budget={"epochs": 1})
            eval_payload, _ = self._run_stage(spec=eval_spec)
            run.evaluate = eval_payload
            if eval_payload.get("status") != "succeeded":
                raise RuntimeError(f"evaluate failed: {eval_payload.get('error')}")
            run.metrics = dict(eval_payload.get("metrics") or {})

            if ood_overrides:
                ood_path, _ = self.build_config(
                    run=WorldModelRun(arm=arm, failure_id=failure_id, seed=seed,
                                      config_path="", config_sha256="",
                                      applied_config={}, experiment_id=rid),
                    overrides=ood_overrides, config_kind="ood",
                    source_config=applied)
                ood_spec = ExperimentSpec(
                    experiment_id=rid + "__ood", objective_id=f"obj_{failure_id}",
                    stage="evaluate", config_path=ood_path, run_id=run.run_id,
                    split="test", seed=seed, budget={"epochs": 1})
                ood_payload, _ = self._run_stage(spec=ood_spec)
                run.ood_evaluate = ood_payload
                if ood_payload.get("status") != "succeeded":
                    raise RuntimeError(f"OOD evaluate failed: {ood_payload.get('error')}")
                run.ood_metrics = dict(ood_payload.get("metrics") or {})

            artifacts = self._find_run_artifacts(experiment_id=rid, run_id=run.run_id)
            run.artifact_paths = artifacts
            run.artifact_hashes = {k: sha256_file(v) for k, v in artifacts.items()
                                   if Path(v).is_file()}
            # Verify the ACTUAL executed config was the one we generated.
            applied_cfg = yaml.safe_load(Path(config_path).read_text(
                encoding="utf-8")) or {}
            mismatches = {k: {"expected": v,
                              "actual": _dotted_get(applied_cfg, k)}
                          for k, v in overrides.items()
                          if _dotted_get(applied_cfg, k) != v}
            if mismatches:
                raise RuntimeError(f"config not applied: {mismatches}")
            run.budget["jobs_used"] = self._jobs
            run.ok = True
        except Exception as exc:
            run.ok = False
            run.error = f"{type(exc).__name__}: {exc}"
            run.infrastructure_failures.append(
                {"arm": arm, "failure_id": failure_id, "seed": seed,
                 "error": run.error[:500]})
        run.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        out = self.run_root / f"{failure_id}__{arm}__s{seed}__run.json"
        out.write_text(json.dumps(run.to_dict(), indent=2, sort_keys=True,
                                  ensure_ascii=False, default=str),
                       encoding="utf-8")
        return run


__all__ = ["WorldModelAdapter", "WorldModelRun", "sha256_file", "sha256_json"]
