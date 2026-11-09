"""P0 research flow: train -> evaluate on the SAME run -> register evidence.

Scope is deliberately narrow: P0 does NOT attempt self-improvement. Its only
job is to prove that the Agent can reliably

    submit a job -> get a job_id -> wait -> read structured metrics
    -> record checkpoint lineage -> produce an auditable report

and to produce a complete FailureReport when anything goes wrong.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from pathlib import Path
from typing import Any

from .evidence_store import EvidenceStore
from .experiment_manager import ExperimentManager
from .failure_diagnoser import FailureDiagnoser
from .model_registry import ModelRegistry
from .schemas import (
    ExperimentResult,
    ExperimentSpec,
    FailureReport,
    ModelVersion,
    ResearchObjective,
)
from .settings import ensure_data_dir, WAIT_TIMEOUT_S


def sha256_file(path: str | Path) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_success_criteria(criteria: dict[str, Any],
                           metrics: dict[str, float]) -> tuple[bool, list[str]]:
    """Generic, engine-agnostic acceptance check.

    ``{"latent_mse@1": 0.05}``            -> upper bound
    ``{"cosine@1": {"min": 0.8}}``        -> lower bound
    ``{"mse": {"max": 0.05, "min": 0.0}}`` -> both
    """
    failures: list[str] = []
    for name, rule in (criteria or {}).items():
        if name not in metrics:
            failures.append(f"metric {name!r} missing from result")
            continue
        value = metrics[name]
        if isinstance(rule, bool) or not isinstance(rule, (int, float, dict)):
            continue
        if isinstance(rule, (int, float)):
            if value > float(rule):
                failures.append(f"{name}={value} exceeds max {rule}")
        else:
            if "max" in rule and value > float(rule["max"]):
                failures.append(f"{name}={value} exceeds max {rule['max']}")
            if "min" in rule and value < float(rule["min"]):
                failures.append(f"{name}={value} below min {rule['min']}")
    return (not failures), failures


class ResearchOrchestrator:
    def __init__(self, manager: ExperimentManager, *,
                 registry: ModelRegistry | None = None,
                 diagnoser: FailureDiagnoser | None = None,
                 evidence_store: EvidenceStore | None = None):
        self.manager = manager
        self.registry = registry or ModelRegistry()
        self.diagnoser = diagnoser or FailureDiagnoser()
        self.evidence = evidence_store or manager.evidence

    # ------------------------------------------------------------- helpers
    def _config_path(self, relative: str) -> Path | None:
        project = getattr(self.manager.adapter, "project_dir", None)
        if project is None:
            return None
        path = Path(relative)
        return path if path.is_absolute() else Path(project) / path

    def _config_sha256(self, relative: str) -> str | None:
        path = self._config_path(relative)
        return sha256_file(path) if path else None

    def _code_version(self) -> str | None:
        project = getattr(self.manager.adapter, "project_dir", None)
        if not project:
            return None
        try:
            proc = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], cwd=str(project),
                capture_output=True, text=True, timeout=5,
            )
            return proc.stdout.strip() if proc.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None

    def _failure(self, result: ExperimentResult,
                 spec: ExperimentSpec | None = None) -> list[dict]:
        reports = self.diagnoser.diagnose_all(result)
        for report in reports:
            self.evidence.append("failure", report.model_dump())
        return [r.model_dump() for r in reports]

    def save_report(self, report: dict) -> str:
        path = ensure_data_dir() / "reports" / f"{report['experiment_id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                   default=str), encoding="utf-8")
        return str(path)

    # ---------------------------------------------------------- single stage
    def run_stage(self, spec: ExperimentSpec, *,
                  timeout: float | None = None) -> dict[str, Any]:
        job_id, result = self.manager.run(spec, timeout=timeout)
        return {
            "job_id": job_id,
            "spec": spec.model_dump(),
            "result": result.model_dump(),
            "failures": self._failure(result, spec),
        }

    # ------------------------------------------------------------- P0 flow
    def run_baseline(self, objective: ResearchObjective, *,
                     config_path: str,
                     epochs: int | None = 1,
                     experiment_id: str | None = None,
                     model_version: str | None = None,
                     dataset_version: str | None = None,
                     wait_timeout: float | None = None,
                     promote: bool = False) -> dict[str, Any]:
        """Train a baseline, evaluate it on the same run_id, register lineage."""
        exp_id = experiment_id or f"exp_{uuid.uuid4().hex[:10]}"
        timeout = wait_timeout if wait_timeout is not None else WAIT_TIMEOUT_S
        budget: dict[str, Any] = {}
        if epochs is not None:
            budget["epochs"] = epochs
        budget.setdefault("timeout_s", timeout)

        train_spec = ExperimentSpec(
            experiment_id=exp_id,
            objective_id=objective.objective_id,
            engine=self.manager.adapter.name,
            stage="train",
            config_path=config_path,
            model_version=model_version,
            dataset_version=dataset_version,
            budget=budget,
        )

        report: dict[str, Any] = {
            "objective": objective.model_dump(),
            "experiment_id": exp_id,
            "engine": self.manager.adapter.name,
            "config_path": config_path,
            "success": False,
            "failure_reports": [],
            "model_version": None,
        }

        # ---- 1) train -----------------------------------------------------
        train_job = self.manager.submit(train_spec)["job_id"]
        train_status = self.manager.wait(train_job, timeout=timeout)
        train_result = self.manager.collect_result(train_job, train_spec)
        report["train"] = {
            "job_id": train_job,
            "status": train_result.status,
            "wait_status": train_status,
            "run_id": train_result.run_id,
            "checkpoint": train_result.checkpoint,
            "artifacts": train_result.artifacts,
            "result_path": train_result.result_path,
        }
        if not train_result.succeeded or not train_result.run_id:
            report["failure_reports"] = self._failure(train_result, train_spec)
            report["report_path"] = self.save_report(report)
            return report

        # ---- 2) evaluate on the SAME run_id (checkpoint lives there) ------
        eval_spec = train_spec.with_run_id(
            train_result.run_id, stage="evaluate", parent_job_id=str(train_job)
        )
        eval_job = self.manager.submit(eval_spec)["job_id"]
        eval_status = self.manager.wait(eval_job, timeout=timeout)
        eval_result = self.manager.collect_result(eval_job, eval_spec)
        report["evaluate"] = {
            "job_id": eval_job,
            "status": eval_result.status,
            "wait_status": eval_status,
            "run_id": eval_result.run_id,
            "metrics": eval_result.metrics,
            "artifacts": eval_result.artifacts,
            "result_path": eval_result.result_path,
        }
        if not eval_result.succeeded:
            report["failure_reports"] = self._failure(eval_result, eval_spec)
            report["report_path"] = self.save_report(report)
            return report

        # ---- 3) lineage + acceptance --------------------------------------
        ckpt = train_result.checkpoint or {}
        version_id = model_version or f"mv_{exp_id}"
        mv = ModelVersion(
            version_id=version_id,
            experiment_id=exp_id,
            job_id=str(train_job),
            engine=train_result.engine,
            checkpoint_path=ckpt.get("path"),
            checkpoint_sha256=ckpt.get("sha256"),
            config_path=config_path,
            config_sha256=self._config_sha256(config_path),
            code_version=self._code_version(),
            dataset_version=(eval_result.dataset_version
                             or train_result.dataset_version or dataset_version),
            parent_version_id=train_spec.model_version,
            metrics=eval_result.metrics,
        )
        self.registry.register(mv)
        if promote:
            mv = self.registry.promote(version_id, "P0 baseline accepted") or mv

        passed, violations = check_success_criteria(
            objective.success_criteria, eval_result.metrics
        )
        report["model_version"] = mv.model_dump()
        report["success_criteria"] = {
            "criteria": objective.success_criteria,
            "passed": passed,
            "violations": violations,
        }
        report["success"] = passed and bool(eval_result.metrics)
        report["failure_reports"] = [
            r.model_dump() for r in self.diagnoser.diagnose_all(eval_result)
        ]
        self.evidence.append("objective_complete", {
            "experiment_id": exp_id,
            "objective_id": objective.objective_id,
            "success": report["success"],
            "model_version": version_id,
            "metrics": eval_result.metrics,
        })
        report["report_path"] = self.save_report(report)
        return report

    # -------------------------------------------------------------- summary
    @staticmethod
    def summarize(report: dict) -> str:
        """Compact human-readable line for an Agent message / log."""
        metrics = (report.get("evaluate") or {}).get("metrics") or {}
        head = ", ".join(f"{k}={v:.5f}" for k, v in list(metrics.items())[:4])
        return (
            f"experiment={report.get('experiment_id')} "
            f"success={report.get('success')} "
            f"model_version={(report.get('model_version') or {}).get('version_id')} "
            f"metrics=[{head}] "
            f"failures={len(report.get('failure_reports') or [])}"
        )
