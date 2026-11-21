"""P0 research flow: train -> evaluate on the SAME run -> register evidence.

Scope is deliberately narrow: P0 does NOT attempt self-improvement. Its only
job is to prove that the Agent can reliably

    submit a job -> get a job_id -> wait -> read structured metrics
    -> record checkpoint lineage -> produce an auditable report

and to produce a complete FailureReport when anything goes wrong.

Two scoreboards are NEVER conflated (see ``evaluate_objective``):

    pipeline_success   train -> evaluate -> lineage completed with metrics
    objective_success  a DECLARED, DECIDABLE research target was met

A perfectly executed run with no target is ``undefined``, not a success --
otherwise the self-improvement loop would stop before it ever started.
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


def _rule_error(rule: Any) -> str | None:
    """Validate ONE acceptance rule. None means the rule is decidable.

    Engine-agnostic by design: a criterion is either a number (upper bound) or
    an object with numeric ``min`` / ``max`` bounds. Anything else (bool, str,
    empty object, non-numeric bound) must NOT be silently skipped -- a silently
    skipped criterion is an acceptance gate that is loose by accident.
    """
    if isinstance(rule, bool) or not isinstance(rule, (int, float, dict)):
        return "must be a number or an object with min/max"
    if isinstance(rule, (int, float)):
        return None
    bounds = {key: rule[key] for key in ("min", "max") if key in rule}
    if not bounds:
        return "object must define 'min' and/or 'max'"
    for key, bound in bounds.items():
        if isinstance(bound, bool) or not isinstance(bound, (int, float)):
            return f"bound {key!r} must be a number"
    return None


def criteria_are_decidable(criteria: dict[str, Any]) -> bool:
    """A research objective is decidable ONLY if it declares >=1 valid rule.

    An empty criteria dict is vacuously true in ``check_success_criteria``, but
    'no target defined' must never be reported as 'target achieved'.
    """
    criteria = criteria or {}
    return bool(criteria) and all(
        _rule_error(rule) is None for rule in criteria.values()
    )


def check_success_criteria(criteria: dict[str, Any],
                           metrics: dict[str, float]) -> tuple[bool, list[str]]:
    """Generic, engine-agnostic acceptance check (PURE COMPARISON ONLY).

    ``{"latent_mse@1": 0.05}``            -> upper bound
    ``{"cosine@1": {"min": 0.8}}``        -> lower bound
    ``{"mse": {"max": 0.05, "min": 0.0}}`` -> both

    This function answers "do the numbers satisfy the declared bounds?". It
    deliberately does NOT answer "was the research objective achieved?" -- that
    also requires a decidable objective and a successful pipeline. Use
    ``evaluate_objective`` for the latter.
    """
    failures: list[str] = []
    for name, rule in (criteria or {}).items():
        error = _rule_error(rule)
        if error:
            failures.append(f"metric {name!r} has invalid criterion: {error}")
            continue
        if name not in metrics:
            failures.append(f"metric {name!r} missing from result")
            continue
        value = metrics[name]
        if isinstance(rule, (int, float)):
            if value > float(rule):
                failures.append(f"{name}={value} exceeds max {rule}")
        else:
            if "max" in rule and value > float(rule["max"]):
                failures.append(f"{name}={value} exceeds max {rule['max']}")
            if "min" in rule and value < float(rule["min"]):
                failures.append(f"{name}={value} below min {rule['min']}")
    return (not failures), failures


def evaluate_objective(criteria: dict[str, Any],
                       metrics: dict[str, float], *,
                       pipeline_success: bool) -> dict[str, Any]:
    """Two scoreboards, never conflated.

    ``pipeline_success``   : did train -> evaluate -> lineage actually run?
    ``objective_success``  : did the model meet a DECLARED, DECIDABLE target?

    A run that executed perfectly with no target (or a broken target) is
    ``undefined``, NOT a success. A run that missed a real target is
    ``not_achieved``. Only ``achieved`` may stop the research loop.
    """
    passed, violations = check_success_criteria(criteria, metrics)
    decidable = criteria_are_decidable(criteria)
    if not pipeline_success:
        status = "pipeline_failed"
    elif not decidable:
        status = "undefined"
    elif not passed:
        status = "not_achieved"
    else:
        status = "achieved"
    return {
        "status": status,
        "objective_success": status == "achieved",
        "pipeline_success": bool(pipeline_success),
        "passed": passed,
        "decidable": decidable,
        "violations": violations,
        "criteria": dict(criteria or {}),
    }


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

    @staticmethod
    def _repo_version(project: str | Path | None) -> dict[str, Any] | None:
        """HEAD + worktree dirtiness for ONE repository.

        ``git status --porcelain -- .`` is scoped to the project directory: a
        parent monorepo may contain unrelated sibling projects whose edits must
        not make this experiment look unreproducible.
        """
        if not project:
            return None
        root = Path(project)
        if not root.is_dir():
            return None
        try:
            head = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], cwd=str(root),
                capture_output=True, text=True, timeout=5,
            )
            if head.returncode != 0:
                return None
            dirty = subprocess.run(
                ["git", "status", "--porcelain", "--", "."], cwd=str(root),
                capture_output=True, text=True, timeout=10,
            )
            return {
                "head": head.stdout.strip() or None,
                "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
                "root": str(root),
            }
        except (OSError, subprocess.TimeoutExpired):
            return None

    def _code_versions(self) -> dict[str, Any]:
        """BOTH repos matter: engine code and Agent runtime code.

        Recording only the engine HEAD lets a report claim reproducibility
        while the orchestrator that produced it was an uncommitted worktree.
        """
        engine = getattr(self.manager.adapter, "project_dir", None)
        runtime = Path(__file__).resolve().parents[1]
        return {
            "engine": self._repo_version(engine),
            "runtime": self._repo_version(runtime),
        }

    def _code_version(self) -> str | None:
        """Backwards-compatible engine HEAD, kept for ``ModelVersion``."""
        info = self._repo_version(
            getattr(self.manager.adapter, "project_dir", None)
        )
        return info["head"] if info else None

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
                     promote: bool = False,
                     evaluation_split: str = "test") -> dict[str, Any]:
        """Train a baseline, evaluate it on the same run_id, register lineage.

        ``evaluation_split`` defaults to the frozen test split. P1 candidate
        selection must pass ``validation`` and keep test untouched.
        """
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

        versions = self._code_versions()
        report: dict[str, Any] = {
            "objective": objective.model_dump(),
            "experiment_id": exp_id,
            "engine": self.manager.adapter.name,
            "config_path": config_path,
            "evaluation_split": evaluation_split,
            "code_versions": versions,
            # Execution scoreboard. Set True only after BOTH stages succeed.
            "pipeline_success": False,
            # Research scoreboard. "pipeline_failed" until proven otherwise.
            "objective_status": "pipeline_failed",
            "objective_success": False,
            # DEPRECATED alias: kept for older Agent prompts / consumers.
            # It now means objective_success, never pipeline_success.
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
            train_result.run_id, stage="evaluate", parent_job_id=str(train_job),
            split=evaluation_split,   # never inherit the train spec's split
        )
        eval_job = self.manager.submit(eval_spec)["job_id"]
        eval_status = self.manager.wait(eval_job, timeout=timeout)
        eval_result = self.manager.collect_result(eval_job, eval_spec)
        report["evaluate"] = {
            "job_id": eval_job,
            "status": eval_result.status,
            "wait_status": eval_status,
            "run_id": eval_result.run_id,
            "evaluation_split": eval_result.evaluation_split,
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
        engine_info = versions.get("engine") or {}
        runtime_info = versions.get("runtime") or {}
        dirty_flags = [info.get("dirty") for info in (engine_info, runtime_info)]
        code_dirty = (None if any(flag is None for flag in dirty_flags)
                      else any(dirty_flags))
        mv = ModelVersion(
            version_id=version_id,
            experiment_id=exp_id,
            job_id=str(train_job),
            engine=train_result.engine,
            checkpoint_path=ckpt.get("path"),
            checkpoint_sha256=ckpt.get("sha256"),
            config_path=config_path,
            config_sha256=self._config_sha256(config_path),
            code_version=engine_info.get("head"),
            runtime_code_version=runtime_info.get("head"),
            code_dirty=code_dirty,
            dataset_version=(eval_result.dataset_version
                             or train_result.dataset_version or dataset_version),
            parent_version_id=train_spec.model_version,
            metrics=eval_result.metrics,
        )
        self.registry.register(mv)
        if promote:
            mv = self.registry.promote(version_id, "P0 baseline accepted") or mv

        # Execution succeeded; the research verdict is a SEPARATE question.
        verdict = evaluate_objective(
            objective.success_criteria, eval_result.metrics,
            pipeline_success=eval_result.succeeded and bool(eval_result.metrics),
        )
        report["pipeline_success"] = True
        report["objective_status"] = verdict["status"]
        report["objective_success"] = verdict["objective_success"]
        report["success"] = verdict["objective_success"]  # deprecated alias
        report["model_version"] = mv.model_dump()
        report["success_criteria"] = {
            "criteria": verdict["criteria"],
            "passed": verdict["passed"],
            "decidable": verdict["decidable"],
            "violations": verdict["violations"],
            "status": verdict["status"],
        }
        report["failure_reports"] = [
            r.model_dump() for r in self.diagnoser.diagnose_all(eval_result)
        ]
        self.evidence.append("objective_complete", {
            "experiment_id": exp_id,
            "objective_id": objective.objective_id,
            "pipeline_success": report["pipeline_success"],
            "objective_success": report["objective_success"],
            "objective_status": report["objective_status"],
            "success": report["success"],  # deprecated alias
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
            f"pipeline_success={report.get('pipeline_success')} "
            f"objective_status={report.get('objective_status')} "
            f"objective_success={report.get('objective_success')} "
            f"success={report.get('success')} "
            f"model_version={(report.get('model_version') or {}).get('version_id')} "
            f"metrics=[{head}] "
            f"failures={len(report.get('failure_reports') or [])}"
        )
