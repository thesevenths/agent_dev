"""LangChain tools exposing the research loop to the Supervisor's agents.

Opt-in: these are only imported when ``AGENT_RESEARCH_TOOLS=1``. With the flag
off, ``agents.py`` produces the EXACT same tool lists as before (zero
regression). See ``agents.py:create_agents``.

Contract with the LLM
---------------------
* Every tool returns a JSON *string*: ``{"ok": true, ...}`` or
  ``{"ok": false, "error": ...}``. Exceptions never escape into the agent's
  quality gate as opaque failures.
* Submit tools return only identifiers (job_id / run_id / paths). Large
  artefacts and full reports are written to disk and referenced by path.
* ``get_experiment_status`` is read-only; submit tools are serialised by a
  module-level lock so concurrent tool calls cannot race on the job store.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from functools import lru_cache
from typing import Any

try:  # keep this module importable (and unit-testable) without LangChain
    from langchain_core.tools import tool
except ImportError:  # pragma: no cover - lightweight/test interpreters
    def tool(func):  # type: ignore[no-redef]
        return func


_SUBMIT_LOCK = threading.Lock()


def research_tools_enabled() -> bool:
    return os.environ.get("AGENT_RESEARCH_TOOLS", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _ok(**payload: Any) -> str:
    return json.dumps({"ok": True, **payload}, ensure_ascii=False, default=str)


def _err(exc: BaseException) -> str:
    return json.dumps(
        {"ok": False, "error": str(exc), "error_type": type(exc).__name__},
        ensure_ascii=False, default=str,
    )


@lru_cache(maxsize=1)
def _manager():
    from .adapters import SciNetAdapter
    from .experiment_manager import ExperimentManager
    return ExperimentManager(SciNetAdapter())


@lru_cache(maxsize=1)
def _orchestrator():
    from .orchestrator import ResearchOrchestrator
    return ResearchOrchestrator(_manager())


def _default_config() -> str:
    from .settings import DEFAULT_CONFIG
    return DEFAULT_CONFIG


@tool
def train_world_model(experiment_id: str, config_path: str = "",
                      epochs: int = 1, model_version: str = "",
                      dataset_version: str = "") -> str:
    """Submit an ASYNC world-model training job; returns a job_id immediately.

    Use get_experiment_status(job_id) to wait for completion. The training runs
    in a separate process with its own Python environment; it never blocks the
    agent. ``config_path`` is relative to the SciNet++ project root; leave empty
    to use SCINET_CONFIG.
    """
    try:
        from .schemas import ExperimentSpec
        spec = ExperimentSpec(
            experiment_id=experiment_id,
            objective_id=experiment_id,
            engine="scinet++",
            stage="train",
            config_path=config_path or _default_config(),
            model_version=model_version or None,
            dataset_version=dataset_version or None,
            budget={"epochs": int(epochs)},
        )
        with _SUBMIT_LOCK:
            record = _manager().submit(spec)
        return _ok(**record)
    except Exception as exc:  # noqa: BLE001 - tool contract: JSON, always
        return _err(exc)


@tool
def evaluate_world_model(experiment_id: str, run_id: str,
                         config_path: str = "", model_version: str = "") -> str:
    """Submit an ASYNC evaluation job for an existing training run.

    ``run_id`` MUST be the run_id returned by the training job -- the checkpoint
    lives in that run directory. Omitting it would silently create a new empty
    run and the evaluation would fail.
    """
    try:
        from .schemas import ExperimentSpec
        if not run_id:
            raise ValueError("run_id is required; use the training job's run_id")
        spec = ExperimentSpec(
            experiment_id=experiment_id,
            objective_id=experiment_id,
            engine="scinet++",
            stage="evaluate",
            config_path=config_path or _default_config(),
            model_version=model_version or None,
            run_id=run_id,
            parent_run_id=run_id,
        )
        with _SUBMIT_LOCK:
            record = _manager().submit(spec)
        return _ok(**record)
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@tool
def get_experiment_status(job_id: str, include_result: bool = True) -> str:
    """Query a research job. Returns status plus structured metrics when done.

    Safe to call repeatedly and after an Agent restart; state lives in the
    engine's SQLite JobStore, not in this process.
    """
    try:
        manager = _manager()
        status = manager.get_status(job_id)
        payload: dict[str, Any] = {"job_id": job_id, "status": status}
        if include_result and status in ("succeeded", "failed", "cancelled", "incomplete"):
            result = manager.collect_result(job_id)
            payload.update({
                "stage": result.stage,
                "run_id": result.run_id,
                "metrics": result.metrics,
                "artifacts": result.artifacts,
                "error_type": result.error_type,
                "error": (result.error or "")[:800] or None,
                "result_path": result.result_path,
                "checkpoint": result.checkpoint,
            })
        return _ok(**payload)
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@tool
def run_research_baseline(description: str = "Train and evaluate a baseline world model",
                          config_path: str = "", epochs: int = 1,
                          success_criteria_json: str = "") -> str:
    """Run the FULL P0 loop in one call: train -> evaluate (same run) -> register.

    Returns only a compact summary plus report_path; the full auditable report
    is written to disk. ``success_criteria_json`` accepts e.g.
    ``{"latent_mse@1": 0.05}`` (upper bound) or ``{"cosine@1": {"min": 0.8}}``.

    Two separate scoreboards are returned: ``pipeline_success`` means the jobs
    ran; ``objective_success`` means a DECLARED, DECIDABLE target was met.
    Omitting ``success_criteria_json`` yields ``objective_status="undefined"``
    and ``success=false`` -- a run with no target is never a research success.
    The legacy ``success`` field is an alias of ``objective_success``.
    """
    try:
        from .orchestrator import ResearchOrchestrator
        from .schemas import ResearchObjective
        criteria = json.loads(success_criteria_json) if success_criteria_json else {}
        if not isinstance(criteria, dict):
            raise ValueError("success_criteria_json must be a JSON object")
        objective = ResearchObjective(
            objective_id=f"obj_{uuid.uuid4().hex[:10]}",
            domain="science",
            description=description,
            success_criteria=criteria,
        )
        report = _orchestrator().run_baseline(
            objective, config_path=config_path or _default_config(),
            epochs=int(epochs),
        )
        return _ok(
            summary=ResearchOrchestrator.summarize(report),
            success=report.get("success"),  # deprecated alias
            pipeline_success=report.get("pipeline_success"),
            objective_success=report.get("objective_success"),
            objective_status=report.get("objective_status"),
            objective_violations=(report.get("success_criteria") or {}).get(
                "violations", []),
            experiment_id=report.get("experiment_id"),
            metrics=(report.get("evaluate") or {}).get("metrics", {}),
            model_version=(report.get("model_version") or {}).get("version_id"),
            failure_count=len(report.get("failure_reports") or []),
            report_path=report.get("report_path"),
        )
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


RESEARCH_TOOLS = [
    train_world_model,
    evaluate_world_model,
    get_experiment_status,
    run_research_baseline,
]
