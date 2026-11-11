"""LangChain tools exposing the research loop to the Supervisor's agents.

Opt-in: these are only imported when ``AGENT_RESEARCH_TOOLS=1``. With the flag
off, ``agents.py`` produces the EXACT same tool lists as before (zero
regression). See ``agents.py:create_agents``.

Two tool families
-----------------
* ``RESEARCH_TOOLS`` (P0, 4 tools): single train / evaluate / status / baseline.
* ``CAMPAIGN_RESEARCH_TOOLS`` (P1-06, 11 tools): the approved research loop.
  These are thin JSON adapters over ``research_service.ResearchService``; the
  service re-reads every trust input from durable storage before calling the
  EXISTING P1-04 CampaignRunner / P1-05 CampaignComparator.

Contract with the LLM
---------------------
* Every tool returns a JSON *string*: ``{"ok": true, ...}`` or
  ``{"ok": false, "error_code": ..., "error": ...}``. Exceptions never escape
  into the agent's quality gate as opaque failures.
* Submit tools return only identifiers (job_id / run_id / paths). Large
  artefacts and full reports are written to disk / SQLite and referenced by id.
* No tool can grant a human approval, submit an arbitrary config payload, or
  evaluate on the reserved frozen final split.
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


# --------------------------------------------------------------------------- #
# P1-06: campaign-level research tools (A2)
# --------------------------------------------------------------------------- #
# The LLM sees a tiny, identifier-only surface. The service re-reads the plan,
# the hash-bound human approval, the persisted Critic verdicts, the config
# whitelist, the frozen dataset and the budget BEFORE calling the EXISTING
# P1-04 CampaignRunner / P1-05 Comparator. There is deliberately NO tool that
# can grant an approval, submit an arbitrary config_json, or evaluate on the
# reserved frozen final split.
@lru_cache(maxsize=1)
def _service():
    from .domain_registry import default_registry
    from .research_environment import build_default_research_service
    # ARR 0.1 / R2: a domain is selected by name, never by editing the generic
    # service. ARR_DOMAIN is an opt-in escape hatch; unknown values fall back to
    # the default SciNet bundle so the agent tool surface is unchanged.
    domain_name = os.environ.get("ARR_DOMAIN", "scinet")
    if not default_registry.contains(domain_name):
        domain_name = "scinet"
    return build_default_research_service(domain_name=domain_name)


def _from_service(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


@tool
def start_research_session(objective_id: str, description: str = "",
                           success_criteria_json: str = "",
                           session_id: str = "") -> str:
    """Register a research objective and open a research session. Safe.

    A session links objective / hypotheses / reviews / plans / approvals /
    campaigns / comparison reports. By default an objective allows at most TWO
    campaign rounds, so an inconclusive result cannot loop forever.
    """
    try:
        return _from_service(_service().start_research_session(
            objective_id=objective_id, description=description,
            success_criteria_json=success_criteria_json, session_id=session_id))
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@tool
def propose_research_hypotheses(objective_id: str, source_experiment_id: str,
                                session_id: str = "",
                                hypotheses_json: str = "") -> str:
    """Propose evidence-grounded hypotheses from a RECORDED experiment. Safe.

    Evidence is re-read by the server from the append-only ledger; the Agent
    cannot inject metrics. ``hypotheses_json`` optionally accepts a structured
    operator/LLM proposal, but its ids and evidence source are overwritten and
    it still needs a deterministic Critic review + human approval.
    """
    try:
        return _from_service(_service().propose_research_hypotheses(
            objective_id=objective_id,
            source_experiment_id=source_experiment_id,
            session_id=session_id, hypotheses_json=hypotheses_json))
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@tool
def review_research_hypotheses(hypothesis_ids_json: str,
                               session_id: str = "") -> str:
    """Run the DETERMINISTIC Critic and persist the real verdict. Safe.

    ``hypothesis_ids_json`` is a JSON array of ids, e.g. '["H1"]'. An LLM
    advisor may add notes but can never turn a rejection into an approval.
    The verdict is bound to the hypothesis content SHA-256.
    """
    try:
        return _from_service(_service().review_research_hypotheses(
            hypothesis_ids_json=hypothesis_ids_json, session_id=session_id))
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@tool
def create_research_plan(objective_id: str, hypothesis_ids_json: str,
                         session_id: str = "", plan_id: str = "") -> str:
    """Create a versioned, SHA-256-addressed ExperimentPlan. Safe/warn.

    Only hypotheses with a PERSISTED passing Critic review may become
    candidates. Dataset, seeds, metric protocol and guardrails come from the
    injected domain policy, NOT from the Agent.
    """
    try:
        return _from_service(_service().create_research_plan(
            objective_id=objective_id,
            hypothesis_ids_json=hypothesis_ids_json,
            session_id=session_id, plan_id=plan_id))
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@tool
def request_plan_approval(plan_id: str, requested_by: str = "research_agent",
                          session_id: str = "") -> str:
    """Request HUMAN approval for a plan; returns a PENDING approval_id. Warn.

    The returned ``human_action`` is a terminal command a human must run. No
    Agent tool can grant an approval, and a pending record cannot execute
    anything.
    """
    try:
        return _from_service(_service().request_plan_approval(
            plan_id=plan_id, requested_by=requested_by, session_id=session_id))
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@tool
def run_research_campaign(plan_id: str, approval_id: str,
                          session_id: str = "") -> str:
    """Execute an APPROVED plan with the existing CampaignRunner. DANGER/HITL.

    The server re-validates plan hash, human approval source + scope, Critic
    verdicts, config whitelist, frozen dataset and budget. Duplicate calls
    return the existing campaign status instead of launching new jobs. This
    call is synchronous (P1-04 runner is serial). No model is promoted.
    """
    try:
        return _from_service(_service().run_research_campaign(
            plan_id=plan_id, approval_id=approval_id, session_id=session_id))
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@tool
def get_campaign_status(campaign_id: str) -> str:
    """Read-only campaign status for a campaign_id. Safe."""
    try:
        return _from_service(_service().get_campaign_status(
            campaign_id=campaign_id))
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@tool
def resume_research_campaign(campaign_id: str, approval_id: str = "",
                             session_id: str = "") -> str:
    """Resume a PAUSED campaign via the existing runner. DANGER/HITL.

    Rejects running, terminal and unbound campaigns; re-checks the bound
    human approval before calling CampaignRunner.resume().
    """
    try:
        return _from_service(_service().resume_research_campaign(
            campaign_id=campaign_id, approval_id=approval_id,
            session_id=session_id))
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@tool
def compare_research_campaign(campaign_id: str, session_id: str = "") -> str:
    """Run the P1-05 paired Comparator on a COMPLETED campaign. Safe.

    Produces immutable ComparisonReports, then the Research Reviewer writes an
    Experience Store record and a structured ResearchReport. A verdict is NOT
    a promotion and never uses the frozen final split.
    """
    try:
        return _from_service(_service().compare_research_campaign(
            campaign_id=campaign_id, session_id=session_id))
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@tool
def get_comparison_report(campaign_id: str, comparison_id: str = "") -> str:
    """Read persisted ComparisonReport(s) for a campaign. Safe.

    With no comparison_id, returns report metadata. A missing report is an
    explicit error -- no result is ever fabricated.
    """
    try:
        return _from_service(_service().get_comparison_report(
            campaign_id=campaign_id, comparison_id=comparison_id))
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@tool
def get_research_report(session_id: str) -> str:
    """Read the immutable final ResearchReport for a session. Safe."""
    try:
        return _from_service(_service().get_research_report(
            session_id=session_id))
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


CAMPAIGN_RESEARCH_TOOLS = [
    start_research_session,
    propose_research_hypotheses,
    review_research_hypotheses,
    create_research_plan,
    request_plan_approval,
    run_research_campaign,
    get_campaign_status,
    resume_research_campaign,
    compare_research_campaign,
    get_comparison_report,
    get_research_report,
]

# What agents.py registers when AGENT_RESEARCH_TOOLS=1. With the flag OFF the
# six role tool lists stay byte-for-byte identical to pre-P1-06 behaviour.
ALL_RESEARCH_TOOLS = RESEARCH_TOOLS + CAMPAIGN_RESEARCH_TOOLS
