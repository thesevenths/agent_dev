"""M4-E2E-01B -- Director recovery (read-only reconciliation).

The Director's persisted state may be stale because the *process* crashed or was
interrupted while the *research engine* kept making progress (or failed). Before
re-executing anything, recovery reads the ACTUAL campaign / session / approval
status from the ``ResearchService`` (which only wraps the trusted stores) and
decides the Director's true current state.

Hard constraints (spec M4-E2E-01B -- "No bypassing gates via recovery")
-----------------------------------------------------------------------
  * READ-ONLY. ``reconcile_director_state`` never grants an approval, never
    starts a campaign, never re-runs anything, never modifies a verdict or a
    historical report.
  * It may only RESOLVE the Director to a concrete *active* state that reflects
    what already happened. It cannot jump a run to COMPLETED unless a real
    ResearchReport exists.
  * M4.3-00 / P0-04: a Report existing does NOT mean Experience exists.
    COMPLETED requires THREE independent proofs: a real ResearchReport, a
    NON-EMPTY ``experience_refs`` list on that report, AND real Experience rows
    in the Experience Store whose ids match those refs. If any is missing, the
    run is resolved back to ``RECORDING_EXPERIENCE`` (the honest in-progress
    state) -- never to COMPLETED.
  * If the reconciliation finds a missing approval, a failed campaign, or an
    unapproved plan, it reports that honestly and lets the (human) operator
    decide how to proceed -- recovery does NOT invent approvals.

The function returns ``(target_state, detail)`` where ``target_state`` is one of
the active states or a halted-recoverable state (e.g. ``FAILED`` when the
campaign itself failed). The Director applies it via the state machine; the
state machine is what guarantees the transition is legal.
"""
from __future__ import annotations

from typing import Any

from .director_state_machine import (
    S_AWAITING_APPROVAL, S_COMPLETED, S_EVALUATING, S_EXECUTING,
    S_FAILED, S_HYPOTHESIS_READY, S_IDLE, S_OBJECTIVE_SELECTED, S_PLAN_READY,
    S_RECORDING_EXPERIENCE,
)


def _campaign_is_terminal(status: str | None) -> bool:
    return status in {"completed", "failed", "cancelled", "incomplete"}


def report_has_real_experience(service: Any, session_id: str,
                               report: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """M4.3-00 / P0-04 -- verify Experience is REAL, not just a report row.

    Returns ``(verified, detail)``. Verified requires ALL THREE:
      (1) a real ResearchReport (``report_id`` present),
      (2) a NON-EMPTY ``experience_refs`` list on the report,
      (3) real rows in the Experience Store whose ``experience_id`` values
          intersect those refs (a report that merely *claims* a ref is not
          sufficient -- the Experience Store is the authority).
    """
    detail: dict[str, Any] = {}
    report_id = report.get("report_id")
    refs = [str(r) for r in (report.get("experience_refs") or []) if str(r)]
    detail["report_id_present"] = bool(report_id)
    detail["experience_refs_count"] = len(refs)
    detail["report_present"] = bool(report)
    if not report_id or not refs:
        return False, detail
    try:
        rows = service.stores.sessions.experience(session_id) or []
    except Exception:  # store unreadable -> fail closed, never claim COMPLETED
        detail["experience_store_error"] = True
        return False, detail
    stored_ids = {str(r.get("experience_id") or "") for r in rows}
    matched = [r for r in refs if r in stored_ids]
    detail["experience_store_rows"] = len(rows)
    detail["experience_refs_matched"] = len(matched)
    return (len(matched) > 0), detail


def reconcile_director_state(service: Any, *,
                             session_id: str = "",
                             plan_id: str = "",
                             approval_id: str = "",
                             campaign_id: str = "") -> tuple[str, dict[str, Any]]:
    """Decide the Director's true state from the live ResearchService.

    ``service`` is the real ``ResearchService`` (or a Harness-backed one). Only
    its READ methods are used: ``get_campaign_status``, ``get_research_report``,
    and the trusted stores accessed through ``service.stores``.

    Returns ``(target_state, detail)``.
    """
    detail: dict[str, Any] = {
        "session_id": session_id, "plan_id": plan_id,
        "approval_id": approval_id, "campaign_id": campaign_id,
    }

    # 1) A campaign exists -> inspect its real status first.
    if campaign_id:
        status = service.get_campaign_status(campaign_id=campaign_id)
        detail["campaign_status"] = status.get("status")
        cs = status.get("status")
        if cs in {"running", "paused"}:
            # Still executing on the engine; resume corresponds to EXECUTING.
            return S_EXECUTING, detail
        if cs in {"completed", "cancelled", "incomplete"}:
            # M4.3-00 / P0-04: a Report existing is NOT proof that Experience
            # was written. COMPLETED requires a real Report + non-empty
            # experience_refs + REAL Experience Store records.
            try:
                rep = service.get_research_report(session_id=session_id)
            except Exception:
                rep = None
            if rep and rep.get("ok") and rep.get("report"):
                verified, xp = report_has_real_experience(
                    service, session_id, rep["report"])
                detail.update(xp)
                detail["experience_verified"] = verified
                if verified:
                    return S_COMPLETED, detail
                # Report exists but Experience is missing / unmatched: resolve
                # back to the honest in-progress state, NEVER to COMPLETED.
                detail["reason"] = (
                    "report exists but Experience is missing or unverified; "
                    "COMPLETED is not legitimate")
                return S_RECORDING_EXPERIENCE, detail
            # Campaign done but no report yet -> need comparison (EVALUATING).
            detail["report_present"] = False
            return S_EVALUATING, detail
        if cs == "failed":
            # The campaign itself failed. We do NOT auto-restart it; we surface
            # the failure as a halted-recoverable state and let the operator
            # decide (re-plan / re-approve / stop).
            detail["reason"] = "campaign failed; operator must decide next step"
            return S_FAILED, detail
        # Unknown campaign status -> be conservative, stay at EXECUTING so the
        # operator can reconcile explicitly.
        detail["reason"] = f"unknown campaign status {cs!r}"
        return S_EXECUTING, detail

    # 2) No campaign yet, but an approval was requested -> inspect approval.
    if approval_id:
        approval = None
        try:
            approval = service.stores.approvals.get(approval_id)
        except Exception:
            approval = None
        detail["approval_found"] = approval is not None
        if approval is not None:
            detail["approval_status"] = getattr(approval, "status", None)
            detail["approval_resource_sha256"] = getattr(
                approval, "resource_sha256", None)
            if getattr(approval, "status", None) == "approved":
                # Approved and ready to execute. The Director's execute() still
                # re-verifies the plan SHA against the approval before running.
                return S_AWAITING_APPROVAL, detail
            if getattr(approval, "status", None) in {"rejected", "revoked"}:
                # Approval was denied -> back to PLAN_READY so a new plan /
                # approval can be produced. Recovery never grants anything.
                detail["reason"] = "approval rejected/revoked; re-plan needed"
                return S_PLAN_READY, detail
            # Pending (or other) -> still waiting for the human.
            return S_AWAITING_APPROVAL, detail
        # Approval id referenced but not found -> re-request from PLAN_READY.
        detail["reason"] = "approval not found; re-request needed"
        return S_PLAN_READY, detail

    # 3) A plan exists -> PLAN_READY (awaiting approval request).
    if plan_id:
        plan = None
        try:
            plan = service.stores.plans.get(plan_id)
        except Exception:
            plan = None
        detail["plan_found"] = plan is not None
        if plan is not None:
            return S_PLAN_READY, detail
        # A plan id we cannot resolve -> something is inconsistent; fall through.

    # 4) Hypotheses reviewed/approved but no plan -> HYPOTHESIS_READY.
    if session_id:
        session = None
        try:
            session = service.stores.sessions.get_session(session_id)
        except Exception:
            session = None
        if session is not None:
            detail["session_status"] = getattr(session, "status", None)
            n_plans = len(getattr(session, "plan_ids", []) or [])
            n_hyp = len(getattr(session, "hypothesis_ids", []) or [])
            if n_plans > 0:
                return S_PLAN_READY, detail
            if n_hyp > 0:
                return S_HYPOTHESIS_READY, detail
            return S_OBJECTIVE_SELECTED, detail

    # 5) Nothing resolved -> back to IDLE (a fresh start is safe; the operator
    #    re-selects the objective).
    detail["reason"] = "no recoverable progress found"
    return S_IDLE, detail


def recovery_target_from_history(service: Any, *, director_context: dict[str, Any]
                                 ) -> tuple[str, dict[str, Any]]:
    """Reconcile using the persisted Director run context (ids)."""
    return reconcile_director_state(
        service,
        session_id=director_context.get("session_id", ""),
        plan_id=director_context.get("plan_id", ""),
        approval_id=director_context.get("approval_id", ""),
        campaign_id=director_context.get("campaign_id", ""),
    )


__all__ = ["reconcile_director_state", "recovery_target_from_history",
           "report_has_real_experience"]
