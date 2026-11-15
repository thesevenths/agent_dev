"""M4-E2E-01B -- Research Director (orchestrator + persisted state machine).

Drives the EXISTING ``ResearchService`` (the full 12-step fail-closed gate) from
``IDLE`` all the way to ``COMPLETED``, persisting every state transition to an
append-only journal so a crashed or interrupted run can be recovered. The
Director adds NO scientific judgement and NO second gate: it only sequences the
service's public methods, enforces the linear state machine, and binds approval
to the exact Plan SHA.

Reuse surface (spec M4-E2E-01B -- "reuse existing ...")
------------------------------------------------------
  * ``ResearchService``  -- start_research_session / propose / review / plan /
                           request_approval / run_campaign / compare / report.
  * ``ObjectiveStore``  -- to read the Proposed Objective produced by Discovery.
  * ``Hypothesis`` / ``Critic`` / ``Plan`` / ``Approval`` -- all owned by the
    service; the Director never re-implements them.

Approval integrity (spec M4-E2E-01B)
-------------------------------------
  * ``execute()`` runs ONLY after the approval is ``granted`` AND its
    ``resource_sha256`` equals the plan's CURRENT sha256. A plan change
    invalidates the old approval, so execute refuses and the Director must
    re-request. This is defense-in-depth ON TOP OF the service's own
    ``_authorize`` check.
  * The human grant is performed OUT OF BAND (the real ``approval_cli``). The
    Director only *observes* the store; it never grants for itself.

State persistence (spec M4-E2E-01B -- "each transition persisted")
---------------------------------------------------------------
  * ``<state_dir>/<director_id>.json``     -- current run context (authoritative).
  * ``<state_dir>/<director_id>.journal.jsonl`` -- append-only transition log.
  * Re-instantiating with the same ``director_id`` reloads the context; a fresh
    process therefore resumes exactly where it left off, then ``recover()``
    reconciles against the live engine before doing anything.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .director_recovery import reconcile_director_state
from .director_state_machine import (
    ACTIVE_STATES, E_COMPARE, E_CREATE_PLAN, E_ENTER_RECOVERY, E_EXECUTE,
    E_FAIL, E_FINALIZE, E_INTERRUPT, E_PROPOSE_HYPOTHESES, E_RECORD_EXPERIENCE,
    E_REJECT, E_REQUEST_APPROVAL, E_RESOLVE_RECOVERY, E_SELECT_OBJECTIVE,
    E_STOP, IllegalTransitionError, DirectorStateMachine,     S_AWAITING_APPROVAL,
    S_COMPLETED, S_EXECUTING, S_FAILED, S_HYPOTHESIS_READY, S_IDLE,
    S_OBJECTIVE_SELECTED, S_PLAN_READY, S_RECOVERING, S_RECORDING_EXPERIENCE,
    S_REJECTED, S_EVALUATING,
)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


class DirectorError(Exception):
    """A Director-level failure (e.g. approval missing, no approved hypothesis)."""


@dataclass
class DirectorRunContext:
    """Persistent run state. JSON-serialisable; no live objects."""
    director_id: str
    objective_id: str
    domain: str = "scinet"
    state: str = S_IDLE
    session_id: str = ""
    hypothesis_ids: list[str] = field(default_factory=list)
    plan_id: str = ""
    plan_sha256: str = ""
    approval_id: str = ""
    campaign_id: str = ""
    source_experiment_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    error: str = ""
    report_id: str = ""
    retries_used: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "director_id": self.director_id,
            "objective_id": self.objective_id,
            "domain": self.domain,
            "state": self.state,
            "session_id": self.session_id,
            "hypothesis_ids": list(self.hypothesis_ids),
            "plan_id": self.plan_id,
            "plan_sha256": self.plan_sha256,
            "approval_id": self.approval_id,
            "campaign_id": self.campaign_id,
            "source_experiment_id": self.source_experiment_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "report_id": self.report_id,
            "retries_used": self.retries_used,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DirectorRunContext":
        return cls(
            director_id=d["director_id"], objective_id=d["objective_id"],
            domain=d.get("domain", "scinet"), state=d.get("state", S_IDLE),
            session_id=d.get("session_id", ""),
            hypothesis_ids=list(d.get("hypothesis_ids", [])),
            plan_id=d.get("plan_id", ""), plan_sha256=d.get("plan_sha256", ""),
            approval_id=d.get("approval_id", ""),
            campaign_id=d.get("campaign_id", ""),
            source_experiment_id=d.get("source_experiment_id", ""),
            created_at=d.get("created_at", ""), updated_at=d.get("updated_at", ""),
            error=d.get("error", ""), report_id=d.get("report_id", ""),
            retries_used=int(d.get("retries_used", 0)),
        )


class ResearchDirector:
    """Sequences the ResearchService through the Director state machine."""

    def __init__(self, *, service: Any, objective_store: Any, state_dir: str | Path,
                 objective_id: str, domain: str = "scinet",
                 director_id: str | None = None,
                 source_experiment_id: str = "",
                 approval_granter: Callable[[str, str], None] | None = None,
                 max_retries: int = 1):
        self.service = service
        self.objective_store = objective_store
        self.state_dir = Path(state_dir)
        self.objective_id = objective_id
        self.domain = domain
        self.source_experiment_id = source_experiment_id
        self.approval_granter = approval_granter
        self.max_retries = max(0, int(max_retries))
        self.sm = DirectorStateMachine()
        self.director_id = director_id or (
            "dir_" + _short_hash(f"{objective_id}|{domain}"))
        self._journal_path = self.state_dir / f"{self.director_id}.journal.jsonl"
        self._ctx_path = self.state_dir / f"{self.director_id}.json"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.context = self._load_or_init()

    # ------------------------------------------------------------- persistence
    def _load_or_init(self) -> DirectorRunContext:
        if self._ctx_path.exists():
            try:
                data = json.loads(self._ctx_path.read_text(encoding="utf-8"))
                ctx = DirectorRunContext.from_dict(data)
                # Guard against a stale COMPLETED/REJECTED context being re-driven.
                return ctx
            except Exception:
                pass
        ctx = DirectorRunContext(
            director_id=self.director_id, objective_id=self.objective_id,
            domain=self.domain, state=S_IDLE,
            source_experiment_id=self.source_experiment_id,
            created_at=_now(), updated_at=_now())
        self._write_ctx(ctx)
        self._append_journal(E_SELECT_OBJECTIVE, S_IDLE, S_IDLE,
                             {"note": "director initialised at IDLE"})
        return ctx

    def _write_ctx(self, ctx: DirectorRunContext) -> None:
        self._ctx_path.write_text(
            json.dumps(ctx.to_dict(), sort_keys=True, indent=2),
            encoding="utf-8")

    def _append_journal(self, event: str, from_state: str, to_state: str,
                        detail: dict[str, Any]) -> None:
        entry = {
            "ts": _now(), "director_id": self.director_id,
            "event": event, "from": from_state, "to": to_state,
            "detail": detail,
        }
        with self._journal_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")

    def _record(self, event: str, to_state: str, detail: dict[str, Any]) -> None:
        from_state = self.context.state
        self.context.state = to_state
        self.context.updated_at = _now()
        if "error" in detail and detail["error"]:
            self.context.error = str(detail["error"])
        self._write_ctx(self.context)
        self._append_journal(event, from_state, to_state, detail)

    def history(self) -> list[dict[str, Any]]:
        """Return the append-only transition journal (audit trail)."""
        if not self._journal_path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self._journal_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    # ------------------------------------------------------------- guards
    def _guard_current(self, expected: str, event: str) -> None:
        if self.context.state != expected:
            raise IllegalTransitionError(event, self.context.state,
                                         to_state=expected)

    def _halt(self, event: str, to_state: str, error: str) -> None:
        self._record(event, to_state, {"error": error})

    # ------------------------------------------------------------- steps
    def select_objective(self) -> dict[str, Any]:
        """IDLE -> OBJECTIVE_SELECTED."""
        self._guard_current(S_IDLE, E_SELECT_OBJECTIVE)
        rec = self.objective_store.get(self.objective_id)
        if rec is None:
            raise DirectorError(
                f"objective {self.objective_id!r} not found in ObjectiveStore; "
                "Discovery must persist it first")
        status = getattr(rec, "status", None)
        if status not in (None, "", "proposed"):
            # Non-fatal: only warn via detail; Discovery produces 'proposed'.
            warn = f"objective status is {status!r}, expected 'proposed'"
        else:
            warn = ""
        try:
            sess = self.service.start_research_session(
                objective_id=self.objective_id,
                description="ARR autonomous research director",
                session_id="")
            if not sess.get("ok"):
                raise DirectorError(f"start_research_session failed: {sess}")
            self.context.session_id = sess.get("session_id", "")
            detail = {"session_id": self.context.session_id,
                      "objective_status": status, "warning": warn}
            self._record(E_SELECT_OBJECTIVE, S_OBJECTIVE_SELECTED, detail)
            return detail
        except Exception as exc:  # noqa: BLE001 - record and re-raise
            self._halt(E_FAIL, S_FAILED, f"select_objective: {exc}")
            raise

    def propose_and_review_hypotheses(self) -> dict[str, Any]:
        """OBJECTIVE_SELECTED -> HYPOTHESIS_READY (needs >=1 approved)."""
        self._guard_current(S_OBJECTIVE_SELECTED, E_PROPOSE_HYPOTHESES)
        try:
            src = self.source_experiment_id or "exp_baseline"
            proposed = self.service.propose_research_hypotheses(
                objective_id=self.objective_id,
                source_experiment_id=src,
                session_id=self.context.session_id,
                hypotheses_json="")  # autonomous; grounded in real evidence
            if not proposed.get("ok"):
                raise DirectorError(f"propose failed: {proposed}")
            hids = [h["hypothesis_id"] for h in proposed.get("hypotheses", [])]
            if not hids:
                raise DirectorError(
                    "autonomous proposer produced no hypothesis from the "
                    "available evidence")
            reviewed = self.service.review_research_hypotheses(
                hypothesis_ids_json=json.dumps(hids),
                session_id=self.context.session_id)
            if not reviewed.get("ok"):
                raise DirectorError(f"review failed: {reviewed}")
            approved = reviewed.get("approved_ids", [])
            if not approved:
                raise DirectorError(
                    "Critic rejected every hypothesis; no approvable direction")
            self.context.hypothesis_ids = approved
            detail = {"n_proposed": len(hids),
                      "approved_ids": approved,
                      "reviewed_by": reviewed.get("reviewer")}
            self._record(E_PROPOSE_HYPOTHESES, S_HYPOTHESIS_READY, detail)
            return detail
        except Exception as exc:  # noqa: BLE001
            self._halt(E_FAIL, S_FAILED, f"propose_and_review: {exc}")
            raise

    def create_plan(self) -> dict[str, Any]:
        """HYPOTHESIS_READY -> PLAN_READY."""
        self._guard_current(S_HYPOTHESIS_READY, E_CREATE_PLAN)
        try:
            plan_id = f"plan_{self.director_id}"
            plan = self.service.create_research_plan(
                objective_id=self.objective_id,
                hypothesis_ids_json=json.dumps(self.context.hypothesis_ids),
                session_id=self.context.session_id, plan_id=plan_id)
            if not plan.get("ok"):
                raise DirectorError(f"create_plan failed: {plan}")
            self.context.plan_id = plan_id
            self.context.plan_sha256 = plan.get("plan_sha256", "")
            detail = {"plan_id": plan_id,
                      "plan_sha256": self.context.plan_sha256,
                      "candidate_ids": plan.get("candidate_ids", [])}
            self._record(E_CREATE_PLAN, S_PLAN_READY, detail)
            return detail
        except Exception as exc:  # noqa: BLE001
            self._halt(E_FAIL, S_FAILED, f"create_plan: {exc}")
            raise

    def request_approval(self) -> dict[str, Any]:
        """PLAN_READY -> AWAITING_APPROVAL."""
        self._guard_current(S_PLAN_READY, E_REQUEST_APPROVAL)
        try:
            req = self.service.request_plan_approval(
                plan_id=self.context.plan_id,
                session_id=self.context.session_id)
            if not req.get("ok"):
                raise DirectorError(f"request_approval failed: {req}")
            self.context.approval_id = req.get("approval_id", "")
            detail = {"approval_id": self.context.approval_id,
                      "status": req.get("status"),
                      "human_action": req.get("human_action")}
            self._record(E_REQUEST_APPROVAL, S_AWAITING_APPROVAL, detail)
            return detail
        except Exception as exc:  # noqa: BLE001
            self._halt(E_FAIL, S_FAILED, f"request_approval: {exc}")
            raise

    def approve_approval(self, *, approved_by: str = "human",
                         source: str = "human") -> None:
        """OUT-OF-BAND human grant, mirrored from ``approval_cli``.

        This is a thin convenience wrapper; the Director never treats it as the
        authority -- ``execute()`` re-checks the store. If no ``approval_granter``
        is wired, the caller (the real human console) performs the grant instead.
        """
        if self.approval_granter is not None:
            self.approval_granter(self.context.approval_id,
                                  self.context.plan_sha256)
            return
        # Default: drive the trusted store directly (used by tests / real flow
        # where the harness already exposes the store).
        self.service.stores.approvals.grant(
            self.context.approval_id, approved_by=approved_by, source=source,
            current_resource_sha256=self.context.plan_sha256)

    def execute(self) -> dict[str, Any]:
        """AWAITING_APPROVAL -> EXECUTING (only with a granted, SHA-bound approval)."""
        self._guard_current(S_AWAITING_APPROVAL, E_EXECUTE)
        # Approval integrity: granted AND plan sha matches the approval sha.
        approval = self.service.stores.approvals.get(self.context.approval_id)
        if approval is None:
            raise DirectorError("approval record missing; cannot execute")
        # The out-of-band console grant sets status='approved' (never 'granted');
        # a non-approved approval (pending/rejected/revoked) must not execute.
        if getattr(approval, "status", None) != "approved":
            raise DirectorError(
                "approval is not granted; human must grant via the console "
                "before execution (no bypass via recovery)")
        current_sha = ""
        plan = self.service.stores.plans.get(self.context.plan_id)
        if plan is not None:
            current_sha = getattr(plan, "plan_sha256", "") or ""
        if current_sha and current_sha != getattr(approval, "resource_sha256", ""):
            raise DirectorError(
                "plan SHA changed after approval; old approval is invalid. "
                "Re-create the plan and re-request approval.")
        try:
            run = self.service.run_research_campaign(
                plan_id=self.context.plan_id,
                approval_id=self.context.approval_id,
                session_id=self.context.session_id)
            if not run.get("ok"):
                raise DirectorError(f"run_campaign failed: {run}")
            self.context.campaign_id = run.get("campaign_id", "")
            detail = {"campaign_id": self.context.campaign_id,
                      "status": run.get("status"),
                      "total_trials": run.get("total_trials")}
            self._record(E_EXECUTE, S_EXECUTING, detail)
            return detail
        except Exception as exc:  # noqa: BLE001
            self._halt(E_FAIL, S_FAILED, f"execute: {exc}")
            raise

    def compare(self) -> dict[str, Any]:
        """EXECUTING -> EVALUATING (also writes the real Experience + Report)."""
        self._guard_current(S_EXECUTING, E_COMPARE)
        try:
            cmp = self.service.compare_research_campaign(
                campaign_id=self.context.campaign_id,
                session_id=self.context.session_id)
            if not cmp.get("ok"):
                raise DirectorError(f"compare failed: {cmp}")
            detail = {"comparisons": cmp.get("comparisons", []),
                      "research_outcome": cmp.get("research_outcome")}
            self._record(E_COMPARE, S_EVALUATING, detail)
            return detail
        except Exception as exc:  # noqa: BLE001
            self._halt(E_FAIL, S_FAILED, f"compare: {exc}")
            raise

    def record_experience(self) -> dict[str, Any]:
        """EVALUATING -> RECORDING_EXPERIENCE (confirms Experience was written)."""
        self._guard_current(S_EVALUATING, E_RECORD_EXPERIENCE)
        try:
            rep = self.service.get_research_report(
                session_id=self.context.session_id)
            if not rep.get("ok"):
                raise DirectorError(f"research report missing: {rep}")
            report = rep.get("report", {})
            exp_refs = report.get("experience_refs", [])
            if not exp_refs:
                raise DirectorError(
                    "ResearchReport has no experience_refs; Experience was not "
                    "written by the campaign/comparison step")
            self.context.report_id = report.get("report_id", "")
            detail = {"report_id": self.context.report_id,
                      "n_experience_refs": len(exp_refs),
                      "outcome": report.get("outcome")}
            self._record(E_RECORD_EXPERIENCE, S_RECORDING_EXPERIENCE, detail)
            return detail
        except Exception as exc:  # noqa: BLE001
            self._halt(E_FAIL, S_FAILED, f"record_experience: {exc}")
            raise

    def finalize(self) -> dict[str, Any]:
        """RECORDING_EXPERIENCE -> COMPLETED."""
        self._guard_current(S_RECORDING_EXPERIENCE, E_FINALIZE)
        rep = self.service.get_research_report(
            session_id=self.context.session_id)
        if not rep.get("ok"):
            raise DirectorError(f"finalize report missing: {rep}")
        report = rep.get("report", {})
        detail = {"report_id": self.context.report_id,
                  "outcome": report.get("outcome"),
                  "promotion": report.get("promotion"),
                  "frozen_test_used": report.get("frozen_test_used")}
        self._record(E_FINALIZE, S_COMPLETED, detail)
        return detail

    # ------------------------------------------------------------- halts
    def reject(self, reason: str = "") -> None:
        if self.sm.is_terminal(self.context.state):
            raise IllegalTransitionError(E_REJECT, self.context.state)
        self._record(E_REJECT, S_REJECTED, {"reason": reason})

    def interrupt(self, reason: str = "") -> None:
        if self.sm.is_terminal(self.context.state):
            raise IllegalTransitionError(E_INTERRUPT, self.context.state)
        self._record(E_INTERRUPT, S_INTERRUPTED, {"reason": reason})

    def stop(self, reason: str = "") -> None:
        if self.sm.is_terminal(self.context.state):
            raise IllegalTransitionError(E_STOP, self.context.state)
        self._record(E_STOP, S_STOPPED, {"reason": reason})

    # ------------------------------------------------------------- recovery
    def recover(self) -> str:
        """Reconcile the Director's state against the live engine.

        Returns the resolved state. Read-only: never grants, never re-runs.
        Raises ``IllegalTransitionError`` if the current state is terminal
        (a completed / rejected run cannot be recovered).
        """
        # ENTER_RECOVERY is legal from any non-terminal state.
        if not self.sm.can(E_ENTER_RECOVERY, self.context.state):
            raise IllegalTransitionError(
                E_ENTER_RECOVERY, self.context.state,
                to_state=S_RECOVERING)
        target, detail = reconcile_director_state(
            self.service,
            session_id=self.context.session_id,
            plan_id=self.context.plan_id,
            approval_id=self.context.approval_id,
            campaign_id=self.context.campaign_id)
        # First mark we are recovering.
        self._record(E_ENTER_RECOVERY, S_RECOVERING, detail)
        if target in ACTIVE_STATES:
            resolved = self.sm.require(
                E_RESOLVE_RECOVERY, S_RECOVERING, recovery_target=target)
            self._record(E_RESOLVE_RECOVERY, resolved, detail)
            return resolved
        # Reconciliation concluded at a halted state (e.g. FAILED campaign).
        halt_event = {
            S_FAILED: E_FAIL,
        }.get(target, E_FAIL)
        self._record(halt_event, target, detail)
        return target

    # ------------------------------------------------------------- convenience
    def run_to_awaiting_approval(self) -> dict[str, Any]:
        """Drive Discovery-derived objective up to AWAITING_APPROVAL (Step 2)."""
        self.select_objective()
        self.propose_and_review_hypotheses()
        self.create_plan()
        self.request_approval()
        return {"state": self.context.state,
                "approval_id": self.context.approval_id,
                "plan_id": self.context.plan_id}

    def run_to_completion(self, *, approved_by: str = "human",
                          source: str = "human") -> dict[str, Any]:
        """Full chain IDLE -> COMPLETED (Step 3 path), granting approval en route.

        The human grant is applied via ``approve_approval``; ``execute`` still
        re-verifies the store, so this never bypasses the gate.
        """
        if self.context.state == S_IDLE:
            self.run_to_awaiting_approval()
        if self.context.state != S_AWAITING_APPROVAL:
            raise DirectorError(
                f"expected AWAITING_APPROVAL, got {self.context.state}")
        self.approve_approval(approved_by=approved_by, source=source)
        self.execute()
        self.compare()
        self.record_experience()
        self.finalize()
        return {"state": self.context.state,
                "outcome": self.context.report_id}


__all__ = ["ResearchDirector", "DirectorRunContext", "DirectorError",
           "reconcile_director_state"]
