"""M4.3-01.1 P0-01 / M4.3-02 P0 -- READ-ONLY gate evidence adapter.

No second Gate
--------------
This module is a **READ-ONLY observer** of the EXISTING ``ResearchService``
12-step fail-closed gate. It does NOT re-implement, relax or replace it. The
gate only ever reports PASSED when its complete, durable output chain exists:

    1. a versioned ExperimentPlan with a stored SHA-256
    2. an ApprovalRecord: status=approved, source=human, resource_id == plan,
       resource_sha256 == the CURRENT plan hash (no post-approval edit)
    3. a persisted CampaignRecord that SUCCEEDED (see below)
    4. a persisted ResearchSession that owns the report and the campaign
    5. a persisted ResearchReport whose session/objective match the run
    6. persisted Experience rows that the report's ``experience_refs`` ACTUALLY
       resolve to (a ref string is a claim, not proof)

M4.3-02 P0 fixes (terminal != successful)
-----------------------------------------
``CAMPAIGN_TERMINAL_STATUSES`` includes ``failed`` and ``cancelled``. The old
adapter treated "terminal" as gate success, so a FAILED or CANCELLED campaign
could produce a ``valid_finding``. Now:

  * ``campaign_terminal``   -- any of completed / completed_with_failures /
                               failed / cancelled (kept for reporting);
  * ``campaign_successful`` -- completed / completed_with_failures AND no
                               recorded trial failed to the point that nothing
                               succeeded (a campaign with only failed trials is
                               NOT a success);
  * ``gate_passed`` requires ``campaign_successful`` -- ``failed`` /
    ``cancelled`` NEVER yield gate success;
  * ``valid_finding`` additionally requires ``valid_finding_evidence`` --
    a successful campaign plus a REAL scientific outcome on the report
    (``not_evaluated`` is not an outcome), plus the complete association chain;
  * ``negative_finding`` only requires ``gate_passed``: a genuinely completed
    experiment that did NOT produce the expected positive finding is still a
    valid, retained result.

Precise association (no cross-scope reuse)
------------------------------------------
Every artefact must point at the SAME run scope. Any of the following is a
``scope_conflict`` (fail-closed):
  * campaign.plan_id != ref.plan_id                -> another plan's campaign
  * campaign.objective_id != session.objective_id   -> another objective
  * report.session_id != ref.session_id             -> another session's report
  * report.objective_id != session.objective_id     -> another objective
  * session.report_id  != report.report_id          -> stale/other report
  * campaign_id not listed in session.campaign_ids  -> another session's campaign
  * campaign_id not listed in report.campaigns      -> another campaign's report
  * plan.objective_id != session.objective_id       -> plan for another objective
  * approval.resource_id != ref.plan_id             -> approval for another plan

Cross-RUN reuse (one campaign's success evidence reused by a second run) is
rejected by ``ExperimentRegistry.finalize``: a campaign_id may bind to exactly
one run (see ``registry_store.RegistryStore.runs_with_campaign``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from ..campaign.schemas import CAMPAIGN_TERMINAL_STATUSES

# The 12-step fail-closed gate is defined in research_service. We only ever
# report it as PASSED when its complete persisted output chain is present.
GATE_STEP_COUNT = 12

# Campaign statuses that mean the campaign actually ran to a successful end.
CAMPAIGN_SUCCESS_STATUSES: frozenset[str] = frozenset({
    "completed", "completed_with_failures",
})

# Report outcomes that constitute a REAL scientific outcome. ``not_evaluated``
# means the comparison never produced a verdict, so it can never support a
# ``valid_finding`` (it is retained as ``inconclusive``).
SCIENTIFIC_REPORT_OUTCOMES: frozenset[str] = frozenset({
    "accepted_for_confirmation", "rejected", "inconclusive",
})

# Outcomes that claim a completed, real experiment. These REQUIRE full gate
# evidence. ``invalid`` / ``failed`` / ``interrupted`` are retained as-is
# (spec: real failures and interruptions must be preserved).
GATED_OUTCOMES = ("valid_finding", "negative_finding", "inconclusive")
UNGATED_OUTCOMES = ("invalid", "failed", "interrupted", "stopped")

# Caller-supplied keys that used to forge a gate verdict. Passing ANY of these
# is a hard protocol violation, not a silently ignored hint.
FORGED_GATE_KEYS = frozenset({
    "gate_steps_passed", "external_approval", "gate_passed", "gate_verdict",
})

# Association checks that must ALL hold for the chain to describe ONE run.
ASSOCIATION_CHECKS: tuple[str, ...] = (
    "plan_objective_matches_session",
    "approval_bound_to_plan",
    "campaign_bound_to_plan",
    "campaign_objective_matches_session",
    "session_owns_campaign",
    "report_belongs_to_session",
    "report_objective_matches_session",
    "session_points_at_report",
    "report_lists_campaign",
)


@dataclass(frozen=True)
class GateRunRef:
    """WHERE to look. IDs ONLY -- never a verdict, never a step count."""
    session_id: str = ""
    plan_id: str = ""
    approval_id: str = ""
    campaign_id: str = ""
    report_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GateEvidence:
    """Gate status DERIVED from real persisted ResearchService records."""
    ref: GateRunRef = field(default_factory=GateRunRef)
    # ---- persisted artefact 1: plan -----------------------------------
    plan_found: bool = False
    plan_sha256: str = ""
    plan_objective_id: str = ""
    # ---- persisted artefact 2: approval -------------------------------
    approval_found: bool = False
    approval_status: str = ""
    approval_source: str = ""
    approval_resource_id: str = ""
    approval_resource_sha256: str = ""
    approval_hash_matches_plan: bool = False
    approval_approved_by: str = ""
    # ---- persisted artefact 3: campaign (terminal vs successful) ------
    campaign_found: bool = False
    campaign_id: str = ""
    campaign_status: str = ""
    campaign_terminal: bool = False
    campaign_successful: bool = False
    campaign_plan_id: str = ""
    campaign_objective_id: str = ""
    campaign_trials: int = 0
    campaign_succeeded_trials: int = 0
    campaign_failed_trials: int = 0
    campaign_skipped_trials: int = 0
    # ---- persisted artefact 4: research session -----------------------
    session_found: bool = False
    session_objective_id: str = ""
    session_report_id: str = ""
    session_campaign_ids: tuple[str, ...] = ()
    # ---- persisted artefact 5: research report ------------------------
    report_found: bool = False
    report_id: str = ""
    report_session_id: str = ""
    report_objective_id: str = ""
    report_outcome: str = ""
    report_experience_refs: tuple[str, ...] = ()
    report_campaign_ids: tuple[str, ...] = ()
    # ---- persisted artefact 6: experience rows ------------------------
    experience_rows: int = 0
    experience_ids: tuple[str, ...] = ()
    experience_refs_resolved: int = 0
    experience_refs_unresolved: tuple[str, ...] = ()
    # ---- association -------------------------------------------------
    association: dict[str, bool] = field(default_factory=dict)
    scope_conflicts: tuple[str, ...] = ()
    # ---- any read error (fail-closed: never treated as evidence of passing)
    errors: tuple[str, ...] = ()

    # ------------------------------------------------------------- derived
    @property
    def external_approval(self) -> bool:
        """A REAL out-of-band human approval bound to the current plan hash."""
        return (
            self.approval_found
            and self.approval_status == "approved"
            and self.approval_source == "human"
            and bool(self.approval_approved_by)
            and self.approval_hash_matches_plan
            and self.approval_resource_id == self.ref.plan_id
        )

    @property
    def trial_evidence(self) -> str:
        """Per-trial detail of the campaign.

        ``succeeded``      at least one trial succeeded
        ``partial``        some succeeded, some failed/skipped
        ``failed_only``    trials exist but NONE succeeded -> not a success
        ``none_recorded``  the runner recorded no trial rows (status only)
        """
        if self.campaign_trials <= 0:
            return "none_recorded"
        if self.campaign_succeeded_trials <= 0:
            return "failed_only"
        if (self.campaign_failed_trials or self.campaign_skipped_trials):
            return "partial"
        return "succeeded"

    @property
    def association_ok(self) -> bool:
        return (
            bool(self.association)
            and all(self.association.get(k, False) for k in ASSOCIATION_CHECKS)
            and not self.scope_conflicts
        )

    @property
    def verified_artefacts(self) -> tuple[str, ...]:
        present: list[str] = []
        if self.plan_found:
            present.append("plan")
        if self.approval_found:
            present.append("approval_record")
        if self.external_approval:
            present.append("external_human_approval")
        if self.campaign_found:
            present.append("campaign_record")
        if self.campaign_terminal:
            present.append("campaign_terminal")
        if self.campaign_successful:
            present.append("campaign_successful")
        if self.session_found:
            present.append("session_record")
        if self.report_found:
            present.append("research_report")
        if self.report_experience_refs:
            present.append("report_experience_refs")
        if self.experience_rows > 0:
            present.append("experience_rows")
        if (self.report_experience_refs
                and not self.experience_refs_unresolved
                and self.experience_refs_resolved > 0):
            present.append("experience_refs_resolved")
        if self.association_ok:
            present.append("association_consistent")
        return tuple(present)

    @property
    def missing_artefacts(self) -> tuple[str, ...]:
        required = (
            "plan", "approval_record", "external_human_approval",
            "campaign_record", "campaign_successful", "session_record",
            "research_report", "report_experience_refs", "experience_rows",
            "experience_refs_resolved", "association_consistent",
        )
        present = set(self.verified_artefacts)
        return tuple(a for a in required if a not in present)

    @property
    def gate_passed(self) -> bool:
        """True ONLY when the gate's complete persisted output chain exists AND
        the campaign actually SUCCEEDED (not merely terminated).

        This is NOT a recomputation of the 12 checks. A fail-closed gate that
        passed and executed leaves exactly these artefacts; any missing link --
        including a failed/cancelled campaign, a cross-scope record or an
        unresolvable experience ref -- means we cannot assert it passed.
        """
        return not self.missing_artefacts

    @property
    def valid_finding_evidence(self) -> bool:
        """Stricter than ``gate_passed``: a REAL positive-capable outcome.

        A ``valid_finding`` additionally requires that the campaign recorded at
        least one successful trial (or, for runners that only persist the
        campaign status, an explicit successful status with no failed trials)
        and that the report carries a real scientific outcome -- a comparison
        that never produced one (``not_evaluated``) cannot support it.
        """
        return (
            self.gate_passed
            and self.campaign_successful
            and self.trial_evidence in ("succeeded", "partial", "none_recorded")
            and self.report_outcome in SCIENTIFIC_REPORT_OUTCOMES
        )

    @property
    def gate_steps_passed(self) -> int:
        """12 iff the gate's persisted output chain is complete, else 0."""
        return GATE_STEP_COUNT if self.gate_passed else 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["external_approval"] = self.external_approval
        d["gate_passed"] = self.gate_passed
        d["gate_steps_passed"] = self.gate_steps_passed
        d["valid_finding_evidence"] = self.valid_finding_evidence
        d["trial_evidence"] = self.trial_evidence
        d["association_ok"] = self.association_ok
        d["verified_artefacts"] = list(self.verified_artefacts)
        d["missing_artefacts"] = list(self.missing_artefacts)
        return d


class GateAdapter(Protocol):
    """The read-only contract the harness depends on."""

    def inspect(self, ref: GateRunRef) -> GateEvidence: ...


def _model_dump(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except Exception:  # noqa: BLE001 - best effort, read-only
            return {}
    if isinstance(obj, dict):
        return dict(obj)
    return {}


class ResearchServiceGateAdapter:
    """READ-ONLY adapter over ONE real ResearchService (P0-01 #1/#2/#8).

    Only existing query methods are used:
      * ``stores.plans.get``                      -> versioned plan + hash
      * ``stores.approvals.get / for_resource``  -> real ApprovalRecord
      * ``deps.campaign_store.load_campaign``    -> persisted CampaignRecord
      * ``service.get_research_report``          -> persisted ResearchReport
      * ``stores.sessions.get_session``          -> persisted ResearchSession
      * ``stores.sessions.experience``           -> persisted Experience rows

    Nothing is written, granted, executed, resumed or recomputed. Any read
    error is captured in ``errors`` and makes the evidence fail closed.
    """

    def __init__(self, service: Any):
        if service is None:
            raise ValueError("ResearchServiceGateAdapter requires a real service")
        self._service = service
        # Diagnostic only: proves A/B/C really went through THIS adapter.
        self.inspect_calls: list[GateRunRef] = []

    @property
    def service(self) -> Any:
        return self._service

    def inspect(self, ref: GateRunRef | None = None) -> GateEvidence:
        ref = ref or GateRunRef()
        self.inspect_calls.append(ref)
        ev: dict[str, Any] = {"ref": ref}
        errors: list[str] = []
        conflicts: list[str] = []

        stores = getattr(self._service, "stores", None)
        deps = getattr(self._service, "deps", None)

        # ---- artefact 1: versioned plan -----------------------------------
        try:
            stored = stores.plans.get(ref.plan_id) if (stores and ref.plan_id) else None
            if stored is not None:
                ev["plan_found"] = True
                ev["plan_sha256"] = str(getattr(stored, "plan_sha256", "") or "")
                ev["plan_objective_id"] = str(
                    getattr(stored, "objective_id", "") or "")
        except Exception as exc:  # noqa: BLE001 - read-only observation
            errors.append(f"plan: {type(exc).__name__}: {exc}")

        # ---- artefact 2: real ApprovalRecord -------------------------------
        try:
            approval = None
            if stores and ref.approval_id:
                approval = stores.approvals.get(ref.approval_id)
            if approval is None and stores and ref.plan_id:
                matches = stores.approvals.for_resource(ref.plan_id)
                approval = matches[-1] if matches else None
            if approval is not None:
                ev["approval_found"] = True
                ev["approval_status"] = str(getattr(approval, "status", "") or "")
                ev["approval_source"] = str(getattr(approval, "source", "") or "")
                ev["approval_resource_id"] = str(
                    getattr(approval, "resource_id", "") or "")
                ev["approval_resource_sha256"] = str(
                    getattr(approval, "resource_sha256", "") or "")
                ev["approval_approved_by"] = str(
                    getattr(approval, "approved_by", "") or "")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"approval: {type(exc).__name__}: {exc}")
        ev["approval_hash_matches_plan"] = bool(
            ev.get("plan_found")
            and ev.get("plan_sha256")
            and ev.get("approval_resource_sha256") == ev["plan_sha256"]
        )

        # ---- artefact 3: persisted CampaignRecord (terminal != successful) -
        try:
            campaign_store = getattr(deps, "campaign_store", None) if deps else None
            record = (campaign_store.load_campaign(ref.campaign_id)
                      if (campaign_store and ref.campaign_id) else None)
            if record is not None:
                schema = _model_dump(record)
                status = str(schema.get("status") or getattr(record, "status", ""))
                trials = list(schema.get("trials") or
                              getattr(record, "trials", []) or [])
                succeeded = sum(
                    1 for t in trials
                    if str(t.get("status") if isinstance(t, dict)
                           else getattr(t, "status", "")) == "succeeded")
                failed = sum(
                    1 for t in trials
                    if str(t.get("status") if isinstance(t, dict)
                           else getattr(t, "status", "")) == "failed")
                skipped = sum(
                    1 for t in trials
                    if str(t.get("status") if isinstance(t, dict)
                           else getattr(t, "status", "")) == "skipped")
                ev["campaign_found"] = True
                ev["campaign_id"] = str(
                    schema.get("campaign_id") or getattr(record, "campaign_id", ""))
                ev["campaign_status"] = status
                ev["campaign_terminal"] = status in CAMPAIGN_TERMINAL_STATUSES
                ev["campaign_plan_id"] = str(
                    schema.get("plan_id") or getattr(record, "plan_id", ""))
                ev["campaign_objective_id"] = str(
                    schema.get("objective_id") or getattr(record, "objective_id", ""))
                ev["campaign_trials"] = len(trials)
                ev["campaign_succeeded_trials"] = succeeded
                ev["campaign_failed_trials"] = failed
                ev["campaign_skipped_trials"] = skipped
                ev["campaign_successful"] = bool(
                    status in CAMPAIGN_SUCCESS_STATUSES
                    and (len(trials) == 0 or succeeded > 0))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"campaign: {type(exc).__name__}: {exc}")

        # ---- artefact 4: ResearchSession -----------------------------------
        try:
            session = None
            if stores and ref.session_id:
                session = stores.sessions.get_session(ref.session_id)
            if session is not None:
                sess = _model_dump(session)
                ev["session_found"] = True
                ev["session_objective_id"] = str(
                    sess.get("objective_id") or getattr(session, "objective_id", ""))
                ev["session_report_id"] = str(
                    sess.get("report_id") or getattr(session, "report_id", "") or "")
                ev["session_campaign_ids"] = tuple(
                    str(x) for x in (sess.get("campaign_ids")
                                     or getattr(session, "campaign_ids", []) or []))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"session: {type(exc).__name__}: {exc}")

        # ---- artefact 5: persisted ResearchReport --------------------------
        try:
            report: dict[str, Any] = {}
            if ref.session_id:
                try:
                    rep = self._service.get_research_report(
                        session_id=ref.session_id)
                except Exception:  # noqa: BLE001 - absent report is not an error
                    rep = {}
                if isinstance(rep, dict) and rep.get("ok"):
                    report = rep.get("report") or {}
                    ev["report_found"] = True
            if not report and stores and ref.report_id:
                stored_report = stores.sessions.get_report(ref.report_id)
                if stored_report is not None:
                    report = _model_dump(stored_report)
                    ev["report_found"] = True
            if ev.get("report_found"):
                ev["report_id"] = str(report.get("report_id") or "")
                ev["report_session_id"] = str(report.get("session_id") or "")
                ev["report_objective_id"] = str(report.get("objective_id") or "")
                ev["report_outcome"] = str(report.get("outcome") or "")
                ev["report_experience_refs"] = tuple(
                    str(x) for x in (report.get("experience_refs") or []))
                ev["report_campaign_ids"] = tuple(
                    str((c or {}).get("campaign_id") or "")
                    for c in (report.get("campaigns") or []) if c)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"report: {type(exc).__name__}: {exc}")

        # ---- artefact 6: persisted Experience rows + ref resolution --------
        try:
            rows: list[dict[str, Any]] = []
            if stores and ref.session_id:
                rows = list(stores.sessions.experience(ref.session_id) or [])
            ids = tuple(str(r.get("experience_id") or "") for r in rows)
            ev["experience_rows"] = len(rows)
            ev["experience_ids"] = ids
            refs = tuple(ev.get("report_experience_refs") or ())
            resolved = [r for r in refs if r in set(ids)]
            unresolved = [r for r in refs if r not in set(ids)]
            ev["experience_refs_resolved"] = len(resolved)
            ev["experience_refs_unresolved"] = tuple(unresolved)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"experience: {type(exc).__name__}: {exc}")

        # ---- association: EVERY artefact must describe the SAME run --------
        sess_obj = ev.get("session_objective_id", "")
        campaign_ids = set(ev.get("session_campaign_ids") or ())
        report_campaign_ids = set(ev.get("report_campaign_ids") or ())
        cid = ref.campaign_id or ev.get("campaign_id", "")
        association = {
            "plan_objective_matches_session":
                bool(sess_obj) and ev.get("plan_objective_id") == sess_obj,
            "approval_bound_to_plan":
                ev.get("approval_resource_id") == ref.plan_id,
            "campaign_bound_to_plan":
                ev.get("campaign_plan_id") == ref.plan_id,
            "campaign_objective_matches_session":
                bool(sess_obj) and ev.get("campaign_objective_id") == sess_obj,
            "session_owns_campaign": bool(cid) and cid in campaign_ids,
            "report_belongs_to_session":
                ev.get("report_session_id") == ref.session_id and bool(ref.session_id),
            "report_objective_matches_session":
                bool(sess_obj) and ev.get("report_objective_id") == sess_obj,
            "session_points_at_report":
                bool(ev.get("report_id"))
                and ev.get("session_report_id") == ev.get("report_id"),
            "report_lists_campaign": bool(cid) and cid in report_campaign_ids,
        }
        ev["association"] = association

        if ev.get("campaign_found") and ev.get("campaign_plan_id") \
                and ref.plan_id and ev["campaign_plan_id"] != ref.plan_id:
            conflicts.append(
                "campaign belongs to another plan "
                f"({ev['campaign_plan_id']!r} != {ref.plan_id!r})")
        if ev.get("campaign_found") and sess_obj \
                and ev.get("campaign_objective_id") != sess_obj:
            conflicts.append(
                "campaign belongs to another objective "
                f"({ev.get('campaign_objective_id')!r} != {sess_obj!r})")
        if ev.get("report_found") and ref.session_id \
                and ev.get("report_session_id") != ref.session_id:
            conflicts.append(
                "report belongs to another session "
                f"({ev.get('report_session_id')!r} != {ref.session_id!r})")
        if ev.get("report_found") and sess_obj \
                and ev.get("report_objective_id") != sess_obj:
            conflicts.append(
                "report belongs to another objective "
                f"({ev.get('report_objective_id')!r} != {sess_obj!r})")
        if ev.get("approval_found") and ref.plan_id \
                and ev.get("approval_resource_id") != ref.plan_id:
            conflicts.append(
                "approval is bound to another resource "
                f"({ev.get('approval_resource_id')!r} != {ref.plan_id!r})")

        ev["scope_conflicts"] = tuple(conflicts)
        ev["errors"] = tuple(errors)
        return GateEvidence(**ev)


__all__ = [
    "GATE_STEP_COUNT", "GATED_OUTCOMES", "UNGATED_OUTCOMES", "FORGED_GATE_KEYS",
    "CAMPAIGN_SUCCESS_STATUSES", "SCIENTIFIC_REPORT_OUTCOMES",
    "ASSOCIATION_CHECKS",
    "GateRunRef", "GateEvidence", "GateAdapter", "ResearchServiceGateAdapter",
]
