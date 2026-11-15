"""M4.3-01.1 P0-01 -- READ-ONLY gate evidence adapter (NO second Gate).

Problem this fixes
------------------
The M4.3-01 harness accepted a ``gate_adapter`` but never called it, and
``record_outcome`` let a caller *claim* ``gate_steps_passed=12`` and
``external_approval=True``. A success record therefore had NO connection to the
real 12-step fail-closed gate: an operator (or a bug) could record a
``valid_finding`` for an arm that never ran, was never approved, or produced no
Report / Experience.

Design (spec P0-01)
-------------------
This module is a **READ-ONLY observer** of the EXISTING ResearchService gate.
It does NOT re-implement, relax, or replace the 12-step gate. The gate is a
fail-closed chain, so if it fully passed AND executed, it necessarily left a set
of durable artefacts in trusted storage:

    1. a versioned ExperimentPlan with a stored SHA-256
    2. an ApprovalRecord: status=approved, source=human, resource_sha256 ==
       the CURRENT plan hash (no post-approval edit)
    3. a persisted CampaignRecord in the terminal state
    4. a persisted ResearchReport with non-empty experience_refs
    5. persisted Experience rows in the session's Experience Store

``ResearchServiceGateAdapter.inspect()`` verifies those persisted OUTPUTS using
ONLY existing read-only query methods. ``GateEvidence.gate_steps_passed`` is
**derived** (12 iff every persisted artefact is present and consistent); it can
never be supplied by a caller. No verdict is recomputed and no gate step is
re-executed.

All three arms (A/B/C) share ONE adapter bound to ONE ResearchService instance,
so they go through the same execution entry point and the same trusted store.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from ..campaign.schemas import CAMPAIGN_TERMINAL_STATUSES

# The 12-step fail-closed gate is defined in research_service. We only ever
# report it as PASSED when its complete persisted output chain is present.
GATE_STEP_COUNT = 12

# Outcomes that claim a completed, real experiment. These REQUIRE full gate
# evidence. ``invalid`` / ``failed`` / ``interrupted`` are retained as-is
# (spec: real failures and interruptions must be preserved).
GATED_OUTCOMES = ("valid_finding", "negative_finding", "inconclusive")
UNGATED_OUTCOMES = ("invalid", "failed", "interrupted")

# Caller-supplied keys that used to forge a gate verdict. Passing ANY of these
# is a hard protocol violation, not a silently ignored hint.
FORGED_GATE_KEYS = frozenset({
    "gate_steps_passed", "external_approval", "gate_passed", "gate_verdict",
})


@dataclass(frozen=True)
class GateRunRef:
    """WHERE to look. IDs ONLY -- never a verdict, never a step count.

    A caller may point the adapter at persisted records, but cannot state what
    those records contain or whether the gate passed.
    """
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
    # persisted artefact 1: plan
    plan_found: bool = False
    plan_sha256: str = ""
    # persisted artefact 2: approval
    approval_found: bool = False
    approval_status: str = ""
    approval_source: str = ""
    approval_resource_sha256: str = ""
    approval_hash_matches_plan: bool = False
    approval_approved_by: str = ""
    # persisted artefact 3: campaign
    campaign_found: bool = False
    campaign_status: str = ""
    campaign_terminal: bool = False
    # persisted artefact 4: research report
    report_found: bool = False
    report_experience_refs: tuple[str, ...] = ()
    # persisted artefact 5: experience rows
    experience_rows: int = 0
    # any read error (fail-closed: never treated as evidence of passing)
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
        if self.report_found:
            present.append("research_report")
        if self.report_experience_refs:
            present.append("report_experience_refs")
        if self.experience_rows > 0:
            present.append("experience_rows")
        return tuple(present)

    @property
    def missing_artefacts(self) -> tuple[str, ...]:
        required = (
            "plan", "approval_record", "external_human_approval",
            "campaign_record", "campaign_terminal", "research_report",
            "report_experience_refs", "experience_rows",
        )
        present = set(self.verified_artefacts)
        return tuple(a for a in required if a not in present)

    @property
    def gate_passed(self) -> bool:
        """True ONLY when the ENTIRE persisted output chain of the gate exists.

        This is NOT a recomputation of the 12 checks. A fail-closed gate that
        passed and executed leaves exactly these artefacts; any missing link
        means we cannot assert it passed (fail-closed).
        """
        return (
            not self.errors
            and self.plan_found
            and self.external_approval
            and self.campaign_found
            and self.campaign_terminal
            and self.report_found
            and bool(self.report_experience_refs)
            and self.experience_rows > 0
        )

    @property
    def gate_steps_passed(self) -> int:
        """12 iff the gate's persisted output chain is complete, else 0.

        Never caller-supplied. No partial credit: an incomplete chain means the
        gate was NOT verified, so ``plan_validity_rate`` must not count it.
        """
        return GATE_STEP_COUNT if self.gate_passed else 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["external_approval"] = self.external_approval
        d["gate_passed"] = self.gate_passed
        d["gate_steps_passed"] = self.gate_steps_passed
        d["verified_artefacts"] = list(self.verified_artefacts)
        d["missing_artefacts"] = list(self.missing_artefacts)
        return d


class GateAdapter(Protocol):
    """The read-only contract the harness depends on."""

    def inspect(self, ref: GateRunRef) -> GateEvidence: ...


class ResearchServiceGateAdapter:
    """READ-ONLY adapter over ONE real ResearchService (P0-01 #1/#2/#8).

    Only existing query methods are used:
      * ``stores.plans.get``                      -> versioned plan + hash
      * ``stores.approvals.get / for_resource``  -> real ApprovalRecord
      * ``deps.campaign_store.load_campaign``    -> persisted CampaignRecord
      * ``service.get_research_report``          -> persisted ResearchReport
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

        stores = getattr(self._service, "stores", None)
        deps = getattr(self._service, "deps", None)

        # ---- artefact 1: versioned plan -----------------------------------
        try:
            stored = stores.plans.get(ref.plan_id) if (stores and ref.plan_id) else None
            if stored is not None:
                ev["plan_found"] = True
                ev["plan_sha256"] = str(getattr(stored, "plan_sha256", "") or "")
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

        # ---- artefact 3: persisted CampaignRecord --------------------------
        try:
            campaign_store = getattr(deps, "campaign_store", None) if deps else None
            record = (campaign_store.load_campaign(ref.campaign_id)
                      if (campaign_store and ref.campaign_id) else None)
            if record is not None:
                ev["campaign_found"] = True
                status = str(getattr(record, "status", "") or "")
                ev["campaign_status"] = status
                ev["campaign_terminal"] = status in CAMPAIGN_TERMINAL_STATUSES
        except Exception as exc:  # noqa: BLE001
            errors.append(f"campaign: {type(exc).__name__}: {exc}")

        # ---- artefact 4: persisted ResearchReport --------------------------
        try:
            report = None
            if ref.session_id:
                rep = self._service.get_research_report(session_id=ref.session_id)
                if isinstance(rep, dict) and rep.get("ok"):
                    report = rep.get("report") or {}
                    ev["report_found"] = True
            if report is None and stores and ref.report_id:
                stored_report = stores.sessions.get_report(ref.report_id)
                if stored_report is not None:
                    report = stored_report.model_dump(mode="json")
                    ev["report_found"] = True
            if ev.get("report_found"):
                ev["report_experience_refs"] = tuple(
                    str(x) for x in (report.get("experience_refs") or []))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"report: {type(exc).__name__}: {exc}")

        # ---- artefact 5: persisted Experience rows -------------------------
        try:
            if stores and ref.session_id:
                rows = stores.sessions.experience(ref.session_id)
                ev["experience_rows"] = len(rows or [])
        except Exception as exc:  # noqa: BLE001
            errors.append(f"experience: {type(exc).__name__}: {exc}")

        ev["errors"] = tuple(errors)
        return GateEvidence(**ev)


__all__ = [
    "GATE_STEP_COUNT", "GATED_OUTCOMES", "UNGATED_OUTCOMES", "FORGED_GATE_KEYS",
    "GateRunRef", "GateEvidence", "GateAdapter", "ResearchServiceGateAdapter",
]
