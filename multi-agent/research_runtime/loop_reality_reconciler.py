"""M5-00.1 / P0-02 -- READ-ONLY reality reconciliation for loop rounds.

A round JSON containing ``campaign_id`` is a CLAIM, not proof. This module
resolves every id against the real stores and the real Director journal:

  Objective -> Hypotheses(+Critic) -> Plan -> Approval -> Campaign binding
  -> CampaignStore record -> ResearchReport -> Experience -> Director journal
  -> original 12-step Gate evidence.

It NEVER creates, grants, executes, finalises or fabricates anything. It only
classifies what is actually persisted so recovery cannot turn an unresolved
round into COMPLETED.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .director_recovery import report_has_real_experience
from .research_store.db import content_sha256

ROUND_COMPLETED = "ROUND_COMPLETED"
AWAITING_APPROVAL = "AWAITING_APPROVAL"
CAMPAIGN_IN_FLIGHT = "CAMPAIGN_IN_FLIGHT"
EXPERIENCE_PENDING = "EXPERIENCE_PENDING"
RECOVERY_REQUIRED = "RECOVERY_REQUIRED"
CORRUPT = "CORRUPT"

CLASSIFICATIONS = (
    ROUND_COMPLETED, AWAITING_APPROVAL, CAMPAIGN_IN_FLIGHT,
    EXPERIENCE_PENDING, RECOVERY_REQUIRED, CORRUPT,
)


@dataclass
class RoundReality:
    round_no: int
    classification: str = RECOVERY_REQUIRED
    resolved: dict[str, str] = field(default_factory=dict)
    unresolved: list[str] = field(default_factory=list)
    corruption: list[str] = field(default_factory=list)
    gate_ok: bool = False
    director_state: str = ""
    campaign_state: str = ""
    budget: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_no": self.round_no,
            "classification": self.classification,
            "resolved": dict(self.resolved),
            "unresolved": list(self.unresolved),
            "corruption": list(self.corruption),
            "gate_ok": self.gate_ok,
            "director_state": self.director_state,
            "campaign_state": self.campaign_state,
            "budget": dict(self.budget),
        }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - unreadable evidence is unresolved
        return {}


def _resolve_director(round_no: int, state_dir: Path,
                      loop_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    director_id = f"m5_round_{round_no}_{loop_id}"
    base = state_dir / f"round_{round_no:02d}"
    ctx = _read_json(base / f"{director_id}.json")
    journal: list[dict[str, Any]] = []
    path = base / f"{director_id}.journal.jsonl"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    journal.append(json.loads(line))
                except ValueError:
                    continue
    return ctx, journal


def reconcile_round(service: Any, round_rec: dict[str, Any], *,
                    rounds_dir: str | Path,
                    state_dir: str | Path | None = None,
                    budget_ledger: Any | None = None,
                    loop_id: str = "",
                    config_sha: str = "") -> RoundReality:
    """Resolve one round against REAL persisted state. Read-only."""
    n = int(round_rec.get("round_no", 0))
    out = RoundReality(round_no=n)
    try:
        stores = service.stores
        for name in ("objectives", "hypotheses", "plans", "approvals", "sessions"):
            getattr(stores, name)
        getattr(service, "deps").campaign_store
    except Exception as exc:  # noqa: BLE001 - a partial service is not proof
        out.unresolved.append(f"stores_unavailable:{type(exc).__name__}")
        out.classification = RECOVERY_REQUIRED
        return out
    ids = {k: str(round_rec.get(k) or "") for k in (
        "objective_id", "plan_id", "approval_id", "campaign_id",
        "report_id", "experience_id")}
    hypotheses = [str(x) for x in (round_rec.get("hypothesis_ids") or [])]

    def need(name: str, value: str, obj: Any) -> Any:
        if not value or obj is None:
            out.unresolved.append(name)
            return None
        out.resolved[name] = value
        return obj

    # 1) Objective / Hypotheses + Critic / Plan / Approval
    objective = need("objective", ids["objective_id"],
                     stores.objectives.get(ids["objective_id"]))
    for hid in hypotheses:
        h = need(f"hypothesis:{hid}", hid, stores.hypotheses.get(hid))
        if h is not None:
            review = stores.hypotheses.latest_review(hid)
            if review is None or not review.passed:
                out.unresolved.append(f"critic_not_passed:{hid}")
    plan = need("plan", ids["plan_id"], stores.plans.get(ids["plan_id"]))
    approval = need("approval", ids["approval_id"],
                    stores.approvals.get(ids["approval_id"]))
    if plan is not None and approval is not None:
        if getattr(approval, "resource_id", "") != ids["plan_id"]:
            out.corruption.append("approval_bound_to_other_plan")
        if (getattr(approval, "resource_sha256", "")
                != getattr(plan, "plan_sha256", "")):
            out.corruption.append("approval_plan_sha_mismatch")
        if getattr(approval, "status", "") != "approved":
            out.unresolved.append("approval_not_approved")

    # 2) Campaign binding + real CampaignStore record
    campaign = need("campaign", ids["campaign_id"],
                    service.deps.campaign_store.load_campaign(ids["campaign_id"]))
    if campaign is not None:
        out.campaign_state = str(campaign.status)
        binding = stores.sessions.binding_for_campaign(ids["campaign_id"])
        if binding is None:
            out.unresolved.append("campaign_binding_missing")
        else:
            for field, expected in (
                ("objective_id", ids["objective_id"]),
                ("plan_id", ids["plan_id"]),
                ("approval_id", ids["approval_id"]),
            ):
                if getattr(binding, field, "") != expected:
                    out.corruption.append(f"campaign_binding_{field}_mismatch")
        if campaign.status != "completed":
            out.unresolved.append(f"campaign_status:{campaign.status}")
        failed = [t.trial_id for t in campaign.trials if t.status == "failed"]
        if failed:
            out.unresolved.append("campaign_has_failed_trials")

    # 3) Session / Report / Experience / original 12-step Gate evidence
    session_id = str(round_rec.get("session_id") or "")
    if campaign is not None and stores.sessions.binding_for_campaign(
            ids["campaign_id"]) is not None:
        session_id = session_id or str(
            stores.sessions.binding_for_campaign(ids["campaign_id"]).session_id or "")
    session = need("session", session_id,
                   stores.sessions.get_session(session_id)) if session_id else None
    report = need("report", ids["report_id"],
                  stores.sessions.get_report(ids["report_id"]))
    if report is not None:
        payload = report.model_dump(mode="json")
        payload.pop("report_sha256", None)
        if content_sha256(payload) != report.report_sha256:
            out.corruption.append("report_sha_mismatch")
        if report.session_id != session_id:
            out.corruption.append("report_session_mismatch")
        if report.objective_id != ids["objective_id"]:
            out.corruption.append("report_objective_mismatch")
        if report.frozen_test_used:
            out.corruption.append("frozen_test_used")
        if report.promotion != "not_requested":
            out.corruption.append("promotion_requested")
        if not report.reviews:
            out.unresolved.append("gate_review_missing")
        elif not all(bool(r.get("passed")) for r in report.reviews):
            out.unresolved.append("gate_review_not_passed")
        if session is not None:
            verified, _detail = report_has_real_experience(
                service, session_id, report.model_dump(mode="json"))
            if not verified:
                out.unresolved.append("experience_refs_unverified")
        else:
            out.unresolved.append("experience_refs_unverified")
    rows = stores.sessions.experience(session_id) if session_id else []
    stored_exp = {str(r.get("experience_id") or "") for r in rows}
    if ids["experience_id"] and ids["experience_id"] in stored_exp:
        out.resolved["experience"] = ids["experience_id"]
    else:
        out.unresolved.append("experience")

    # 4) Director journal (authoritative for the round's own state machine)
    base = Path(state_dir) if state_dir else Path(rounds_dir).parent / "director"
    ctx, journal = _resolve_director(n, base, loop_id)
    out.director_state = str(ctx.get("state") or "")
    if ctx.get("state") != "COMPLETED":
        out.unresolved.append(f"director_state:{out.director_state or 'missing'}")
    last = journal[-1] if journal else {}
    if last.get("to") != "COMPLETED" or last.get("event") != "FINALIZE":
        out.unresolved.append("director_journal_not_finalized")

    # 5) Budget reality: a consumed campaign must be charged exactly once.
    if budget_ledger is not None and loop_id and ids["campaign_id"]:
        try:
            out.budget = budget_ledger.snapshot(loop_id)
            with budget_ledger.db.session() as conn:
                entry = conn.execute(
                    """SELECT state FROM loop_budget_entries
                       WHERE loop_id=? AND campaign_id=?""",
                    (loop_id, ids["campaign_id"])).fetchone()
            if entry is None:
                out.unresolved.append("budget_not_reserved")
            elif entry["state"] != "consumed":
                out.unresolved.append(f"budget_state:{entry['state']}")
        except Exception as exc:  # noqa: BLE001
            out.corruption.append(f"budget_unreadable:{type(exc).__name__}")

    out.gate_ok = (
        not out.corruption
        and not out.unresolved
        and bool(report is not None and campaign is not None
                 and out.director_state == "COMPLETED")
    )

    # 6) Conservative classification (no inference from a non-empty id).
    if out.corruption:
        out.classification = CORRUPT
    elif out.gate_ok:
        out.classification = ROUND_COMPLETED
    elif campaign is None and approval is not None and \
            getattr(approval, "status", "") != "approved":
        out.classification = AWAITING_APPROVAL
    elif campaign is not None and campaign.status not in (
            "completed", "completed_with_failures", "failed", "cancelled"):
        out.classification = CAMPAIGN_IN_FLIGHT
    elif campaign is not None and campaign.status == "completed" and \
            (report is None or out.director_state != "COMPLETED"
             or ids["experience_id"] not in stored_exp):
        out.classification = EXPERIENCE_PENDING
    else:
        out.classification = RECOVERY_REQUIRED
    return out


__all__ = [
    "ROUND_COMPLETED", "AWAITING_APPROVAL", "CAMPAIGN_IN_FLIGHT",
    "EXPERIENCE_PENDING", "RECOVERY_REQUIRED", "CORRUPT", "CLASSIFICATIONS",
    "RoundReality", "reconcile_round",
]
