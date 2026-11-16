"""M5-01 -- thin cross-domain scheduler (supervised multi-domain).

It uses ONLY existing components: DomainAdapterRegistry -> ResearchDirector ->
ResearchService -> original 12-step Gate -> Experience. The generic scheduler
contains NO domain metric, dataset or config key; routing is by capability and
budget. A domain failure is recorded and isolated from the other domains.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .discovery_context_provider import (
    DISCOVERY_SNAPSHOT_SCHEMA_VERSION, DiscoverySnapshot,
)
from .domain_budget_manager import DomainBudgetManager
from .domain_run_record import DomainRunRecordStore
from .research_director import ResearchDirector
from .research_service import make_campaign_id


class DomainCapabilityError(RuntimeError):
    pass


class CrossDomainScheduler:
    def __init__(self, *, run_id: str, registry: Any,
                 budgets: DomainBudgetManager, records: DomainRunRecordStore,
                 services: dict[str, Any], approver: Callable[[str, str], None],
                 out_dir: str | Path, max_campaigns_per_domain: int = 1):
        self.run_id = run_id
        self.registry = registry
        self.budgets = budgets
        self.records = records
        self.services = dict(services)
        self.approver = approver
        self.out_dir = Path(out_dir)
        self.max_campaigns_per_domain = max_campaigns_per_domain
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- preflight
    def preflight(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in self.registry.names():
            adapter = self.registry.get(name)
            try:
                detail = adapter.preflight()
                budget = self.budgets.initialize(
                    self.run_id, name,
                    max_campaigns=self.max_campaigns_per_domain)
                out[name] = {"ready": bool(detail.get("ready")),
                             "detail": detail, "budget": budget,
                             "capabilities": sorted(adapter.capabilities)}
            except Exception as exc:  # noqa: BLE001 - one domain must not stop others
                out[name] = {"ready": False,
                             "error": f"{type(exc).__name__}: {exc}"}
        return out

    # ---------------------------------------------------------------- routing
    @staticmethod
    def _capabilities(objective: Any,
                      override: list[str] | None = None) -> list[str]:
        if override is not None:
            return list(override)
        constraints = getattr(objective, "constraints", None)
        if isinstance(constraints, dict):
            return list(constraints.get("required_capabilities") or ())
        if isinstance(objective, dict):
            return list((objective.get("constraints") or {}).get(
                "required_capabilities") or ())
        return []

    def _locate_domain(self, objective_id: str) -> str | None:
        for domain_id, service in sorted(self.services.items()):
            try:
                if service.stores.objectives.get(objective_id) is not None:
                    return domain_id
            except Exception:  # noqa: BLE001 - skip an unavailable domain
                continue
        return None

    def route(self, objective: Any, *, domain_hint: str | None = None,
              required: list[str] | None = None) -> str:
        required = self._capabilities(objective, required)
        domain = domain_hint or self._locate_domain(
            getattr(objective, "objective_id", "") or
            (objective.get("objective_id") if isinstance(objective, dict) else ""))
        if domain is None:
            raise DomainCapabilityError(
                "objective not found in any registered domain store")
        adapter = self.registry.get(domain)
        if not adapter.supports(required):
            raise DomainCapabilityError(
                f"domain {domain!r} lacks capabilities "
                f"{sorted(set(required) - set(adapter.capabilities))}")
        if domain not in self.services:
            raise DomainCapabilityError(f"no service bound for domain {domain!r}")
        return domain

    # -------------------------------------------------------------- execution
    def run_objective(self, objective_id: str, *, domain_hint: str | None = None,
                      required: list[str] | None = None,
                      source_experiment_id: str = "") -> dict[str, Any]:
        """Run ONE real smoke campaign after an external human approval."""
        service = None
        domain = ""
        candidate = None
        for name, svc in sorted(self.services.items()):
            obj = svc.stores.objectives.get(objective_id)
            if obj is not None:
                service, domain, candidate = svc, name, obj
                break
        if candidate is None:
            raise DomainCapabilityError(
                f"objective {objective_id!r} not found in any domain store")
        domain = self.route(candidate, domain_hint=domain_hint or domain,
                            required=required)
        service = self.services[domain]
        adapter = self.registry.get(domain)
        preflight = adapter.preflight()
        if not preflight.get("ready"):
            raise DomainCapabilityError(
                f"domain {domain!r} preflight failed: {preflight}")

        state_dir = self.out_dir / "directors" / domain / objective_id
        director = ResearchDirector(
            service=service, objective_store=service.stores.objectives,
            state_dir=state_dir, objective_id=objective_id, domain=domain,
            director_id=f"m5sched_{self.run_id}_{domain}_{objective_id}",
            source_experiment_id=source_experiment_id)
        director.run_to_awaiting_approval()
        approval_id = director.context.approval_id
        self.approver(approval_id, director.context.plan_id)
        if not director._approval_granted_and_bound():
            raise RuntimeError(
                f"domain {domain}: approval not granted/SHA-bound; no campaign")

        plan = service.stores.plans.get(director.context.plan_id)
        planned_campaign_id = make_campaign_id(
            director.context.plan_id, approval_id, plan.plan_sha256)
        self.budgets.reserve(self.run_id, domain, planned_campaign_id)
        try:
            final = director.run_to_completion()
            ctx = director.context
            if ctx.campaign_id != planned_campaign_id:
                raise RuntimeError(
                    f"campaign id drift in {domain}: reserved="
                    f"{planned_campaign_id}, executed={ctx.campaign_id}")
            spent = self.budgets.consume(
                self.run_id, domain, planned_campaign_id,
                outcome=str(final.get("state") or "unknown"))
        except Exception:
            if director.context.campaign_id:
                self.budgets.consume(self.run_id, domain, planned_campaign_id,
                                     outcome="interrupted")
            else:
                self.budgets.release(self.run_id, domain, planned_campaign_id,
                                     reason="campaign_not_created")
            raise

        result = adapter.collect_result(service, ctx.session_id)
        rec = self.records.write(
            run_id=self.run_id, objective_id=objective_id, domain_id=domain,
            campaign_id=ctx.campaign_id, approval_id=approval_id,
            plan_id=ctx.plan_id, session_id=ctx.session_id,
            report_id=ctx.report_id,
            status="COMPLETED" if final.get("state") == "COMPLETED" else final.get("state", ""),
            sources={"preflight": preflight,
                     "director_state": final.get("state", ""),
                     "budget": spent,
                     "report_experience_refs": list(
                         ((result.get("report") or {}).get("experience_refs") or []))})
        return {"domain_id": domain, "objective_id": objective_id,
                "campaign_id": ctx.campaign_id,
                "report_id": ctx.report_id,
                "director_state": final.get("state", ""),
                "budget": self.budgets.snapshot(self.run_id, domain),
                "record": rec, "result": result}

    def run_objectives(self, objectives: list[str], **kwargs: Any) -> dict[str, Any]:
        """Run all objectives; one domain failure never aborts another domain."""
        out: list[dict[str, Any]] = []
        for oid in objectives:
            try:
                out.append(self.run_objective(oid, **kwargs))
            except Exception as exc:  # noqa: BLE001 - isolation is the contract
                out.append({"objective_id": oid, "ok": False,
                            "error": f"{type(exc).__name__}: {exc}"})
        return {"run_id": self.run_id, "results": out,
                "budgets": {d: self.budgets.snapshot(self.run_id, d)
                            for d in self.services}}

    # ----------------------------------------------------------- shared reads
    def read_experience(self, objective_ids: list[str] | None = None
                        ) -> list[dict[str, Any]]:
        """Unified cross-domain Experience READ, source preserved per row."""
        wanted = set(objective_ids or ())
        out: list[dict[str, Any]] = []
        for rec in self.records.all(self.run_id):
            if wanted and rec.get("objective_id") not in wanted:
                continue
            service = self.services.get(rec["domain_id"])
            if service is None:
                continue
            for row in service.stores.sessions.experience(rec["session_id"]):
                out.append({**row, "source_domain": rec["domain_id"],
                            "source_run_id": self.run_id,
                            "objective_id": rec["objective_id"]})
        return out

    def build_cross_domain_snapshot(self, run_id: str,
                                    objective_ids: list[str] | None = None
                                    ) -> DiscoverySnapshot:
        """Merge REAL per-domain snapshots with explicit source provenance."""
        merged: dict[str, Any] = {
            "run_id": run_id, "objectives": [], "sessions": [],
            "experience_available": False, "unresolved_gaps": []}
        for domain_id, service in sorted(self.services.items()):
            ids = [oid for oid in (objective_ids or ())
                   if (service.stores.objectives.get(oid) is not None)]
            try:
                snap = self.registry.get(domain_id).build_snapshot(run_id, ids)
            except Exception as exc:  # noqa: BLE001 - never fabricate a snapshot
                merged["unresolved_gaps"].append(
                    {"domain_id": domain_id,
                     "reason": f"{type(exc).__name__}: {exc}"})
                continue
            for obj in snap.objectives:
                merged["objectives"].append({**obj, "source_domain": domain_id})
            for session in snap.sessions:
                merged["sessions"].append({**session, "source_domain": domain_id})
            merged["experience_available"] = (
                merged["experience_available"] or snap.experience_available)
        return DiscoverySnapshot(
            run_id=run_id, schema_version=DISCOVERY_SNAPSHOT_SCHEMA_VERSION,
            generated_at="", evidence_view={"cross_domain": True},
            objectives=merged["objectives"], links={}, sessions=merged["sessions"],
            sessions_by_objective={},
            experience_available=merged["experience_available"],
            failures={}, provenance={"source_domains": sorted(self.services)},
            unresolved_gaps=merged["unresolved_gaps"])


__all__ = ["CrossDomainScheduler", "DomainCapabilityError"]
