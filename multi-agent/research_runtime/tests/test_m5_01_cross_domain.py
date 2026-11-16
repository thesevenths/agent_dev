"""M5-01 -- cross-domain adapter/routing/budget contracts (ENGINE-FREE)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from research_runtime.cross_domain_scheduler import (
    CrossDomainScheduler, DomainCapabilityError,
)
from research_runtime.domain_adapter_registry import (
    build_default_adapters,
)
from research_runtime.domain_budget_manager import (
    DomainBudgetExhausted, DomainBudgetManager,
)
from research_runtime.domain_run_record import DomainRunRecordStore
from research_runtime.research_store import (
    ResearchStores, from_research_objective,
)
from research_runtime.schemas import ResearchObjective


def _objective(store, oid):
    obj = ResearchObjective(objective_id=oid, domain="test",
                            description=oid,
                            constraints={"required_capabilities": ["backtest"]})
    store.objectives.save(from_research_objective(obj, status="proposed"))
    return obj


def _scheduler(tmp_path, *, services=None):
    services = services or {}
    db = ResearchStores(tmp_path / "sched.sqlite").db
    registry = build_default_adapters(service_holders={
        name: {"service": svc} for name, svc in services.items()})
    budgets = DomainBudgetManager(db=db)
    records = DomainRunRecordStore(db=db)
    return CrossDomainScheduler(
        run_id="M5S1", registry=registry, budgets=budgets, records=records,
        services=services, approver=lambda *a: None,
        out_dir=tmp_path / "out", max_campaigns_per_domain=1)


def test_three_domain_adapters_register_and_preflight():
    registry = build_default_adapters()
    assert registry.names() == ["embodied", "quant", "scinet"]
    for name in registry.names():
        assert registry.get(name).capabilities
        assert {"domain_id": name}.items() <= registry.get(name).to_dict().items()
        detail = registry.get(name).preflight()
        assert isinstance(detail.get("ready"), bool)
    assert registry.route(["simulation"], domain_hint="embodied").domain_id == "embodied"
    assert registry.route(["backtest"], domain_hint="quant").domain_id == "quant"


def test_objective_routes_to_owning_domain_and_capability_rejects(tmp_path):
    svc = SimpleNamespace(stores=ResearchStores(tmp_path / "q.sqlite"))
    _objective(svc.stores, "obj_q")
    sched = _scheduler(tmp_path, services={"quant": svc})
    assert sched.route(svc.stores.objectives.get("obj_q")) == "quant"
    with pytest.raises(DomainCapabilityError):
        sched.route(svc.stores.objectives.get("obj_q"),
                    required=["simulation"])


def test_domain_budgets_are_independent(tmp_path):
    sched = _scheduler(tmp_path, services={
        "quant": SimpleNamespace(stores=ResearchStores(tmp_path / "q.sqlite")),
        "scinet": SimpleNamespace(stores=ResearchStores(tmp_path / "s.sqlite")),
    })
    sched.preflight()
    sched.budgets.reserve("M5S1", "scinet", "camp_s")
    sched.budgets.consume("M5S1", "scinet", "camp_s", outcome="COMPLETED")
    assert sched.budgets.snapshot("M5S1", "scinet")["remaining"] == 0
    assert sched.budgets.snapshot("M5S1", "quant")["remaining"] == 1
    with pytest.raises(DomainBudgetExhausted):
        sched.budgets.reserve("M5S1", "scinet", "camp_s2")


def test_one_domain_failure_does_not_abort_others(tmp_path, monkeypatch):
    sched = _scheduler(tmp_path, services={
        "quant": SimpleNamespace(stores=ResearchStores(tmp_path / "q.sqlite")),
        "scinet": SimpleNamespace(stores=ResearchStores(tmp_path / "s.sqlite")),
    })
    sched.preflight()

    def fake_run(oid, **kwargs):
        if oid == "bad":
            raise RuntimeError("domain failed")
        return {"objective_id": oid, "ok": True}

    monkeypatch.setattr(sched, "run_objective", fake_run)
    out = sched.run_objectives(["bad", "good"])
    assert out["results"][0]["ok"] is False
    assert out["results"][1]["ok"] is True
    assert set(out["budgets"]) == {"quant", "scinet"}


def test_cross_domain_experience_keeps_source_and_run(tmp_path):
    services = {}
    for domain in ("embodied", "quant"):
        stores = ResearchStores(tmp_path / f"{domain}.sqlite")
        obj = ResearchObjective(objective_id=f"obj_{domain}", domain=domain,
                                description=domain)
        stores.objectives.save(from_research_objective(obj, status="proposed"))
        session = stores.sessions.create_session(obj, session_id=f"session_{domain}")
        stores.sessions.record_experience(
            session_id=session.session_id, objective_id=obj.objective_id,
            kind="research_outcome", payload={"outcome": "inconclusive"})
        services[domain] = SimpleNamespace(stores=stores)

    sched = _scheduler(tmp_path, services=services)
    sched.preflight()
    for domain in ("embodied", "quant"):
        sched.records.write(
            run_id="M5S1", objective_id=f"obj_{domain}", domain_id=domain,
            campaign_id=f"camp_{domain}", approval_id=f"apr_{domain}",
            plan_id=f"plan_{domain}", session_id=f"session_{domain}",
            report_id=f"rep_{domain}", status="COMPLETED")
    rows = sched.read_experience()
    assert {r["source_domain"] for r in rows} == {"embodied", "quant"}
    assert all(r["source_run_id"] == "M5S1" for r in rows)
    filtered = sched.read_experience(["obj_quant"])
    assert filtered and {r["source_domain"] for r in filtered} == {"quant"}
