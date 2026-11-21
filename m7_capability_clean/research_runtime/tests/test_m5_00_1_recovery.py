"""M5-00.1 / P0-02..P0-04 -- real-store recovery + immutable round evidence (ENGINE-FREE)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from research_runtime.autonomous_loop_state import (
    LoopEventStore, LoopRecord, S_DISCOVERING,
)
from research_runtime.autonomous_research_loop import (
    AutonomousResearchLoop, LoopConfig, LoopRecoveryRequired,
)
from research_runtime.campaign.campaign_store import CampaignStore
from research_runtime.campaign.schemas import CampaignRecord
from research_runtime.loop_budget_ledger import LoopBudgetLedger, config_sha256
from research_runtime.loop_reality_reconciler import (
    AWAITING_APPROVAL, CAMPAIGN_IN_FLIGHT, CORRUPT, RECOVERY_REQUIRED,
    ROUND_COMPLETED, reconcile_round,
)
from research_runtime.research_store import (
    ResearchStores, ResearchStoreConflict, from_research_objective,
)
from research_runtime.research_store.schemas import (
    CampaignBinding, ResearchReport,
)
from research_runtime.schemas import ResearchObjective


OID = "obj_recovery_test"
HID = "H_RECOVERY"
PLAN = "plan_recovery"
APPROVAL = "approval_recovery"
CAMPAIGN = "campaign_recovery"
SESSION = "session_recovery"
REPORT = "report_recovery"
EXP = "exp_recovery"


def _fake_service(tmp_path):
    root = tmp_path / "db"
    root.mkdir(parents=True, exist_ok=True)
    stores = ResearchStores(root / "research.sqlite")
    svc = SimpleNamespace(
        stores=stores,
        deps=SimpleNamespace(
            campaign_store=CampaignStore(root / "campaigns.sqlite")),
    )
    svc.get_research_report = lambda session_id: {
        "ok": True,
        "report": json.loads(json.dumps(
            stores.sessions.get_report(REPORT).model_dump(mode="json")))
    }
    return svc


def _build_complete_round(tmp_path, *, report_passed=True,
                          campaign_status="completed", approval_status="approved"):
    svc = _fake_service(tmp_path)
    stores = svc.stores
    objective = ResearchObjective(objective_id=OID, domain="test",
                                  description="recovery fixture")
    stores.objectives.save(from_research_objective(objective, status="proposed"))
    stores.hypotheses.save({
        "hypothesis_id": HID, "objective_id": OID,
        "source_experiment_id": "src_recovery", "title": "t",
        "statement": "s", "mechanism": "m",
        "evidence_refs": ["experiment:src:result:x"],
        "intervention": {"train.lr": 0.0001},
        "predicted_effects": [{"metric": "m", "direction": "decrease",
                               "expected_change_pct": 1.0}],
        "falsification_criteria": ["f"], "confidence": 0.5,
    })
    stores.hypotheses.save_review(HID, passed=report_passed)
    plan = stores.plans.save({
        "plan_id": PLAN, "objective_id": OID,
        "baseline_experiment_id": "base", "dataset_version": "ds",
        "candidates": [{
            "candidate_id": "cand", "hypothesis_id": HID,
            "config_overrides": {"train.lr": 0.0001}, "seeds": [42],
            "dataset_version": "ds", "evaluation_split": "validation"}],
        "primary_metric": "m", "paired_seeds": [42],
    })
    approval = stores.approvals.request(
        resource_type="experiment_plan", resource_id=PLAN,
        resource_sha256=plan.plan_sha256)
    if approval_status == "approved":
        stores.approvals.grant(approval.approval_id, approved_by="human",
                               current_resource_sha256=plan.plan_sha256)
    stores.sessions.create_session(objective, session_id=SESSION)
    svc.deps.campaign_store.create_campaign(CampaignRecord(
        campaign_id=CAMPAIGN, plan_id=PLAN, objective_id=OID,
        status=campaign_status, dataset_version="ds"))
    real_exp_id = stores.sessions.record_experience(
        session_id=SESSION, objective_id=OID,
        kind="research_outcome", payload={"outcome": "inconclusive"})
    stores.sessions.save_report(ResearchReport(
        report_id=REPORT, session_id=SESSION, objective_id=OID,
        outcome="inconclusive", campaign_count=1, comparison_count=1,
        reviews=[{"passed": report_passed, "reviewer": "deterministic_critic"}],
        experience_refs=[real_exp_id], evidence_refs=["experiment:src:result:x"],
        promotion="not_requested", frozen_test_used=False))
    # Rebind with the REAL approval id (request() generates its own id).
    stores.sessions.bind_campaign(CampaignBinding(
        campaign_id=CAMPAIGN, plan_id=PLAN,
        approval_id=approval.approval_id,
        plan_sha256=plan.plan_sha256, objective_id=OID, session_id=SESSION))

    director_dir = tmp_path / "out" / "director" / "round_01"
    director_dir.mkdir(parents=True, exist_ok=True)
    did = "m5_round_1_LOOP"
    (director_dir / f"{did}.json").write_text(json.dumps({
        "state": "COMPLETED", "objective_id": OID, "hypothesis_ids": [HID],
        "plan_id": PLAN, "approval_id": approval.approval_id,
        "campaign_id": CAMPAIGN, "session_id": SESSION,
        "report_id": REPORT}), encoding="utf-8")
    (director_dir / f"{did}.journal.jsonl").write_text(
        json.dumps({"event": "FINALIZE", "to": "COMPLETED"}) + "\n",
        encoding="utf-8")

    round_rec = {
        "round_no": 1, "objective_id": OID, "hypothesis_ids": [HID],
        "plan_id": PLAN, "approval_id": approval.approval_id,
        "session_id": SESSION, "director_id": did,
        "campaign_id": CAMPAIGN, "report_id": REPORT,
        "experience_id": real_exp_id, "director_state": "COMPLETED",
    }
    return svc, round_rec, approval.approval_id


def _ledger_for(svc, loop_id="LOOP", *, config_sha="sha"):
    ledger = LoopBudgetLedger(db=svc.stores.db)
    ledger.initialize(loop_id, config_sha=config_sha, max_campaigns=2,
                      max_wall_time_s=0)
    ledger.reserve(loop_id, CAMPAIGN, config_sha=config_sha)
    ledger.consume(loop_id, CAMPAIGN, outcome="COMPLETED", config_sha=config_sha)
    return ledger


def _reality(tmp_path, round_rec, approval_id, **helpers):
    svc = _fake_service(tmp_path)
    # _build_complete_round already wrote matching records; reopen them here.
    return reconcile_round(
        svc, {**round_rec, "approval_id": approval_id},
        rounds_dir=tmp_path / "out" / "rounds",
        state_dir=tmp_path / "out" / "director",
        budget_ledger=_ledger_for(svc), loop_id="LOOP", config_sha="sha",
        **helpers)


def test_real_store_reconciliation_accepts_only_proven_completion(tmp_path):
    svc, rec, approval_id = _build_complete_round(tmp_path)
    ledger = _ledger_for(svc)
    reality = reconcile_round(
        svc, rec, rounds_dir=tmp_path / "out" / "rounds",
        state_dir=tmp_path / "out" / "director",
        budget_ledger=ledger, loop_id="LOOP", config_sha="sha")
    assert reality.classification == ROUND_COMPLETED
    assert reality.gate_ok is True

    # A campaign that merely exists as an id but is still running is NOT complete.
    svc2, rec2, approval2 = _build_complete_round(tmp_path / "running")
    svc2.deps.campaign_store.save_campaign(
        svc2.deps.campaign_store.load_campaign(CAMPAIGN).model_copy(
            update={"status": "running"}))
    running = reconcile_round(
        svc2, rec2, rounds_dir=tmp_path / "running" / "out" / "rounds",
        state_dir=tmp_path / "running" / "out" / "director",
        budget_ledger=_ledger_for(svc2), loop_id="LOOP", config_sha="sha")
    assert running.classification == CAMPAIGN_IN_FLIGHT
    assert running.gate_ok is False

    # Unresolved report id must never be reported complete.
    missing = reconcile_round(
        svc, {**rec, "report_id": "report_missing"},
        rounds_dir=tmp_path / "out" / "rounds",
        state_dir=tmp_path / "out" / "director",
        budget_ledger=ledger, loop_id="LOOP", config_sha="sha")
    assert missing.classification != ROUND_COMPLETED
    assert "report" in missing.unresolved

    # Gate review failure or plan/approval SHA conflict cannot pass.
    bad_gate = reconcile_round(
        svc, rec, rounds_dir=tmp_path / "out" / "rounds",
        state_dir=tmp_path / "out" / "director", budget_ledger=ledger,
        loop_id="LOOP", config_sha="sha")
    assert bad_gate.gate_ok is True   # baseline before tamper
    with svc.stores.db.session() as conn:
        conn.execute("UPDATE approvals SET resource_sha256='bad_sha' "
                     "WHERE approval_id=?", (approval_id,))
    corrupt = reconcile_round(
        svc, rec, rounds_dir=tmp_path / "out" / "rounds",
        state_dir=tmp_path / "out" / "director",
        budget_ledger=ledger, loop_id="LOOP", config_sha="sha")
    assert corrupt.classification == CORRUPT


def test_round_evidence_is_immutable(tmp_path):
    db = ResearchStores(tmp_path / "round.sqlite").db
    svc = SimpleNamespace(stores=SimpleNamespace(db=db))
    loop = AutonomousResearchLoop(service=svc,
                                  config=LoopConfig(loop_id="L_IMM"),
                                  out_dir=tmp_path / "out")
    payload = {"round_no": 1, "objective_id": "o", "campaign_id": "c"}
    first = loop._write_round(1, payload)
    same = loop._write_round(1, payload)
    assert first["evidence_sha256"] == same["evidence_sha256"]
    with pytest.raises(ResearchStoreConflict):
        loop._write_round(1, {**payload, "campaign_id": "other"})


def test_run_fails_closed_when_recovery_cannot_resolve_round(tmp_path):
    svc = _fake_service(tmp_path / "failclosed")
    db = svc.stores.db
    cfg = LoopConfig(loop_id="L_FAIL", domain="test")
    loop = AutonomousResearchLoop(service=svc, config=cfg,
                                  out_dir=tmp_path / "out")
    loop.preflight = lambda: {"ready": True, "detail": {}}
    store = LoopEventStore(db=db)
    rec = LoopRecord(loop_id="L_FAIL", state=S_DISCOVERING,
                     round_no=1, config=cfg.__dict__.copy(),
                     config_sha=config_sha256(cfg))
    store.save(rec)
    loop.record = rec
    loop.budget.initialize("L_FAIL", config_sha=loop.config_sha,
                           max_campaigns=2, max_wall_time_s=0)
    (tmp_path / "out" / "rounds").mkdir(parents=True, exist_ok=True)
    (tmp_path / "out" / "rounds" / "round_01.json").write_text(json.dumps({
        "round_no": 1, "objective_id": "unresolvable",
        "campaign_id": "campaign_missing"}), encoding="utf-8")
    called = []
    loop._run_round = lambda *a, **k: called.append(True)
    with pytest.raises(LoopRecoveryRequired):
        loop.run()
    assert not called
