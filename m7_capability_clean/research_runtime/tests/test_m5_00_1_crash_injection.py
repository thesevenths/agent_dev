"""M5-00.1 / P0-05 -- crash injection: at-most-once + resume-in-place (ENGINE-FREE).

These are TEST DOUBLES of persistence states, not FakeRunner E2E: no claim of
real-engine execution is made here. The proof target is narrower and exact --
after a crash the loop either (a) resumes a PROVEN completed round without
re-executing its campaign, or (b) fails closed and waits for a human.
"""
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
    AWAITING_APPROVAL, CAMPAIGN_IN_FLIGHT, EXPERIENCE_PENDING,
)
from research_runtime.research_store import (
    ResearchStores, from_research_objective,
)
from research_runtime.research_store.schemas import (
    CampaignBinding, ResearchReport,
)
from research_runtime.schemas import ResearchObjective


OID = "obj_crash"
HID = "H_CRASH"
PLAN = "plan_crash"
CAMPAIGN = "campaign_crash"
SESSION = "session_crash"
REPORT = "report_crash"


def _service(root):
    root.mkdir(parents=True, exist_ok=True)
    stores = ResearchStores(root / "research.sqlite")
    svc = SimpleNamespace(
        stores=stores,
        deps=SimpleNamespace(campaign_store=CampaignStore(root / "campaigns.sqlite")))
    svc.get_research_report = lambda session_id: {
        "ok": True, "report": json.loads(json.dumps(
            stores.sessions.get_report(REPORT).model_dump(mode="json")))}
    return svc


def _objective(stores):
    obj = ResearchObjective(objective_id=OID, domain="test",
                            description="crash fixture")
    stores.objectives.save(from_research_objective(obj, status="proposed"))
    return obj


def _plan(stores):
    stores.hypotheses.save({
        "hypothesis_id": HID, "objective_id": OID,
        "source_experiment_id": "src", "title": "t", "statement": "s",
        "mechanism": "m", "evidence_refs": ["experiment:src:result:x"],
        "intervention": {"train.lr": 0.0001},
        "predicted_effects": [{"metric": "m", "direction": "decrease",
                               "expected_change_pct": 1.0}],
        "falsification_criteria": ["f"], "confidence": 0.5})
    stores.hypotheses.save_review(HID, passed=True)
    return stores.plans.save({
        "plan_id": PLAN, "objective_id": OID, "baseline_experiment_id": "base",
        "dataset_version": "ds", "candidates": [{
            "candidate_id": "cand", "hypothesis_id": HID,
            "config_overrides": {"train.lr": 0.0001}, "seeds": [42],
            "dataset_version": "ds", "evaluation_split": "validation"}],
        "primary_metric": "m", "paired_seeds": [42]})


def _loop(svc, out, *, max_rounds=1):
    cfg = LoopConfig(loop_id="L_CRASH", domain="test",
                     max_rounds=max_rounds, max_campaigns=2)
    loop = AutonomousResearchLoop(service=svc, config=cfg, out_dir=out)
    loop.preflight = lambda: {"ready": True, "detail": {}}
    return loop


def _persist_loop(loop, *, round_no=1, rounds=None, state=S_DISCOVERING):
    rec = LoopRecord(loop_id=loop.config.loop_id, state=state, round_no=round_no,
                     config=loop.config.__dict__.copy(),
                     config_sha=loop.config_sha, rounds=list(rounds or []))
    loop.store.save(rec)
    loop.record = rec
    loop.budget.initialize(rec.loop_id, config_sha=loop.config_sha,
                           max_campaigns=loop.config.max_campaigns,
                           max_wall_time_s=0)
    return rec


def _write_round_file(out, rec):
    d = out / "rounds"
    d.mkdir(parents=True, exist_ok=True)
    (d / "round_01.json").write_text(
        json.dumps({"round_no": 1, **rec}), encoding="utf-8")


def _complete_round(svc, out, *, with_round_file=True, with_loop_rounds=False):
    stores = svc.stores
    obj = _objective(stores)
    plan = _plan(stores)
    approval = stores.approvals.request(
        resource_type="experiment_plan", resource_id=PLAN,
        resource_sha256=plan.plan_sha256)
    stores.approvals.grant(approval.approval_id, approved_by="human",
                           current_resource_sha256=plan.plan_sha256)
    stores.sessions.create_session(obj, session_id=SESSION)
    stores.sessions.bind_campaign(CampaignBinding(
        campaign_id=CAMPAIGN, plan_id=PLAN, approval_id=approval.approval_id,
        plan_sha256=plan.plan_sha256, objective_id=OID, session_id=SESSION))
    svc.deps.campaign_store.create_campaign(CampaignRecord(
        campaign_id=CAMPAIGN, plan_id=PLAN, objective_id=OID,
        status="completed", dataset_version="ds"))
    exp_id = stores.sessions.record_experience(
        session_id=SESSION, objective_id=OID, kind="research_outcome",
        payload={"outcome": "inconclusive"})
    stores.sessions.save_report(ResearchReport(
        report_id=REPORT, session_id=SESSION, objective_id=OID,
        outcome="inconclusive", campaign_count=1, comparison_count=1,
        reviews=[{"passed": True, "reviewer": "deterministic_critic"}],
        experience_refs=[exp_id], evidence_refs=["experiment:src:result:x"],
        promotion="not_requested", frozen_test_used=False))
    did = "m5_round_1_L_CRASH"
    dd = out / "director" / "round_01"
    dd.mkdir(parents=True, exist_ok=True)
    (dd / f"{did}.json").write_text(json.dumps({
        "state": "COMPLETED", "objective_id": OID, "hypothesis_ids": [HID],
        "plan_id": PLAN, "approval_id": approval.approval_id,
        "campaign_id": CAMPAIGN, "session_id": SESSION,
        "report_id": REPORT}), encoding="utf-8")
    (dd / f"{did}.journal.jsonl").write_text(
        json.dumps({"event": "FINALIZE", "to": "COMPLETED"}) + "\n",
        encoding="utf-8")
    rec = {"round_no": 1, "objective_id": OID, "hypothesis_ids": [HID],
           "plan_id": PLAN, "approval_id": approval.approval_id,
           "session_id": SESSION, "director_id": did,
           "campaign_id": CAMPAIGN, "report_id": REPORT,
           "experience_id": exp_id, "director_state": "COMPLETED"}
    loop = _loop(svc, out)
    _persist_loop(loop, round_no=1, rounds=[rec] if with_loop_rounds else [])
    loop.budget.reserve("L_CRASH", CAMPAIGN, config_sha=loop.config_sha)
    loop.budget.consume("L_CRASH", CAMPAIGN, outcome="COMPLETED",
                        config_sha=loop.config_sha)
    if with_round_file:
        _write_round_file(out, rec)
    return loop, rec


def _classification(loop):
    return loop.recover(apply=False).classification


def test_crash_1_objective_created_only_fails_closed(tmp_path):
    svc = _service(tmp_path / "one")
    _objective(svc.stores)
    loop = _loop(svc, tmp_path / "one" / "out")
    _persist_loop(loop, round_no=1)
    _write_round_file(loop.out_dir, {"objective_id": OID})
    report = loop.recover(apply=False)
    assert report.classification not in ("ROUND_COMPLETED", AWAITING_APPROVAL)
    assert svc.deps.campaign_store.campaign_ids() == []


def test_crash_2_plan_created_only_fails_closed(tmp_path):
    svc = _service(tmp_path / "two")
    _objective(svc.stores)
    plan = _plan(svc.stores)
    loop = _loop(svc, tmp_path / "two" / "out")
    _persist_loop(loop, round_no=1)
    _write_round_file(loop.out_dir, {"objective_id": OID, "plan_id": PLAN})
    assert loop.recover(apply=False).classification not in ("ROUND_COMPLETED",)
    assert plan.plan_sha256
    assert svc.deps.campaign_store.campaign_ids() == []


def test_crash_3_approval_before_campaign_is_awaiting_approval(tmp_path):
    svc = _service(tmp_path / "three")
    obj = _objective(svc.stores)
    plan = _plan(svc.stores)
    approval = svc.stores.approvals.request(
        resource_type="experiment_plan", resource_id=PLAN,
        resource_sha256=plan.plan_sha256)
    svc.stores.sessions.create_session(obj, session_id=SESSION)
    loop = _loop(svc, tmp_path / "three" / "out")
    _persist_loop(loop, round_no=1)
    _write_round_file(loop.out_dir, {
        "objective_id": OID, "plan_id": PLAN,
        "approval_id": approval.approval_id})
    assert _classification(loop) == AWAITING_APPROVAL
    assert svc.deps.campaign_store.campaign_ids() == []


def test_crash_4_campaign_in_flight_is_not_completed(tmp_path):
    svc = _service(tmp_path / "four")
    loop, rec = _complete_round(svc, tmp_path / "four" / "out",
                                with_loop_rounds=True)
    svc.deps.campaign_store.save_campaign(
        svc.deps.campaign_store.load_campaign(CAMPAIGN).model_copy(
            update={"status": "running"}))
    assert _classification(loop) == CAMPAIGN_IN_FLIGHT


def test_crash_5_report_without_confirmed_experience_is_pending(tmp_path):
    svc = _service(tmp_path / "five")
    loop, rec = _complete_round(svc, tmp_path / "five" / "out",
                                with_loop_rounds=True)
    bad = dict(rec)
    bad["experience_id"] = "exp_missing"
    _write_round_file(loop.out_dir, bad)
    loop.record.rounds = [bad]
    loop.store.save(loop.record)
    assert _classification(loop) == EXPERIENCE_PENDING


def test_crash_6_experience_written_round_json_lost_resumes_in_place(tmp_path):
    svc = _service(tmp_path / "six")
    loop, _rec = _complete_round(svc, tmp_path / "six" / "out",
                                 with_round_file=False)
    called = []
    loop._run_round = lambda *a, **k: called.append(True)
    result = loop.run()
    assert result["n_rounds"] == 1
    assert not called                      # campaign never re-executed
    assert (loop.out_dir / "rounds" / "round_01.json").is_file()
    assert svc.deps.campaign_store.campaign_ids() == [CAMPAIGN]


def test_crash_7_round_json_written_loop_record_lost_resumes_in_place(tmp_path):
    svc = _service(tmp_path / "seven")
    loop, _rec = _complete_round(svc, tmp_path / "seven" / "out",
                                 with_round_file=True, with_loop_rounds=False)
    called = []
    loop._run_round = lambda *a, **k: called.append(True)
    result = loop.run()
    assert result["n_rounds"] == 1
    assert not called
    assert svc.deps.campaign_store.campaign_ids() == [CAMPAIGN]


def test_crash_8_9_10_restart_keeps_budget_and_never_repeats_campaign(tmp_path):
    root = tmp_path / "eight"
    svc = _service(root)
    loop, _rec = _complete_round(svc, root / "out", with_round_file=True,
                                 with_loop_rounds=True)
    loop.run()
    before = svc.deps.campaign_store.campaign_ids()

    # Fresh process: only SQLite + immutable files survive.
    svc2 = _service(root)
    restarted = _loop(svc2, root / "out")
    restarted._run_round = lambda *a, **k: pytest.fail("campaign re-executed")
    result = restarted.run()
    assert result["n_rounds"] == 1
    assert svc2.deps.campaign_store.campaign_ids() == before == [CAMPAIGN]
    snap = restarted.budget.snapshot("L_CRASH")
    assert (snap["consumed"], snap["reserved"], snap["remaining"]) == (1, 0, 1)
    assert config_sha256(restarted.config) == restarted.record.config_sha
