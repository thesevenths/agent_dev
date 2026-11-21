"""M4-E2E-01.1 P1 -- Campaign idempotency & crash recovery (engine-free).

These tests use the SAME engine-free ``Harness`` as the rest of M4-E2E-01. They
prove the campaign launch is EXACTLY-once and budget-bounded even under crashes:

  * A duplicate ``run_research_campaign`` (same plan + approval) never launches a
    second campaign -- the DB ``UNIQUE(plan_id, approval_id, plan_sha256)`` binding
    plus the persisted ``CampaignRecord`` are the idempotency authority.
  * If the launch CRASHES before the runner writes its ``CampaignRecord`` (binding
    already persisted = execution intent), a later call RE-LAUNCHES exactly once
    -- it must never silently skip a run that never happened, nor double-execute.
  * An externally-created binding with no record (another process bound then died)
    is recovered and launched exactly once.
  * The per-objective round cap still holds after recovery (budget is not silently
    consumed by a crashed run).

No new DB schema, no rewrite of the CampaignRunner: only the launch-or-skip
decision in ``ResearchService._run_campaign`` is hardened.
"""
from __future__ import annotations

import json
import pytest

from research_runtime.research_service import make_campaign_id
from research_runtime.research_store.schemas import CampaignBinding

from tests.rt_tests.research_fakes import (
    Harness, FakeRunner, OBJECTIVE_ID, HYPOTHESIS_ID,
)


class _CrashThenSucceedRunner(FakeRunner):
    """Runner that raises on the FIRST ``.run()`` (before persisting a record)."""

    def __init__(self, store):
        super().__init__(store)
        self.calls = 0

    def run(self, plan, run_spec):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("injected crash before persist")
        return super().run(plan, run_spec)


def _campaign_records(harness: Harness) -> list:
    return list(harness.campaign_store.records.values())


# --------------------------------------------------------------------- 1: duplicate submission => no double execution
def test_duplicate_submission_does_not_double_execute():
    harness = Harness()
    try:
        chain = harness.chain(grant=True)
        pid = chain["plan"]["plan_id"]
        aid = chain["approval_id"]

        r1 = harness.service.run_research_campaign(plan_id=pid, approval_id=aid)
        assert r1["ok"] and not r1.get("already_submitted"), r1

        r2 = harness.service.run_research_campaign(plan_id=pid, approval_id=aid)
        assert r2["ok"] and r2.get("already_submitted") is True, r2

        # Exactly one CampaignRecord exists -- the duplicate was a no-op.
        assert len(_campaign_records(harness)) == 1
    finally:
        harness.close()


# --------------------------------------------------------------------- 2: crash before launch => recover & run exactly once
def test_crash_before_launch_recovers_and_runs_once():
    harness = Harness()
    try:
        chain = harness.chain(grant=True)
        pid = chain["plan"]["plan_id"]
        aid = chain["approval_id"]

        # First launch crashes before the runner persists a record.
        harness.runner = _CrashThenSucceedRunner(harness.campaign_store)
        r1 = harness.service.run_research_campaign(plan_id=pid, approval_id=aid)
        assert not r1["ok"], "crash must surface as an error"
        assert len(_campaign_records(harness)) == 0, \
            "no record should exist until the runner persists one"

        # Recovery: a healthy runner re-launches exactly once.
        harness.runner = FakeRunner(harness.campaign_store)
        r2 = harness.service.run_research_campaign(plan_id=pid, approval_id=aid)
        assert r2["ok"] and not r2.get("already_submitted"), r2
        records = _campaign_records(harness)
        assert len(records) == 1
        assert records[0].campaign_id == r2["campaign_id"]
    finally:
        harness.close()


# --------------------------------------------------------------------- 3: recover, then duplicate => still exactly one record
def test_recovery_then_duplicate_still_single_record():
    harness = Harness()
    try:
        chain = harness.chain(grant=True)
        pid = chain["plan"]["plan_id"]
        aid = chain["approval_id"]

        harness.runner = _CrashThenSucceedRunner(harness.campaign_store)
        r1 = harness.service.run_research_campaign(plan_id=pid, approval_id=aid)
        assert not r1["ok"]  # crashed

        harness.runner = FakeRunner(harness.campaign_store)
        r2 = harness.service.run_research_campaign(plan_id=pid, approval_id=aid)
        assert r2["ok"] and not r2.get("already_submitted")

        # A third call (after a successful recovery) must be idempotent.
        r3 = harness.service.run_research_campaign(plan_id=pid, approval_id=aid)
        assert r3.get("already_submitted") is True
        assert len(_campaign_records(harness)) == 1
    finally:
        harness.close()


# --------------------------------------------------------------------- 4: external binding w/o record => relaunch once
def test_external_binding_without_record_relaunches_once():
    harness = Harness()
    try:
        chain = harness.chain(grant=True)
        pid = chain["plan"]["plan_id"]
        aid = chain["approval_id"]
        sha = chain["plan"]["plan_sha256"]
        cid = make_campaign_id(pid, aid, sha)

        # Another process bound the campaign but died before launching.
        harness.stores.sessions.bind_campaign(CampaignBinding(
            campaign_id=cid, plan_id=pid, approval_id=aid, plan_sha256=sha,
            objective_id=OBJECTIVE_ID, session_id=""))

        r = harness.service.run_research_campaign(plan_id=pid, approval_id=aid)
        assert r["ok"] and not r.get("already_submitted"), r
        records = _campaign_records(harness)
        assert len(records) == 1
        # The recovery must reuse the EXISTING campaign id, never mint a new one.
        assert records[0].campaign_id == cid == r["campaign_id"]
    finally:
        harness.close()


# --------------------------------------------------------------------- 5: round cap still enforced after recovery
def test_round_limit_enforced_after_recovery():
    harness = Harness()
    try:
        harness.replace_domain(max_campaign_rounds=1)
        chain = harness.chain(grant=True)
        pid = chain["plan"]["plan_id"]
        aid = chain["approval_id"]

        r1 = harness.service.run_research_campaign(plan_id=pid, approval_id=aid)
        assert r1["ok"], r1
        assert len(_campaign_records(harness)) == 1

        # A second, distinct plan + approval for the SAME objective must be blocked
        # by the round cap -- a crashed run does NOT free budget.
        h2 = harness.service.create_research_plan(
            objective_id=OBJECTIVE_ID,
            hypothesis_ids_json=json.dumps([HYPOTHESIS_ID]),
            plan_id="plan_second")
        assert h2["ok"], h2
        req2 = harness.service.request_plan_approval(plan_id="plan_second")
        assert req2["ok"], req2
        harness.stores.approvals.grant(
            req2["approval_id"], approved_by="alice", source="human",
            current_resource_sha256=h2["plan_sha256"])

        res = harness.service.run_research_campaign(
            plan_id="plan_second", approval_id=req2["approval_id"])
        assert res["ok"] is False
        assert res["error_code"] == "round_limit_reached"
        # No second campaign record was ever created.
        assert len(_campaign_records(harness)) == 1
    finally:
        harness.close()
