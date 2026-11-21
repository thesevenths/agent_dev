"""ARR v1.0 WP-01: production runtime integrity — durable campaign claims.

Tests the REAL (engine-free backend) FormalRuntimeAdapter + RuntimeState campaign
state machine. Every assertion maps to a WP-01 acceptance requirement.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from research_runtime.discovery.discovery_coordinator import DiscoveryCoordinator
from research_runtime.discovery.formal_runtime import FormalRuntimeAdapter
from research_runtime.harness_rsi.harness_loader import LoadedHarness
from research_runtime.research_store.db import ResearchDatabase
from research_runtime.runtime_state import (
    RuntimeState, RuntimeStateError)

from test_m7_discovery_coordinator import (FakeAgent, FakeProvider,
                                           fake_executor)


class _Loader:
    def resolve_for_campaign(self, *, campaign_id, domain):
        return LoadedHarness(harness_sha="a" * 64,
                             component_shas={"critic": "b" * 64},
                             parent_sha="c" * 64, activation_id="act")


def _runtime(tmp_path, run_id="wp01"):
    rt = RuntimeState(db=ResearchDatabase(tmp_path / "rt.sqlite"))
    if not rt.run(run_id):
        rt.initialize(run_id, config=SimpleNamespace(), dataset_shas={},
                      max_rounds=8, global_budget=8)
    return rt


def _adapter(rt, backend, *, executor_id="exec_wp01"):
    return FormalRuntimeAdapter(
        runtime=rt, harness_loader=_Loader(), backend=backend,
        domain="embodied", run_id="wp01", executor_id=executor_id)


def _dispatch(adapter, rid=1, experiment_id="e1"):
    return adapter.dispatch(
        experiment_id=experiment_id, preregistration_sha="f" * 64,
        arms=["baseline"], seeds=[42], splits=["validation"],
        out_dir="", stage="EXPERIMENT_EXECUTION",
        expected_harness_sha="a" * 64,
        config_sha=adapter.runtime.run("wp01")["config_sha"], round_no=rid)


# --- 1. duplicate dispatch must NOT re-execute backend ---------------------

def test_duplicate_dispatch_reuses_result_without_backend(tmp_path):
    rt = _runtime(tmp_path)
    calls = []

    def backend(**kw):
        calls.append(kw)
        return {"ok": True, "summary_sha256": "d" * 64}

    a = _adapter(rt, backend)
    r1 = _dispatch(a)
    r2 = _dispatch(a)
    assert len(calls) == 1, "duplicate dispatch re-executed the backend"
    assert r1["campaign_id"] == r2["campaign_id"]
    assert r2["cache_hit"] is True
    assert r2["result_sha256"] == r1["result_sha256"]
    assert len(rt.reservations("wp01")) == 1
    assert rt.verify_journal("wp01")["ok"]


# --- 2. concurrent/different executor is refused --------------------------

def test_second_executor_cannot_claim_same_campaign(tmp_path):
    rt = _runtime(tmp_path)
    cid = "camp_concurrent"
    c1 = rt.claim_campaign(campaign_id=cid, run_id="wp01",
                           executor_id="exec_a", config_sha="x" * 64,
                           harness_sha="a" * 64)
    assert c1["ok"] is True
    c2 = rt.claim_campaign(campaign_id=cid, run_id="wp01",
                           executor_id="exec_b", config_sha="x" * 64,
                           harness_sha="a" * 64)
    assert c2["ok"] is False and c2["reason"] == "claimed_by_another_executor"
    # same executor re-claim is idempotent
    c3 = rt.claim_campaign(campaign_id=cid, run_id="wp01",
                           executor_id="exec_a", config_sha="x" * 64,
                           harness_sha="a" * 64)
    assert c3["ok"] is True and c3["resume_in_place"] is True


# --- 3. state machine transitions validated --------------------------------

def test_campaign_state_transitions_validated(tmp_path):
    rt = _runtime(tmp_path)
    cid = "camp_states"
    rt.claim_campaign(campaign_id=cid, run_id="wp01", executor_id="e",
                      config_sha="x" * 64, harness_sha="a" * 64)
    rt.set_campaign_state(cid, "RUNNING", executor_id="e")
    assert rt.campaign_state(cid)["state"] == "RUNNING"
    rt.set_campaign_state(cid, "COMPLETED", executor_id="e",
                          result={"ok": True}, result_sha="z" * 64)
    assert rt.campaign_state(cid)["state"] == "COMPLETED"
    # terminal is immutable: no transition out of COMPLETED
    with pytest.raises(RuntimeStateError):
        rt.set_campaign_state(cid, "RUNNING", executor_id="e")


# --- 4. crash leaves RECOVERY_REQUIRED; never auto-retry -------------------

def test_interrupted_campaign_is_recovery_required_not_retried(tmp_path):
    rt = _runtime(tmp_path)
    cid = "camp_crash"
    rt.claim_campaign(campaign_id=cid, run_id="wp01", executor_id="e",
                      config_sha="x" * 64, harness_sha="a" * 64)
    rt.set_campaign_state(cid, "RUNNING", executor_id="e")
    out = rt.recover_interrupted_campaigns("wp01", executor_id="e")
    assert any(o["campaign_id"] == cid for o in out)
    assert rt.campaign_state(cid)["state"] == "RECOVERY_REQUIRED"
    # a claim on a RECOVERY_REQUIRED campaign is refused (no blind retry)
    c = rt.claim_campaign(campaign_id=cid, run_id="wp01", executor_id="e",
                          config_sha="x" * 64, harness_sha="a" * 64)
    assert c["ok"] is False and "unsafe_recovery" in c["reason"]


# --- 5. failed backend settles as FAILED, not COMPLETED --------------------

def test_backend_failure_settles_failed_not_success(tmp_path):
    rt = _runtime(tmp_path)

    def bad(**kw):
        raise RuntimeError("engine down")

    a = _adapter(rt, bad)
    with pytest.raises(RuntimeError):
        _dispatch(a, experiment_id="e_fail")
    states = [r["state"] for r in rt.reservations("wp01")]
    assert all(s == "consumed" for s in states)
    # find the failed campaign state
    failed = [r for r in rt.db.rows(
        "SELECT campaign_id,state FROM rt_campaign_exec")
        if r["state"] == "FAILED"]
    assert failed, "failed backend did not mark campaign FAILED"


# --- 6. exactly-once HONESTY: no exactly-once claim in source --------------

def test_no_exactly_once_claim_in_runtime_source():
    src = open(__import__("research_runtime.runtime_state", fromlist=["x"])
               .__file__, encoding="utf-8").read()
    low = src.lower().replace("\n", " ")
    assert "exactly-once" in low or "exactly once" in low
    # must be a DISCLAIMER, not an assertion of guarantee
    assert "no exactly-once claim" in low
    assert "at-most-once" in low


# --- 7. claim graph: INCONCLUSIVE must NOT become REFUTES ------------------

def test_inconclusive_verdict_emits_neutral_edge(tmp_path):
    from research_runtime.discovery.knowledge.graph import EDGE_TYPES
    assert "INCONCLUSIVE_EVIDENCE" in EDGE_TYPES
    # Drive a coordinator with INCONCLUSIVE verdicts and assert the edge map.
    from research_runtime.tests.test_m7_discovery_coordinator import make_coordinator
    c = make_coordinator(tmp_path, pilot_effects=None)  # -> INCONCLUSIVE verdict
    r = c.run()
    assert r["ok"] and r["status"] == "COMPLETED"
    mapping = c.state.get("claim_edge_mapping", {})
    edges = {v["edge"] for v in mapping.values()}
    assert edges and all(e != "REFUTES" for e in edges), mapping
    assert c.state["scientific_verdict"] == "INCONCLUSIVE"
