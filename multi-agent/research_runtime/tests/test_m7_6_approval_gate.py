"""M7.6 P0-A: approval gate negative matrix + out-of-band CLI tests.

Every case here is engine-free (no LLM/engine). Fixtures are explicitly stamped
and NEVER accepted in production mode. The production path re-reads the M4.3
ApprovalStore and rejects auto/in-memory approvals.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from research_runtime.discovery.approval_gate import verify_external_approval
from research_runtime.discovery.discovery_coordinator import (
    DiscoveryBlocked, DiscoveryCoordinator)
from research_runtime.harness_rsi.identity import sha256_file
from research_runtime.research_store import ResearchStores
from research_runtime.research_store.db import ResearchDatabase

from test_m7_discovery_coordinator import (FakeAgent, FakeProvider,
                                           fake_executor)

PROTOCOL_SHA = "c" * 64


def _fixture_coordinator(tmp_path, **overrides):
    kw = dict(run_id="m76_approval_test",
              db=ResearchDatabase(tmp_path / "approval.sqlite"),
              out_dir=tmp_path / "discovery", providers=[FakeProvider()],
              agent=FakeAgent(), executor=fake_executor,
              evidence_map={
                  "eig": {"hypotheses": ["H0", "H1"], "priors": [0.5, 0.5],
                          "likelihoods": {"H0": [0.9, 0.1], "H1": [0.2, 0.8]}},
                  "scientific_significance": {"independent_review": [
                      {"reviewer": "test-fixture", "verdict": "RELEVANT"}]},
                  "practical_impact": {"independent_review": [
                      {"reviewer": "test-fixture", "verdict": "RELEVANT"}]}},
              pilot_effects={"e1": 0.1, "e2": 0.12, "e3": 0.11, "e4": 0.13})
    kw.update(overrides)
    return DiscoveryCoordinator(**kw)


def _gate(store, approval_id, protocol_sha=PROTOCOL_SHA):
    return verify_external_approval(
        approval_store=store, approval_id=approval_id,
        protocol_sha=protocol_sha)


def _grant(store, protocol_sha=PROTOCOL_SHA, approved_by="Dr. Hu",
           status="approved"):
    # M7 approvals bind the frozen PREREGISTRATION, not an M5 experiment_plan;
    # the gate verifies resource_sha256/status/source/scope, not resource_type.
    rec = store.request(resource_type="m7_preregistration",
                        resource_id="plan_m76", resource_sha256=protocol_sha)
    if status == "approved":
        return store.grant(rec.approval_id, approved_by=approved_by)
    if status == "rejected":
        return store.reject(rec.approval_id, decided_by=approved_by)
    return rec


def _stores(tmp_path):
    return ResearchStores(tmp_path / "trusted.sqlite")


class _FakeLoader:
    def resolve_for_campaign(self, *, campaign_id, domain):
        from research_runtime.harness_rsi.harness_loader import LoadedHarness
        return LoadedHarness(harness_sha="a" * 64,
                             component_shas={"critic": "b" * 64},
                             parent_sha="c" * 64, activation_id="act")


def _production_kwargs(tmp_path):
    """Minimal formal seam so approval-only tests reach the approval stage."""
    from research_runtime.discovery.formal_runtime import FormalRuntimeAdapter
    from research_runtime.runtime_state import RuntimeState
    from types import SimpleNamespace
    runtime = RuntimeState(db=ResearchDatabase(tmp_path / "runtime.sqlite"))
    if not runtime.run("m76_approval_test"):
        runtime.initialize("m76_approval_test",
                           config=SimpleNamespace(), dataset_shas={},
                           max_rounds=6, global_budget=4)
    adapter = FormalRuntimeAdapter(
        runtime=runtime, harness_loader=_FakeLoader(),
        backend=fake_executor, domain="embodied_world_model",
        run_id="m76_approval_test")
    return dict(harness_sha="a" * 64, formal_loader=_FakeLoader(),
                production_runtime=adapter)


# --- verify_external_approval matrix ---------------------------------------

def test_missing_approval_blocks(tmp_path):
    v = _gate(_stores(tmp_path).approvals, "approval_nonexistent")
    assert v["ok"] is False and v["reason"] == "approval_not_found"


def test_pending_approval_does_not_authorize(tmp_path):
    stores = _stores(tmp_path)
    rec = _grant(stores.approvals, status="pending")
    assert _gate(stores.approvals, rec.approval_id)["reason"] == "approval_status_pending"


def test_rejected_approval_does_not_authorize(tmp_path):
    stores = _stores(tmp_path)
    rec = _grant(stores.approvals, status="rejected")
    assert _gate(stores.approvals, rec.approval_id)["reason"] == "approval_status_rejected"


def test_revoked_approval_does_not_authorize(tmp_path):
    stores = _stores(tmp_path)
    rec = _grant(stores.approvals)
    stores.approvals.revoke(rec.approval_id, decided_by="Dr. Hu")
    assert _gate(stores.approvals, rec.approval_id)["reason"] == "approval_status_revoked"


def test_wrong_protocol_sha_rejected(tmp_path):
    stores = _stores(tmp_path)
    rec = _grant(stores.approvals)
    assert _gate(stores.approvals, rec.approval_id, "f" * 64)["reason"] == \
        "approval_not_bound_to_protocol_sha"


def test_agent_cannot_approve_itself(tmp_path):
    stores = _stores(tmp_path)
    rec = _grant(stores.approvals, approved_by="research_agent")
    assert _gate(stores.approvals, rec.approval_id)["reason"] == \
        "approval_approver_forbidden_or_missing"


def test_legacy_human_operator_string_rejected(tmp_path):
    stores = _stores(tmp_path)
    rec = _grant(stores.approvals, approved_by="human-operator")
    assert _gate(stores.approvals, rec.approval_id)["ok"] is False


def test_system_policy_source_rejected(tmp_path):
    stores = _stores(tmp_path)
    rec = stores.approvals.request(
        resource_type="experiment_plan", resource_id="plan_m76",
        resource_sha256=PROTOCOL_SHA)
    stores.approvals.grant(rec.approval_id, approved_by="ops",
                           source="system_policy")
    assert "source_rejected" in _gate(stores.approvals, rec.approval_id)["reason"]


def test_missing_scope_rejected(tmp_path):
    stores = _stores(tmp_path)
    rec = stores.approvals.request(
        resource_type="experiment_plan", resource_id="plan_m76",
        resource_sha256=PROTOCOL_SHA, scope=["resume_campaign"])
    stores.approvals.grant(rec.approval_id, approved_by="Dr. Hu")
    assert _gate(stores.approvals, rec.approval_id)["reason"] == \
        "approval_scope_missing:run_campaign"


def test_valid_external_human_approval_passes(tmp_path):
    stores = _stores(tmp_path)
    rec = _grant(stores.approvals)
    verdict = _gate(stores.approvals, rec.approval_id)
    assert verdict["ok"] is True
    assert verdict["approved_by"] == "Dr. Hu"
    assert verdict["evidence_sha"]


# --- coordinator production gate ------------------------------------------

def test_production_blocks_auto_approval_callback(tmp_path):
    stores = _stores(tmp_path)
    c = _fixture_coordinator(
        tmp_path,
        approval=lambda sha: {"protocol_sha": sha, "approver": "human-operator",
                              "decision": "APPROVED"},
        production_mode=True, **_production_kwargs(tmp_path))
    result = c.run()
    assert result["status"] == "BLOCKED"
    assert "approval" in result["reason"]
    assert not (c.root / "experiments" / "selection").exists()


def test_production_requires_trusted_store_record(tmp_path):
    stores = _stores(tmp_path)
    # Phase 1: no approval -> freezes preregistration and blocks at the gate.
    c1 = _fixture_coordinator(tmp_path, production_mode=True,
                              **_production_kwargs(tmp_path))
    r1 = c1.run()
    assert r1["status"] == "BLOCKED" and r1["stage"] == "HUMAN_APPROVAL"
    sha = c1.state["preregistration_sha"]
    # Out-of-band: a human grants an approval bound to the FROZEN sha.
    rec = _grant(stores.approvals, protocol_sha=sha)
    # Phase 2: resume with the trusted store; the gate re-reads and verifies.
    c2 = _fixture_coordinator(
        tmp_path, approval_store=stores.approvals,
        approval_id=rec.approval_id, production_mode=True,
        **_production_kwargs(tmp_path))
    result = c2.run()
    assert result["ok"] and result["status"] == "COMPLETED", result
    assert c2.state["approval"]["trust"] == "EXTERNAL_STORE"
    assert c2.state["approval"]["approved_by"] == "Dr. Hu"


def test_production_blocked_when_approval_revoked_before_resume(tmp_path):
    stores = _stores(tmp_path)
    rec = _grant(stores.approvals)
    c = _fixture_coordinator(
        tmp_path, approval_store=stores.approvals,
        approval_id=rec.approval_id, production_mode=True,
        **_production_kwargs(tmp_path))
    stores.approvals.revoke(rec.approval_id, decided_by="Dr. Hu")
    result = c.run()
    assert result["status"] == "BLOCKED"
    assert "revoked" in result["reason"]


def test_test_fixture_without_explicit_stamp_rejected(tmp_path):
    c = _fixture_coordinator(
        tmp_path,
        approval=lambda sha: {"protocol_sha": sha, "approver": "engineer-A",
                              "decision": "APPROVED"})
    result = c.run()
    assert result["status"] == "BLOCKED"
    assert "fixture" in result["reason"]


def test_fixture_cannot_be_used_in_production(tmp_path):
    c = _fixture_coordinator(
        tmp_path,
        approval=lambda sha: {"protocol_sha": sha, "approver": "engineer-A",
                              "decision": "APPROVED"},
        fixture_approval=True, production_mode=True,
        **_production_kwargs(tmp_path))
    result = c.run()
    assert result["status"] == "BLOCKED"


def test_auto_approval_bypass_count_is_zero(tmp_path):
    """No production E2E entry may fabricate approver/decision in-process."""
    from pathlib import Path
    driver = (Path(__file__).resolve().parents[1] / "run_m7_e2e.py").read_text(
        encoding="utf-8")
    assert 'approver": "human-operator"' not in driver
    assert '"decision": "APPROVED"' not in driver
    assert "def approve(" not in driver


# --- out-of-band CLI -------------------------------------------------------

def test_approval_cli_grant_success(tmp_path):
    stores = _stores(tmp_path)
    rec = _grant(stores.approvals, status="pending")
    out = subprocess.run(
        [sys.executable, "-m", "research_runtime.research_store.approval_cli",
         "--db", str(tmp_path / "trusted.sqlite"), "grant", rec.approval_id,
         "--approved-by", "Dr. CLI", "--yes"],
        capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert stores.approvals.get(rec.approval_id).status == "approved"
    assert _gate(stores.approvals, rec.approval_id)["approved_by"] == "Dr. CLI"


def test_approval_cli_failure_unknown_id(tmp_path):
    out = subprocess.run(
        [sys.executable, "-m", "research_runtime.research_store.approval_cli",
         "--db", str(tmp_path / "trusted.sqlite"), "grant", "approval_missing",
         "--approved-by", "Dr. CLI", "--yes"],
        capture_output=True, text=True, timeout=60)
    assert out.returncode == 1
    assert "unknown approval" in out.stderr
