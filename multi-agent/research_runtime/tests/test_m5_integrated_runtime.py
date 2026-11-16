"""M5-INTEGRATED -- top-level runtime state/coordinator contracts (ENGINE-FREE)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from research_runtime.autonomous_runtime import (
    AutonomousRuntime, RuntimeConfig, RuntimeApprovalRequired,
    RuntimeRecoveryRequired,
)
from research_runtime.research_store.db import (
    ResearchDatabase, ResearchStoreConflict,
)
from research_runtime.runtime_coordinator import RuntimeCoordinator
from research_runtime.runtime_state import (
    RuntimeState, config_sha, S_COMPLETED, S_EXHAUSTED, C_COMPLETE, C_CORRUPT,
)
from research_runtime.runtime_report import RuntimeReport


def _state(tmp_path):
    return RuntimeState(db=ResearchDatabase(tmp_path / "runtime.sqlite"))


def _fake_scheduler(fail_domain=None):
    class S:
        services = {
            "scinet": SimpleNamespace(stores=SimpleNamespace(
                objectives=SimpleNamespace(all=lambda: []))),
            "embodied": SimpleNamespace(stores=SimpleNamespace(
                objectives=SimpleNamespace(all=lambda: []))),
        }

        def run_objective(self, oid, source_experiment_id=""):
            domain = "scinet" if "sci" in oid else "embodied"
            if domain == fail_domain:
                raise RuntimeError("injected domain failure")
            return {"campaign_id": f"camp_{oid}", "report_id": f"rep_{oid}",
                    "director_state": "COMPLETED", "domain_id": domain,
                    "budget": {}, "result": {}}

        def build_cross_domain_snapshot(self, run_id, objective_ids=None):
            return SimpleNamespace(
                objectives=[{"objective_id": o, "source_domain": "scinet"}
                            for o in (objective_ids or [])],
                sessions=[], experience_available=True,
                provenance={"source_domains": ["scinet"], "artifact_shas": {}},
                unresolved_gaps=[])

    return S()


def _fake_discovery():
    class D:
        def discover(self, snapshot, objective_store=None, domain=""):
            return SimpleNamespace(proposed_objective_ids=["obj_next"])
    return D()


def _coordinator(tmp_path, *, approver=lambda a, p: None, fail_domain=None,
                 budget=12, structured=True):
    cfg = RuntimeConfig(run_id="R1", max_rounds=2, global_budget=budget,
                        require_structured_holdout=structured,
                        domain_order=("scinet", "embodied"))
    return RuntimeCoordinator(
        config=cfg, state=_state(tmp_path),
        scheduler=_fake_scheduler(fail_domain),
        discovery=_fake_discovery(), rrsi=None,
        dataset_shas={"scinet": "a" * 64}, approver=approver,
        out_dir=tmp_path / "out")


def test_runtime_init_and_config_sha_binding(tmp_path):
    c = _coordinator(tmp_path)
    c.start()
    r = c.state.run("R1")
    assert r["config_sha"] == config_sha(c.config)
    assert r["state"] in ("INIT",)
    # config drift refused
    c.config.global_budget = 99
    with pytest.raises(Exception):
        c.start()


def test_journal_hash_chain_and_tamper_detection(tmp_path):
    c = _coordinator(tmp_path)
    c.start()
    c.state.append("R1", 1, "e1", {"a": 1})
    c.state.append("R1", 1, "e2", {"a": 2})
    assert c.state.verify_journal("R1")["ok"]
    with c.state.db.session() as conn:
        conn.execute("UPDATE rt_journal SET payload_json='{}' WHERE seq=1")
    assert not c.state.verify_journal("R1")["ok"]
    assert c.state.classify("R1")["classification"] == C_CORRUPT


def test_global_budget_reservation_idempotent_and_exhaustion(tmp_path):
    c = _coordinator(tmp_path, budget=2)
    c.start()
    c.state.reserve_global("R1", "camp_1", config_sha=config_sha(c.config))
    c.state.reserve_global("R1", "camp_1", config_sha=config_sha(c.config))
    snap = c.state.global_budget_snapshot("R1")
    assert snap["reserved"] == 1 and snap["remaining"] == 1
    c.state.consume_global("R1", "camp_1", outcome="COMPLETED")
    c.state.consume_global("R1", "camp_1", outcome="COMPLETED")
    snap = c.state.global_budget_snapshot("R1")
    assert snap["consumed"] == 1 and snap["reserved"] == 0
    # config drift on reserve refused
    with pytest.raises(Exception):
        c.state.reserve_global("R1", "camp_2", config_sha="0" * 64)


def test_checkpoint_immutable(tmp_path):
    c = _coordinator(tmp_path)
    c.start()
    cp = c.state.checkpoint("R1", "cp1")
    cp2 = c.state.checkpoint("R1", "cp1")
    assert cp["state_sha"] == cp2["state_sha"]
    c.state.append("R1", 1, "later_event", {"x": 1})
    with pytest.raises(ResearchStoreConflict):
        c.state.checkpoint("R1", "cp1")  # state changed -> refuse overwrite


def test_approval_boundary_and_domain_isolation(tmp_path):
    # No approver -> real run must fail closed before any campaign.
    c0 = _coordinator(tmp_path, approver=None)
    c0.start()
    with pytest.raises(RuntimeApprovalRequired):
        c0._approve("apr", "plan")
    # One domain fails; other domain still completes.
    c = _coordinator(tmp_path / "iso", fail_domain="scinet")
    c.start()
    out = c.run_round(1, objective_specs={
        "scinet": {"objective_id": "obj_sci_x"},
        "embodied": {"objective_id": "obj_emb_x"}})
    assert any(r["domain"] == "scinet" and not r["ok"] for r in out["results"])
    assert any(r["domain"] == "embodied" and r["ok"] for r in out["results"])


def test_cross_domain_discovery_and_finalize(tmp_path):
    c = _coordinator(tmp_path)
    c.start()
    c.run_round(1, objective_specs={
        "scinet": {"objective_id": "obj_sci_1"},
        "embodied": {"objective_id": "obj_emb_1"}})
    cd = c.cross_domain_discovery(
        round_no=2, objective_ids=["obj_sci_1", "obj_emb_1"])
    assert cd["summary"]["proposed_objective_ids"] == ["obj_next"]
    assert cd["summary"]["experience_available"] is True
    fin = c.finalize(round_no=2, cross_summary=cd["summary"])
    assert fin["state"] == S_COMPLETED
    assert fin["journal"]["ok"]
    # Evidence package written
    pkg = RuntimeReport().write(coordinator=c, out_dir=tmp_path / "evidence")
    assert pkg["package_sha256"] and pkg["journal"]["ok"]


def test_human_stop_exhaustion_and_recovery_classification(tmp_path):
    c = _coordinator(tmp_path, budget=1)
    c.start()
    c.request_stop("operator halt")
    assert c.state.run("R1")["state"] == "STOPPED"
    # Recovery of an incomplete run is fail-closed
    c2 = _coordinator(tmp_path / "rec")
    c2.start()
    c2.state.put_round("R1", 1, "scinet", "DISPATCHED", objective_id="obj_s")
    with pytest.raises(RuntimeRecoveryRequired):
        c2.recover_or_raise()
