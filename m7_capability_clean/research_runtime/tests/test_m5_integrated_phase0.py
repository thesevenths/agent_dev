"""M5-INTEGRATED Phase 0 -- snapshot merge + structured holdout gate (ENGINE-FREE)."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from research_runtime.cross_domain_scheduler import CrossDomainScheduler
from research_runtime.domain_adapter_registry import build_default_adapters
from research_runtime.domain_budget_manager import DomainBudgetManager
from research_runtime.domain_run_record import DomainRunRecordStore
from research_runtime.research_store import (
    ResearchStores, from_research_objective,
)
from research_runtime.rrsi.holdout_evidence import (
    gate_allows_promotion_verified, verify_holdout_evidence,
)
from research_runtime.rrsi import (
    RRSICoordinator, EvolutionRegistry, DOMAIN_MODEL_LAYER,
)
from research_runtime.research_store.db import ResearchDatabase
from research_runtime.schemas import ResearchObjective

SHA = "a" * 64


def _svc(tmp_path, domain, oid):
    stores = ResearchStores(tmp_path / f"{domain}.sqlite")
    obj = ResearchObjective(objective_id=oid, domain=domain, description=oid)
    stores.objectives.save(from_research_objective(obj, status="proposed"))
    return SimpleNamespace(stores=stores)


def _sched(tmp_path, services):
    db = ResearchStores(tmp_path / "sched.sqlite").db
    return CrossDomainScheduler(
        run_id="M5I", registry=build_default_adapters(service_holders={
            n: {"service": s} for n, s in services.items()}),
        budgets=DomainBudgetManager(db=db),
        records=DomainRunRecordStore(db=db),
        services=services, approver=lambda *a: None,
        out_dir=tmp_path / "out", max_campaigns_per_domain=1)


def test_snapshot_none_vs_empty_list_preserves_domain_context(tmp_path):
    services = {
        "scinet": _svc(tmp_path, "scinet", "obj_s"),
        "quant": _svc(tmp_path, "quant", "obj_q"),
    }
    sched = _sched(tmp_path, services)
    snap_all = sched.build_cross_domain_snapshot("M5I")
    assert {o["objective_id"] for o in snap_all.objectives} == {"obj_s", "obj_q"}
    assert snap_all.provenance["source_domains"] == ["quant", "scinet"]
    for d in ("quant", "scinet"):
        assert d in snap_all.provenance["artifact_shas"]
    # Empty list -> explicit gap, but no silent global context loss for None case.
    snap_empty = sched.build_cross_domain_snapshot("M5I", objective_ids=[])
    gaps = [g for g in snap_empty.unresolved_gaps
            if g.get("kind") == "no_objectives_in_scope"]
    assert {g["source_domain"] for g in gaps} == {"quant", "scinet"}


def test_snapshot_partial_domain_and_source_isolation(tmp_path):
    services = {
        "scinet": _svc(tmp_path, "scinet", "obj_s"),
        "quant": _svc(tmp_path, "quant", "obj_q"),
    }
    sched = _sched(tmp_path, services)
    snap = sched.build_cross_domain_snapshot("M5I", objective_ids=["obj_s"])
    domains = {o["source_domain"] for o in snap.objectives}
    assert "scinet" in domains and "quant" not in domains
    assert all(o["source_run_id"] == "M5I" for o in snap.objectives)
    assert "scinet" in snap.evidence_view and "quant" in snap.evidence_view
    assert "scinet" in snap.failures["by_domain"]
    # Same objective id in two domains stays isolated per source.
    services["quant"].stores.objectives.save(from_research_objective(
        ResearchObjective(objective_id="obj_s", domain="quant",
                          description="collision"), status="proposed"))
    snap2 = sched.build_cross_domain_snapshot("M5I")
    pairs = {(o["objective_id"], o["source_domain"]) for o in snap2.objectives}
    assert ("obj_s", "quant") in pairs and ("obj_s", "scinet") in pairs


def _valid_evidence(candidate_sha, parent_sha=""):
    return {
        "dataset_manifest_sha": SHA, "split_id": "holdout_seed_60",
        "candidate_sha": candidate_sha, "parent_sha": parent_sha,
        "evaluation_task_ids": ["task_h1", "task_h2"],
        "selection_task_ids": ["task_s1", "task_s2"],
        "result_shas": [SHA, "b" * 64],
        "metrics": {"mean_improvement": 0.12},
        "statistics": {"n_holdout_seeds": 5, "holdout_direction_fraction": 0.8},
        "independent": True,
    }


def test_structured_holdout_verification_accepts_rejects_and_is_fail_closed():
    ev = _valid_evidence(SHA)
    assert verify_holdout_evidence(ev, expected_candidate_sha=SHA)["verified"]
    # Mismatched candidate sha
    bad = dict(ev, candidate_sha="c" * 64)
    v = verify_holdout_evidence(bad, expected_candidate_sha=SHA)
    assert not v["verified"] and "candidate_sha_mismatch" in v["reasons"]
    # Overlapping holdout/selection tasks -> not independent
    overlap = dict(ev, selection_task_ids=["task_s1", "task_h1"])
    v = verify_holdout_evidence(overlap, expected_candidate_sha=SHA)
    assert not v["verified"] and "holdout_overlaps_selection" in v["reasons"]
    # A TRAINING split cannot stand in for an independent holdout.
    train_split = dict(ev, split_id="train")
    v = verify_holdout_evidence(train_split, expected_candidate_sha=SHA)
    assert not v["verified"] and "split_is_training_not_holdout" in v["reasons"]
    # Bare boolean cannot promote under strict gate
    ok, reasons = gate_allows_promotion_verified(
        "ACCEPTED", holdout_confirmed=True, holdout_evidence=None,
        expected_candidate_sha=SHA)
    assert not ok and "evidence_not_an_object" in reasons
    ok, _ = gate_allows_promotion_verified(
        "ACCEPTED", holdout_confirmed=True, holdout_evidence=ev,
        expected_candidate_sha=SHA)
    assert ok


def test_coordinator_strict_mode_denies_boolean_only_and_records_rejection(tmp_path):
    coord = RRSICoordinator(
        EvolutionRegistry(db=ResearchDatabase(tmp_path / "rrsi.sqlite")))
    cand = {"candidate_id": "dom_strict", "target": "train.lr", "new": 0.001}
    import research_runtime.rrsi.coordinator as coord_mod
    sha = coord_mod.RRSICoordinator._candidate_sha(cand)
    out = coord.run(
        experiment_id="STRICT1", dominant_failure="dynamics_error_accumulation",
        evidence={"changed_layers": [DOMAIN_MODEL_LAYER]},
        generator=lambda **kw: [cand],
        evaluator=lambda **kw: {"decision": "ACCEPTED",
                                "holdout_confirmed": True},
        layer=DOMAIN_MODEL_LAYER, require_structured_holdout=True)
    assert out["outcomes"][0]["status"] != "PROMOTED"
    assert out["outcomes"][0]["gate_passed"] is False
    rec = coord.registry.all(DOMAIN_MODEL_LAYER)[0]
    assert rec["evidence"]["promotion_gate"]["structured_required"] is True
    assert rec["candidate_sha"] == sha  # failed candidate still retained with SHA
