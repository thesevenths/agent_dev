"""ARR-06 engine-free tests (no network, no full engine campaign)."""
from __future__ import annotations

import pytest

from research_runtime.arr06 import (belief, benchmark, core, engine, evidence,
                                    memory, orchestrator, protocol, statistics as st)
from research_runtime.arr06.core import IntegrityError, assert_no_leak


def test_blind_views_no_leak():
    for t in benchmark.build_tasks()[:20]:
        v = t.blind_view()
        assert_no_leak(v)
        assert t.cause not in str(v) and t.task_id not in str(v)


def test_all_causes_discriminative_and_fresh_ids():
    for c in benchmark.CAUSES:
        assert any(e.discriminates == (c,) for e in benchmark.EXPERIMENTS)
    assert benchmark.EXP_BY_ID["EXP_PLACEBO"].discriminates == ()
    for t in benchmark.build_tasks()[:20]:
        assert "arr06::" in t.task_id and t.variant >= 20


def test_determinism_modes_and_repro_contract(tmp_path):
    s1 = engine.set_deterministic_mode("det", 1, True)
    s2 = engine.set_deterministic_mode("nondet", 4, False)
    assert s1["mode"] == "det" and s2["mode"] == "nondet"
    assert s1["torch_num_threads_actual"] == 1
    r = orchestrator.reproducibility_contract(tmp_path)
    assert set(r) >= {"R0_artifact_integrity", "R1_statistical_recalculation",
                      "R2_experiment_replay", "R3_full_campaign_reproduction"}
    assert not r["R0_artifact_integrity"]


def test_belief_update_bayesian_and_ig():
    prior = belief.prior_uniform()
    assert abs(sum(prior.values()) - 1.0) < 1e-9
    up = belief.belief_update(prior, "EXP_CAP", 0.5)
    assert abs(sum(up["posterior"].values()) - 1.0) < 1e-9
    assert up["prediction"] == "insufficient_model_capacity"
    assert up["information_gain"] > 0
    up0 = belief.belief_update(prior, "EXP_CAP", 0.0)
    assert up0["information_gain"] < up["information_gain"]


def test_belief_no_false_certainty_from_placebo():
    prior = belief.prior_uniform()
    up = belief.belief_update(prior, "EXP_PLACEBO", 0.0)
    assert 0 <= up["confidence"] <= 1


def test_evidence_claim_system(tmp_path):
    raw = tmp_path / "raw.json"
    raw.write_text("{}", encoding="utf-8")
    store = evidence.EvidenceStore(tmp_path)
    ev = evidence.Evidence("e1", "exp1", "claim", "mse", 0.1, str(raw),
                           core.sha256_file(raw), "proto")
    store.add_evidence(ev)
    c = evidence.Claim("c1", "LLM claim", "LLM", "h1", supporting_evidence=["e1"])
    with pytest.raises(RuntimeError):
        store.register_claim(evidence.Claim("c2", "x", "LLM", "h1", status="VERIFIED"))
    store.register_claim(c)
    out = store.verify_claim("c1")
    assert out["status"] in ("VERIFIED", "INCONCLUSIVE")


def test_memory_status_and_applicability_gate():
    items = memory.build_memory_items()
    assert all(i.provenance_hash for i in items)
    assert {i.verification_status for i in items} <= {
        "PROVISIONAL", "VERIFIED", "CONTRADICTED", "REJECTED", "INCONCLUSIVE"}
    task = benchmark.build_tasks("heldout")[0]
    g = memory.applicability_gate(items, task)
    assert "abstain" in g and "transfer_risk" in g and "selected" in g


def test_statistics_and_holm():
    s = st.paired_stats([0.1, 0.2, 0.05, 0.15], n_boot=50, n_perm=50)
    assert s["n"] == 4 and 0 <= s["p"] <= 1
    assert len(st.holm([0.01, 0.04])) == 2
    cal = st.calibration([{"confidence": 0.9, "correct": True},
                          {"confidence": 0.7, "correct": False}])
    assert 0 <= cal["ece"] <= 1 and "ace" in cal


def test_budget_controller():
    b = orchestrator.BudgetController(max_formal=2)
    b.consume("formal"); b.consume("formal")
    with pytest.raises(RuntimeError):
        b.consume("formal")


def test_checkpoint_resume(tmp_path):
    s = orchestrator.CampaignState(tmp_path)
    s.checkpoint("x", {"a": 1})
    s.run_phase("p", lambda: {"v": 1})
    called = []
    s.run_phase("p", lambda: called.append(1) or {"v": 1})
    assert not called


def test_seal_leak(tmp_path):
    seal = core.SealedStore(tmp_path)
    info = seal.seal("p", {"a": 1})
    assert seal.load("p", info["sha256"]) == {"a": 1}
    with pytest.raises(IntegrityError):
        seal.load("p", "0" * 64)
    with pytest.raises(IntegrityError):
        assert_no_leak({"ground_truth": 1})
