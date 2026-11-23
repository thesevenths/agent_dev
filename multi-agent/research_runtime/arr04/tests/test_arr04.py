"""ARR-04 engine-free unit tests (no simulator, no LLM network, no engine)."""
from __future__ import annotations

import pytest

from research_runtime.arr04 import (audit, benchmark, core, knowledge,
                                    orchestrator, preflight, statistics as st,
                                    tracks)
from research_runtime.arr04.core import IntegrityError, assert_no_leak, sha256_json


def test_public_task_views_no_leak():
    for t in benchmark.build_tasks():
        v = t.blind_view()
        assert_no_leak(v)
        assert "cause" not in v and "truth" not in v
        assert t.task_id not in str(v)


def test_all_causes_have_discriminative_experiment():
    for cause in benchmark.CAUSES:
        exps = [e for e in benchmark.DIAGNOSTIC_EXPERIMENTS
                if e.discriminates == (cause,)]
        assert exps, cause
    placebo = benchmark.EXP_BY_ID["EXP_PLACEBO"]
    assert placebo.discriminates == ()


def test_task_signatures_distinct_per_cause():
    sigs = {}
    for t in benchmark.tasks_for_split("train"):
        sigs[t.cause] = t.expected_signature
    assert len(sigs) == 5
    # optimization vs capacity must differ on train_loss / lr_scale
    assert sigs["optimization_instability"]["lr_scale"] != \
        sigs["insufficient_model_capacity"]["lr_scale"]


def test_iv_to_config_and_matched_pool():
    task = benchmark.tasks_for_split("heldout")[0]
    tr, mo = tracks.iv_to_config(task, {"hidden": [64, 64]})
    assert mo["hidden"] == [64, 64]
    tr2, mo2 = tracks.iv_to_config(task, {"__placebo__": True})
    assert tr2 == task.failure_train and mo2 == task.failure_model
    pool = knowledge.INTERVENTION_POOL
    assert len(pool) >= 4 and all(isinstance(x, dict) for x in pool)


def test_resolve_intervention_real():
    for c in benchmark.CAUSES:
        iv = knowledge.resolve_intervention(c)
        assert iv and any(k in ("hidden", "lr", "n_episodes", "action_mode",
                                "action_scale") for k in iv)


def test_diagnosis_parse_helpers():
    from research_runtime.arr04.diagnosis_agent import DiagnosisAgent
    assert DiagnosisAgent._cause("optimization_instability") == "optimization_instability"
    assert DiagnosisAgent._cause("bogus") == "UNKNOWN"
    assert DiagnosisAgent._cause(None) == "UNKNOWN"


def test_statistics_paired_holm_wilson_calibration():
    s = st.paired_stats([0.1, 0.2, 0.05, 0.15, 0.12], n_boot=100, n_perm=100)
    assert s["n"] == 5 and 0 <= s["p"] <= 1
    assert s["bootstrap_ci95"][0] <= s["bootstrap_ci95"][1]
    h = st.holm([0.01, 0.04, 0.03])
    assert len(h) == 3 and all(0 <= x["holm_p"] <= 1 for x in h)
    lo, hi = st.wilson(8, 10)
    assert 0 <= lo <= 0.8 <= hi <= 1
    cal = st.calibration([{"confidence": 0.9, "correct": True},
                          {"confidence": 0.8, "correct": False}])
    assert 0 <= cal["ece"] <= 1 and 0 <= cal["brier"] <= 1


def test_verdicts_logic():
    proto = dict(preflight.PROTOCOL_FROZEN)
    m1 = {s: {"accuracy": 0.9 if s == "active_arr" else 0.5,
              "ece": 0.05 if s == "active_arr" else 0.3,
              "brier": 0.1, "information_gain": 0.4}
          for s in ("passive_arr", "active_arr", "rule_experiment")}
    m2 = {"arr_vs_random": {"significant": True, "mean": 0.1},
          "arr_vs_rule": {"mean": 0.05}}
    m3 = {"transfer_vs_fresh": {"significant": True, "mean": 0.1},
          "negative_transfer_rate": 0.0}
    rec = {"improvement_significant": True, "regression": False}
    v = st.verdicts(engineering={"n_runs": 10, "ok": True},
                    integrity={"gt": True}, m1=m1, m2=m2, m3=m3,
                    recursive=rec, audit={"a": True}, protocol=proto)
    assert v["m1_active_diagnosis"] == "PASS"
    assert v["engineering"] == "PASS"


def test_checkpoint_resume(tmp_path):
    state = orchestrator.CampaignState(tmp_path)
    state.start_phase("phase0_audit")
    h = state.checkpoint("t", {"x": 1})
    assert state.load_checkpoint("t") == {"x": 1}
    state.complete_phase("phase0_audit", {"ok": True})
    assert state.phase_done("phase0_audit")
    # Idempotent: rerunning a completed phase is skipped.
    called = []
    state.run_phase("phase0_audit", lambda: called.append(1) or {"ok": True})
    assert not called


def test_budget_controller():
    b = orchestrator.BudgetController(max_formal_runs=2, max_preflight_runs=1)
    b.consume(kind="formal"); b.consume(kind="formal")
    with pytest.raises(RuntimeError):
        b.consume(kind="formal")
    assert b.stopping_condition()["formal_exhausted"]


def test_seal_and_leak(tmp_path):
    seal = core.SealedStore(tmp_path)
    info = seal.seal("proto", {"a": 1})
    assert seal.load("proto", info["sha256"]) == {"a": 1}
    with pytest.raises(IntegrityError):
        seal.load("proto", "0" * 64)
    with pytest.raises(IntegrityError):
        assert_no_leak({"ground_truth": 1})


def test_integrity_audit_empty(tmp_path):
    (tmp_path / "raw_data" / "immutable").mkdir(parents=True)
    r = audit.integrity_audit(tmp_path)
    assert r["n_records"] == 0 and not r["pass"]


def test_verified_memory_only_ci_positive():
    m2 = {"oracle_vs_base": {"n": 5, "mean": 0.5, "ci95": [0.2, 0.8]}}
    m1 = {"active_arr": {"accuracy": 0.8}}
    items = knowledge.build_verified_memory(m2, m1)
    assert len(items) == 5 and all(x.provenance_hash for x in items)
    m2_bad = {"oracle_vs_base": {"n": 5, "mean": 0.5, "ci95": [-0.1, 0.8]}}
    assert knowledge.build_verified_memory(m2_bad, m1) == []
