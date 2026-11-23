"""ARR-05 engine-free tests (no simulator, no LLM network)."""
from __future__ import annotations

import pytest

from research_runtime.arr05 import (benchmark, core, orchestrator, preflight,
                                    protocol, source_reproducibility as sr,
                                    statistics as st)
from research_runtime.arr05.core import IntegrityError, assert_no_leak, sha256_json


def test_blind_views_no_leak():
    for t in benchmark.build_tasks()[:20]:
        v = t.blind_view()
        assert_no_leak(v)
        assert t.cause not in str(v)
        assert t.task_id not in str(v)


def test_fresh_task_ids_not_arr04():
    for t in benchmark.build_tasks()[:20]:
        assert t.variant >= 5
        assert "arr05::" in t.task_id


def test_discriminative_experiments_and_signatures():
    for c in benchmark.CAUSES:
        assert any(e.discriminates == (c,) for e in benchmark.DIAGNOSTIC_EXPERIMENTS)
    assert benchmark.EXP_BY_ID["EXP_PLACEBO"].discriminates == ()
    sigs = {t.cause: t.expected_signature for t in benchmark.build_tasks("calibration")
            if t.variant == 5}
    assert len(sigs) == 5
    assert sigs["optimization_instability"]["lr_scale"] != \
        sigs["insufficient_model_capacity"]["lr_scale"]


def test_paired_holm_wilson_calibration():
    s = st.paired_stats([0.1, 0.2, 0.05, 0.15, 0.12], n_boot=100, n_perm=100)
    assert s["n"] == 5 and 0 <= s["p"] <= 1
    assert len(st.holm([0.01, 0.04, 0.03])) == 3
    lo, hi = st.wilson(8, 10)
    assert 0 <= lo <= 0.8 <= hi <= 1
    cal = st.calibration([{"confidence": 0.9, "correct": True},
                          {"confidence": 0.8, "correct": False},
                          {"confidence": 0.7, "correct": False}])
    assert 0 <= cal["ece"] <= 1 and 0 <= cal["brier"] <= 1
    assert "log_loss" in cal and "calibration_slope" in cal and "ace" in cal


def test_temperature_scale():
    t = st.temperature_scale([0.9, 0.7, 0.6, 0.8], [1, 0, 0, 1])
    assert 0.2 <= t <= 5.0


def test_verdicts_reproducibility_flag():
    proto = dict(protocol.FROZEN)
    def ps(m, s=True):
        return {"n": 10, "mean": m, "p": 0.01, "significant": s, "ci95": [0.01, 0.1]}
    m1 = {"active": {"calibrated_ece": 0.05, "false_certainty_rate": 0.0,
                     "abstention_precision": 0.9},
          "passive": {"false_certainty_rate": 0.3},
          "active_vs_passive": ps(0.05),
          "realized_information_gain": 0.1}
    m2 = {"arr_vs_random": ps(0.1), "arr_vs_rule": ps(0.05)}
    m3 = {"gated_vs_fresh": ps(0.1), "raw_vs_fresh": ps(0.01, False),
          "negative_transfer_rate": 0.0, "applicability_match": 0.9}
    rq = {"arr_vs_random": ps(0.2), "falsifiability": 0.9, "independent_evaluation": True}
    rsi = {"actual_methodology_changes": 2, "evolved_vs_frozen": ps(0.1),
           "evolved_vs_memory_only": ps(0.1), "ood_regression": False}
    v = st.verdicts(engineering={"n_runs": 10, "ok": True},
                    integrity={"gt": True}, m1=m1, m2=m2, m3=m3, rq=rq, rsi=rsi,
                    audit={"a": True}, reproduction={"reproduction_success": True},
                    protocol=proto)
    assert v["source_reproducibility"] in ("PASS", "PARTIAL")
    assert v["engineering"] == "PASS"


def test_checkpoint_resume_budget(tmp_path):
    state = orchestrator.CampaignState(tmp_path)
    state.checkpoint("x", {"a": 1})
    assert state.data["checkpoints"]["x"]["sha256"]
    b = orchestrator.BudgetController(max_formal=2, max_llm=3)
    b.consume("formal"); b.consume("formal")
    with pytest.raises(RuntimeError):
        b.consume("formal")
    assert b.to_dict()["formal"] == 2


def test_seal_leak(tmp_path):
    seal = core.SealedStore(tmp_path)
    i = seal.seal("p", {"a": 1})
    assert seal.load("p", i["sha256"]) == {"a": 1}
    with pytest.raises(IntegrityError):
        seal.load("p", "0" * 64)
    with pytest.raises(IntegrityError):
        assert_no_leak({"ground_truth": 1})


def test_arr04_source_reproducibility_gap():
    from pathlib import Path
    root = Path("experiments/campaigns/ARR-05-2026-001")
    a04 = Path("experiments/campaigns/ARR-04-2026-001")
    if not a04.is_dir():
        pytest.skip("ARR-04 artifacts absent")
    s = sr.source_audit(root, a04)
    assert s["declared_arr04_source_files"] == 0
    assert s["artifact_arr04_source_files"] > 0
    assert s["source_recovered"] is True
    r = sr.recompute_arr04_primary(root, a04)
    assert r["reproduction_success"] is True and r["abs_mean_delta"] < 1e-6
