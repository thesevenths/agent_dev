"""ARR-07 engine-free tests (no LLM network, no full engine runs)."""
from __future__ import annotations

import pytest

from research_runtime.arr07 import benchmark, core, engine_gate, loop
from research_runtime.arr07.core import IntegrityError, assert_no_leak


def test_blind_observation_no_config_no_cause():
    t = benchmark.build_tasks("heldout")[0]
    v = t.blind_observation({"one_step_mse": 0.5, "train_final_loss": 0.4})
    assert_no_leak(v)
    assert "configuration" not in v
    assert t.cause not in str(v)
    assert v["measurement"]["one_step_mse"] == 0.5


def test_leaky_observation_has_config():
    t = benchmark.build_tasks("heldout")[0]
    v = t.leaky_observation({"one_step_mse": 0.5})
    assert "configuration" in v and "model" in v["configuration"]


def test_frozen_evaluator_independent_of_llm():
    # With correct mechanism: met.
    met, note, ok = loop.frozen_evaluate(0.4, target=0.3,
        experiment_id="EXP_COV", true_cause="insufficient_state_action_coverage")
    assert met is True and ok is True and "0.400" in note
    # Spurious convergence: high improvement but wrong mechanism -> NOT met.
    met2, note2, ok2 = loop.frozen_evaluate(0.4, target=0.3,
        experiment_id="EXP_COV", true_cause="insufficient_model_capacity")
    assert met2 is False and ok2 is False and "spurious" in note2
    assert loop.frozen_evaluate(0.1, target=0.3)[0] is False
    assert loop.frozen_evaluate(None, target=0.3)[0] is False


def test_repaired_gate_abstains_not_placebo():
    class Item:
        def __init__(self, mechanism, status="VERIFIED"):
            self.applicability_conditions = {"mechanism": mechanism,
                                              "action_conditioned": True}
            self.verification_status = status
            self.supporting_evidence = ["e"]
            self.confidence = 0.8
    t = benchmark.build_tasks("heldout")[0]
    m = engine_gate.repaired_applicability_gate([Item(t.cause)], t)
    assert m["should_transfer"] is True and len(m["selected"]) == 1
    w = engine_gate.repaired_applicability_gate([Item("other")], t)
    assert w["should_transfer"] is False and w["selected"] == []
    assert "no VERIFIED memory" in w["abstain_reason"]


def test_all_experiments_registered():
    for c in benchmark.CAUSES:
        assert any(e.discriminates == (c,) for e in benchmark.EXPERIMENTS)


def test_leak_detection():
    with pytest.raises(IntegrityError):
        assert_no_leak({"configuration": {"hidden": [4]}})
    with pytest.raises(IntegrityError):
        assert_no_leak({"cause": "capacity"})
