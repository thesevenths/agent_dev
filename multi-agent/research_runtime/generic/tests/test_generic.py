"""Tests for domain-agnostic identifiability + stopping modules."""
from __future__ import annotations

import pytest
from pathlib import Path

from research_runtime.generic.identifiability import (
    RegisteredExperiment, audit, apply_override, effective_config_hash,
    mechanism_label_is_falsifiable)
from research_runtime.generic.stopping import GoalCriterion, GoalGate


def test_apply_override_is_deep_and_non_mutating():
    base = {"train": {"lr": 0.1, "epochs": 5}, "model": {"h": 32}}
    out = apply_override(base, {"train": {"lr": 0.01}})
    assert out["train"] == {"lr": 0.01, "epochs": 5}
    assert base["train"]["lr"] == 0.1
    assert out is not base


def test_detects_collision_and_labels_unfalsifiable():
    ctxs = [("task1", {"train": {"mode": "biased", "scale": 1.0}})]
    exps = [RegisteredExperiment("E1", "m1", {"train": {"mode": "random"}}),
            RegisteredExperiment("E2", "m2", {"train": {"mode": "random",
                                                         "scale": 1.0}})]
    rep = audit(ctxs, exps)
    assert not rep.all_identifiable and len(rep.collisions) == 1
    assert rep.collisions[0].experiment_ids == ["E1", "E2"]
    assert not mechanism_label_is_falsifiable(rep, "E1")


def test_no_collision_when_configs_differ():
    ctxs = [("t", {"train": {"mode": "biased"}})]
    exps = [RegisteredExperiment("E1", "m1", {"train": {"mode": "random"}}),
            RegisteredExperiment("E2", "m2", {"model": {"lr": 0.01}})]
    rep = audit(ctxs, exps)
    assert rep.all_identifiable
    assert mechanism_label_is_falsifiable(rep, "E1")


def test_goal_gate_stops_on_meet_no_residual_chasing():
    gate = GoalGate(GoalCriterion(target_improvement=0.3, max_iterations=4))
    ds = [gate.observe(iteration=i, improvement=v)
          for i, v in enumerate([0.13, 0.91, 0.16, -0.03])]
    assert ds[0].should_stop is False
    assert ds[1].should_stop is True and ds[1].goal_met is True
    assert gate.consecutive == 1


def test_goal_gate_respects_budget_and_mechanism():
    g1 = GoalGate(GoalCriterion(0.5, 3))
    ds = [g1.observe(iteration=i, improvement=0.1) for i in range(3)]
    assert ds[-1].should_stop is True and ds[-1].goal_met is False
    g2 = GoalGate(GoalCriterion(0.3, 3, mechanism_hit_required=True))
    d = g2.observe(iteration=0, improvement=0.9, mechanism_correct=False)
    assert d.goal_met is False and d.should_stop is False


def test_design_discriminative_experiment():
    from research_runtime.generic.experiment_design import (
        Hypothesis, Operator, design_discriminative, check_unfalsifiable)
    h1 = Hypothesis("H1", "mechanism one", {"mse": "down", "loss": "flat"})
    h2 = Hypothesis("H2", "mechanism two", {"mse": "down", "loss": "up"})
    op = Operator("OP", "do something", lambda c, v: c, [0.5])
    d = design_discriminative(h1, h2, [op])
    assert d is not None and d.predicted_separation > 0
    assert "loss" in d.separating_metrics
    audit = check_unfalsifiable([h1, h2], [op])
    assert audit["pairs"][0]["falsifiable"] is True


def test_unfalsifiable_hypotheses_flagged():
    from research_runtime.generic.experiment_design import (
        Hypothesis, Operator, check_unfalsifiable)
    h1 = Hypothesis("H1", "same", {"mse": "down"})
    h2 = Hypothesis("H2", "same", {"mse": "down"})
    op = Operator("OP", "x", lambda c, v: c, [1])
    audit = check_unfalsifiable([h1, h2], [op])
    assert audit["unfalsifiable"] == [["H1", "H2"]]


def test_scinet_backend_config_materialisation():
    from research_runtime.arr07.scinet_backend import materialise_config, ALLOWED
    from research_runtime.adapters.scinet_adapter import SciNetAdapter
    import tempfile
    a = SciNetAdapter(strict=False)
    with tempfile.TemporaryDirectory() as td:
        p = materialise_config(a.project_dir / "configs/pendulum_smoke.yaml",
                               {"train.lr": 0.0005}, Path(td) / "c.yaml")
        cfg = __import__("yaml").safe_load(Path(p).read_text(encoding="utf-8"))
        assert cfg["train"]["lr"] == 0.0005
        try:
            materialise_config(a.project_dir / "configs/pendulum_smoke.yaml",
                               {"bad.key": 1}, Path(td) / "x.yaml")
            assert False, "unsupported knob must raise"
        except ValueError:
            pass


def test_scinet_loop_wiring_uses_generic_modules():
    import inspect
    from research_runtime.arr07 import scinet_loop
    src = inspect.getsource(scinet_loop)
    assert "GoalGate" in src and "design_discriminative" in src
    assert "run_scinet_experiment" in src
