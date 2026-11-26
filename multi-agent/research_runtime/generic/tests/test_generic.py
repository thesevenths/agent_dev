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


def test_llm_autonomy_validates_proposals_and_finish():
    from research_runtime.generic.llm_autonomy import (
        LLMResearcher, validate_proposal)
    allowed = {"train.lr": (0.0, 1.0), "model.latent_dim": None}
    ok, why, clean = validate_proposal({"train.lr": 0.01}, allowed)
    assert ok and clean == {"train.lr": 0.01}
    bad, why, _ = validate_proposal({"train.lr": 5.0}, allowed)
    assert not bad and "outside" in why
    bad2, why2, _ = validate_proposal({"fake.key": 1}, allowed)
    assert not bad2 and "not in allowed" in why2
    calls = []
    def gen(prompt):
        calls.append(prompt)
        return '{"finish": true, "reasoning": "mse 0.01 < 0.026", "quoted_metrics": {"latent_mse@1": 0.01}}'
    r = LLMResearcher(gen, Path("experiments/campaigns/ARR-07-AUTONOMY-TEST"))
    d = r.decide_finish(objective="minimize mse", target=0.1,
                        baseline={"latent_mse@1": 0.026},
                        latest={"latent_mse@1": 0.01}, improvement=0.6,
                        iteration=0, max_iterations=3)
    assert d.finish is True and d.quoted_metrics["latent_mse@1"] == 0.01


def test_budget_does_not_rewrite_llm_finish_verdict():
    from research_runtime.generic.llm_autonomy import LLMResearcher
    calls = []
    def gen(prompt):
        calls.append(prompt)
        return '{"finish": false, "reasoning": "target not met", "quoted_metrics": {}}'
    r = LLMResearcher(gen, Path("experiments/campaigns/ARR-07-AUTONOMY-TEST2"))
    d = r.decide_finish(objective="x", target=0.1, baseline={}, latest={},
                        improvement=0.05, iteration=1, max_iterations=2)
    assert d.finish is False, "LLM said continue; budget exhaustion must NOT rewrite it"


def test_frozen_contract_multi_objective():
    from research_runtime.generic.research_contract import (
        MetricGoal, ResearchContract)
    c = ResearchContract(goals=(MetricGoal("mse", "min", 0.10),
                                MetricGoal("cos", "max", 0.0)), budget=3)
    good = c.evaluate({"mse": 0.02, "cos": 0.6}, {"mse": 0.015, "cos": 0.7})
    assert good["contract_met"] is True
    bad = c.evaluate({"mse": 0.02, "cos": 0.6}, {"mse": 0.015, "cos": 0.5})
    assert bad["contract_met"] is False


def test_sanity_check_rejects_oversized_structure():
    from research_runtime.generic.research_contract import sanity_check
    ok = sanity_check({"model.latent_dim": 8, "model.predictor_hidden": [16]},
                      n_train_samples=64)
    bad = sanity_check({"model.latent_dim": 32, "model.predictor_hidden": [256, 256]},
                       n_train_samples=64)
    assert ok["ok"] is True and bad["ok"] is False


def test_search_memory_dedup_and_hint():
    from research_runtime.generic.research_contract import SearchMemory
    m = SearchMemory()
    m.record(iteration=0, overrides={"train.lr": 0.01},
             metrics={"latent_mse@1": 0.02}, verdict={"contract_met": True},
             engine_ok=True)
    assert m.seen({"train.lr": 0.01}) and not m.seen({"train.lr": 0.02})
    hint = m.direction_hint(metric="latent_mse@1")
    assert hint["n"] == 1


def test_paired_seed_ci():
    from research_runtime.generic.research_contract import paired_seed_verdict
    r = paired_seed_verdict([(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)],
                            [(0.8, 0.0), (0.82, 0.0), (0.81, 0.0)])
    assert r["n_seeds"] == 3 and r["significantly_positive"] is True


def test_hidden_failure_detector():
    from research_runtime.arr07.scinet_backend import _assert_no_hidden_failure
    class R:
        error = "engine CLI error: Traceback (most recent call last): TypeError: bad"
    import pytest as _p
    with _p.raises(RuntimeError):
        _assert_no_hidden_failure(R(), "train")


def test_contract_externalised_and_frozen():
    import tempfile
    from research_runtime.generic.research_contract import (
        ResearchContract, MetricGoal, freeze_contract, load_contract,
        load_dataset_scale)
    with tempfile.TemporaryDirectory() as td:
        c1 = ResearchContract(goals=(MetricGoal("mse", "min", 0.1),
                                     MetricGoal("cos", "max", 0.2)), budget=5)
        freeze_contract(td, c1)
        loaded = load_contract(Path(td) / "sub")
        # load_contract without local file falls back to default; load from same dir:
        loaded2 = load_contract(td)
        assert loaded2.goals[1].min_relative_gain == 0.2 and loaded2.budget == 5
        # Different campaign can have DIFFERENT thresholds (generalisation).
        c2 = ResearchContract(goals=(MetricGoal("other_metric", "max", 0.9),), budget=2)
        with tempfile.TemporaryDirectory() as td2:
            freeze_contract(td2, c2)
            assert load_contract(td2).goals[0].metric == "other_metric"
        # Freeze: refuse to change mid-campaign.
        try:
            freeze_contract(td, c2)
            assert False, "must refuse to overwrite frozen contract"
        except ValueError:
            pass


def test_scinet_config_seed_propagation():
    """SciNet train.py reads cfg['seed'] from YAML; backend must write it."""
    from research_runtime.arr07.scinet_backend import materialise_config
    from research_runtime.adapters.scinet_adapter import SciNetAdapter
    import yaml, tempfile
    a = SciNetAdapter(strict=False)
    with tempfile.TemporaryDirectory() as td:
        p = materialise_config(a.project_dir / "configs/pendulum_smoke.yaml",
                               {"train.lr": 0.01}, Path(td) / "c.yaml", seed=43)
        assert yaml.safe_load(Path(p).read_text(encoding="utf-8"))["seed"] == 43


def test_change_budget_limits_llm_proposal():
    from research_runtime.generic.llm_autonomy import LLMResearcher
    calls = []
    def gen(prompt):
        calls.append(prompt)
        return ('{"reasoning": "change many", "overrides": {"train.lr": 0.01, '
                '"jepa.pred_weight": 5.0, "model.latent_dim": 8}}')
    r = LLMResearcher(gen, Path("experiments/campaigns/ARR-07-CHANGEBUDGET-TEST"))
    p = r.propose_experiment(objective="x", allowed={
        "train.lr": (0.0, 1.0), "jepa.pred_weight": (0.0, 100.0),
        "model.latent_dim": (2, 32)}, baseline_metrics={}, history=[],
        max_changes=2)
    assert p.valid is False and "max 2" in p.rejection_reason


def test_screen_rank_selects_joint_improvement():
    """Agent screening must rank lr=0.01 above dim=2 (MSE-only trap)."""
    from research_runtime.generic.research_contract import (
        ResearchContract, MetricGoal)
    c = ResearchContract(goals=(MetricGoal("latent_mse@1", "min", 0.10),
                                MetricGoal("cosine@1", "max", 0.0)), budget=3)
    base = {"latent_mse@1": 0.0264, "cosine@1": 0.5437}
    def rank(r):
        m = {"latent_mse@1": r["mse"], "cosine@1": r["cosine"]}
        v = c.evaluate(base, m)
        return (int(v["contract_met"]),
                sum(g.get("relative_gain", 0) for g in v["per_goal"]))
    lr = {"mse": 0.0127, "cosine": 0.928}
    dim = {"mse": 0.0108, "cosine": -0.042}
    assert rank(lr) > rank(dim)
