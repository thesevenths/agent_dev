"""M3.4 HARDENING / P0-1 -- Embodied planning-horizon regression tests.

Root cause (reproduced before the fix, see M3_4_HARDENING_ACCEPTANCE_REPORT.md):
``VALUE_RULES`` allowed ``model.rollout_horizon`` in 2..8, but the evaluate
stage used that SAME number both as the rollout-window length and as a key into
a fixed ``HORIZONS = (1,2,3,4,5)`` error table. Result:

  * h=2,3,4 -> the window was too short for ``state_mse@5`` -> IndexError;
  * h=6,7,8 -> ``errors[horizon]`` missing -> KeyError;
  * h=5     -> the ONLY working value (and the default), which hid the bug.

These tests call the REAL production ``embodied_compute`` on the frozen
``robot_sim_v1`` dataset (pure stdlib, no FakeRunner, no engine checkout).
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest
import yaml

from research_runtime.domains import embodied as E
from research_runtime.domains.embodied import (
    BASELINE_CONFIG,
    EVALUATION_HORIZONS,
    HORIZONS,
    MAX_PLANNING_HORIZON,
    MIN_PLANNING_HORIZON,
    PROJECT_DIR,
    REQUIRED_METRICS,
    VALUE_RULES,
    embodied_compute,
)

# Exact metrics produced by the DEFAULT config (rollout_horizon=5) BEFORE the
# P0-1 fix, with train seed=50 / validation split / frozen robot_sim_v1.
# Locking these numbers proves "keep the existing default behaviour unchanged".
# Comparison uses a 1e-12 relative tolerance: CPython 3.11 vs 3.12+ differ by
# ONE ULP in the float accumulation order of action_prediction_mse
# (0.2594051985526708 vs 0.25940519855267075, ~5e-17). That is a platform
# detail, NOT a behaviour change. 1e-12 is still ~9 orders of magnitude
# tighter than any REAL behaviour change (e.g. a different window count shifts
# state_mse@1 by ~4e-5); the test below proves that discriminating power.
GOLDEN_DEFAULT = {
    "action_prediction_mse": 0.25940519855267075,
    "physics_consistency": 0.9984451639021401,
    "safety_violations": 0.0,
    "state_mse@1": 0.001557257377844594,
    "state_mse@2": 0.004952532049567222,
    "state_mse@3": 0.010604796315103665,
    "state_mse@4": 0.018194163095929642,
    "state_mse@5": 0.028265953783731465,
    "task_success_rate": 1.0,
}
GOLDEN_N_WINDOWS = 180
# 1 ULP-level tolerance for cross-CPython-version float accumulation only.
GOLDEN_RTOL = 1e-12

ALL_HORIZONS = tuple(range(2, 9))  # 2,3,4,5,6,7,8


def _cfg(horizon: int) -> dict:
    cfg = yaml.safe_load((PROJECT_DIR / BASELINE_CONFIG).read_text(encoding="utf-8"))
    cfg["model"]["rollout_horizon"] = horizon
    return cfg


def _train_and_evaluate(horizon: int, tmp: Path) -> tuple[dict, dict]:
    """Real TRAIN then real EVALUATE reusing the checkpoint by run_id."""
    cfg = _cfg(horizon)
    job = tmp / f"h{horizon}"
    job.mkdir(parents=True, exist_ok=True)
    spec_base = {"project_dir": str(PROJECT_DIR), "seed": 50, "run_id": "hz"}
    train = embodied_compute(cfg, {"stage": "train", **spec_base}, job)
    assert train["status"] == "succeeded", train
    out = embodied_compute(
        cfg, {"stage": "evaluate", "run_id": "hz", "split": "validation",
              "project_dir": str(PROJECT_DIR)}, job)
    return out, json.loads((job / "rollout_metrics.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("horizon", ALL_HORIZONS)
def test_every_planning_horizon_succeeds_without_index_or_key_error(horizon):
    """The seven P0-1 cases: 2/3/4 (was IndexError), 5 (default), 6/7/8
    (was KeyError). Any exception here is a regression -- fail-closed allowed,
    but a crash is not."""
    with tempfile.TemporaryDirectory() as td:
        out, doc = _train_and_evaluate(horizon, Path(td))
    assert out["status"] == "succeeded", out
    assert out["n_rollout_windows"] > 0

    metrics = out["metrics"]
    # Every contract metric is present and finite: no missing keys, no NaN.
    for name in REQUIRED_METRICS:
        assert name in metrics, f"missing {name} at h={horizon}"
        value = metrics[name]
        assert isinstance(value, (int, float)) and math.isfinite(value), \
            f"non-finite {name}={value!r} at h={horizon}"
    assert math.isfinite(metrics["physics_consistency"])
    assert 0.0 <= metrics["task_success_rate"] <= 1.0

    # The evaluation metric set is FIXED and must not leak the planning horizon.
    reported = {k for k in metrics if k.startswith("state_mse@")}
    assert reported == {f"state_mse@{k}" for k in EVALUATION_HORIZONS}
    if horizon > max(EVALUATION_HORIZONS):
        assert f"state_mse@{horizon}" not in metrics, \
            "planning horizon must not leak into the reported metric contract"

    # Provenance written by the engine records BOTH horizon notions explicitly.
    assert doc["planning_horizon"] == horizon
    assert doc["evaluation_horizons"] == list(EVALUATION_HORIZONS)
    assert doc["n_windows"] == out["n_rollout_windows"]
    assert doc["checkpoint_sha256"]


@pytest.mark.parametrize("horizon", ALL_HORIZONS)
def test_no_array_out_of_bounds_for_short_episodes(horizon):
    """An episode shorter than the required window must skip (not crash).

    The frozen dataset has 48-step episodes, so build a tiny synthetic split
    with 2 episodes of 4 steps and monkeypatch ``load_split`` -- this exercises
    the exact slicing path that used to raise IndexError at h=2/3/4.
    """
    import research_runtime.domains.embodied as mod

    tiny = []
    for ep in (0, 1):
        for t in range(4):
            tiny.append({"episode": ep, "t": t, "x": 0.01 * t,
                         "v": 0.001 * t, "action": 0.1 * t})
    original = mod.load_split

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        job = tmp / "j"
        job.mkdir()
        cfg = _cfg(horizon)
        # TRAIN must use the REAL frozen split (it needs >=20 transitions).
        # Only the EVALUATE split is replaced with an episode too short for any
        # rollout window, so we exercise the exact slicing path in isolation.
        train = embodied_compute(
            cfg, {"stage": "train", "project_dir": str(PROJECT_DIR),
                  "seed": 50, "run_id": "short"}, job)
        assert train["status"] == "succeeded"
        mod.load_split = lambda *_a, **_kw: tiny
        try:
            out = embodied_compute(
                cfg, {"stage": "evaluate", "run_id": "short",
                      "split": "validation", "project_dir": str(PROJECT_DIR)}, job)
        finally:
            mod.load_split = original
        # No rollout window can fit -> explicit fail-closed, NEVER an
        # IndexError / KeyError / silent zero-division.
        assert out["status"] == "failed"
        assert out["error_type"] == "missing_artifact"
        assert "no rollout windows" in out["error"]


def test_default_config_behaviour_is_bit_for_bit_unchanged():
    """The shipped baseline (h=5) must produce the exact pre-fix numbers."""
    with tempfile.TemporaryDirectory() as td:
        out, doc = _train_and_evaluate(5, Path(td))
    assert out["n_rollout_windows"] == GOLDEN_N_WINDOWS
    assert set(out["metrics"]) == set(GOLDEN_DEFAULT)
    for name, expected in GOLDEN_DEFAULT.items():
        assert out["metrics"][name] == pytest.approx(
            expected, rel=GOLDEN_RTOL, abs=1e-15), \
            f"default behaviour changed for {name}: {out['metrics'][name]} != {expected}"
    assert doc["planning_horizon"] == 5
    assert doc["evaluation_horizons"] == list(EVALUATION_HORIZONS)


def test_golden_tolerance_still_detects_real_behaviour_change():
    """The 1-ULP tolerance must NOT be able to hide an actual change.

    A longer planning horizon (6) shrinks the window count 180 -> 168 and
    moves state_mse@1 by ~4e-5, i.e. ~8 orders of magnitude above the
    tolerance. If this ever stops being true, the golden test has become
    vacuous and must be tightened.
    """
    with tempfile.TemporaryDirectory() as td:
        _, h5 = _train_and_evaluate(5, Path(td))
    with tempfile.TemporaryDirectory() as td:
        out6, _ = _train_and_evaluate(6, Path(td))
    baseline = h5["metrics"]["state_mse@1"]
    changed = out6["metrics"]["state_mse@1"]
    assert h5["n_windows"] == 180 and out6["n_rollout_windows"] == 168
    assert changed != pytest.approx(baseline, rel=GOLDEN_RTOL, abs=1e-15), \
        "golden tolerance is too loose: it can no longer detect a real change"


def test_evaluation_horizons_are_fixed_and_aliased():
    """The metric contract is immutable; HORIZONS stays as a compatibility alias."""
    assert EVALUATION_HORIZONS == (1, 2, 3, 4, 5)
    assert HORIZONS == EVALUATION_HORIZONS
    assert MIN_PLANNING_HORIZON == 2 and MAX_PLANNING_HORIZON == 8
    # VALUE_RULES must be derived from the SAME constants, so the allowed range
    # and the fail-closed guard in embodied_compute cannot drift apart again.
    rule = VALUE_RULES["model.rollout_horizon"]
    assert rule(float(MIN_PLANNING_HORIZON)) is True
    assert rule(float(MAX_PLANNING_HORIZON)) is True
    assert rule(float(MIN_PLANNING_HORIZON) - 1) is not True
    assert rule(float(MAX_PLANNING_HORIZON) + 1) is not True


@pytest.mark.parametrize("bad", [0, 1, 9, 20, -3])
def test_out_of_contract_horizon_fails_closed_not_crash(bad):
    """Direct callers bypass VALUE_RULES; the compute layer must guard itself."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        job = tmp / "j"
        job.mkdir()
        cfg = _cfg(bad)
        t = embodied_compute(cfg, {"stage": "train",
                                   "project_dir": str(PROJECT_DIR),
                                   "seed": 50, "run_id": "bad"}, job)
        assert t["status"] == "succeeded"
        out = embodied_compute(
            cfg, {"stage": "evaluate", "run_id": "bad", "split": "validation",
                  "project_dir": str(PROJECT_DIR)}, job)
        assert out["status"] == "failed"
        assert out["error_type"] == "invalid_planning_horizon"
        assert "outside" in out["error"]


def test_evaluation_metrics_independent_of_planning_horizon_2_to_5():
    """For h <= max(EVALUATION_HORIZONS) the window length is the same, so the
    fixed evaluation metrics must be identical across those configs -- the
    planning horizon may only move the task-success decision point."""
    docs = {}
    for h in (2, 3, 4, 5):
        with tempfile.TemporaryDirectory() as td:
            out, _ = _train_and_evaluate(h, Path(td))
        docs[h] = out["metrics"]
    for name in ("state_mse@1", "state_mse@2", "state_mse@3",
                 "state_mse@4", "state_mse@5", "action_prediction_mse",
                 "safety_violations"):
        values = [docs[h][name] for h in (2, 3, 4, 5)]
        assert all(v == pytest.approx(values[0], rel=GOLDEN_RTOL, abs=1e-15)
                   for v in values), \
            f"{name} drifted with planning horizon: {values}"
