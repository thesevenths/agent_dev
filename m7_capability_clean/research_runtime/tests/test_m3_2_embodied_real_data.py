"""ARR 1.1 / M3.2 -- Embodied REAL-data regression tests.

ENGINE-FREE of the statistical gate (that is covered by test_m3_0_statistics),
these tests lock the M3.2 FACTS about the domain layer:

* The embodied domain runs on the REAL frozen ``robot_sim_v1`` dataset with
  PURE STDLIB (csv/hashlib/random/math + pyyaml). It does NOT need
  gymnasium / mujoco / lerobot -- a blocker that was wrongly attributed to M3.2
  in a stale memory. These tests would FAIL loudly if that regression crept in.
* The domain's genuine model fix ``model.drift_correction`` produces a real,
  direction-positive improvement over baseline under the existing runner, which
  is what the M3.0 gate formally ACCEPTED at 10+10 seeds.

No FakeRunner: these call the production ``embodied_compute`` directly.
Deterministic injected seeds keep the runs fast and reproducible.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from research_runtime.domains.embodied import PROJECT_DIR, embodied_compute
from research_runtime.domains.local_engine import (
    load_split, read_frozen_manifest,
)


def _base_cfg():
    return {
        "seed": 42,
        "data": {"frozen_dir": "datasets/robot_sim_v1",
                 "dataset_seed": 20261112, "train_split": "train"},
        "model": {"drift_correction": 0.0, "rollout_horizon": 5,
                  "safety_bound": 2.0},
        "train": {"lr": 0.02, "epochs": 80},
    }


def _job(tmp, name):
    j = tmp / name
    j.mkdir(parents=True, exist_ok=True)
    return j


def test_embodied_real_data_needs_no_robotics_runtime():
    """The embodied domain loads the frozen robot_sim_v1 CSV and computes a real
    metric with stdlib only -- no gymnasium/mujoco/lerobot import anywhere."""
    # Importing + running embodied_compute must not pull in any robotics stack.
    import sys
    banned = {"gymnasium", "mujoco", "lerobot", "torch"}
    leaked = banned & set(sys.modules)
    assert not leaked, f"robotics/runtime deps leaked into sys.modules: {leaked}"
    cfg = _base_cfg()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        t = embodied_compute(cfg, {"stage": "train", "project_dir": str(PROJECT_DIR),
                                   "seed": 42, "run_id": "smoke"}, _job(tmp, "tr"))
        assert t["status"] == "succeeded"
        e = embodied_compute(cfg, {"stage": "evaluate", "project_dir": str(PROJECT_DIR),
                                   "run_id": "smoke", "split": "validation"},
                             _job(tmp, "ev"))
        assert e["status"] == "succeeded"
        q = e["metrics"]["state_mse@1"]
        assert isinstance(q, float) and q >= 0.0 and q < 1.0


def test_embodied_frozen_manifest_is_sha_pinned():
    cfg = _base_cfg()
    manifest = read_frozen_manifest(PROJECT_DIR, cfg)
    assert manifest["dataset_version"] == "robot_sim_v1-4979652dd897"
    # Every split file passes its on-disk SHA-256 check (fail-closed provenance).
    for name in ("train", "validation", "test", "ood"):
        rows = load_split(PROJECT_DIR, cfg, name)
        assert len(rows) > 0
        assert all(c in rows[0] for c in ("episode", "t", "x", "v", "action"))


def test_drift_correction_is_a_real_positive_improvement():
    """The genuine TRAIN-ONLY model fix lowers state_mse@1.

    Exercises the REAL production pipeline (CandidateConfigBuilder ->
    real embodied_compute runner -> evaluate_domain_candidate), the same path the
    M3.2 driver uses, so the verdict is produced by the production wiring rather
    than a hand-rolled train/evaluate handoff.
    """
    from research_runtime.cli import _build_real_domain_runner
    from research_runtime.domains.embodied import read_dataset_version
    from research_runtime.evolution.domain_evolver import (
        evaluate_domain_candidate, DomainModelCandidate, CandidateType,
    )
    project_dir = str(PROJECT_DIR)
    base_cfg = str(PROJECT_DIR / "config.yaml")
    ds_version = read_dataset_version()
    runner = _build_real_domain_runner("embodied", project_dir)
    cand = DomainModelCandidate(
        candidate_id="", domain="embodied", target="model.drift_correction",
        old=None, new=0.75, taxonomy="dynamics_error_accumulation",
        hypothesis="TRAIN-ONLY residual drift correction; reduces long-horizon "
                  "rollout error.")
    rec = evaluate_domain_candidate(
        domain="embodied", candidate=cand, base_config=base_cfg,
        project_dir=project_dir, campaign_id="m3_2_test", seeds=(42, 43),
        dataset_version=ds_version, runner=runner,
        require_frozen_manifest=True, holdout_seeds=(60, 61))
    assert rec["n_invalid"] == 0, "every paired + holdout run must be valid"
    assert rec["mean_improvement"] is not None and rec["mean_improvement"] > 0.0, \
        "drift_correction must lower state_mse@1 (metric is minimized)"
    # Holdout must confirm the same positive direction (no silent contradiction).
    assert rec["holdout_direction_fraction"] == 1.0

