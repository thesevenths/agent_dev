"""
tests.dynamics.test_regime_audit
================================
P1-B1.1 / P8: independent regime audit.

Confirms the audit (a) recomputes regime from RAW observations using the YAML
declared thresholds -- not just a stored-label format check -- and (b) FAILS on
corrupted labels/counts and on threshold changes that alter classification. Also
locks down ``classify_regime``'s single-source-of-truth threshold handling.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.controlled_pendulum import ScenarioSpec  # noqa: E402
from data.controlled_pendulum_v2 import (  # noqa: E402
    DEFAULT_REGIME_CRITERIA, classify_regime,
)
import scripts.audit_controlled_pendulum_v2 as AUDIT  # noqa: E402

V2 = REPO_ROOT / "datasets" / "controlled_pendulum_v2_pilot"


def _make_obs_constant_omega(winding_target, T=200, dt=0.02):
    """Trajectory with constant omega -> winding=winding_target, 0 sign changes."""
    th = np.linspace(0.0, winding_target * 2.0 * np.pi, T + 1)
    om = np.full(T + 1, (winding_target * 2.0 * np.pi) / (T * dt))
    return np.stack([th, om], axis=1)


def _make_obs_two_sign_changes(A=1.0, T=300, dt=0.02):
    """Omega +A/-A/+A over equal thirds -> exactly 2 sign changes."""
    third = T // 3
    om = np.concatenate([
        np.full(third, A),
        np.full(third, -A),
        np.full(T + 1 - 2 * third, A),
    ])
    th = np.zeros(T + 1)
    for k in range(1, T + 1):
        th[k] = th[k - 1] + om[k - 1] * dt
    return np.stack([th, om], axis=1)


def _scenario():
    return ScenarioSpec(name="t", g=9.81, L=1.0, m=1.0, b=0.1,
                        theta0=0.0, omega0=0.0)


class TestClassifyRegimeThresholdSource(unittest.TestCase):
    def test_default_matches_yaml_values(self):
        obs = _make_obs_two_sign_changes()
        r = classify_regime(obs, _scenario())  # uses DEFAULT criteria
        self.assertIn(r["regime"], ("bounded_oscillation", "transitional",
                                    "continuous_rotation"))

    def test_missing_criterion_raises(self):
        obs = _make_obs_two_sign_changes()
        with self.assertRaises(ValueError):
            classify_regime(obs, _scenario(),
                            criteria={"bounded_oscillation": {"max_theta_range_rad": 6.28,
                                                             "min_omega_sign_changes": 2}})

    def test_invalid_criterion_raises(self):
        obs = _make_obs_two_sign_changes()
        with self.assertRaises(ValueError):
            classify_regime(obs, _scenario(), criteria={
                "continuous_rotation": {"min_abs_winding": "not_a_number",
                                        "max_omega_sign_changes": 2},
                "bounded_oscillation": {"max_theta_range_rad": 6.28,
                                        "min_omega_sign_changes": 2},
                "transitional": "otherwise"})


class TestThresholdSensitivity(unittest.TestCase):
    def test_winding_threshold_changes_classification(self):
        obs = _make_obs_constant_omega(winding_target=1.5)  # sign_changes=0
        default = classify_regime(obs, _scenario(),
                                  criteria=DEFAULT_REGIME_CRITERIA)["regime"]
        self.assertEqual(default, "continuous_rotation")
        modified = dict(DEFAULT_REGIME_CRITERIA)
        modified["continuous_rotation"] = {"min_abs_winding": 2.0,
                                           "max_omega_sign_changes": 2}
        changed = classify_regime(obs, _scenario(), criteria=modified)["regime"]
        self.assertNotEqual(default, changed)
        self.assertEqual(changed, "transitional")

    def test_theta_range_threshold_changes_classification(self):
        obs = _make_obs_two_sign_changes()
        default = classify_regime(obs, _scenario(),
                                  criteria=DEFAULT_REGIME_CRITERIA)["regime"]
        self.assertEqual(default, "bounded_oscillation")
        modified = dict(DEFAULT_REGIME_CRITERIA)
        # Force the theta_range threshold below the actual range -> not bounded.
        actual_range = float(obs[:, 0].max() - obs[:, 0].min())
        modified["bounded_oscillation"] = {"max_theta_range_rad": actual_range * 0.5,
                                           "min_omega_sign_changes": 2}
        changed = classify_regime(obs, _scenario(), criteria=modified)["regime"]
        self.assertNotEqual(default, changed)
        self.assertEqual(changed, "transitional")

    def test_sign_changes_threshold_changes_classification(self):
        obs = _make_obs_two_sign_changes()
        default = classify_regime(obs, _scenario(),
                                  criteria=DEFAULT_REGIME_CRITERIA)["regime"]
        self.assertEqual(default, "bounded_oscillation")
        modified = dict(DEFAULT_REGIME_CRITERIA)
        modified["bounded_oscillation"] = {"max_theta_range_rad": 2.0 * np.pi,
                                           "min_omega_sign_changes": 3}
        changed = classify_regime(obs, _scenario(), criteria=modified)["regime"]
        self.assertNotEqual(default, changed)
        self.assertEqual(changed, "transitional")


class TestIndependentRegimeAuditOnRealData(unittest.TestCase):
    def _load(self):
        with np.load(V2 / "pilot_episodes.npz", allow_pickle=False) as z:
            obs = z["observations"]
        man = json.loads((V2 / "manifest.json").read_text(encoding="utf-8"))
        import yaml
        cfg = yaml.safe_load(
            (REPO_ROOT / "configs" / "dynamics"
             / "controlled_pendulum_v2_pilot.yaml").read_text(encoding="utf-8"))
        return obs, man, cfg

    def test_positive_pass_on_historical_pilot(self):
        obs, man, cfg = self._load()
        passed, detail = AUDIT.independent_regime_audit(obs, man, cfg)
        self.assertTrue(passed, msg=detail)
        self.assertEqual(detail["n_label_mismatches"], 0)
        self.assertTrue(detail["counts_match"])
        self.assertTrue(detail["config_consistent_with_yaml"])

    def test_corrupted_label_fails(self):
        obs, man, cfg = self._load()
        bad = json.loads(json.dumps(man))
        bad["episodes"][0]["regime"] = "continuous_rotation"  # was bounded
        passed, detail = AUDIT.independent_regime_audit(obs, bad, cfg)
        self.assertFalse(passed)
        self.assertGreaterEqual(detail["n_label_mismatches"], 1)

    def test_corrupted_count_fails(self):
        obs, man, cfg = self._load()
        bad = json.loads(json.dumps(man))
        bad["regime_counts"] = {"bounded_oscillation": 19,
                                "transitional": 0, "continuous_rotation": 1}
        passed, detail = AUDIT.independent_regime_audit(obs, bad, cfg)
        self.assertFalse(passed)
        self.assertFalse(detail["counts_match"])

    def test_config_inconsistency_fails(self):
        obs, man, cfg = self._load()
        bad_cfg = json.loads(json.dumps(cfg))
        # Tamper the YAML-declared criteria so they no longer match the manifest.
        bad_cfg["regime_criteria"]["bounded_oscillation"]["min_omega_sign_changes"] = 99
        passed, detail = AUDIT.independent_regime_audit(obs, man, bad_cfg)
        self.assertFalse(passed)
        self.assertFalse(detail["config_consistent_with_yaml"])


if __name__ == "__main__":
    unittest.main()
