"""
tests.dynamics.test_controlled_pendulum_v2
==========================================
P1-A' review fix P4: physical-parameter validation, per-step finite guards,
torque/profile validation, and zero-regression proof against the v1 core.

The v1 numerical core is REUSED by inheritance, so for all legal inputs the
v2 simulator must be BITWISE identical to v1. That is asserted here.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from data.controlled_pendulum import ControlledPendulum, ProfileSpec
from data.controlled_pendulum_v2 import (
    ValidatedPendulum, classify_regime, generate_torque_sequence_v2,
    validate_physical_params, validate_profile, V2_PROFILE_TYPES,
)


class TestPhysicalParameterValidation(unittest.TestCase):
    def test_rejects_non_positive(self):
        bad = [
            dict(g=0), dict(g=-1), dict(L=0), dict(L=-1),
            dict(m=0), dict(m=-2), dict(dt=0), dict(dt=-0.1),
            dict(b=-0.1),
        ]
        for kw in bad:
            with self.subTest(kw=kw):
                base = dict(g=9.81, L=1.0, m=1.0, b=0.1, dt=0.02)
                base.update(kw)
                with self.assertRaises(ValueError):
                    validate_physical_params(**base)

    def test_accepts_zero_damping(self):
        v = validate_physical_params(g=9.81, L=1.0, m=1.0, b=0.0, dt=0.02)
        self.assertEqual(v["b"], 0.0)

    def test_rejects_non_finite_and_bool(self):
        for kw in (dict(g=float("nan")), dict(L=float("inf")),
                   dict(m=float("-inf")), dict(dt=float("nan"))):
            with self.subTest(kw=kw):
                base = dict(g=9.81, L=1.0, m=1.0, b=0.1, dt=0.02)
                base.update(kw)
                with self.assertRaises((ValueError, TypeError)):
                    validate_physical_params(**base)
        with self.assertRaises(TypeError):
            validate_physical_params(g=True, L=1.0, m=1.0, b=0.1, dt=0.02)

    def test_constructor_rejects_bad_params(self):
        for kw in (dict(m=0), dict(L=-1), dict(dt=0)):
            with self.subTest(kw=kw):
                base = dict(g=9.81, L=1.0, m=1.0, b=0.1, dt=0.02)
                base.update(kw)
                with self.assertRaises(ValueError):
                    ValidatedPendulum(**base)


class TestPerStepFiniteGuard(unittest.TestCase):
    def test_nonfinite_state_raises(self):
        p = ValidatedPendulum(torque_bounds=(-1.0, 1.0))
        with self.assertRaises(FloatingPointError):
            p.rk4_step(np.array([np.nan, 0.0]), 0.5)
        with self.assertRaises(FloatingPointError):
            p.rk4_step(np.array([0.0, np.inf]), 0.5)

    def test_nonfinite_torque_raises(self):
        p = ValidatedPendulum(torque_bounds=(-1.0, 1.0))
        with self.assertRaises((ValueError, TypeError)):
            p.simulate(np.array([np.nan]), 0.0, 0.0)

    def test_finite_trajectory_passes(self):
        p = ValidatedPendulum(torque_bounds=(-1.0, 1.0))
        tor = np.linspace(-1.0, 1.0, 200)
        obs = p.simulate(tor, 2.0, 0.0)
        self.assertEqual(obs.shape, (201, 2))
        self.assertTrue(np.all(np.isfinite(obs)))


class TestTorqueAndProfileValidation(unittest.TestCase):
    def test_torque_out_of_bounds_raises(self):
        p = ValidatedPendulum(torque_bounds=(-1.0, 1.0))
        with self.assertRaises(ValueError):
            p.simulate(np.array([2.0]), 0.0, 0.0)

    def test_profile_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            validate_profile(ProfileSpec(name="x", type="bogus"), [-1.0, 1.0])

    def test_profile_bad_fraction_magnitude_raises(self):
        for kw in (dict(fraction=1.5), dict(magnitude=-0.1),
                   dict(amplitude=2.0), dict(frequency_hz=-1.0)):
            with self.subTest(kw=kw):
                base = dict(name="x", type="step")
                base.update(kw)
                with self.assertRaises(ValueError):
                    validate_profile(ProfileSpec(**base), [-1.0, 1.0])

    def test_torque_bounds_must_bracket_zero(self):
        with self.assertRaises(ValueError):
            validate_profile(ProfileSpec(name="x", type="zero"), [0.0, 1.0])

    def test_zero_profile_is_all_zero(self):
        pr = ProfileSpec(name="zero", type="zero")
        seq = generate_torque_sequence_v2(pr, 100, 0.02, seed=1,
                                          hold_steps=5, torque_bounds=[-1.0, 1.0])
        self.assertTrue(np.all(seq == 0.0))

    def test_hold_steps_repeat(self):
        pr = ProfileSpec(name="random", type="random")
        seq = generate_torque_sequence_v2(pr, 20, 0.02, seed=42,
                                          hold_steps=5, torque_bounds=[-1.0, 1.0])
        for k in range(0, 20, 5):
            np.testing.assert_array_equal(seq[k:k + 5], np.full(5, seq[k]))

    def test_all_profile_types_within_bounds(self):
        for typ in V2_PROFILE_TYPES:
            with self.subTest(typ=typ):
                pr = ProfileSpec(name=typ, type=typ)
                seq = generate_torque_sequence_v2(
                    pr, 200, 0.02, seed=7, hold_steps=3,
                    torque_bounds=[-1.0, 1.0])
                self.assertEqual(seq.shape, (200,))
                self.assertTrue(np.all(np.abs(seq) <= 1.0 + 1e-9))


class TestZeroRegressionAgainstV1(unittest.TestCase):
    """v2 must be bitwise identical to v1 for all legal inputs."""

    def test_bitwise_identical_rollouts(self):
        rng = np.random.RandomState(3)
        for _ in range(6):
            g, L, m, b, dt = (9.81, float(rng.uniform(0.8, 1.2)),
                              float(rng.uniform(0.8, 1.2)),
                              float(rng.uniform(0.05, 0.15)), 0.02)
            tor = rng.uniform(-1.0, 1.0, size=100)
            th0, om0 = float(rng.uniform(-3, 3)), float(rng.uniform(-1, 1))
            v1 = ControlledPendulum(g=g, L=L, m=m, b=b, dt=dt)
            v2 = ValidatedPendulum(g=g, L=L, m=m, b=b, dt=dt,
                                   torque_bounds=(-1.0, 1.0))
            np.testing.assert_array_equal(
                v1.simulate(tor, th0, om0), v2.simulate(tor, th0, om0))

    def test_derivatives_bitwise_identical(self):
        v1 = ControlledPendulum()
        v2 = ValidatedPendulum(torque_bounds=(-5.0, 5.0))
        for th, om, tau in [(2.0, 0.0, 2.5), (0.0, 0.0, -2.5),
                            (-1.0, 3.0, 0.0)]:
            np.testing.assert_array_equal(
                v1.derivatives(np.array([th, om]), tau),
                v2.derivatives(np.array([th, om]), tau))


class TestRegimeClassification(unittest.TestCase):
    def test_no_rotation_for_spec_compliant_pilot(self):
        """With torque_bounds=[-1,1] and train-range params, no full rotation."""
        p = ValidatedPendulum(torque_bounds=(-1.0, 1.0))
        obs = p.simulate(np.ones(1000) * 1.0, theta0=2.0, omega0=0.0)
        from data.controlled_pendulum import ScenarioSpec
        sc = ScenarioSpec(name="t", g=9.81, L=1.0, m=1.0, b=0.1)
        r = classify_regime(obs, sc)
        self.assertLess(abs(r["winding_number"]), 1.0)


if __name__ == "__main__":
    unittest.main()
