"""
tests.dynamics.test_action_intervention_analytic
================================================
P1-A' review fix P1 -- analytic-level unit tests for the D-A Action
Intervention acceptance condition in P2_03A_READINESS_PLAN.md §5.6.

The original acceptance text asserted "positive/negative torque produce
OPPOSITE angular acceleration" using the nominal initial condition
theta0=2.0, omega0=0. That is WRONG: the gravity term dominates
(-(g/L)sin(2) = -8.920208 rad/s^2), so both +tau and -tau still give a
NEGATIVE angular acceleration. The corrected condition uses theta0=0
(zero gravity torque), where alpha = tau/(m L^2) and the direction test is
valid.

These tests pin BOTH facts so the mistake cannot silently return.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from data.controlled_pendulum import ControlledPendulum


class TestActionInterventionAnalyticSigns(unittest.TestCase):
    """alpha = -(g/L) sin(theta) - b*omega + tau/(m*L^2)."""

    def setUp(self):
        # nominal scenario from configs/dynamics/controlled_pendulum_jepa.yaml
        self.g, self.L, self.m, self.b = 9.81, 1.0, 1.0, 0.1
        self.I = self.m * self.L ** 2
        self.sim = ControlledPendulum(g=self.g, L=self.L, m=self.m,
                                      b=self.b, dt=0.02)

    def analytic_alpha(self, theta, omega, tau):
        return -(self.g / self.L) * math.sin(theta) - self.b * omega + tau / self.I

    def test_theta0_zero_positive_negative_zero_torque_have_expected_signs(self):
        """Corrected D-A initial state: theta0=0, omega0=0 -> alpha = tau/I."""
        theta, omega = 0.0, 0.0
        for tau, expected_sign in ((2.5, 1), (-2.5, -1), (0.0, 0)):
            with self.subTest(tau=tau):
                alpha = self.analytic_alpha(theta, omega, tau)
                if expected_sign > 0:
                    self.assertGreater(alpha, 0.0)
                elif expected_sign < 0:
                    self.assertLess(alpha, 0.0)
                else:
                    self.assertAlmostEqual(alpha, 0.0, places=12)
                # analytic == numeric derivatives (elementwise)
                deriv = self.sim.derivatives(np.array([theta, omega]), tau)
                self.assertAlmostEqual(float(deriv[1]), alpha, places=12)
                self.assertAlmostEqual(float(deriv[0]), omega, places=12)

    def test_theta0_zero_alpha_equals_tau_over_inertia(self):
        for tau in (-2.5, -1.0, 0.0, 1.0, 2.5):
            with self.subTest(tau=tau):
                self.assertAlmostEqual(
                    self.analytic_alpha(0.0, 0.0, tau), tau / self.I, places=12)

    def test_original_theta0_two_condition_is_indeed_invalid(self):
        """Regression guard: theta0=2 gives SAME (negative) sign for +/- tau."""
        theta, omega = 2.0, 0.0
        base = self.analytic_alpha(theta, omega, 0.0)
        self.assertAlmostEqual(base, -9.81 * math.sin(2.0), places=12)
        a_pos = self.analytic_alpha(theta, omega, 2.5)
        a_neg = self.analytic_alpha(theta, omega, -2.5)
        self.assertAlmostEqual(a_pos, -6.420208, places=5)
        self.assertAlmostEqual(a_neg, -11.420208, places=5)
        self.assertLess(a_pos, 0.0)
        self.assertLess(a_neg, 0.0)                 # both negative -> NOT opposite
        self.assertEqual(math.copysign(1, a_pos), math.copysign(1, a_neg))
        # but the DIFFERENCE still equals Delta tau / I (what should be asserted)
        self.assertAlmostEqual(a_pos - a_neg, 5.0 / self.I, places=12)

    def test_general_theta_direction_rule_is_delta_alpha_not_absolute_sign(self):
        """For general theta, only Delta alpha = Delta tau / I is guaranteed."""
        rng = np.random.RandomState(0)
        for _ in range(10):
            theta = float(rng.uniform(-math.pi, math.pi))
            omega = float(rng.uniform(-1.0, 1.0))
            t1, t2 = float(rng.uniform(-2, 2)), float(rng.uniform(-2, 2))
            d_alpha = (self.analytic_alpha(theta, omega, t1)
                       - self.analytic_alpha(theta, omega, t2))
            self.assertAlmostEqual(d_alpha, (t1 - t2) / self.I, places=12)

    def test_numeric_derivatives_match_analytic_for_random_states(self):
        rng = np.random.RandomState(7)
        for _ in range(20):
            theta = float(rng.uniform(-3.0, 3.0))
            omega = float(rng.uniform(-5.0, 5.0))
            tau = float(rng.uniform(-5.0, 5.0))
            d = self.sim.derivatives(np.array([theta, omega]), tau)
            self.assertAlmostEqual(float(d[0]), omega, places=12)
            self.assertAlmostEqual(
                float(d[1]), self.analytic_alpha(theta, omega, tau), places=12)


if __name__ == "__main__":
    unittest.main()
