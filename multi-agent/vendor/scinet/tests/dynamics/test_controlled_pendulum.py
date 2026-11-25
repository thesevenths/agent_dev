"""
tests.dynamics.test_controlled_pendulum
=======================================
Unit tests for the P1-A controlled-pendulum simulator
(``data.controlled_pendulum``). These are pure (numpy + torch only) and mirror the
8-item audit checklist at a single-episode / single-step granularity so the
simulator logic is covered by the normal regression suite, not just the
end-to-end audit script.
"""

import math
import unittest

import numpy as np
import torch

from data.controlled_pendulum import (
    ControlledPendulum,
    ScenarioSpec,
    ProfileSpec,
    generate_torque_sequence,
    generate_episode,
    derive_episode_seed,
    episode_id_for,
)


class TestActionSequenceDeterminism(unittest.TestCase):
    """The torque generators must be pure functions of (profile, T, dt, seed)."""

    def test_random_is_seeded_reproducible(self):
        pr = ProfileSpec(name="random", type="random", torque_max=5.0)
        a = generate_torque_sequence(pr, 200, 0.02, seed=12345)
        b = generate_torque_sequence(pr, 200, 0.02, seed=12345)
        self.assertEqual(a.dtype, np.float64)
        np.testing.assert_array_equal(a, b)

    def test_random_differs_across_seeds(self):
        pr = ProfileSpec(name="random", type="random", torque_max=5.0)
        a = generate_torque_sequence(pr, 200, 0.02, seed=1)
        b = generate_torque_sequence(pr, 200, 0.02, seed=2)
        self.assertGreater(np.abs(a - b).max(), 0.0)

    def test_nonrandom_profiles_are_deterministic_formulas(self):
        for typ in ("step", "sinusoid", "alternating"):
            pr = ProfileSpec(name=typ, type=typ, torque_max=5.0)
            a = generate_torque_sequence(pr, 200, 0.02, seed=999)
            b = generate_torque_sequence(pr, 200, 0.02, seed=7)  # seed ignored
            np.testing.assert_array_equal(a, b)

    def test_torque_within_bounds(self):
        for typ in ("random", "step", "sinusoid", "alternating"):
            pr = ProfileSpec(name=typ, type=typ, torque_max=5.0)
            seq = generate_torque_sequence(pr, 200, 0.02, seed=42)
            self.assertTrue(np.all(np.abs(seq) <= 5.0 + 1e-9))


class TestRK4Correctness(unittest.TestCase):
    """Physics-level correctness of the integrator."""

    def test_small_angle_matches_analytic(self):
        # zero torque, zero damping, small initial angle -> linearized pendulum
        g, L = 9.81, 1.0
        sim = ControlledPendulum(g=g, L=L, m=1.0, b=0.0, dt=0.02)
        theta0, omega0 = 0.05, 0.0
        T = 300
        obs = sim.simulate(np.zeros(T, dtype=np.float64), theta0, omega0)
        w = math.sqrt(g / L)
        t = np.arange(T + 1) * 0.02
        analytic = theta0 * np.cos(w * t) + (omega0 / w) * np.sin(w * t)
        self.assertLess(np.max(np.abs(obs[:, 0] - analytic)), 1e-3)

    def test_zero_torque_energy_conservation(self):
        g, L, m = 9.81, 1.0, 1.0
        sim = ControlledPendulum(g=g, L=L, m=m, b=0.0, dt=0.02)
        T = 200
        obs = sim.simulate(np.zeros(T, dtype=np.float64), 2.0, 1.0)
        th, om = obs[:, 0], obs[:, 1]
        E = 0.5 * m * (L ** 2) * om ** 2 + m * g * L * (1.0 - np.cos(th))
        rel = np.max(np.abs(E - E[0])) / (abs(E[0]) + 1e-12)
        self.assertLess(rel, 1e-5)

    def test_rk4_is_fourth_order(self):
        sim = ControlledPendulum(g=9.81, L=1.0, m=1.0, b=0.1, dt=0.02)
        ref = sim.simulate_substep(np.full(60, 1.3), 2.0, 0.0, 16)
        coarse = sim.simulate(np.full(60, 1.3), 2.0, 0.0)
        fine = sim.simulate_substep(np.full(60, 1.3), 2.0, 0.0, 2)
        e_dt = float(np.max(np.abs(coarse - ref)))
        e_dt2 = float(np.max(np.abs(fine - ref)))
        self.assertGreater(e_dt / e_dt2, 10.0)  # ~16x expected (order 4)

    def test_unwrapped_theta_not_reduced_mod_2pi(self):
        # The simulator must NEVER reduce theta modulo 2*pi. Start one episode
        # at 2*pi + 0.1 and another at 0.1. Because sin(theta) is periodic, their
        # omega trajectories are identical, so their theta trajectories must keep
        # a constant +2*pi offset (continuous unwrapping) -- NOT fold the second
        # episode back into (-pi, pi].
        sim = ControlledPendulum(g=9.81, L=1.0, m=1.0, b=0.0, dt=0.02)
        zeros = np.zeros(50, dtype=np.float64)
        obs_large = sim.simulate(zeros, theta0=2.0 * math.pi + 0.1, omega0=0.0)
        obs_small = sim.simulate(zeros, theta0=0.1, omega0=0.0)
        self.assertAlmostEqual(float(obs_large[0, 0]), 2.0 * math.pi + 0.1, places=9)
        # omega dynamics identical -> theta offset stays exactly 2*pi.
        self.assertAlmostEqual(
            float(obs_large[-1, 0] - obs_small[-1, 0]), 2.0 * math.pi, places=6
        )
        self.assertTrue(np.all(np.isfinite(obs_large[:, 0])))


class TestEpisodeGeneration(unittest.TestCase):
    """End-to-end episode generation + DynamicsBatch contract."""

    def test_episode_shape_and_dtype(self):
        sc = ScenarioSpec(name="nominal", g=9.81, L=1.0, m=1.0, b=0.1)
        pr = ProfileSpec(name="random", type="random", torque_max=5.0)
        ep = generate_episode(sc, pr, seed=42, T=200, dt=0.02)
        self.assertEqual(ep.observations.shape, (201, 2))
        self.assertEqual(ep.conditions.shape, (200, 1))
        self.assertEqual(ep.observations.dtype, np.float64)

    def test_episode_id_and_seed_derivation(self):
        self.assertEqual(episode_id_for(0, 1, "nominal", "step"), "S0_step")
        # seed stays within numpy's 32-bit RandomState range
        for s in range(10):
            seed = derive_episode_seed(20261111, s, 0)
            self.assertTrue(0 <= seed < 2 ** 32)

    def test_builds_dynamics_batch_float64(self):
        from dynamics.contracts import DynamicsBatch
        sc = ScenarioSpec(name="nominal", g=9.81, L=1.0, m=1.0, b=0.1)
        pr = ProfileSpec(name="random", type="random", torque_max=5.0)
        ep = generate_episode(sc, pr, seed=7, T=200, dt=0.02)
        # DynamicsBatch expects a batched [B, T+1, D] tensor (single episode -> B=1).
        batch = DynamicsBatch(
            torch.as_tensor(ep.observations[None], dtype=torch.float64),
            torch.as_tensor(ep.conditions[None], dtype=torch.float64),
            torch.ones(1, 200, dtype=torch.bool),
            ["S0_random"],
        )
        self.assertEqual(batch.obs_dim, 2)
        self.assertEqual(batch.cond_dim, 1)
        self.assertEqual(batch.num_steps, 200)


if __name__ == "__main__":
    unittest.main()
