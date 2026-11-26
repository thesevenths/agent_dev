"""P2-02 tests for dynamics.metrics (masked metrics + episode bootstrap)."""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from dynamics.metrics import (
    describe,
    episode_errors,
    masked_mae,
    masked_mse,
    masked_mse_loss,
    one_step_metrics,
    paired_bootstrap_ci,
    per_dimension_rmse,
    persistence_skill_score,
)


def _data():
    pred = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]],
        [[2.0, 1.0], [88.0, 88.0], [77.0, 77.0]],
    ])
    true = torch.tensor([
        [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    ])
    mask = torch.tensor([
        [True, True, False],
        [True, False, False],
    ])
    return pred, true, mask


class TestMaskedMetrics(unittest.TestCase):
    def setUp(self):
        self.p, self.t, self.m = _data()

    def test_masked_mse_excludes_padding(self):
        p, t, m = self.p, self.t, self.m
        valid = p[m] - t[m]            # [[1,2],[3,4],[2,1]]
        expected = float((valid ** 2).mean())
        self.assertAlmostEqual(masked_mse(p, t, m), expected, places=6)
        # If padding were included, the huge 99/88 values would dominate.
        self.assertLess(masked_mse(p, t, m), 100.0)

    def test_masked_mse_loss_is_differentiable(self):
        p = self.p.clone().requires_grad_(True)
        loss = masked_mse_loss(p, self.t, self.m)
        loss.backward()
        self.assertIsNotNone(p.grad)
        self.assertEqual(p.grad[self.m].abs().sum().item() > 0, True)
        self.assertEqual(float(p.grad[~self.m].abs().sum()), 0.0)

    def test_masked_mae(self):
        expected = float((self.p[self.m] - self.t[self.m]).abs().mean())
        self.assertAlmostEqual(masked_mae(self.p, self.t, self.m), expected, places=6)

    def test_per_dimension_rmse(self):
        vals = per_dimension_rmse(self.p, self.t, self.m)
        self.assertEqual(len(vals), 2)
        d = self.p[self.m] - self.t[self.m]
        for j in range(2):
            self.assertAlmostEqual(vals[j], float(torch.sqrt((d[:, j] ** 2).mean())),
                                   places=6)

    def test_one_step_metrics(self):
        met = one_step_metrics(self.p, self.t, self.m)
        self.assertEqual(met["n_valid"], 3)
        self.assertAlmostEqual(met["rmse"], math.sqrt(met["mse"]), places=6)
        self.assertEqual(len(met["per_dim_rmse"]), 2)

    def test_shape_and_mask_validation(self):
        with self.assertRaises(ValueError):
            masked_mse(self.p[..., :-1], self.t, self.m)
        with self.assertRaises(TypeError):
            masked_mse(self.p, self.t, self.m.float())
        with self.assertRaises(TypeError):  # non-bool mask is a type error
            masked_mse(self.p, self.t, torch.ones(2, 3))
        with self.assertRaises(ValueError):
            masked_mse(self.p, self.t, torch.zeros(2, 3, dtype=torch.bool))

    def test_persistence_skill_score(self):
        self.assertAlmostEqual(persistence_skill_score(0.25, 1.0), 0.75)
        self.assertAlmostEqual(persistence_skill_score(1.0, 1.0), 0.0)
        self.assertLess(persistence_skill_score(2.0, 1.0), 0.0)
        with self.assertRaises(ValueError):
            persistence_skill_score(1.0, 0.0)


class TestEpisodeAggregation(unittest.TestCase):
    def test_episode_errors_length_and_mask(self):
        p, t, m = _data()
        errs = episode_errors(p, t, m)
        self.assertEqual(len(errs), 2)
        self.assertAlmostEqual(errs[0], float(((p[0][m[0]] - t[0][m[0]]) ** 2).mean()),
                               places=6)

    def test_episode_errors_rejects_empty_episode(self):
        p, t, m = _data()
        m2 = m.clone()
        m2[1] = False
        with self.assertRaises(ValueError):
            episode_errors(p, t, m2)

    def test_describe(self):
        d = describe([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(d["mean"], 2.5)
        self.assertAlmostEqual(d["median"], 2.5)
        self.assertAlmostEqual(d["std"], 1.11803398875, places=6)
        self.assertEqual(d["n"], 4)


class TestPairedBootstrap(unittest.TestCase):
    def test_reproducible_and_ci_order(self):
        a = [0.10, 0.12, 0.09, 0.11, 0.10, 0.13]
        b = [0.30, 0.35, 0.28, 0.33, 0.31, 0.36]
        r1 = paired_bootstrap_ci(a, b, n_boot=2000, seed=42)
        r2 = paired_bootstrap_ci(a, b, n_boot=2000, seed=42)
        self.assertEqual(r1["ci_low"], r2["ci_low"])
        self.assertLess(r1["ci_low"], r1["mean_diff"])
        self.assertLess(r1["mean_diff"], r1["ci_high"])
        self.assertTrue(r1["significant_a_better"])
        self.assertFalse(r1["significant_b_better"])
        self.assertEqual(r1["n_pairs"], 6)

    def test_no_significant_difference(self):
        a = [0.10, 0.20, 0.30, 0.40]
        r = paired_bootstrap_ci(a, a, n_boot=1000, seed=1)
        self.assertAlmostEqual(r["mean_diff"], 0.0)
        self.assertFalse(r["significant_a_better"])
        self.assertFalse(r["significant_b_better"])

    def test_validation(self):
        with self.assertRaises(ValueError):
            paired_bootstrap_ci([1.0, 2.0], [1.0])
        with self.assertRaises(ValueError):
            paired_bootstrap_ci([1.0], [2.0])
        with self.assertRaises(ValueError):
            paired_bootstrap_ci([1.0, 2.0], [1.0, 2.0], alpha=1.5)
        with self.assertRaises(ValueError):
            paired_bootstrap_ci([1.0, 2.0], [1.0, 2.0], n_boot=10)


if __name__ == "__main__":
    unittest.main()
