"""P2-02 tests for dynamics.normalization (train-only statistics)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from dynamics import dynamics_batch_from_episodes, fit_normalization
from dynamics.evaluation import training_tensors
from dynamics.normalization import NormalizationStats


def _batch(seed=0, shift=0.0, scale=1.0, B=3, T=8, D=5, U=2):
    g = torch.Generator().manual_seed(seed)
    eps, conds = [], []
    for i in range(B):
        x = torch.randn(T + 1, D, generator=g) * scale + shift
        u = torch.randn(T, U, generator=g)
        # make one dimension constant to exercise the std clamp
        x[:, -1] = 3.0
        eps.append(x)
        conds.append(u)
    return dynamics_batch_from_episodes(eps, conds, [f"e{i}" for i in range(B)])


class TestFitNormalization(unittest.TestCase):
    def setUp(self):
        self.b = _batch()

    def test_source_is_train_only(self):
        st = fit_normalization(self.b)
        self.assertEqual(st.source_split, "train")
        self.assertEqual(st.n_valid_transitions, self.b.num_valid_transitions())

    def test_shapes(self):
        st = fit_normalization(self.b)
        self.assertEqual(tuple(st.state_mean.shape), (5,))
        self.assertEqual(tuple(st.state_std.shape), (5,))
        self.assertEqual(tuple(st.cond_mean.shape), (2,))
        self.assertEqual(tuple(st.cond_std.shape), (2,))

    def test_constant_dimension_is_clamped(self):
        st = fit_normalization(self.b)
        self.assertGreater(float(st.state_std.min()), 0.0)
        # float32 representation of 1e-8 is 9.999999939e-09; allow 1% tolerance.
        self.assertGreaterEqual(float(st.state_std[-1]), st.eps * 0.99)
        self.assertTrue(torch.isfinite(st.state_std).all())

    def test_residual_identity_holds(self):
        """x_norm_next - x_norm == delta/state_std (the core math of P2-02)."""
        st = fit_normalization(self.b)
        x, _u, target, _m = training_tensors(self.b, st)
        xt, _ut, xnext, _mask = self.b.to_training_pairs()
        lhs = st.normalize_state(xnext) - x
        rhs = st.normalize_delta(xnext - xt)
        self.assertLess(float((lhs - rhs).abs().max()), 1e-5)

    def test_state_roundtrip(self):
        st = fit_normalization(self.b)
        x = self.b.observations
        self.assertTrue(torch.allclose(
            st.denormalize_state(st.normalize_state(x)), x, atol=1e-5
        ))

    def test_delta_roundtrip(self):
        st = fit_normalization(self.b)
        d = torch.randn(4, 5)
        self.assertTrue(torch.allclose(
            st.denormalize_delta(st.normalize_delta(d)), d, atol=1e-5
        ))

    def test_train_only_statistics_differ_from_pooled(self):
        """Statistics must come ONLY from the train batch, not all splits."""
        train = _batch(seed=1, shift=0.0)
        other = _batch(seed=2, shift=10.0)
        st = fit_normalization(train)
        # The train-only mean must be near 0, not pulled toward the other split.
        self.assertLess(float(st.state_mean[:4].abs().mean()), 0.6)
        pooled = torch.cat([train.states()[train.valid_mask],
                            other.states()[other.valid_mask]], dim=0)
        self.assertGreater(float(pooled.mean()), 1.0)

    def test_mask_excludes_padded_transitions(self):
        g = torch.Generator().manual_seed(3)
        eps = [torch.randn(11, 5, generator=g) + 2.0,
               torch.randn(5, 5, generator=g) - 2.0]
        conds = [torch.randn(10, 2, generator=g),
                 torch.randn(4, 2, generator=g)]
        b = dynamics_batch_from_episodes(eps, conds, ["long", "short"])
        st = fit_normalization(b)
        self.assertEqual(st.n_valid_transitions, 14)
        # If padding (zeros) were included, the mean would be pulled to ~0.
        self.assertTrue(torch.isfinite(st.state_mean).all())

    def test_unconditioned_batch_has_no_cond_stats(self):
        g = torch.Generator().manual_seed(4)
        eps = [torch.randn(9, 5, generator=g)]
        b = dynamics_batch_from_episodes(eps, None, ["only"])
        st = fit_normalization(b)
        self.assertIsNone(st.cond_mean)
        self.assertIsNone(st.cond_std)
        with self.assertRaises(ValueError):
            st.normalize_condition(torch.zeros(2, 2))

    def test_rejects_nan(self):
        b = self.b
        b.observations[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            fit_normalization(b)

    def test_rejects_empty_mask(self):
        b = self.b
        b.valid_mask.zero_()
        with self.assertRaises(ValueError):
            fit_normalization(b)

    def test_rejects_bad_eps(self):
        for bad in (0, -1, True, "x"):
            with self.assertRaises(ValueError):
                fit_normalization(self.b, eps=bad)


class TestSerialization(unittest.TestCase):
    def test_save_load_roundtrip(self):
        st = fit_normalization(_batch())
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "normalization.json"
            st.save(p)
            st2 = NormalizationStats.load(p)
        self.assertTrue(torch.equal(st.state_mean, st2.state_mean))
        self.assertTrue(torch.equal(st.state_std, st2.state_std))
        self.assertTrue(torch.equal(st.cond_mean, st2.cond_mean))
        self.assertEqual(st.source_split, st2.source_split)
        self.assertEqual(st.n_valid_transitions, st2.n_valid_transitions)

    def test_from_dict_unconditioned(self):
        d = {
            "state_mean": [0.0] * 5, "state_std": [1.0] * 5,
            "cond_mean": None, "cond_std": None, "eps": 1e-8,
            "source_split": "train", "n_valid_transitions": 10,
        }
        st = NormalizationStats.from_dict(d)
        self.assertIsNone(st.cond_mean)


if __name__ == "__main__":
    unittest.main()
