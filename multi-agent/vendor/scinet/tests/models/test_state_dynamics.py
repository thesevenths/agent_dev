"""P2-02 tests for models.state_dynamics (raw-state dynamics baselines)."""

from __future__ import annotations

import io
import unittest

import torch

from models.state_dynamics import (
    PersistenceBaseline,
    ResidualStatePredictor,
    RidgeDynamics,
    shuffle_conditions,
)

D, U = 6, 3


def _pair(B=4, T=7):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(B, T, D, generator=g)
    u = torch.randn(B, T, U, generator=g)
    return x, u


class TestForwardAndResidual(unittest.TestCase):
    def test_unconditioned_forward(self):
        m = ResidualStatePredictor(D, None, (16, 16))
        x, _ = _pair()
        self.assertEqual(tuple(m(x).shape), (4, 7, D))
        self.assertTrue(torch.isfinite(m(x)).all())

    def test_conditioned_forward(self):
        m = ResidualStatePredictor(D, U, (16, 16))
        x, u = _pair()
        self.assertEqual(tuple(m(x, u).shape), (4, 7, D))

    def test_single_step_forward(self):
        m = ResidualStatePredictor(D, U, (16,))
        x, u = _pair()
        self.assertEqual(tuple(m(x[:, 0], u[:, 0]).shape), (4, D))

    def test_residual_prediction_semantics(self):
        m = ResidualStatePredictor(D, U, (16, 16))
        x, u = _pair()
        delta = m.predict_delta(x, u)
        self.assertTrue(torch.allclose(m(x, u), x + delta, atol=1e-6))

    def test_hidden_dims_empty_is_linear(self):
        m = ResidualStatePredictor(D, U, ())
        x, u = _pair()
        self.assertEqual(tuple(m(x, u).shape), (4, 7, D))
        self.assertEqual(len(list(m.net.parameters())), 2)  # weight + bias


class TestInputValidation(unittest.TestCase):
    def setUp(self):
        self.m = ResidualStatePredictor(D, U, (16, 16))
        self.x, self.u = _pair()

    def test_wrong_state_dim(self):
        with self.assertRaises(ValueError):
            self.m(self.x[..., :-1], self.u)

    def test_wrong_condition_dim(self):
        with self.assertRaises(ValueError):
            self.m(self.x, self.u[..., :-1])

    def test_leading_dim_mismatch(self):
        with self.assertRaises(ValueError):
            self.m(self.x, self.u[:, :-1])

    def test_rank_mismatch_no_broadcast(self):
        with self.assertRaises(ValueError):
            self.m(self.x, self.u[:, 0])  # [B,U] broadcast across T forbidden

    def test_b1_rejects_condition(self):
        m1 = ResidualStatePredictor(D, None, (16,))
        with self.assertRaises(ValueError):
            m1(self.x, self.u)

    def test_b2_rejects_missing_condition(self):
        with self.assertRaises(TypeError):
            self.m(self.x)

    def test_nan_rejection(self):
        x = self.x.clone()
        x[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.m(x, self.u)

    def test_inf_rejection(self):
        u = self.u.clone()
        u[0, 0, 0] = float("inf")
        with self.assertRaises(ValueError):
            self.m(self.x, u)

    def test_non_tensor_rejection(self):
        with self.assertRaises(TypeError):
            self.m([[0.0] * D], [[0.0] * U])

    def test_int_dtype_rejection(self):
        with self.assertRaises(TypeError):
            self.m(torch.zeros(2, D, dtype=torch.int64),
                   torch.zeros(2, U))

    def test_constructor_validation(self):
        for bad in (0, -1, True, 3.5, "6"):
            with self.assertRaises(ValueError):
                ResidualStatePredictor(bad, U)
        for bad in (0, -2, True, 2.5):
            with self.assertRaises(ValueError):
                ResidualStatePredictor(D, bad)
        for bad in (None, "abc", (16, -1), (0,)):
            with self.assertRaises(Exception):
                ResidualStatePredictor(D, U, bad)


class TestGradientAndRollout(unittest.TestCase):
    def test_gradient_propagation(self):
        m = ResidualStatePredictor(D, U, (16, 16))
        x, u = _pair()
        out = m(x, u)
        out.sum().backward()
        for p in m.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())

    def test_rollout_shape_and_alignment(self):
        m = ResidualStatePredictor(D, U, (16,))
        x, u = _pair(B=3, T=9)
        x0 = x[:, 0]
        preds = m.rollout(x0, u)
        self.assertEqual(tuple(preds.shape), (3, 9, D))
        manual = []
        z = x0
        for t in range(u.shape[1]):
            z = m(z, u[:, t])
            manual.append(z)
        self.assertTrue(torch.allclose(preds, torch.stack(manual, dim=1), atol=1e-6))

    def test_rollout_gradient_flows(self):
        m = ResidualStatePredictor(D, U, (16,))
        x, u = _pair(B=2, T=4)
        preds = m.rollout(x[:, 0], u)
        preds.sum().backward()
        self.assertIsNotNone(next(m.parameters()).grad)

    def test_rollout_without_conditions_requires_horizon(self):
        m = ResidualStatePredictor(D, None, (16,))
        x, _ = _pair(B=2, T=4)
        with self.assertRaises(ValueError):
            m.rollout(x[:, 0])
        self.assertEqual(tuple(m.rollout(x[:, 0], horizon=5).shape), (2, 5, D))

    def test_rollout_rejects_nan(self):
        m = ResidualStatePredictor(D, U, (16,))
        x, u = _pair(B=2, T=4)
        u2 = u.clone()
        u2[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            m.rollout(x[:, 0], u2)


class TestPersistenceBaseline(unittest.TestCase):
    def test_identity(self):
        x, u = _pair()
        p = PersistenceBaseline(D)
        self.assertTrue(torch.equal(p(x, u), x))

    def test_rollout_constant(self):
        x, u = _pair(B=2, T=5)
        p = PersistenceBaseline(D)
        r = p.rollout(x[:, 0], u)
        self.assertEqual(tuple(r.shape), (2, 5, D))
        self.assertTrue(torch.allclose(r, x[:, :1].expand(-1, 5, -1)))


class TestRidgeDynamics(unittest.TestCase):
    def test_recovers_known_linear_map(self):
        torch.manual_seed(0)
        x, u = _pair(B=8, T=12)
        W = torch.randn(D + U, D) * 0.2
        b = torch.randn(D) * 0.1
        delta = torch.cat([x, u], dim=-1) @ W + b
        mask = torch.ones(x.shape[:2], dtype=torch.bool)
        m = RidgeDynamics(D, U, alpha=0.0)
        m.fit(x, delta, u, mask, alpha=0.0)
        pred = m.predict_delta(x, u)
        self.assertLess(float((pred - delta).abs().max()), 1e-4)

    def test_mask_excludes_padding(self):
        x, u = _pair(B=2, T=6)
        mask = torch.zeros(2, 6, dtype=torch.bool)
        mask[:, :4] = True
        delta = torch.zeros_like(x)
        delta[1, 5] = 999.0  # invalid position, must be ignored
        m = RidgeDynamics(D, U, alpha=1e-6)
        m.fit(x, delta, u, mask)
        self.assertLess(float(m.predict_delta(x, u).abs().max()), 1e-3)

    def test_predict_before_fit_rejected(self):
        m = RidgeDynamics(D, U)
        x, u = _pair()
        with self.assertRaises(RuntimeError):
            m(x, u)

    def test_rollout(self):
        x, u = _pair(B=3, T=5)
        m = RidgeDynamics(D, U)
        m.fit(x, torch.zeros_like(x), u)
        self.assertEqual(tuple(m.rollout(x[:, 0], u).shape), (3, 5, D))


class TestShuffleNegativeControl(unittest.TestCase):
    def test_reproducible_and_changes(self):
        _, u = _pair(B=4, T=10)
        a = shuffle_conditions(u, seed=7)
        b = shuffle_conditions(u, seed=7)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, u))

    def test_mask_positions_untouched(self):
        _, u = _pair(B=2, T=6)
        mask = torch.zeros(2, 6, dtype=torch.bool)
        mask[:, :4] = True
        out = shuffle_conditions(u, mask, seed=3)
        self.assertTrue(torch.equal(out[~mask], u[~mask]))
        self.assertEqual(float(out[mask].mean()), float(u[mask].mean()))

    def test_too_few_valid_rejected(self):
        _, u = _pair(B=1, T=2)
        mask = torch.tensor([[True, False]])
        with self.assertRaises(ValueError):
            shuffle_conditions(u, mask, seed=1)


class TestCheckpointAndDeterminism(unittest.TestCase):
    def test_state_dict_roundtrip(self):
        m = ResidualStatePredictor(D, U, (16, 16))
        buf = io.BytesIO()
        torch.save(m.state_dict(), buf)
        buf.seek(0)
        m2 = ResidualStatePredictor(D, U, (16, 16))
        m2.load_state_dict(torch.load(buf, weights_only=True))
        x, u = _pair()
        m.eval(); m2.eval()
        self.assertTrue(torch.allclose(m(x, u), m2(x, u), atol=1e-7))

    def test_deterministic_evaluation(self):
        torch.manual_seed(11)
        m1 = ResidualStatePredictor(D, U, (16, 16)).eval()
        torch.manual_seed(11)
        m2 = ResidualStatePredictor(D, U, (16, 16)).eval()
        x, u = _pair()
        self.assertTrue(torch.equal(m1(x, u), m2(x, u)))

    def test_new_model_keys_do_not_collide_with_legacy(self):
        """A2/P2-02 models must never alter WorldModel state_dict keys."""
        from models.world_model import WorldModel
        wm = WorldModel(block_size=5, state_dim=2, latent_dim=4,
                        encoder_hidden=(8,), predictor_hidden=(8,))
        legacy = set(wm.state_dict())
        self.assertEqual(
            {k.split(".")[0] for k in legacy},
            {"encoder", "predictor", "target_encoder"},
        )
        m = ResidualStatePredictor(D, U, (16,))
        self.assertFalse(set(m.state_dict()) & legacy)
        self.assertFalse(hasattr(wm, "residual_predictor"))


if __name__ == "__main__":
    unittest.main()
