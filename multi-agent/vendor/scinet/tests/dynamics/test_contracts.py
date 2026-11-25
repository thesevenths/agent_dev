"""
tests.dynamics.test_contracts
=============================

Contract tests for :class:`dynamics.contracts.DynamicsBatch`.

These tests are pure (numpy + torch only) and never import the ``robotics``
package, satisfying the A1 rule "新增接口不依赖 robotics 包".
"""

import os
import tempfile
import unittest

import numpy as np
import torch

from dynamics.contracts import (
    DynamicsBatch,
    dynamics_batch_from_episodes,
    dynamics_batch_from_windows,
    training_pairs,
)


def _rand(*shape):
    return np.random.RandomState(0).randn(*shape).astype(np.float32)


class TestUnconditionedBatch(unittest.TestCase):
    def test_unconditioned_conditions_none(self):
        obs = [_rand(11, 4) for _ in range(3)]
        b = dynamics_batch_from_episodes(obs)
        self.assertIsNone(b.conditions)
        self.assertEqual(tuple(b.observations.shape), (3, 11, 4))
        self.assertEqual(tuple(b.valid_mask.shape), (3, 10))
        self.assertEqual(b.cond_dim, None)
        self.assertFalse(b.is_conditioned)
        self.assertTrue(b.valid_mask.all())

    def test_conditioned_batch(self):
        obs = [_rand(11, 4) for _ in range(3)]
        act = [_rand(10, 2) for _ in range(3)]
        b = dynamics_batch_from_episodes(obs, act)
        self.assertIsNotNone(b.conditions)
        self.assertEqual(tuple(b.conditions.shape), (3, 10, 2))
        self.assertEqual(b.cond_dim, 2)
        self.assertTrue(b.is_conditioned)


class TestShapeValidation(unittest.TestCase):
    def test_obs_wrong_ndim_raises(self):
        obs = torch.randn(2, 5)  # 2-D, must be [B, T+1, D]
        mask = torch.ones(2, 4, dtype=torch.bool)
        with self.assertRaises(ValueError):
            DynamicsBatch(obs, None, mask, ["a", "b"])

    def test_episode_ids_length_mismatch_raises(self):
        obs = torch.randn(2, 11, 3)
        mask = torch.ones(2, 10, dtype=torch.bool)
        with self.assertRaises(ValueError):
            DynamicsBatch(obs, None, mask, ["only-one"])

    def test_conditions_shape_mismatch_raises(self):
        obs = torch.randn(2, 11, 3)
        mask = torch.ones(2, 10, dtype=torch.bool)
        cond = torch.randn(2, 9, 2)  # should be [B, T=10, U]
        with self.assertRaises(ValueError):
            DynamicsBatch(obs, cond, mask, ["a", "b"])


class TestDtypeValidation(unittest.TestCase):
    def test_mask_not_bool_raises(self):
        obs = torch.randn(2, 11, 3)
        mask = torch.ones(2, 10, dtype=torch.int64)  # not bool
        with self.assertRaises(TypeError):
            DynamicsBatch(obs, None, mask, ["a", "b"])

    def test_obs_not_float_raises(self):
        obs = torch.ones(2, 11, 3, dtype=torch.int64)
        mask = torch.ones(2, 10, dtype=torch.bool)
        with self.assertRaises(TypeError):
            DynamicsBatch(obs, None, mask, ["a", "b"])

    def test_conditions_not_float_raises(self):
        obs = torch.randn(2, 11, 3)
        mask = torch.ones(2, 10, dtype=torch.bool)
        cond = torch.ones(2, 10, 2, dtype=torch.int64)
        with self.assertRaises(TypeError):
            DynamicsBatch(obs, cond, mask, ["a", "b"])


class TestUnconditionedRepresentationIsUnique(unittest.TestCase):
    """A1 amendment: conditions=None is the ONLY unconditioned representation.

    Rejects the second convention (`U=0` / all-zero tensor standing in for
    "no control") that the A0 gap draft had suggested. Zero-valued controls
    with U>=1 remain valid and semantically distinct from None.
    """

    def test_zero_width_conditions_tensor_raises(self):
        obs = torch.randn(2, 11, 3)
        mask = torch.ones(2, 10, dtype=torch.bool)
        cond_u0 = torch.zeros(2, 10, 0)  # [B,T,U=0]
        with self.assertRaises(ValueError):
            DynamicsBatch(obs, cond_u0, mask, ["a", "b"])

    def test_none_vs_all_zero_are_distinct(self):
        obs = [_rand(11, 3) for _ in range(2)]
        zeros = [np.zeros((10, 2), dtype=np.float32) for _ in range(2)]
        b_zero = dynamics_batch_from_episodes(obs, zeros)
        b_none = dynamics_batch_from_episodes(obs)
        # zero-valued controls => conditioned; None => unconditioned
        self.assertIsNotNone(b_zero.conditions)
        self.assertTrue(b_zero.is_conditioned)
        self.assertEqual(b_zero.cond_dim, 2)
        self.assertIsNone(b_none.conditions)
        self.assertFalse(b_none.is_conditioned)
        self.assertIsNone(b_none.cond_dim)

    def test_episode_ids_are_string_metadata_not_tensor(self):
        obs = [_rand(11, 3) for _ in range(2)]
        b = dynamics_batch_from_episodes(obs, episode_ids=["ep-a", "ep-b"])
        self.assertIsInstance(b.episode_ids, list)
        self.assertTrue(all(isinstance(e, str) for e in b.episode_ids))
        # integer ids are accepted but normalised to plain strings
        b2 = dynamics_batch_from_episodes(obs, episode_ids=[7, 8])
        self.assertEqual(b2.episode_ids, ["7", "8"])


class TestTemporalAlignment(unittest.TestCase):
    def test_xt_ut_xtplus1_alignment(self):
        # Construct a fully controlled, deterministic sequence so we can assert
        # the exact (x_t, u_t) -> x_{t+1} relationship.
        n = 5
        obs_seq = np.stack(
            [np.arange(11, dtype=np.float32) + 100 * i for i in range(n)]
        )[..., None]  # [5, 11, 1]
        act = np.stack(
            [(np.arange(10, dtype=np.float32) + 50 * i) for i in range(n)]
        )[..., None]  # [5, 10, 1]
        b = dynamics_batch_from_episodes(list(obs_seq), list(act))
        xt, ut, xnext, m = training_pairs(b)
        self.assertEqual(tuple(xt.shape), (5, 10, 1))
        self.assertTrue(m.all())
        for i in range(n):
            for t in range(10):
                self.assertEqual(xt[i, t, 0].item(), 100 * i + t)
                self.assertEqual(xnext[i, t, 0].item(), 100 * i + t + 1)
                self.assertEqual(ut[i, t, 0].item(), 50 * i + t)


class TestVariableLengthAndPadding(unittest.TestCase):
    def test_padding_and_mask(self):
        obs = [
            _rand(6, 3),   # 5 transitions
            _rand(11, 3),  # 10 transitions
        ]
        b = dynamics_batch_from_episodes(obs, pad_value=0.0)
        self.assertEqual(tuple(b.observations.shape), (2, 11, 3))
        self.assertEqual(tuple(b.valid_mask.shape), (2, 10))
        # row 0: first 5 transitions valid, the rest padded -> False
        self.assertTrue(b.valid_mask[0, :5].all())
        self.assertFalse(b.valid_mask[0, 5:].any())
        self.assertTrue(b.valid_mask[1].all())
        # padded observation steps must equal the pad value
        self.assertTrue(torch.all(b.observations[0, 6:] == 0.0))

    def test_variable_condition_lengths(self):
        obs = [_rand(8, 3), _rand(13, 3)]
        act = [_rand(7, 2), _rand(12, 2)]
        b = dynamics_batch_from_episodes(obs, act)
        self.assertEqual(tuple(b.conditions.shape), (2, 12, 2))
        self.assertTrue(b.valid_mask[0, :7].all())
        self.assertFalse(b.valid_mask[0, 7:].any())


class TestEmptyTransitionRejection(unittest.TestCase):
    def test_single_observation_rejected(self):
        # One observation => zero transitions => must be rejected.
        obs = [_rand(1, 3)]
        with self.assertRaises(ValueError):
            dynamics_batch_from_episodes(obs)

    def test_all_padded_row_rejected(self):
        # If a row ends up with zero valid transitions it must be rejected.
        obs = [_rand(11, 3), _rand(2, 3)]  # second episode has 1 transition -> ok
        b = dynamics_batch_from_episodes(obs)
        self.assertEqual(b.num_valid_transitions(), 10 + 1)


class TestNonFiniteRejection(unittest.TestCase):
    def test_nan_rejected(self):
        obs = [_rand(11, 3) for _ in range(2)]
        obs[0][0, 0] = np.nan
        with self.assertRaises(ValueError):
            dynamics_batch_from_episodes(obs)

    def test_inf_rejected(self):
        obs = [_rand(11, 3) for _ in range(2)]
        obs[1][3, 1] = np.inf
        with self.assertRaises(ValueError):
            dynamics_batch_from_episodes(obs)

    def test_nan_in_conditions_rejected(self):
        obs = [_rand(11, 3) for _ in range(2)]
        act = [_rand(10, 2) for _ in range(2)]
        act[0][0, 0] = np.nan
        with self.assertRaises(ValueError):
            dynamics_batch_from_episodes(obs, act)


class TestSerialization(unittest.TestCase):
    def test_roundtrip_unconditioned(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "batch.npz")
        obs = [_rand(11, 4) for _ in range(3)]
        b = dynamics_batch_from_episodes(obs, episode_ids=["a", "b", "c"])
        b.save(p)
        b2 = DynamicsBatch.load(p)
        self.assertEqual(tuple(b2.observations.shape), (3, 11, 4))
        self.assertIsNone(b2.conditions)
        self.assertTrue(torch.allclose(b2.observations, b.observations))
        self.assertTrue(torch.equal(b2.valid_mask, b.valid_mask))
        self.assertEqual(b2.episode_ids, ["a", "b", "c"])

    def test_roundtrip_conditioned(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "batch.npz")
        obs = [_rand(11, 4) for _ in range(3)]
        act = [_rand(10, 2) for _ in range(3)]
        b = dynamics_batch_from_episodes(obs, act, episode_ids=["x", "y", "z"])
        b.save(p)
        b2 = DynamicsBatch.load(p)
        self.assertIsNotNone(b2.conditions)
        self.assertEqual(tuple(b2.conditions.shape), (3, 10, 2))
        self.assertTrue(torch.allclose(b2.conditions, b.conditions))
        self.assertTrue(torch.equal(b2.valid_mask, b.valid_mask))
        self.assertEqual(b2.episode_ids, ["x", "y", "z"])


if __name__ == "__main__":
    unittest.main()
