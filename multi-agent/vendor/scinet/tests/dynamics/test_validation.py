"""
tests.dynamics.test_validation
==============================

Direct unit tests for :func:`dynamics.validation.validate_dynamics_batch` and
:func:`dynamics.validation.summarize_batch`.

These construct :class:`DynamicsBatch` objects that bypass ``__post_init__``
(via ``object.__new__``) so the validator is exercised in isolation, including
its rejection paths.
"""

import unittest

import torch

from dynamics.contracts import (
    DynamicsBatch,
    dynamics_batch_from_episodes,
)
from dynamics.validation import summarize_batch, validate_dynamics_batch


def _raw(observations, conditions, valid_mask, episode_ids):
    """Build a DynamicsBatch without running __post_init__ validation."""
    b = object.__new__(DynamicsBatch)
    b.observations = observations
    b.conditions = conditions
    b.valid_mask = valid_mask
    b.episode_ids = episode_ids
    return b


class TestValidatorPasses(unittest.TestCase):
    def test_valid_batch_passes(self):
        obs = [_rand(11, 3) for _ in range(2)]
        b = dynamics_batch_from_episodes(obs)
        self.assertTrue(validate_dynamics_batch(b))

    def test_valid_conditioned_passes(self):
        import numpy as np

        obs = [np.random.randn(11, 3).astype(np.float32) for _ in range(2)]
        act = [np.random.randn(10, 2).astype(np.float32) for _ in range(2)]
        b = dynamics_batch_from_episodes(obs, act)
        self.assertTrue(validate_dynamics_batch(b))


class TestValidatorRejections(unittest.TestCase):
    def test_wrong_obs_ndim(self):
        b = _raw(torch.randn(2, 5), None, torch.ones(2, 4, dtype=torch.bool), ["a", "b"])
        with self.assertRaises(ValueError):
            validate_dynamics_batch(b)

    def test_mask_not_bool(self):
        b = _raw(
            torch.randn(2, 11, 3),
            None,
            torch.ones(2, 10, dtype=torch.int64),
            ["a", "b"],
        )
        with self.assertRaises(TypeError):
            validate_dynamics_batch(b)

    def test_obs_not_float(self):
        b = _raw(
            torch.ones(2, 11, 3, dtype=torch.int64),
            None,
            torch.ones(2, 10, dtype=torch.bool),
            ["a", "b"],
        )
        with self.assertRaises(TypeError):
            validate_dynamics_batch(b)

    def test_mask_shape_mismatch(self):
        b = _raw(
            torch.randn(2, 11, 3),
            None,
            torch.ones(2, 9, dtype=torch.bool),  # should be [2, 10]
            ["a", "b"],
        )
        with self.assertRaises(ValueError):
            validate_dynamics_batch(b)

    def test_conditions_shape_mismatch(self):
        b = _raw(
            torch.randn(2, 11, 3),
            torch.randn(2, 9, 2),  # should be [2, 10, U]
            torch.ones(2, 10, dtype=torch.bool),
            ["a", "b"],
        )
        with self.assertRaises(ValueError):
            validate_dynamics_batch(b)

    def test_episode_ids_length_mismatch(self):
        b = _raw(
            torch.randn(2, 11, 3),
            None,
            torch.ones(2, 10, dtype=torch.bool),
            ["only-one"],
        )
        with self.assertRaises(ValueError):
            validate_dynamics_batch(b)

    def test_zero_width_conditions_rejected(self):
        # U=0 is a second "unconditioned" representation and must NOT be
        # accepted; conditions=None is the only unconditioned representation.
        b = _raw(
            torch.randn(2, 11, 3),
            torch.zeros(2, 10, 0),
            torch.ones(2, 10, dtype=torch.bool),
            ["a", "b"],
        )
        with self.assertRaises(ValueError):
            validate_dynamics_batch(b)

    def test_zero_valued_conditions_are_still_conditioned(self):
        # A genuine [B,T,U>=1] all-zero tensor is VALID: the control channel
        # exists but the input is zero. It is a different process from None.
        import numpy as np

        obs = [np.random.randn(11, 3).astype(np.float32) for _ in range(2)]
        act = [np.zeros((10, 2), dtype=np.float32) for _ in range(2)]
        b = dynamics_batch_from_episodes(obs, act)
        self.assertIsNotNone(b.conditions)
        self.assertTrue(b.is_conditioned)
        self.assertEqual(b.cond_dim, 2)
        self.assertTrue(validate_dynamics_batch(b))
        # None, by contrast, is unconditioned.
        b_none = dynamics_batch_from_episodes(obs)
        self.assertIsNone(b_none.conditions)
        self.assertFalse(b_none.is_conditioned)
        self.assertIsNone(b_none.cond_dim)

    def test_empty_transition_rejected(self):
        # Row 0 has 1 transition, row 1 has 0 -> must be rejected.
        b = _raw(
            torch.randn(2, 3, 3),  # T_obs=3 -> T=2 transitions each
            None,
            torch.tensor([[True, True], [False, False]], dtype=torch.bool),
            ["a", "b"],
        )
        with self.assertRaises(ValueError):
            validate_dynamics_batch(b)

    def test_nan_rejected(self):
        obs = torch.randn(2, 11, 3)
        obs[0, 0, 0] = float("nan")
        b = _raw(obs, None, torch.ones(2, 10, dtype=torch.bool), ["a", "b"])
        with self.assertRaises(ValueError):
            validate_dynamics_batch(b)

    def test_inf_rejected(self):
        obs = torch.randn(2, 11, 3)
        obs[1, 3, 1] = float("inf")
        b = _raw(obs, None, torch.ones(2, 10, dtype=torch.bool), ["a", "b"])
        with self.assertRaises(ValueError):
            validate_dynamics_batch(b)


class TestSummarize(unittest.TestCase):
    def test_summary_keys(self):
        import numpy as np

        obs = [np.random.randn(11, 3).astype(np.float32) for _ in range(2)]
        act = [np.random.randn(10, 2).astype(np.float32) for _ in range(2)]
        b = dynamics_batch_from_episodes(obs, act)
        s = summarize_batch(b)
        self.assertEqual(s["batch_size"], 2)
        self.assertEqual(s["num_steps"], 10)
        self.assertEqual(s["obs_dim"], 3)
        self.assertTrue(s["conditioned"])
        self.assertEqual(s["cond_dim"], 2)
        self.assertEqual(s["num_valid_transitions"], 20)
        self.assertEqual(s["unique_episode_ids"], 2)


def _rand(*shape):
    import numpy as np

    return np.random.RandomState(1).randn(*shape).astype(np.float32)


if __name__ == "__main__":
    unittest.main()
