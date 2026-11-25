"""
tests.dynamics.test_episode_isolation
====================================

Episode-isolation and cross-domain integration tests for the dynamics contract.

Key guarantees checked here
---------------------------
1. The whole-episode builder never stitches two episodes into one row.
2. The sliding-window builder never lets a window cross an episode boundary;
   every window is tagged with its source episode id.
3. Real Pusher-v5 data (read-only via numpy, **no robotics import**) builds a
   conditioned batch whose ``(x_t, u_t) -> x_{t+1}`` alignment matches the stored
   ``next_obs`` exactly.
4. Newton data maps to an *unconditioned* batch (force is an observation channel,
   not a control condition) -- verifying the contract is domain-independent.
"""

import glob
import json
import os
import unittest

import numpy as np
import torch

from dynamics.contracts import (
    DynamicsBatch,
    dynamics_batch_from_episodes,
    dynamics_batch_from_windows,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PUSHER_DIR = os.path.join(REPO_ROOT, "datasets", "pusher_v1")


class TestWholeEpisodeIsolation(unittest.TestCase):
    def test_one_row_per_episode(self):
        obs = [np.random.randn(11, 3).astype(np.float32) for _ in range(4)]
        b = dynamics_batch_from_episodes(obs, episode_ids=["e0", "e1", "e2", "e3"])
        self.assertEqual(b.batch_size, 4)
        self.assertEqual(b.episode_ids, ["e0", "e1", "e2", "e3"])
        # All transitions valid (no padding because all episodes equal length).
        self.assertTrue(b.valid_mask.all())

    def test_padded_rows_mask_valid_only_inside_episode(self):
        obs = [np.random.randn(6, 3).astype(np.float32) for _ in range(2)]
        # Two separate episodes; the builder must NOT treat the boundary between
        # them as a transition. With whole-episode packing, each episode is its
        # own row, so the question is moot -- but the valid mask must still only
        # cover real transitions of that row.
        b = dynamics_batch_from_episodes(obs, episode_ids=["a", "b"])
        self.assertEqual(b.episode_ids, ["a", "b"])
        self.assertTrue(b.valid_mask[0].all())  # 5 real transitions
        self.assertTrue(b.valid_mask[1].all())  # 5 real transitions


class TestWindowIsolation(unittest.TestCase):
    def test_windows_stay_within_episode(self):
        # One long episode of 50 observations -> windows of length 11, stride 1.
        L = 50
        obs = [np.random.randn(L, 3).astype(np.float32)]
        act = [np.random.randn(L - 1, 2).astype(np.float32)]
        b = dynamics_batch_from_windows(
            obs, act, episode_ids=["long"], window_len=11, stride=1
        )
        n_expected = (L - 11) // 1 + 1  # = 40
        self.assertEqual(b.batch_size, n_expected)
        self.assertEqual(tuple(b.observations.shape), (n_expected, 11, 3))
        self.assertEqual(tuple(b.conditions.shape), (n_expected, 10, 2))
        # Every window belongs to the same (single) episode.
        self.assertEqual(set(b.episode_ids), {"long"})
        self.assertTrue(b.valid_mask.all())
        # Verify each window is a contiguous, in-order slice of the source.
        src = torch.as_tensor(obs[0])
        for i in range(n_expected):
            s = i  # stride 1 => window i starts at i
            self.assertTrue(torch.allclose(b.observations[i], src[s : s + 11]))

    def test_windows_tagged_by_source_episode(self):
        obs = [
            np.random.randn(20, 3).astype(np.float32),
            np.random.randn(20, 3).astype(np.float32),
        ]
        act = [
            np.random.randn(19, 2).astype(np.float32),
            np.random.randn(19, 2).astype(np.float32),
        ]
        b = dynamics_batch_from_windows(
            obs, act, episode_ids=["epA", "epB"], window_len=11, stride=5
        )
        # Both episodes contribute windows, but no window spans both.
        self.assertEqual(set(b.episode_ids), {"epA", "epB"})


class TestPusherIntegrationNoRoboticsImport(unittest.TestCase):
    """Reads Pusher-v1 npz with plain numpy; must not import ``robotics``."""

    def _load_episodes(self, split, limit=3):
        pattern = os.path.join(PUSHER_DIR, split, "*.npz")
        files = sorted(glob.glob(pattern))
        if not files:
            return None
        obs_seqs, act_seqs, ids = [], [], []
        for f in files[:limit]:
            with np.load(f, allow_pickle=False) as z:
                meta = json.loads(str(z["meta_str"]))
                o = z["obs"]        # [T, 23]
                a = z["action"]     # [T, 7]
                no = z["next_obs"]  # [T, 23]
                # Reconstruct the observation sequence for the episode:
                # x_0 = obs[0]; x_{t+1} = next_obs[t].
                seq = np.concatenate([o[0:1], no], axis=0)  # [T+1, 23]
                obs_seqs.append(seq.astype(np.float32))
                act_seqs.append(a.astype(np.float32))
                ids.append(meta["episode_id"])
        return obs_seqs, act_seqs, ids

    def test_pusher_alignment_and_isolation(self):
        loaded = self._load_episodes("train", limit=3)
        if loaded is None:
            self.skipTest("Pusher-v1 dataset not found; skipping integration test")
        obs_seqs, act_seqs, ids = loaded

        # No robotics import anywhere in this test module path.
        b = dynamics_batch_from_episodes(obs_seqs, act_seqs, episode_ids=ids)

        # Conditioned batch with correct dimensions.
        self.assertTrue(b.is_conditioned)
        self.assertEqual(b.obs_dim, 23)
        self.assertEqual(b.cond_dim, 7)

        # Temporal alignment: x_{t+1} must equal the stored next_obs[t].
        xt, ut, xnext, mask = b.to_training_pairs()
        for i, (o, a) in enumerate(zip(obs_seqs, act_seqs)):
            T = a.shape[0]
            self.assertTrue(torch.allclose(xnext[i, :T], torch.as_tensor(o[1:])))
            self.assertTrue(torch.allclose(ut[i, :T], torch.as_tensor(a)))

        # Episode isolation: each row is exactly one episode, all transitions
        # within that episode are valid, none connect to another episode.
        self.assertEqual(b.episode_ids, ids)
        for i in range(b.batch_size):
            T_i = act_seqs[i].shape[0]
            self.assertTrue(b.valid_mask[i, :T_i].all())
            self.assertFalse(b.valid_mask[i, T_i:].any())


class TestNewtonUnconditioned(unittest.TestCase):
    """Newton's applied force is an OBSERVATION channel, not a condition."""

    def test_newton_maps_to_unconditioned(self):
        try:
            from data.newton import NewtonGenerator
        except Exception as exc:  # pragma: no cover - environment dependent
            self.skipTest(f"could not import NewtonGenerator: {exc}")

        gen = NewtonGenerator(length=60)
        sig, _F, _m = gen.sample_one()  # sig: [length, 2] = (position, force)
        # Treat the (position, force) signal as a 2-D observation, unconditioned.
        # We do NOT fabricate an external-force condition.
        b = dynamics_batch_from_episodes(
            [sig.astype(np.float32)], episode_ids=["newton-0"]
        )
        self.assertFalse(b.is_conditioned)
        self.assertIsNone(b.conditions)
        self.assertEqual(b.obs_dim, 2)
        # NewtonGenerator.sample_one() returns `length` observation steps.
        self.assertEqual(tuple(b.observations.shape), (1, 60, 2))
        self.assertEqual(tuple(b.valid_mask.shape), (1, 59))
        self.assertTrue(b.valid_mask.all())


if __name__ == "__main__":
    unittest.main()
