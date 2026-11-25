import tempfile
import unittest

import numpy as np
from pathlib import Path

from robotics.data.trajectory import DEFAULT_DTYPE, RobotTransition, Trajectory


def _t(seed=0, contact=None, reward=0.0):
    return RobotTransition(
        obs=np.zeros(23), action=np.zeros(7), next_obs=np.zeros(23),
        reward=reward, terminated=False, truncated=False,
        episode_id="e1", timestep=0, seed=seed, contact_state=contact,
    )


class TrajectoryTests(unittest.TestCase):
    def test_copy_semantics(self):
        raw = np.arange(23, dtype=np.float64)
        t = RobotTransition(
            obs=raw, action=np.zeros(7), next_obs=np.zeros(23),
            reward=1.0, terminated=False, truncated=False,
            episode_id="e", timestep=3, seed=5,
        )
        raw[:] = 999.0  # mutate the caller's buffer
        self.assertNotEqual(float(t.obs[0]), 999.0)  # stored copy is independent
        self.assertEqual(t.obs.dtype, DEFAULT_DTYPE)

    def test_save_load_roundtrip(self):
        traj = Trajectory("ep1", 42, "random", "train", "Pusher-v5")
        for _ in range(3):
            traj.append(_t(seed=42, reward=1.0))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "train" / "ep1.npz"
            traj.save_npz(p)
            loaded = Trajectory.load_npz(p)
            self.assertEqual(len(loaded), 3)
            self.assertEqual(loaded.episode_id, "ep1")
            self.assertEqual(loaded.policy, "random")
            self.assertEqual(loaded.split, "train")
            self.assertEqual(loaded.transitions[0].reward, 1.0)
            self.assertTrue(np.allclose(loaded.transitions[0].obs, 0.0))

    def test_contact_state_optional_roundtrip(self):
        traj = Trajectory("ep2", 1, "guided", "validation", "Pusher-v5")
        traj.append(_t(contact=1))
        traj.append(_t(contact=0))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "validation" / "ep2.npz"
            traj.save_npz(p)
            loaded = Trajectory.load_npz(p)
            self.assertEqual([t.contact_state for t in loaded.transitions], [1, 0])

        traj2 = Trajectory("ep3", 1, "random", "train", "Pusher-v5")
        traj2.append(_t())
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "train" / "ep3.npz"
            traj2.save_npz(p)
            loaded = Trajectory.load_npz(p)
            self.assertIsNone(loaded.transitions[0].contact_state)


if __name__ == "__main__":
    unittest.main()
