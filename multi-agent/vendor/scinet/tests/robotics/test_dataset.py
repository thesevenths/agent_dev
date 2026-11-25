import tempfile
import unittest

import numpy as np
from pathlib import Path

from robotics.data.dataset import load_split, split_episode_ids, summarize
from robotics.data.trajectory import RobotTransition, Trajectory


def _write(root: Path, split: str, n_eps: int, with_contact: bool = False):
    for e in range(n_eps):
        d = root / split
        d.mkdir(parents=True, exist_ok=True)
        eid = f"{split}_random_{e:05d}"
        traj = Trajectory(eid, 1, "random", split, "Pusher-v5")
        for i in range(5):
            traj.append(RobotTransition(
                obs=np.zeros(23), action=np.zeros(7), next_obs=np.zeros(23),
                reward=float(i), terminated=False, truncated=False,
                episode_id=eid, timestep=i, seed=1,
                contact_state=(1 if i % 2 == 0 else 0) if with_contact else None,
            ))
        traj.save_npz(d / f"{eid}.npz")


class DatasetTests(unittest.TestCase):
    def test_load_split_and_summary(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write(root, "train", 3, with_contact=True)
            _write(root, "validation", 1, with_contact=True)
            loaded = load_split(root, "train")
            self.assertEqual(len(loaded), 3)
            self.assertEqual(len(loaded[0]), 5)
            ids = split_episode_ids(root)
            self.assertEqual(
                set(ids["train"]), {f"train_random_{i:05d}" for i in range(3)}
            )
            s = summarize(root)
            self.assertEqual(s["total_episodes"], 4)
            self.assertEqual(s["total_transitions"], 20)
            self.assertIsNotNone(s["contact_ratio"])
            self.assertAlmostEqual(s["reward"]["mean"], 2.0)  # mean of 0..4
            self.assertIn("action", s)
            self.assertIn("object_goal_distance", s)

    def test_split_isolation_no_id_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write(root, "train", 2)
            _write(root, "frozen_test", 2)
            ids = split_episode_ids(root)
            self.assertEqual(set(ids["train"]) & set(ids["frozen_test"]), set())


if __name__ == "__main__":
    unittest.main()
