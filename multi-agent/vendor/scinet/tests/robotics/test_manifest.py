import tempfile
import unittest

import numpy as np
from pathlib import Path

from robotics.data.manifest import (
    build_manifest,
    dataset_version,
    read_manifest,
    verify_config_match,
    verify_manifest,
    write_manifest,
)
from robotics.data.trajectory import RobotTransition, Trajectory

CONFIG = {
    "env": {
        "id": "Pusher-v5", "obs_dim": 23, "action_dim": 7,
        "action_low": -2.0, "action_high": 2.0, "max_episode_steps": 100,
    },
    "dataset_seed": 20261110,
    "collection": {
        "policies": {"random": 2, "guided": 0},
        "splits": {"train": 0.5, "validation": 0.5, "frozen_test": 0, "ood_test": 0},
        "record_contact": False,
    },
}


def _write_tiny_dataset(root: Path):
    for split, _ in (("train", 1), ("validation", 1)):
        d = root / split
        d.mkdir(parents=True, exist_ok=True)
        eid = f"{split}_random_00000"
        traj = Trajectory(eid, 1, "random", split, "Pusher-v5")
        traj.append(RobotTransition(
            obs=np.zeros(23), action=np.zeros(7), next_obs=np.zeros(23),
            reward=0.0, terminated=False, truncated=False,
            episode_id=eid, timestep=0, seed=1,
        ))
        traj.save_npz(d / f"{eid}.npz")


class ManifestTests(unittest.TestCase):
    def test_build_and_verify(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_tiny_dataset(root)
            m = build_manifest(
                CONFIG, root, gym_version="1.2.1", mujoco_version="3.3.7",
                policy_versions={"random": "1.0.0", "guided": "1.0.0", "mixture": "1.0.0"},
            )
            required = (
                "dataset_version", "environment_id", "gymnasium_version",
                "mujoco_version", "python_version", "observation_dim",
                "action_dim", "action_bounds", "obs_dtype", "collector_version",
                "policy_versions", "dataset_seed", "collection_config_hash",
                "split_episode_ids", "split_file_hashes", "created_at",
            )
            for k in required:
                self.assertIn(k, m)
            self.assertEqual(m["observation_dim"], 23)
            self.assertEqual(m["action_bounds"], [-2.0, 2.0])
            self.assertEqual(m["split_episode_ids"]["train"], ["train_random_00000"])
            verify_manifest(m, root)  # must pass on a clean dataset
            write_manifest(root / "manifest.json", m)
            self.assertEqual(
                read_manifest(root / "manifest.json")["dataset_version"],
                m["dataset_version"],
            )

    def test_tampered_file_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_tiny_dataset(root)
            m = build_manifest(
                CONFIG, root, gym_version="1.2.1", mujoco_version="3.3.7",
                policy_versions={"random": "1.0.0"},
            )
            f = root / "train" / "train_random_00000.npz"
            with open(f, "ab") as fh:
                fh.write(b"\x00\x01")  # corrupt the file
            with self.assertRaises(RuntimeError):
                verify_manifest(m, root)

    def test_config_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_tiny_dataset(root)
            m = build_manifest(
                CONFIG, root, gym_version="1.2.1", mujoco_version="3.3.7",
                policy_versions={"random": "1.0.0"},
            )
            with self.assertRaises(RuntimeError):
                verify_config_match(m, "pusher-wrongversion")

    def test_dataset_version_changes_with_seed(self):
        v1 = dataset_version(CONFIG)
        c2 = dict(CONFIG)
        c2["dataset_seed"] = 999
        self.assertNotEqual(v1, dataset_version(c2))
        self.assertEqual(v1, dataset_version(CONFIG))  # stable


if __name__ == "__main__":
    unittest.main()
