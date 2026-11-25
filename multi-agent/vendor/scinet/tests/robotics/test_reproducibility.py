import tempfile
import unittest

import numpy as np
from pathlib import Path

try:
    import gymnasium  # noqa: F401
    import mujoco  # noqa: F401
    HAVE_MUJOCO = True
except Exception:  # pragma: no cover
    HAVE_MUJOCO = False


def setUpModule():
    if not HAVE_MUJOCO:
        raise unittest.SkipTest("gymnasium[mujoco] not installed; run in the env that provides it")


from robotics.data.dataset import collect_dataset  # noqa: E402
from robotics.data.manifest import dataset_version  # noqa: E402
from robotics.data.policies import RandomPolicy  # noqa: E402
from robotics.data.splits import plan_collection  # noqa: E402
from robotics.envs.pusher_env import ACTION_DIM, create_pusher_env  # noqa: E402

SMALL_CONFIG = {
    "env": {
        "id": "Pusher-v5", "obs_dim": 23, "action_dim": 7,
        "action_low": -2.0, "action_high": 2.0, "max_episode_steps": 20,
    },
    "dataset_seed": 20261110,
    "collection": {
        "policies": {"random": 2, "guided": 0},
        "splits": {"train": 0.5, "validation": 0.5, "frozen_test": 0, "ood_test": 0},
        "record_contact": False,
    },
    "output_dir": "unused",
}


class ReproTests(unittest.TestCase):
    def test_reset_seed_reproducibility(self):
        env = create_pusher_env()
        try:
            o1, _ = env.reset(seed=42)
            o2, _ = env.reset(seed=42)
            np.testing.assert_array_equal(o1, o2)
            o3, _ = env.reset(seed=43)
            self.assertFalse(np.array_equal(o1, o3))  # different seed -> different init
        finally:
            env.close()

    def test_same_config_same_plan(self):
        p1 = plan_collection(
            SMALL_CONFIG["collection"]["policies"],
            SMALL_CONFIG["collection"]["splits"],
            SMALL_CONFIG["dataset_seed"],
        )
        p2 = plan_collection(
            SMALL_CONFIG["collection"]["policies"],
            SMALL_CONFIG["collection"]["splits"],
            SMALL_CONFIG["dataset_seed"],
        )
        self.assertEqual(
            [(x["episode_id"], x["reset_seed"]) for x in p1],
            [(x["episode_id"], x["reset_seed"]) for x in p2],
        )

    def test_policy_determinism(self):
        env = create_pusher_env()
        try:
            env.reset(seed=5)
            p = RandomPolicy()
            r1 = np.random.RandomState(0)
            r2 = np.random.RandomState(0)
            a1 = p.select_action(env, 0, r1)
            a2 = p.select_action(env, 0, r2)
            np.testing.assert_allclose(a1, a2)
        finally:
            env.close()

    def test_existing_dataset_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = dict(SMALL_CONFIG)
            cfg["output_dir"] = d
            collect_dataset(cfg, d)
            with self.assertRaises(FileExistsError):
                collect_dataset(cfg, d, resume=False)
            written = collect_dataset(cfg, d, resume=True)
            self.assertTrue(all(w["skipped"] for w in written))

    def test_same_config_same_dataset_identity(self):
        v1 = dataset_version(SMALL_CONFIG)
        self.assertEqual(v1, dataset_version(SMALL_CONFIG))
        cfg2 = dict(SMALL_CONFIG)
        cfg2["dataset_seed"] = 1
        self.assertNotEqual(v1, dataset_version(cfg2))


if __name__ == "__main__":
    unittest.main()
