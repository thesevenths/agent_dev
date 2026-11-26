import unittest

import numpy as np

try:
    import gymnasium  # noqa: F401
    import mujoco  # noqa: F401
    HAVE_MUJOCO = True
except Exception:  # pragma: no cover - exercised only where MuJoCo is absent
    HAVE_MUJOCO = False


def setUpModule():
    if not HAVE_MUJOCO:
        raise unittest.SkipTest("gymnasium[mujoco] not installed; run in the env that provides it")


from robotics.envs.pusher_env import (  # noqa: E402
    ACTION_DIM,
    ACTION_HIGH,
    ACTION_LOW,
    MAX_EPISODE_STEPS,
    OBS_DIM,
    RobotState,
    clip_action,
    create_pusher_env,
    parse_observation,
    verify_observation_contract,
)


class PusherEnvTests(unittest.TestCase):
    def setUp(self):
        self.env = create_pusher_env()

    def tearDown(self):
        self.env.close()

    def test_env_creation(self):
        self.assertIsNotNone(self.env)

    def test_observation_shape(self):
        obs, _ = self.env.reset(seed=0)
        self.assertEqual(obs.shape, (OBS_DIM,))
        self.assertEqual(obs.dtype, np.float64)

    def test_action_shape_and_bounds(self):
        self.assertEqual(self.env.action_space.shape, (ACTION_DIM,))
        self.assertTrue(np.all(self.env.action_space.low == ACTION_LOW))
        self.assertTrue(np.all(self.env.action_space.high == ACTION_HIGH))

    def test_step_basic(self):
        self.env.reset(seed=1)
        nobs, r, term, trunc, _info = self.env.step(np.zeros(ACTION_DIM))
        self.assertEqual(nobs.shape, (OBS_DIM,))
        self.assertIsInstance(float(r), float)
        self.assertFalse(term)  # Pusher never terminates
        self.assertIsInstance(trunc, bool)

    def test_terminated_false_truncated_at_limit(self):
        self.env.reset(seed=2)
        saw_trunc = False
        for _ in range(MAX_EPISODE_STEPS + 5):
            _n, _r, term, trunc, _i = self.env.step(self.env.action_space.sample())
            self.assertFalse(term)
            if trunc:
                saw_trunc = True
                break
        self.assertTrue(saw_trunc)

    def test_observation_field_parsing_matches_model(self):
        # Fails if the installed Gymnasium/MuJoCo version changes the layout.
        verify_observation_contract(self.env)

    def test_parse_roundtrip(self):
        obs, _ = self.env.reset(seed=3)
        st = parse_observation(obs)
        self.assertIsInstance(st, RobotState)
        self.assertEqual(st.qpos.shape, (7,))
        self.assertEqual(st.fingertip.shape, (3,))
        np.testing.assert_allclose(st.to_array(), obs)

    def test_action_clipping(self):
        a = clip_action(np.array([10.0, -10.0, 0.0, 1.0, 2.0, -1.0, 3.0]))
        self.assertTrue(np.all(a >= ACTION_LOW) and np.all(a <= ACTION_HIGH))


if __name__ == "__main__":
    unittest.main()
