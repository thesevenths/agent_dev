"""
robotics.envs.pusher_env
========================
Adapter for Gymnasium MuJoCo ``Pusher-v5``.

Observation contract (verified against the installed source in
``gymnasium/envs/mujoco/pusher_v5.py`` -- NOT assumed):

    obs[0:7]    qpos       -- 7 joint angles (rad)
    obs[7:14]   qvel       -- 7 joint angular velocities (rad/s)
    obs[14:17]  fingertip  -- ``tips_arm`` body centre-of-mass, world coords
    obs[17:20]  object     -- ``object`` body centre-of-mass, world coords
    obs[20:23]  goal       -- ``goal`` body centre-of-mass, world coords

The environment emits ``float64`` arrays of shape ``(23,)``; actions are
``Box(-2, 2, (7,))``. The episode truncates after 100 steps (``TimeLimit``
wrapper) and never terminates (``terminated`` is always ``False``).

:func:`verify_observation_contract` asserts the slice-based extraction matches
the *live* MuJoCo model, so the layout is tested against the locked Gymnasium /
MuJoCo versions rather than trusted blindly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# --- Dimensions / constants (locked to Pusher-v5) ------------------------- #
OBS_DIM = 23
ACTION_DIM = 7
QPOS_DIM = 7
QVEL_DIM = 7
ACTION_LOW = -2.0
ACTION_HIGH = 2.0
ENV_ID = "Pusher-v5"
FINGERTIP_BODY = "tips_arm"
OBJECT_BODY = "object"
GOAL_BODY = "goal"
MAX_EPISODE_STEPS = 100  # Pusher-v5 TimeLimit truncation


class RobotState:
    """Named, copy-on-construct view of a 23-dim Pusher observation."""

    __slots__ = ("qpos", "qvel", "fingertip", "object", "goal")

    def __init__(self, qpos, qvel, fingertip, object, goal):
        self.qpos = np.ascontiguousarray(qpos, dtype=np.float64)
        self.qvel = np.ascontiguousarray(qvel, dtype=np.float64)
        self.fingertip = np.ascontiguousarray(fingertip, dtype=np.float64)
        self.object = np.ascontiguousarray(object, dtype=np.float64)
        self.goal = np.ascontiguousarray(goal, dtype=np.float64)

    @classmethod
    def from_observation(cls, obs: np.ndarray) -> "RobotState":
        obs = np.asarray(obs, dtype=np.float64)
        if obs.shape != (OBS_DIM,):
            raise ValueError(f"expected obs shape {(OBS_DIM,)}, got {obs.shape}")
        return cls(
            qpos=obs[0:7].copy(),
            qvel=obs[7:14].copy(),
            fingertip=obs[14:17].copy(),
            object=obs[17:20].copy(),
            goal=obs[20:23].copy(),
        )

    def to_array(self) -> np.ndarray:
        return np.concatenate(
            [self.qpos, self.qvel, self.fingertip, self.object, self.goal]
        ).astype(np.float64)

    @property
    def object_goal_distance(self) -> float:
        return float(np.linalg.norm(self.object - self.goal))

    @property
    def fingertip_object_distance(self) -> float:
        return float(np.linalg.norm(self.fingertip - self.object))


def create_pusher_env(*, render_mode=None, **kwargs):
    """Create a Pusher-v5 environment (lazy import keeps this module importable
    even where ``gymnasium[mujoco]`` is not installed)."""
    import gymnasium as gym

    return gym.make(ENV_ID, render_mode=render_mode, **kwargs)


def parse_observation(obs: np.ndarray) -> RobotState:
    """Extract named fields from a raw 23-dim observation."""
    return RobotState.from_observation(obs)


def extract_state_from_env(env) -> RobotState:
    """Authoritative named fields read directly from the live MuJoCo model.

    Used both for contact-aware collection and as the ground truth that
    :func:`verify_observation_contract` checks the slice-based parser against.
    """
    d = env.unwrapped.data
    return RobotState(
        qpos=np.asarray(d.qpos[:QPOS_DIM]).copy(),
        qvel=np.asarray(d.qvel[:QVEL_DIM]).copy(),
        fingertip=np.asarray(env.unwrapped.get_body_com(FINGERTIP_BODY)).copy(),
        object=np.asarray(env.unwrapped.get_body_com(OBJECT_BODY)).copy(),
        goal=np.asarray(env.unwrapped.get_body_com(GOAL_BODY)).copy(),
    )


def verify_observation_contract(env) -> None:
    """Assert slice-based extraction equals the live model.

    Raises AssertionError if the installed Gymnasium / MuJoCo version changes
    the observation layout, so a version bump cannot silently break the field
    mapping.
    """
    obs, _ = env.reset(seed=0)
    parsed = parse_observation(obs)
    live = extract_state_from_env(env)
    for name in ("qpos", "qvel", "fingertip", "object", "goal"):
        a = getattr(parsed, name)
        b = getattr(live, name)
        if not np.allclose(a, b, atol=1e-9):
            raise AssertionError(
                f"observation contract mismatch on '{name}': "
                f"sliced={a.tolist()} live={b.tolist()}"
            )


def detect_contact(env, object_body: str = OBJECT_BODY, other_bodies: Any = None) -> int:
    """Return 1 if the object body is in contact with a *manipulation* body, else 0.

    Uses the genuine MuJoCo contact array (``data.contact`` / ``data.ncon``) and the
    geom->body mapping (``model.geom_bodyid``). Contact is *never* inferred from
    position proximity.

    ``other_bodies`` selects which partners count:
      * ``None``            -> any partner (object in any contact pair)
      * tuple of names     -> only those bodies (e.g. ``("tips_arm",)``)
      * ``"robot"``         -> every body that is not world / object / goal
                              (i.e. the arm is touching the object)

    A string other than ``"robot"`` is treated as a single body name.
    """
    d = env.unwrapped.data
    m = env.unwrapped.model
    if d.ncon == 0:
        return 0
    obj_id = m.body(object_body).id

    if other_bodies is None:
        other_ids = None
    elif other_bodies == "robot":
        other_ids = frozenset(
            i for i in range(m.nbody)
            if i not in (0, obj_id, m.body(GOAL_BODY).id)
        )
    elif isinstance(other_bodies, str):
        other_ids = frozenset({int(m.body(other_bodies).id)})
    else:
        other_ids = frozenset(int(m.body(b).id) for b in other_bodies)

    g1 = d.contact.geom1[: d.ncon]
    g2 = d.contact.geom2[: d.ncon]
    b1 = m.geom_bodyid[g1]
    b2 = m.geom_bodyid[g2]
    for x, y in zip(b1.tolist(), b2.tolist()):
        if obj_id not in (x, y):
            continue
        if other_ids is None or bool(other_ids & {x, y}):
            return 1
    return 0


def clip_action(action: np.ndarray) -> np.ndarray:
    a = np.asarray(action, dtype=np.float64)
    return np.clip(a, ACTION_LOW, ACTION_HIGH)
