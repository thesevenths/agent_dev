"""
robotics.data.policies
======================
Episode collection policies for Pusher-v5.

All policies are deterministic given the RNG handed to
:meth:`Policy.select_action`. The collector creates one RNG per episode (seeded
from the episode's reset seed) and passes it in, so a full collection is a pure
function of the configuration + dataset seed.

* ``RandomPolicy``   -- uniform torque noise (baseline coverage).
* ``GuidedPolicy``  -- damped-least-squares end-effector controller that drives
  the fingertip toward the object, then pushes the object toward the goal. Uses
  the real MuJoCo Jacobian (``mj_jacBodyCom``); no inverse-kinematics guesswork.
* ``MixturePolicy`` -- mixes the two by a fixed fraction.
"""

from __future__ import annotations

import numpy as np

from ..envs.pusher_env import (
    ACTION_DIM,
    ACTION_HIGH,
    ACTION_LOW,
    FINGERTIP_BODY,
    clip_action,
    extract_state_from_env,
)

POLICY_VERSIONS = {
    "random": "1.0.0",
    "guided": "1.0.0",
    "mixture": "1.0.0",
}


class Policy:
    name = "base"

    def select_action(self, env, timestep: int, rng: np.random.RandomState) -> np.ndarray:
        raise NotImplementedError


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, noise_scale: float = 1.0) -> None:
        self.noise_scale = float(noise_scale)

    def select_action(self, env, timestep: int, rng: np.random.RandomState) -> np.ndarray:
        a = rng.uniform(ACTION_LOW, ACTION_HIGH, size=ACTION_DIM) * self.noise_scale
        return clip_action(a)


def _actuated_dof_indices(m) -> list:
    """Robustly map actuators to their generalized-coordinate (DOF) indices.

    The Pusher model has more DOFs than actuators (the object and goal bodies add
    slides), so the actuated columns of the Jacobian are NOT a priori the first
    N. We read the mapping from the model instead of guessing.
    """
    import mujoco

    dofs: list = []
    for i in range(m.nu):
        trntype = m.actuator_trntype[i]
        trnid = int(m.actuator_trnid[i, 0])
        if trntype == mujoco.mjtTrn.mjTRN_JOINT:
            dofs.append(int(m.jnt_dofadr[trnid]))
    if dofs:
        return sorted(dofs)
    # Fallback: assume the first ``nu`` DOFs are actuated (matches the verified
    # obs contract where qpos[:7] / qvel[:7] are the arm joints).
    return list(range(m.nu))


class GuidedPolicy(Policy):
    """Damped-least-squares end-effector controller.

    Phase 1: move the fingertip to the object (approach).
    Phase 2: once close, push the object toward the goal.

    The fingertip Jacobian is read from the live MuJoCo model each step; the
    action is a PD torque on the *actuated* joint velocities with a small
    reproducible noise term.
    """

    name = "guided"

    def __init__(
        self,
        kp: float = 2.0,
        kd: float = 0.5,
        lam: float = 0.05,
        noise_scale: float = 0.1,
        approach_threshold: float = 0.05,
    ) -> None:
        self.kp = float(kp)
        self.kd = float(kd)
        self.lam = float(lam)
        self.noise_scale = float(noise_scale)
        self.approach_threshold = float(approach_threshold)

    def select_action(self, env, timestep: int, rng: np.random.RandomState) -> np.ndarray:
        import mujoco

        d = env.unwrapped.data
        m = env.unwrapped.model
        state = extract_state_from_env(env)

        # Where to point the fingertip: at the object while far, at the goal
        # (i.e. push the object) once we are already in contact range.
        if state.fingertip_object_distance > self.approach_threshold:
            target = state.object
        else:
            target = state.goal

        jid = m.body(FINGERTIP_BODY).id
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacBodyCom(m, d, jacp, jacr, jid)

        # Restrict to the actuated DOFs (see _actuated_dof_indices).
        act_dofs = _actuated_dof_indices(m)
        jac_arm = jacp[:, act_dofs]  # (3, nu)

        err = (target - state.fingertip).reshape(3, 1)
        a_mat = jac_arm @ jac_arm.T + self.lam * np.eye(3)
        dq = jac_arm.T @ np.linalg.solve(a_mat, err)  # (nu,)
        dq = np.asarray(dq).reshape(-1)

        qvel_arm = np.asarray(d.qvel)[act_dofs]  # (nu,)
        action = self.kp * dq - self.kd * qvel_arm
        if self.noise_scale:
            action = action + rng.uniform(
                -self.noise_scale, self.noise_scale, size=ACTION_DIM
            )
        return clip_action(action)


class MixturePolicy(Policy):
    name = "mixture"

    def __init__(
        self,
        random_frac: float = 0.5,
        random_policy: Optional[Policy] = None,
        guided_policy: Optional[Policy] = None,
    ) -> None:
        self.random_frac = float(random_frac)
        self._random = random_policy or RandomPolicy()
        self._guided = guided_policy or GuidedPolicy()

    def select_action(self, env, timestep: int, rng: np.random.RandomState) -> np.ndarray:
        if rng.random() < self.random_frac:
            return self._random.select_action(env, timestep, rng)
        return self._guided.select_action(env, timestep, rng)
