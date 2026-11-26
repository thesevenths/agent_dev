"""
robotics.data.collector
=======================
Rollout collection for Pusher-v5.

:func:`collect_episode` runs one episode under a policy and returns a
:class:`~robotics.data.trajectory.Trajectory` whose transitions are all stored as
copies. :func:`collect_plan` materialises a plan (from
:mod:`robotics.data.splits`) to disk, one ``.npz`` per episode, with a hard
no-overwrite guard unless ``--resume`` is requested.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..envs.pusher_env import ENV_ID, MAX_EPISODE_STEPS, create_pusher_env, detect_contact
from .policies import GuidedPolicy, MixturePolicy, Policy, RandomPolicy
from .trajectory import RobotTransition, Trajectory


def collect_episode(
    env,
    policy: Policy,
    episode_id: str,
    seed: int,
    max_steps: int,
    *,
    record_contact: bool = False,
    policy_rng_seed: Optional[int] = None,
) -> Trajectory:
    """Collect a single episode. ``policy_rng_seed`` drives action noise so the
    episode is reproducible independently of the environment reset seed."""
    rng = np.random.RandomState(policy_rng_seed if policy_rng_seed is not None else int(seed))
    obs, _ = env.reset(seed=int(seed))
    traj = Trajectory(episode_id, int(seed), policy.name, "", ENV_ID)
    for t in range(int(max_steps)):
        action = policy.select_action(env, t, rng)
        next_obs, reward, terminated, truncated, _info = env.step(action)
        contact = detect_contact(env, other_bodies="robot") if record_contact else None
        traj.append(
            RobotTransition(
                obs=np.asarray(obs),
                action=np.asarray(action),
                next_obs=np.asarray(next_obs),
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                episode_id=episode_id,
                timestep=t,
                seed=int(seed),
                contact_state=contact,
            )
        )
        obs = next_obs
        if terminated or truncated:
            break
    return traj


def collect_plan(
    plan: List[dict],
    *,
    max_steps: int,
    output_dir,
    record_contact: bool = False,
    policies: Optional[Dict[str, Policy]] = None,
    resume: bool = False,
) -> List[dict]:
    """Materialise a collection plan to ``<output_dir>/<split>/<episode_id>.npz``.

    Raises FileExistsError on the first already-present file when ``resume`` is
    False (the default), so an accidental re-run never silently clobbers data.
    """
    output_dir = Path(output_dir)
    policies = policies or {
        "random": RandomPolicy(),
        "guided": GuidedPolicy(),
        "mixture": MixturePolicy(),
    }
    written: List[dict] = []
    for item in plan:
        split_dir = output_dir / item["split"]
        fpath = split_dir / f"{item['episode_id']}.npz"
        if fpath.exists():
            if not resume:
                raise FileExistsError(
                    f"refusing to overwrite existing episode {fpath}; "
                    f"pass resume=True to continue an interrupted collection"
                )
            written.append({"episode_id": item["episode_id"], "path": str(fpath), "skipped": True})
            continue
        env = create_pusher_env()
        try:
            traj = collect_episode(
                env,
                policies[item["policy"]],
                item["episode_id"],
                item["reset_seed"],
                max_steps,
                record_contact=record_contact,
                policy_rng_seed=(item["reset_seed"] ^ 0x5BD1E995) & 0x7FFFFFFF,
            )
            traj.split = item["split"]
            traj.save_npz(fpath)
        finally:
            env.close()
        written.append({"episode_id": item["episode_id"], "path": str(fpath), "skipped": False})
    return written
