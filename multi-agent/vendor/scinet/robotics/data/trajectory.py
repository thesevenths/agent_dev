"""
robotics.data.trajectory
========================
Transition and trajectory containers for Pusher rollouts.

Every NumPy array is *copied* on ingestion. This is the single most important
defence against a classic MuJoCo bug: ``data.qpos`` / ``data.body_com`` return
views into the simulator's memory, so if a transition stored the raw view, an
entire episode's history would silently collapse to the final timestep after the
simulator advanced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

#: Storage dtype. The environment emits float64; we store float32 to match the
#: SciNet++ convention for trajectory tensors and keep files compact. The choice
#: is recorded in the dataset manifest (``obs_dtype``) so it is never implicit.
DEFAULT_DTYPE = np.float32


@dataclass
class RobotTransition:
    obs: np.ndarray
    action: np.ndarray
    next_obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    episode_id: str
    timestep: int
    seed: int
    contact_state: Optional[int] = None

    def __post_init__(self) -> None:
        self.obs = np.ascontiguousarray(self.obs, dtype=DEFAULT_DTYPE)
        self.action = np.ascontiguousarray(self.action, dtype=DEFAULT_DTYPE)
        self.next_obs = np.ascontiguousarray(self.next_obs, dtype=DEFAULT_DTYPE)
        self.reward = float(self.reward)
        self.terminated = bool(self.terminated)
        self.truncated = bool(self.truncated)
        self.timestep = int(self.timestep)
        self.seed = int(self.seed)
        if self.contact_state is not None:
            self.contact_state = int(self.contact_state)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "obs": self.obs,
            "action": self.action,
            "next_obs": self.next_obs,
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "episode_id": self.episode_id,
            "timestep": self.timestep,
            "seed": self.seed,
        }
        if self.contact_state is not None:
            d["contact_state"] = self.contact_state
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RobotTransition":
        return cls(
            obs=np.asarray(d["obs"]),
            action=np.asarray(d["action"]),
            next_obs=np.asarray(d["next_obs"]),
            reward=float(d["reward"]),
            terminated=bool(d["terminated"]),
            truncated=bool(d["truncated"]),
            episode_id=str(d["episode_id"]),
            timestep=int(d["timestep"]),
            seed=int(d["seed"]),
            contact_state=(
                int(d["contact_state"])
                if d.get("contact_state") is not None
                else None
            ),
        )


class Trajectory:
    """An ordered list of :class:`RobotTransition` for one episode."""

    def __init__(
        self,
        episode_id: str,
        seed: int,
        policy: str,
        split: str,
        env_id: str,
        transitions: Optional[List[RobotTransition]] = None,
    ) -> None:
        self.episode_id = episode_id
        self.seed = int(seed)
        self.policy = policy
        self.split = split
        self.env_id = env_id
        self.transitions: List[RobotTransition] = list(transitions or [])

    def __len__(self) -> int:
        return len(self.transitions)

    def append(self, t: RobotTransition) -> None:
        self.transitions.append(t)

    # -- serialization ------------------------------------------------------ #
    def save_npz(self, path) -> None:
        path = Path(path)
        if not self.transitions:
            raise ValueError(f"refusing to save empty trajectory {self.episode_id!r}")
        path.parent.mkdir(parents=True, exist_ok=True)
        obs = np.stack([t.obs for t in self.transitions]).astype(DEFAULT_DTYPE)
        action = np.stack([t.action for t in self.transitions]).astype(DEFAULT_DTYPE)
        next_obs = np.stack([t.next_obs for t in self.transitions]).astype(DEFAULT_DTYPE)
        reward = np.array([t.reward for t in self.transitions], dtype=np.float32)
        terminated = np.array([t.terminated for t in self.transitions], dtype=bool)
        truncated = np.array([t.truncated for t in self.transitions], dtype=bool)
        timestep = np.array([t.timestep for t in self.transitions], dtype=np.int64)
        seed = np.array([t.seed for t in self.transitions], dtype=np.int64)
        has_contact = self.transitions[0].contact_state is not None
        meta = {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "policy": self.policy,
            "split": self.split,
            "env_id": self.env_id,
            "n_transitions": len(self.transitions),
            "has_contact_state": has_contact,
            "obs_dtype": str(np.dtype(DEFAULT_DTYPE)),
        }
        meta_str = np.array(json.dumps(meta, sort_keys=True))
        if has_contact:
            contact = np.array(
                [t.contact_state for t in self.transitions], dtype=np.int64
            )
            np.savez(
                path,
                obs=obs,
                action=action,
                next_obs=next_obs,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                timestep=timestep,
                seed=seed,
                contact_state=contact,
                meta_str=meta_str,
            )
        else:
            np.savez(
                path,
                obs=obs,
                action=action,
                next_obs=next_obs,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                timestep=timestep,
                seed=seed,
                meta_str=meta_str,
            )

    @classmethod
    def load_npz(cls, path) -> "Trajectory":
        path = Path(path)
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta_str"]))
            n = int(meta["n_transitions"])
            obs = z["obs"]
            action = z["action"]
            next_obs = z["next_obs"]
            reward = z["reward"]
            terminated = z["terminated"]
            truncated = z["truncated"]
            timestep = z["timestep"]
            seed = z["seed"]
            contact = z["contact_state"] if "contact_state" in z.files else None
            traj = cls(
                meta["episode_id"],
                meta["seed"],
                meta["policy"],
                meta["split"],
                meta["env_id"],
            )
            for i in range(n):
                traj.append(
                    RobotTransition(
                        obs=obs[i],
                        action=action[i],
                        next_obs=next_obs[i],
                        reward=float(reward[i]),
                        terminated=bool(terminated[i]),
                        truncated=bool(truncated[i]),
                        episode_id=meta["episode_id"],
                        timestep=int(timestep[i]),
                        seed=int(seed[i]),
                        contact_state=(int(contact[i]) if contact is not None else None),
                    )
                )
        return traj
