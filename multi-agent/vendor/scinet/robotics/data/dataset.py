"""
robotics.data.dataset
=====================
Load Pusher trajectories from disk and drive a full collection from a config dict.

The other stages (dynamics model, JEPA, MPC) are intentionally out of scope for
P2-01; this module only produces and reads the raw (obs, action, next_obs,
reward, ...) tuples plus per-episode metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..envs.pusher_env import ENV_ID
from .collector import collect_plan
from .policies import GuidedPolicy, MixturePolicy, Policy, RandomPolicy
from .splits import plan_collection
from .trajectory import RobotTransition, Trajectory


def load_trajectory(path) -> Trajectory:
    return Trajectory.load_npz(path)


def load_split(output_dir, split: str) -> List[Trajectory]:
    d = Path(output_dir) / split
    if not d.is_dir():
        return []
    return [Trajectory.load_npz(f) for f in sorted(d.glob("*.npz"))]


def split_episode_ids(output_dir) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    base = Path(output_dir)
    if not base.is_dir():
        return out
    for sub in sorted(base.iterdir()):
        if sub.is_dir():
            out[sub.name] = sorted(p.stem for p in sub.glob("*.npz"))
    return out


def collect_dataset(
    config: dict,
    output_dir,
    *,
    resume: bool = False,
    policies: Optional[Dict[str, Policy]] = None,
) -> List[dict]:
    """Plan and collect a full dataset from a robotics config dict."""
    policies = policies or {
        "random": RandomPolicy(),
        "guided": GuidedPolicy(),
        "mixture": MixturePolicy(),
    }
    ds = int(config["dataset_seed"])
    policy_counts = dict(config["collection"]["policies"])
    split_ratios = dict(config["collection"]["splits"])
    max_steps = int(config["env"]["max_episode_steps"])
    record_contact = bool(config["collection"].get("record_contact", False))
    plan = plan_collection(policy_counts, split_ratios, ds)
    return collect_plan(
        plan,
        max_steps=max_steps,
        output_dir=output_dir,
        record_contact=record_contact,
        policies=policies,
        resume=resume,
    )


def summarize(output_dir) -> dict:
    """Compute collection statistics used by the run report.

    Returns episode counts, transition counts, per-split policy mix, contact
    ratio (when recorded), and reward / action / object-goal-distance stats.
    """
    out: Dict[str, object] = {
        "splits": {},
        "total_episodes": 0,
        "total_transitions": 0,
        "policies": {},
        "contact_ratio": None,
        "reward": {},
        "action": {},
        "object_goal_distance": {},
    }
    any_contact = False
    contact_hits = 0
    contact_total = 0
    rewards = []
    actions = []
    ogd = []
    for split, ids in split_episode_ids(output_dir).items():
        eps = load_split(output_dir, split)
        n_trans = sum(len(e) for e in eps)
        policy_mix: Dict[str, int] = {}
        for e in eps:
            policy_mix[e.policy] = policy_mix.get(e.policy, 0) + 1
        out["splits"][split] = {  # type: ignore[index]
            "episodes": len(eps),
            "transitions": n_trans,
            "policies": policy_mix,
        }
        out["total_episodes"] += len(eps)  # type: ignore[operator]
        out["total_transitions"] += n_trans  # type: ignore[operator]
        for key in policy_mix:
            out["policies"][key] = out["policies"].get(key, 0) + policy_mix[key]  # type: ignore[index]
        for e in eps:
            for t in e.transitions:
                rewards.append(t.reward)
                actions.append(t.action)
                if t.contact_state is not None:
                    any_contact = True
                    contact_total += 1
                    contact_hits += int(t.contact_state)
                if t.obs.shape[0] == 23:
                    ogd.append(float(np.linalg.norm(t.obs[17:20] - t.obs[20:23])))
    if any_contact:
        out["contact_ratio"] = (contact_hits / contact_total) if contact_total else 0.0
    if rewards:
        r = np.asarray(rewards, dtype=np.float32)
        out["reward"] = {  # type: ignore[index]
            "mean": float(r.mean()),
            "std": float(r.std()),
            "min": float(r.min()),
            "max": float(r.max()),
        }
    if actions:
        a = np.asarray(actions, dtype=np.float32)
        out["action"] = {  # type: ignore[index]
            "mean_abs": float(np.abs(a).mean()),
            "mean": float(a.mean()),
            "std": float(a.std()),
            "min": float(a.min()),
            "max": float(a.max()),
        }
    if ogd:
        g = np.asarray(ogd, dtype=np.float32)
        out["object_goal_distance"] = {  # type: ignore[index]
            "mean": float(g.mean()),
            "std": float(g.std()),
            "min": float(g.min()),
            "max": float(g.max()),
        }
    return out
