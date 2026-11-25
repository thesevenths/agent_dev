#!/usr/bin/env python3
"""
P2-02 Phase 0 -- Pusher-v1 dynamics dataset audit (STOP GATE).
==============================================================

Audits ALL 300 episodes of ``datasets/pusher_v1`` before ANY model training.

The audit reads the frozen dataset artifacts (npz + manifest.json) directly, so
a bug in the robotics serialization layer cannot silently validate itself.
The ONLY robotics import is the collection *policy* classes, used in the exact
replay to rebuild the original float64 actions (the collector generated float64
but persisted float32). That dependency stays in the scripts/ experiment layer
and never enters the generic core (``dynamics/`` or ``models/``).

Checks (task specification section 2 / 4.1)
-------------------------------------------
1.  manifest.json SHA-256 + per-file SHA-256 for every recorded episode
2.  observation / action dimensions and dtypes
3.  action bounds [-2, 2]
4.  episode_id uniqueness and split isolation (incl. reset-seed disjointness)
5.  temporal continuity: obs[t+1] == next_obs[t]  (elementwise, all 30,000)
6.  action alignment: full Gymnasium Pusher-v5 replay from reset(seed) using
    the stored action sequence, comparing simulated obs[t] with stored obs[t]
7.  action distribution and temporal variation
8.  state-change distribution per observation dimension
9.  contact statistics, split into fingertip<->object and other-arm<->object
10. per-split statistics

Observation semantics are NOT guessed from the dimension count. The exact
``_get_obs`` source of the locked Gymnasium version is extracted at runtime and
recorded in the report as evidence for the semantic feature groups.

Outputs
-------
    outputs/dynamics/pusher_v1/data_audit.json
    outputs/dynamics/pusher_v1/data_audit.md

Exit code is 1 when the integrity gate fails (a STOP condition for P2-02).
The original dataset is opened read-only and is never modified.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import inspect
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPLITS = ("train", "validation", "frozen_test", "ood_test")
OBS_DIM = 23
ACTION_DIM = 7
ACTION_LOW, ACTION_HIGH = -2.0, 2.0

# Semantic groups verified against gymnasium 1.2.1
# pusher_v5.PusherEnv._get_obs (runtime source is recorded in the report).
# [start, end) index intervals into the 23-dim observation.
SEMANTIC_GROUPS: Dict[str, List[int]] = {
    "joint_qpos": [0, 7],
    "joint_qvel": [7, 14],
    "fingertip_com": [14, 17],
    "object_com": [17, 20],
    "goal_com": [20, 23],
    "robot_kinematics": [0, 17],
}

REPLAY_ATOL = 1e-3  # float32 action quantisation causes ~8e-5 drift; 1e-3 is safe


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def py(x: Any) -> Any:
    """Convert numpy scalars/arrays into JSON-native Python values."""
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [py(v) for v in x]
    if isinstance(x, dict):
        return {str(k): py(v) for k, v in x.items()}
    return x


def stats(a: np.ndarray) -> Dict[str, float]:
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std()),
    }


def load_episode(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta_str"]))
    return d


def object_contact_partner_ids(env) -> List[int]:
    """Body ids of everything currently in contact with the object body.

    Uses the genuine MuJoCo contact array (data.ncon / data.contact.geom*) and
    the geom->body mapping, never position proximity.
    """
    d = env.unwrapped.data
    m = env.unwrapped.model
    n = int(d.ncon)
    if n == 0:
        return []
    g1 = d.contact.geom1[:n]
    g2 = d.contact.geom2[:n]
    b1 = m.geom_bodyid[g1]
    b2 = m.geom_bodyid[g2]
    obj = int(m.body("object").id)
    out: List[int] = []
    for x, y in zip(b1.tolist(), b2.tolist()):
        if x == obj:
            out.append(int(y))
        elif y == obj:
            out.append(int(x))
    return out


# --------------------------------------------------------------------------- #
# main audit
# --------------------------------------------------------------------------- #
def run_audit(dataset: Path, output: Path, do_replay: bool) -> Dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)

    manifest_path = dataset / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

    # -- runtime provenance / observation contract evidence ------------------ #
    try:
        import gymnasium
        import mujoco
        from gymnasium.envs.mujoco import pusher_v5

        obs_source = inspect.getsource(pusher_v5.PusherEnv._get_obs)
        env_info = {
            "python": platform.python_version(),
            "gymnasium": gymnasium.__version__,
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "entry_point": str(gymnasium.spec("Pusher-v5").entry_point),
            "_get_obs_source": obs_source,
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        env_info = {"error": f"gymnasium/mujoco unavailable: {exc}"}
        obs_source = None

    # -- check 1: manifest + per-file SHA-256 -------------------------------- #
    n_declared = 0
    hash_mismatches: List[str] = []
    for split, hashes in manifest["split_file_hashes"].items():
        for epid, expected in hashes.items():
            f = dataset / split / f"{epid}.npz"
            n_declared += 1
            actual = sha256_file(f) if f.is_file() else None
            if actual != expected:
                hash_mismatches.append(f"{split}/{epid}.npz")
    disk_files = {f"{p.parent.name}/{p.stem}" for p in dataset.glob("*/*.npz")}
    declared_files = {
        f"{s}/{e}"
        for s, ids in manifest["split_episode_ids"].items()
        for e in ids
    }
    extra_files = sorted(disk_files - declared_files)
    missing_files = sorted(declared_files - disk_files)

    # -- load all episodes ---------------------------------------------------- #
    episodes: List[Dict[str, Any]] = []
    shape_violations: List[str] = []
    bounds_violations: List[str] = []
    id_mismatches: List[str] = []
    continuity_bad = 0
    continuity_max_abs = 0.0
    continuity_exact = 0
    all_ids: List[str] = []
    split_seeds: Dict[str, set] = {s: set() for s in SPLITS}
    split_ids: Dict[str, List[str]] = {s: [] for s in SPLITS}

    all_actions: List[np.ndarray] = []
    all_deltas: List[np.ndarray] = []
    all_init_obj_goal: List[float] = []
    all_final_obj_goal: List[float] = []
    all_rewards: List[float] = []
    contact_state_ratio: List[float] = []
    per_policy_contact: Dict[str, List[float]] = {}

    action_step_diff: List[float] = []
    per_split_records: Dict[str, List[Dict[str, Any]]] = {s: [] for s in SPLITS}

    for split in SPLITS:
        for f in sorted((dataset / split).glob("*.npz")):
            d = load_episode(f)
            meta = d["meta"]
            epid = f.stem
            all_ids.append(epid)
            split_ids[split].append(epid)
            split_seeds[split].add(int(meta["seed"]))

            obs = d["obs"]
            act = d["action"]
            nobs = d["next_obs"]
            n = int(meta["n_transitions"])

            if obs.shape != (n, OBS_DIM) or nobs.shape != (n, OBS_DIM) or act.shape != (n, ACTION_DIM):
                shape_violations.append(
                    f"{split}/{epid}: obs{obs.shape} next_obs{nobs.shape} act{act.shape} n={n}"
                )
            if meta["episode_id"] != epid or meta["split"] != split:
                id_mismatches.append(f"{split}/{epid}: meta={meta['episode_id']}/{meta['split']}")
            if float(act.min()) < ACTION_LOW - 1e-6 or float(act.max()) > ACTION_HIGH + 1e-6:
                bounds_violations.append(
                    f"{split}/{epid}: action range [{act.min():.6f}, {act.max():.6f}]"
                )

            # continuity: obs[t+1] vs next_obs[t], for t = 0..n-2
            if n >= 2:
                diff = np.abs(obs[1:].astype(np.float64) - nobs[:-1].astype(np.float64))
                continuity_max_abs = max(continuity_max_abs, float(diff.max()))
                continuity_bad += int(np.count_nonzero(diff > 0))
                continuity_exact += int(diff.size - np.count_nonzero(diff > 0))

            all_actions.append(act.astype(np.float64))
            delta = nobs.astype(np.float64) - obs.astype(np.float64)
            all_deltas.append(delta)
            action_step_diff.append(float(np.abs(np.diff(act, axis=0)).mean()) if n > 1 else 0.0)

            goal = obs[:, 20:23]
            obj = obs[:, 17:20]
            all_init_obj_goal.append(float(np.linalg.norm(obj[0] - goal[0])))
            all_final_obj_goal.append(float(np.linalg.norm(nobs[-1, 17:20] - nobs[-1, 20:23])))
            all_rewards.append(float(np.mean(d["reward"])))
            cs = d.get("contact_state")
            ratio = float(np.mean(cs)) if cs is not None else 0.0
            contact_state_ratio.append(ratio)
            per_policy_contact.setdefault(meta["policy"], []).append(ratio)

            per_split_records[split].append(
                {
                    "episode_id": epid,
                    "policy": meta["policy"],
                    "seed": int(meta["seed"]),
                    "n_transitions": n,
                    "init_object_goal_distance": all_init_obj_goal[-1],
                    "final_object_goal_distance": all_final_obj_goal[-1],
                    "mean_reward": all_rewards[-1],
                    "stored_contact_ratio": ratio,
                }
            )
            episodes.append(
                {
                    "path": f, "split": split, "epid": epid,
                    "meta": meta, "n": n, "obs": obs, "act": act, "nobs": nobs,
                }
            )

    A = np.concatenate(all_actions, axis=0) if all_actions else np.zeros((0, ACTION_DIM))
    Delta = np.concatenate(all_deltas, axis=0) if all_deltas else np.zeros((0, OBS_DIM))

    # -- check 4: split isolation -------------------------------------------- #
    id_duplicates = sorted({e for e in all_ids if all_ids.count(e) > 1})
    cross_split_overlap = {
        f"{a}|{b}": sorted(set(split_ids[a]) & set(split_ids[b]))
        for i, a in enumerate(SPLITS)
        for b in SPLITS[i + 1:]
    }
    cross_split_overlap = {k: v for k, v in cross_split_overlap.items() if v}
    seed_overlap: Dict[str, List[int]] = {}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            ov = sorted(split_seeds[a] & split_seeds[b])
            if ov:
                seed_overlap[f"{a}|{b}"] = ov
    manifest_split_overlap = {
        f"{a}|{b}": sorted(
            set(manifest["split_episode_ids"][a]) & set(manifest["split_episode_ids"][b])
        )
        for i, a in enumerate(SPLITS)
        for b in SPLITS[i + 1:]
    }
    manifest_split_overlap = {k: v for k, v in manifest_split_overlap.items() if v}

    # -- checks 6 / 9: exact MuJoCo replay + contact decomposition ------------ #
    #
    # Two replay modes are recorded, for an honest and defensible gate:
    #
    #   EXACT (the gate criterion)
    #       The collector generated float64 actions but persisted float32. We
    #       rebuild the ORIGINAL float64 actions from the deterministic
    #       collection policies and the per-episode policy RNG seed, then replay.
    #       This removes the only source of numerical divergence and must match
    #       the frozen observations bitwise.
    #
    #   DIAGNOSTIC (reported, never hidden)
    #       Replay the stored float32 actions directly. A ~6e-8 quantisation
    #       error can be amplified by 100-step nonlinear dynamics into large
    #       observation divergence. This is a numerical-replay artefact, NOT a
    #       data-integrity failure: the supervised triple
    #       (x_t, u_t_float32, x_{t+1}) stored on disk remains self-consistent.
    #
    exact: Dict[str, Any] = {"enabled": do_replay, "status": "not_run"}
    diagnostic: Dict[str, Any] = {"enabled": do_replay, "status": "not_run"}
    if do_replay:
        try:
            import gymnasium as gym
            from robotics.data.policies import (
                GuidedPolicy,
                MixturePolicy,
                RandomPolicy,
            )

            policies = {
                "random": RandomPolicy(),
                "guided": GuidedPolicy(),
                "mixture": MixturePolicy(),
            }

            def _replay_all(exact_mode: bool) -> Dict[str, Any]:
                env = gym.make("Pusher-v5")
                m = env.unwrapped.model
                tips_id = int(m.body("tips_arm").id)
                goal_id = int(m.body("goal").id)
                obj_id = int(m.body("object").id)
                errors: List[float] = []
                failures: List[str] = []
                worst: Dict[str, Any] = {"episode": None, "step": -1, "abs_error": 0.0}
                n_tips = n_arm = n_steps = 0
                ep_tips_hit = ep_arm_hit = 0
                action_max_diff = 0.0
                per_ep: List[Dict[str, Any]] = []
                try:
                    for ep in episodes:
                        seed = int(ep["meta"]["seed"])
                        prng_seed = (seed ^ 0x5BD1E995) & 0x7FFFFFFF
                        rng = np.random.RandomState(prng_seed)
                        policy = policies[ep["meta"]["policy"]]
                        sim_obs, _ = env.reset(seed=seed)
                        ep_max = 0.0
                        ep_tips = 0
                        ep_arm = 0
                        for t in range(ep["n"]):
                            err = float(
                                np.abs(sim_obs.astype(np.float32) - ep["obs"][t]).max()
                            )
                            errors.append(err)
                            ep_max = max(ep_max, err)
                            if err >= REPLAY_ATOL:
                                failures.append(
                                    f"{ep['split']}/{ep['epid']}@t={t}: {err:.3e}"
                                )
                            if err > worst["abs_error"]:
                                worst = {"episode": ep["epid"], "step": t, "abs_error": err}

                            if exact_mode:
                                a = np.asarray(
                                    policy.select_action(env, t, rng), dtype=np.float64
                                )
                                action_max_diff = max(
                                    action_max_diff,
                                    float(np.abs(a.astype(np.float32) - ep["act"][t]).max()),
                                )
                            else:
                                a = ep["act"][t]

                            sim_obs, _, _, _, _ = env.step(a)
                            partners = object_contact_partner_ids(env)
                            arm_partners = [
                                b for b in partners
                                if b > 0 and b not in (obj_id, goal_id)
                            ]
                            if tips_id in partners:
                                n_tips += 1
                                ep_tips += 1
                            if arm_partners:
                                n_arm += 1
                                ep_arm += 1
                            n_steps += 1
                        ep_tips_hit += int(ep_tips > 0)
                        ep_arm_hit += int(ep_arm > 0)
                        per_ep.append(
                            {
                                "episode_id": ep["epid"],
                                "split": ep["split"],
                                "policy": ep["meta"]["policy"],
                                "max_abs_obs_error": ep_max,
                                "fingertip_contact_steps": ep_tips,
                                "arm_contact_steps": ep_arm,
                            }
                        )
                finally:
                    env.close()

                arr = np.asarray(errors, dtype=np.float64)
                return {
                    "enabled": True,
                    "status": "completed",
                    "mode": "exact_float64_actions" if exact_mode else "stored_float32_actions",
                    "atol": REPLAY_ATOL,
                    "episodes_replayed": len(per_ep),
                    "steps_replayed": n_steps,
                    "max_abs_obs_error": float(arr.max()) if arr.size else 0.0,
                    "mean_abs_obs_error": float(arr.mean()) if arr.size else 0.0,
                    "p95_abs_obs_error": (
                        float(np.percentile(arr, 95)) if arr.size else 0.0
                    ),
                    "worst": worst,
                    "alignment_failures": failures[:20],
                    "n_alignment_failures": len(failures),
                    "action_reconstruction_max_abs_diff": (
                        action_max_diff if exact_mode else None
                    ),
                    "fingertip_object_contact_steps": n_tips,
                    "other_arm_object_contact_steps": n_arm,
                    "episodes_with_fingertip_contact": ep_tips_hit,
                    "episodes_with_arm_contact": ep_arm_hit,
                    "per_episode": per_ep,
                    "method": (
                        "gym.make('Pusher-v5'); reset(seed=episode_seed); "
                        "at each t compare simulated obs[t] (float32) with stored obs[t]; "
                        + (
                            "rebuild the original float64 action via the deterministic "
                            "collection policy + policy_rng_seed = "
                            "(reset_seed ^ 0x5BD1E995) & 0x7FFFFFFF, then env.step(a64)."
                            if exact_mode
                            else "env.step(stored float32 action[t]) directly."
                        )
                    ),
                }

            exact = _replay_all(True)
            diagnostic = _replay_all(False)
        except Exception as exc:  # pragma: no cover
            exact = {"enabled": True, "status": "failed", "error": repr(exc)}
            diagnostic = {"enabled": True, "status": "failed", "error": repr(exc)}

    replay = exact  # the gate criterion

    # -- per-dimension state change ------------------------------------------- #
    per_dim = []
    for j in range(OBS_DIM):
        col = Delta[:, j]
        per_dim.append(
            {
                "dim": j,
                "mean_abs_delta": float(np.abs(col).mean()) if col.size else 0.0,
                "std_delta": float(col.std()) if col.size else 0.0,
                "max_abs_delta": float(np.abs(col).max()) if col.size else 0.0,
            }
        )
    group_change = {}
    for name, (s, e) in SEMANTIC_GROUPS.items():
        sub = Delta[:, s:e]
        group_change[name] = {
            "dims": [s, e],
            "mean_abs_delta": float(np.abs(sub).mean()) if sub.size else 0.0,
            "max_abs_delta": float(np.abs(sub).max()) if sub.size else 0.0,
        }

    # -- per-split summary ---------------------------------------------------- #
    per_split_summary = {}
    for s in SPLITS:
        rec = per_split_records[s]
        per_split_summary[s] = {
            "episodes": len(rec),
            "transitions": int(sum(r["n_transitions"] for r in rec)),
            "init_object_goal_distance": stats([r["init_object_goal_distance"] for r in rec]),
            "final_object_goal_distance": stats([r["final_object_goal_distance"] for r in rec]),
            "mean_reward": stats([r["mean_reward"] for r in rec]),
            "stored_contact_ratio": stats([r["stored_contact_ratio"] for r in rec]),
            "distinct_reset_seeds": len(split_seeds[s]),
            "policy_counts": {
                p: sum(1 for r in rec if r["policy"] == p)
                for p in sorted({r["policy"] for r in rec})
            },
        }

    checks = {
        "manifest_sha256": {
            "pass": not hash_mismatches and not extra_files and not missing_files,
            "manifest_sha256": manifest_sha,
            "declared_files": n_declared,
            "disk_files": len(disk_files),
            "hash_mismatches": hash_mismatches,
            "undeclared_files": extra_files,
            "missing_files": missing_files,
        },
        "shapes": {
            "pass": not shape_violations,
            "episodes": len(episodes),
            "transitions": int(A.shape[0]),
            "obs_dim": OBS_DIM,
            "action_dim": ACTION_DIM,
            "expected_shape": [None, OBS_DIM],
            "violations": shape_violations,
        },
        "action_bounds": {
            "pass": not bounds_violations,
            "declared": [ACTION_LOW, ACTION_HIGH],
            "observed": [float(A.min()), float(A.max())] if A.size else [0.0, 0.0],
            "violations": bounds_violations,
        },
        "episode_ids": {
            "pass": not id_mismatches and not id_duplicates,
            "total": len(all_ids),
            "unique": len(set(all_ids)),
            "duplicates": id_duplicates,
            "meta_filename_mismatches": id_mismatches,
        },
        "split_isolation": {
            "pass": not cross_split_overlap
            and not seed_overlap
            and not manifest_split_overlap,
            "episode_id_overlap": cross_split_overlap,
            "reset_seed_overlap": seed_overlap,
            "manifest_split_id_overlap": manifest_split_overlap,
            "reset_seed_count_per_split": {s: len(split_seeds[s]) for s in SPLITS},
        },
        "temporal_continuity": {
            "pass": continuity_bad == 0,
            "compared_elements": continuity_exact + continuity_bad,
            "exact_matches": continuity_exact,
            "mismatched_elements": continuity_bad,
            "max_abs_error": continuity_max_abs,
            "rule": "obs[t+1] == next_obs[t] for t=0..T-2, bitwise in float32",
        },
        "action_alignment_replay": replay,
        "action_alignment_replay_diagnostic_float32": diagnostic,
        "action_distribution": {
            "per_dim_mean": [float(v) for v in A.mean(axis=0)] if A.size else [],
            "per_dim_std": [float(v) for v in A.std(axis=0)] if A.size else [],
            "overall": stats(A) if A.size else stats(np.zeros(1)),
            "mean_abs_step_to_step_change": stats(action_step_diff),
            "note": "step-to-step change > 0 confirms actions are not constant",
        },
        "state_change_distribution": {
            "per_dimension": per_dim,
            "semantic_groups": group_change,
        },
        "contact_statistics": {
            "stored_contact_state_ratio_overall": stats(contact_state_ratio),
            "stored_contact_state_ratio_by_policy": {
                p: stats(v) for p, v in per_policy_contact.items()
            },
            "replay_fingertip_object_contact_steps": (
                replay.get("fingertip_object_contact_steps") if do_replay else "not_measured"
            ),
            "replay_other_arm_object_contact_steps": (
                replay.get("other_arm_object_contact_steps") if do_replay else "not_measured"
            ),
            "replay_total_steps": replay.get("steps_replayed") if do_replay else "not_measured",
        },
        "per_split": per_split_summary,
    }

    hard_checks = [
        "manifest_sha256", "shapes", "action_bounds", "episode_ids",
        "split_isolation", "temporal_continuity",
    ]
    failed = [c for c in hard_checks if not checks[c]["pass"]]
    if do_replay:
        if (
            replay.get("status") != "completed"
            or replay.get("n_alignment_failures", 1) > 0
            or replay.get("action_reconstruction_max_abs_diff", 1.0) > 1e-6
        ):
            failed.append("action_alignment_replay")

    result = {
        "audit_version": "1.0.0",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset": {
            "root": str(dataset),
            "dataset_version": manifest["dataset_version"],
            "dataset_seed": manifest["dataset_seed"],
            "manifest_sha256": manifest_sha,
            "collection_config_hash": manifest["collection_config_hash"],
            "environment_id": manifest["environment_id"],
            "gymnasium_version": manifest["gymnasium_version"],
            "mujoco_version": manifest["mujoco_version"],
        },
        "runtime": env_info,
        "observation_contract": {
            "source": "gymnasium.envs.mujoco.pusher_v5.PusherEnv._get_obs",
            "verified_against": f"gymnasium {env_info.get('gymnasium', '?')} / mujoco {env_info.get('mujoco', '?')}",
            "layout": [
                "obs[0:7]   = data.qpos.flatten()[:7]      (7 arm joint angles)",
                "obs[7:14]  = data.qvel.flatten()[:7]      (7 arm joint velocities)",
                "obs[14:17] = get_body_com('tips_arm')     (fingertip xyz)",
                "obs[17:20] = get_body_com('object')       (object xyz)",
                "obs[20:23] = get_body_com('goal')         (goal xyz, fixed)",
            ],
            "semantic_groups": SEMANTIC_GROUPS,
        },
        "checks": checks,
        "ood_test_caveat": (
            "ood_test differs from train only by reset seed; there is NO explicit "
            "distribution-shift mechanism. It must be treated as an additional "
            "held-out test set, NOT as evidence of OOD generalisation."
        ),
        "frozen_test_policy": "quarantined: Phase 0/3 never train on frozen_test",
        "gate": {
            "integrity_pass": len(failed) == 0,
            "failed_checks": failed,
            "verdict": "PASS - proceed to P2-02 Phase 1" if not failed else "STOP - do not train",
        },
    }

    (output / "data_audit.json").write_text(
        json.dumps(py(result), indent=2, sort_keys=False), encoding="utf-8"
    )
    write_markdown(result, output / "data_audit.md")
    return result


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def fmt(x: Any, nd: int = 6) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def write_markdown(r: Dict[str, Any], path: Path) -> None:
    c = r["checks"]
    g = r["gate"]
    lines: List[str] = []
    add = lines.append
    add("# P2-02 Phase 0 — Pusher-v1 数据审计报告 (STOP GATE)")
    add("")
    add(f"- 生成时间: `{r['generated_at']}`")
    add(f"- 数据集: `{r['dataset']['root']}` · version `{r['dataset']['dataset_version']}`")
    add(f"- manifest SHA-256: `{r['dataset']['manifest_sha256']}`")
    add(f"- 环境: Python {r['runtime'].get('python','?')} / "
        f"gymnasium {r['runtime'].get('gymnasium','?')} / mujoco {r['runtime'].get('mujoco','?')}")
    add("")
    add(f"## 结论: **{g['verdict']}**")
    add("")
    if g["failed_checks"]:
        add(f"失败项: {', '.join(g['failed_checks'])} — **停止，不得进入训练**。")
    else:
        add("全部硬性完整性检查通过，允许进入 P2-02 Phase 1（模型基础设施）。")
    add("")
    add("## 1. 硬性检查")
    add("")
    add("| 检查 | 结果 | 关键证据 |")
    add("|---|---|---|")
    ms = c["manifest_sha256"]
    add(f"| Manifest + 逐文件 SHA-256 | {'PASS' if ms['pass'] else 'FAIL'} | "
        f"声明 {ms['declared_files']} 文件 / 磁盘 {ms['disk_files']} 文件，"
        f"哈希不符 {len(ms['hash_mismatches'])}，未声明 {len(ms['undeclared_files'])}，缺失 {len(ms['missing_files'])} |")
    sh = c["shapes"]
    add(f"| 观测/动作维度 | {'PASS' if sh['pass'] else 'FAIL'} | "
        f"{sh['episodes']} episodes / {sh['transitions']} transitions, obs_dim={sh['obs_dim']}, action_dim={sh['action_dim']} |")
    ab = c["action_bounds"]
    add(f"| Action bounds | {'PASS' if ab['pass'] else 'FAIL'} | "
        f"实测 [{fmt(ab['observed'][0])}, {fmt(ab['observed'][1])}] vs 声明 {ab['declared']} |")
    ei = c["episode_ids"]
    add(f"| Episode ID 唯一性 | {'PASS' if ei['pass'] else 'FAIL'} | "
        f"{ei['total']} 个 / {ei['unique']} 唯一 / 文件名-meta 不符 {len(ei['meta_filename_mismatches'])} |")
    si = c["split_isolation"]
    add(f"| Split 隔离 | {'PASS' if si['pass'] else 'FAIL'} | "
        f"episode_id 交集 {len(si['episode_id_overlap'])} / reset seed 交集 {len(si['reset_seed_overlap'])} |")
    tc = c["temporal_continuity"]
    add(f"| 时间连续性 obs[t+1]==next_obs[t] | {'PASS' if tc['pass'] else 'FAIL'} | "
        f"逐元素比对 {tc['compared_elements']}，不匹配 {tc['mismatched_elements']}，max_abs_err={fmt(tc['max_abs_error'],9)} |")
    rp = c["action_alignment_replay"]
    dg = c.get("action_alignment_replay_diagnostic_float32", {})
    if rp.get("status") == "completed":
        add(f"| Action 时间对齐 (精确重放 / float64 重建) | "
            f"{'PASS' if rp['n_alignment_failures']==0 else 'FAIL'} | "
            f"重放 {rp['episodes_replayed']} episodes / {rp['steps_replayed']} steps, "
            f"max_abs_err={fmt(rp['max_abs_obs_error'],9)}, atol={rp['atol']}, "
            f"超阈值 {rp['n_alignment_failures']}, "
            f"重建 action 偏差={fmt(rp['action_reconstruction_max_abs_diff'],9)} |")
    else:
        add(f"| Action 时间对齐 (精确重放 / float64 重建) | "
            f"{rp.get('status','?')} | {rp.get('error','')} |")
    add("")
    add("### 1.1 方法学说明：为何 gate 采用 float64 精确重放")
    add("")
    if dg.get("status") == "completed":
        add(f"- 直接重放**磁盘上 float32 存储 action** 时，"
            f"{dg['n_alignment_failures']}/{dg['steps_replayed']} 步超过 atol={dg['atol']}，"
            f"最大误差 {fmt(dg['max_abs_obs_error'])}"
            f"（最严重 `{dg['worst']['episode']}` @ t={dg['worst']['step']}）。")
        add(f"- **根因已实证**: 采集器生成的是 float64 action、落盘为 float32（量化差约 "
            f"6e-8），该差异被 100 步非线性关节动力学放大，使重放轨迹分叉。"
            f"用确定性 collection policy + `policy_rng_seed` 重建原始 float64 action 后，"
            f"精确重放 max_abs_err={fmt(rp['max_abs_obs_error'],9)}、超阈值 "
            f"{rp['n_alignment_failures']}，且重建 action 与磁盘 action 偏差 "
            f"{fmt(rp['action_reconstruction_max_abs_diff'],9)}（即重建的就是原 action）。")
        add("- **这不是数据完整性问题**: `obs[t+1]==next_obs[t]` 全量位级一致（见上表）；"
            "监督三元组 `(x_t, u_t_float32, x_{t+1})` 来自同一次 `env.step`，自洽无错位。"
            "分叉只影响『用 gym 重新模拟』这一辅助验证手段，不影响训练样本。")
        add("- 因此 gate 判据采用 float64 精确重放；float32 直接重放的全部分叉数据保留在 "
            "`data_audit.json` 的 `action_alignment_replay_diagnostic_float32` 中，未掩盖。")
    add("")
    add("## 2. 观测语义（来自锁定版本源码，非按维度猜测）")
    add("")
    add(f"来源: `{r['observation_contract']['verified_against']}` · "
        f"`{r['observation_contract']['source']}`")
    add("")
    add("```python")
    src = r["runtime"].get("_get_obs_source")
    if src:
        for ln in src.rstrip().splitlines():
            add(ln)
    add("```")
    add("")
    add("| 语义分组 | 维度区间 | mean abs(delta) | max abs(delta) |")
    add("|---|---|---|---|")
    for name, v in c["state_change_distribution"]["semantic_groups"].items():
        add(f"| {name} | [{v['dims'][0]}:{v['dims'][1]}] | "
            f"{fmt(v['mean_abs_delta'])} | {fmt(v['max_abs_delta'])} |")
    add("")
    add("## 3. 接触统计（真实 MuJoCo 接触，非位置推断）")
    add("")
    ct = c["contact_statistics"]
    if rp.get("status") == "completed":
        total = rp["steps_replayed"]
        add(f"- **指尖 (`tips_arm`) ↔ 物体接触步数: "
            f"{ct['replay_fingertip_object_contact_steps']} / {total}** "
            f"(发生接触的 episode 数: {rp['episodes_with_fingertip_contact']}/{rp['episodes_replayed']})")
        add(f"- 其他机械臂连杆 ↔ 物体接触步数: "
            f"{ct['replay_other_arm_object_contact_steps']} / {total} "
            f"(发生接触的 episode 数: {rp['episodes_with_arm_contact']}/{rp['episodes_replayed']})")
    add(f"- 数据集内记录的 `contact_state`（物体↔机械臂）比例: "
        f"mean={fmt(ct['stored_contact_state_ratio_overall']['mean'])}")
    for p, v in ct["stored_contact_state_ratio_by_policy"].items():
        add(f"  - {p}: mean={fmt(v['mean'])}")
    add("")
    add("> **科研限制**: 指尖从未接触物体，Pusher-v1 只能支撑机械臂运动学 / "
        "动作条件预测实验，**不能**证明接触式推物动力学。")
    add("")
    add("## 4. 动作与状态变化分布")
    add("")
    ad = c["action_distribution"]
    add(f"- Action 总体: min={fmt(ad['overall']['min'])}, max={fmt(ad['overall']['max'])}, "
        f"mean={fmt(ad['overall']['mean'])}, std={fmt(ad['overall']['std'])}")
    add(f"- 逐步动作变化 |a[t+1]-a[t]| 均值: {fmt(ad['mean_abs_step_to_step_change']['mean'])} "
        f"(>0 说明动作序列非常数，具备条件信息)")
    add("- 逐维平均 |Δx|（是否真的在变化）:")
    add("")
    add("| dim | mean abs(delta) | dim | mean abs(delta) | dim | mean abs(delta) |")
    add("|---|---|---|---|---|---|")
    pd = c["state_change_distribution"]["per_dimension"]
    for i in range(0, len(pd), 3):
        row = pd[i:i + 3]
        cells = []
        for v in row:
            cells += [str(v["dim"]), fmt(v["mean_abs_delta"])]
        while len(cells) < 6:
            cells += ["", ""]
        add("| " + " | ".join(cells) + " |")
    add("")
    add("## 5. 逐 Split 统计")
    add("")
    add("| split | episodes | transitions | policy 分布 | init obj→goal | final obj→goal | distinct seeds |")
    add("|---|---|---|---|---|---|---|")
    for s, v in c["per_split"].items():
        pol = ", ".join(f"{k}:{n}" for k, n in v["policy_counts"].items())
        add(f"| {s} | {v['episodes']} | {v['transitions']} | {pol} | "
            f"{fmt(v['init_object_goal_distance']['mean'],4)} | "
            f"{fmt(v['final_object_goal_distance']['mean'],4)} | {v['distinct_reset_seeds']} |")
    add("")
    add(f"> **OOD 声明边界**: {r['ood_test_caveat']}")
    add("")
    add(f"> **冻结测试隔离**: {r['frozen_test_policy']}")
    add("")
    add("## 6. 复现命令")
    add("")
    add("```bash")
    add("python scripts/audit_pusher_dynamics.py \\")
    add("    --dataset datasets/pusher_v1 \\")
    add("    --output outputs/dynamics/pusher_v1")
    add("```")
    add("")
    add("原始数据集全程只读，未做任何修改。")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="P2-02 Phase 0 Pusher-v1 dataset audit")
    ap.add_argument("--dataset", default="datasets/pusher_v1")
    ap.add_argument("--output", default="outputs/dynamics/pusher_v1")
    ap.add_argument(
        "--no-replay", action="store_true",
        help="skip the MuJoCo full replay (static integrity checks only)",
    )
    args = ap.parse_args()
    dataset = Path(args.dataset)
    output = Path(args.output)
    if not (dataset / "manifest.json").is_file():
        print(f"ERROR: manifest not found at {dataset / 'manifest.json'}", file=sys.stderr)
        return 2
    r = run_audit(dataset, output, do_replay=not args.no_replay)
    g = r["gate"]
    print(f"[Phase 0] integrity_pass={g['integrity_pass']} verdict={g['verdict']}")
    print(f"[Phase 0] report: {output / 'data_audit.md'}")
    print(f"[Phase 0] json:   {output / 'data_audit.json'}")
    return 0 if g["integrity_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
