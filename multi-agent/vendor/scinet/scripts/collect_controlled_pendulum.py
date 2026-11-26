"""
scripts/collect_controlled_pendulum.py
======================================
Collect the P1-A Controlled-Pendulum pilot dataset.

    python scripts/collect_controlled_pendulum.py \
        --config configs/dynamics/controlled_pendulum_jepa.yaml

Produces EXACTLY ``n_scenarios * n_profiles`` episodes (20 for the default
config): 5 physical scenarios x 4 action profiles. Each episode is a
deterministic RK4 roll-out of the torque-controlled pendulum; the per-episode
seed is derived purely from the config so the whole pilot is byte-reproducible.

Outputs (under ``experiment.output_root``)
------------------------------------------
  pilot_episodes.npz   -- float64 tensors
                          observations [B, T+1, 2]   (theta_unwrapped, omega)
                          conditions   [B, T,   1]   (torque)
                          valid_mask   [B, T]        bool
                          initial_state[B, 2]        float64
                          scenario_index[B]          int64
                          profile_index [B]          int64
                          meta_str      (JSON string with full metadata)
  manifest.json        -- content-addressed dataset id, SHA-256 of the npz,
                          per-episode records (seed, scenario, profile, stats),
                          and integrity summary.

Refuses to overwrite an existing dataset unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import yaml


# --------------------------------------------------------------------------- #
# Reproducible identity helpers
# --------------------------------------------------------------------------- #
def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_spec(cfg: dict) -> dict:
    """Canonical, data-affecting snapshot of the config (excludes output paths)."""
    sim = cfg["simulator"]
    scenarios = [
        {k: sc.get(k) for k in ("name", "g", "L", "m", "b", "theta0", "omega0")}
        for sc in cfg["scenarios"]
    ]
    default_tau = float(sim.get("action_max_torque", 5.0))
    profiles = []
    for pr in cfg["action_profiles"]:
        profiles.append({
            "name": pr["name"],
            "type": pr["type"],
            "torque_max": float(pr.get("torque_max", default_tau)),
            "fraction": pr.get("fraction", 0.5),
            "magnitude": pr.get("magnitude", 0.6),
            "amplitude": pr.get("amplitude", 0.7),
            "frequency_hz": pr.get("frequency_hz", 0.5),
            "phase": pr.get("phase", 0.0),
        })
    return {
        "generator": "controlled_pendulum",
        "dt": float(sim["dt"]),
        "horizon_steps": int(sim["horizon_steps"]),
        "action_max_torque": default_tau,
        "master_seed": int(cfg["experiment"]["seed"]),
        "scenarios": scenarios,
        "profiles": profiles,
    }


def dataset_version(cfg: dict) -> str:
    return f"cpilot-{_sha_text(_canonical(dataset_spec(cfg)))[:12]}"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Collect P1-A controlled-pendulum pilot")
    parser.add_argument("--config", required=True, type=Path, help="YAML pilot config")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override experiment.output_root")
    parser.add_argument("--seed", type=int, default=None, help="Override master seed")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing dataset")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import torch  # only for version reporting

    from data.controlled_pendulum import (
        ControlledPendulum,
        ScenarioSpec,
        ProfileSpec,
        config_to_specs,
        plan_pilot,
        generate_episode,
    )

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed

    output_dir = Path(args.output_dir) if args.output_dir else \
        Path(cfg["experiment"]["output_root"])
    npz_path = output_dir / "pilot_episodes.npz"
    manifest_path = output_dir / "manifest.json"

    if npz_path.exists() and not args.force:
        print(f"ERROR: {npz_path} already exists. Refusing to overwrite.\n"
              f"       Use --force to start fresh (this deletes the old pilot).",
              file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios, profiles, T, dt, master_seed = config_to_specs(cfg)
    torque_max = float(cfg["simulator"]["action_max_torque"])
    plan = plan_pilot(scenarios, profiles, master_seed)

    n_expected = int(cfg["pilot"]["n_scenarios"]) * int(cfg["pilot"]["n_profiles"])
    if cfg["pilot"].get("require_exactly", True) and len(plan) != n_expected:
        print(f"ERROR: pilot plan size {len(plan)} != expected {n_expected}",
              file=sys.stderr)
        return 2

    print(f"[collect] dataset_version={dataset_version(cfg)}")
    print(f"[collect] master_seed={master_seed} dt={dt} T={T} torque_max={torque_max}")
    print(f"[collect] scenarios={[s.name for s in scenarios]}")
    print(f"[collect] profiles={[p.name for p in profiles]}")
    print(f"[collect] episodes={len(plan)} -> {output_dir}")

    # Generate all episodes.
    obs_list, cond_list = [], []
    init_list, scen_idx_list, prof_idx_list = [], [], []
    episode_meta = []
    for item in plan:
        s_idx = item["scenario_idx"]
        p_idx = item["profile_idx"]
        sc = item["scenario"]
        pr = item["profile"]
        seed = item["seed"]

        ep = generate_episode(sc, pr, seed, T, dt)
        obs_list.append(ep.observations)            # (T+1, 2)
        cond_list.append(ep.conditions)            # (T, 1)
        init_list.append([sc.theta0, sc.omega0])
        scen_idx_list.append(s_idx)
        prof_idx_list.append(p_idx)

        torques = ep.conditions[:, 0]
        step_dtheta = np.abs(np.diff(ep.observations[:, 0]))
        episode_meta.append({
            "episode_id": item["episode_id"],
            "scenario_index": s_idx,
            "profile_index": p_idx,
            "scenario_name": sc.name,
            "profile_name": pr.name,
            "seed": int(seed),
            "initial_state": [float(sc.theta0), float(sc.omega0)],
            "scenario": sc.as_dict(),
            "profile": pr.as_dict(),
            "action_stats": {
                "mean": float(torques.mean()),
                "std": float(torques.std()),
                "min": float(torques.min()),
                "max": float(torques.max()),
                "abs_max": float(np.abs(torques).max()),
                "in_bounds": bool(np.all(np.abs(torques) <= torque_max + 1e-9)),
            },
            "final_state": [float(ep.observations[-1, 0]), float(ep.observations[-1, 1])],
            "max_abs_theta": float(np.abs(ep.observations[:, 0]).max()),
            "max_abs_omega": float(np.abs(ep.observations[:, 1]).max()),
            "max_theta_step": float(step_dtheta.max()),
            "torque_in_bounds": bool(np.all(np.abs(torques) <= torque_max + 1e-9)),
        })
        print(f"  + {item['episode_id']:18s} "
              f"seed={seed:6d} tau_absmax={episode_meta[-1]['action_stats']['abs_max']:.3f} "
              f"theta_f={ep.observations[-1,0]:+.3f} omega_f={ep.observations[-1,1]:+.3f}")

    # Stack into arrays.
    observations = np.stack(obs_list, axis=0).astype(np.float64)      # (B, T+1, 2)
    conditions = np.stack(cond_list, axis=0).astype(np.float64)      # (B, T, 1)
    initial_state = np.asarray(init_list, dtype=np.float64)          # (B, 2)
    scenario_index = np.asarray(scen_idx_list, dtype=np.int64)
    profile_index = np.asarray(prof_idx_list, dtype=np.int64)
    valid_mask = np.ones((observations.shape[0], T), dtype=bool)      # all real

    meta = {
        "dt": dt,
        "T": T,
        "obs_dim": observations.shape[-1],
        "cond_dim": conditions.shape[-1],
        "master_seed": int(master_seed),
        "torque_max": torque_max,
        "storage_dtype": "float64",
        "episode_ids": [m["episode_id"] for m in episode_meta],
        "dataset_version": dataset_version(cfg),
        "config_hash": _sha_text(_canonical(dataset_spec(cfg))),
    }
    meta_str = np.array(json.dumps(meta, sort_keys=True))

    np.savez(
        npz_path,
        observations=observations,
        conditions=conditions,
        valid_mask=valid_mask,
        initial_state=initial_state,
        scenario_index=scenario_index,
        profile_index=profile_index,
        meta_str=meta_str,
    )

    # Manifest (content-addressed, with integrity).
    npz_sha = sha256_file(npz_path)
    manifest = {
        "dataset_version": dataset_version(cfg),
        "config_hash": _sha_text(_canonical(dataset_spec(cfg))),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "n_episodes": int(observations.shape[0]),
        "T": int(T),
        "obs_dim": int(observations.shape[-1]),
        "cond_dim": int(conditions.shape[-1]),
        "dt": float(dt),
        "master_seed": int(master_seed),
        "torque_max": float(torque_max),
        "storage_dtype": "float64",
        "npz_file": npz_path.name,
        "npz_sha256": npz_sha,
        "scenarios": [s.as_dict() for s in scenarios],
        "profiles": [p.as_dict() for p in profiles],
        "episodes": episode_meta,
        "integrity": {
            "observations_shape": list(observations.shape),
            "conditions_shape": list(conditions.shape),
            "valid_mask_shape": list(valid_mask.shape),
            "n_valid_transitions": int(valid_mask.sum()),
            "all_valid": bool(valid_mask.all()),
            "unique_episode_ids": len(set(m["episode_id"] for m in episode_meta))
                                  == len(episode_meta),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                             encoding="utf-8")

    print("\n=== collection summary ===")
    print(f"episodes          : {manifest['n_episodes']}")
    print(f"observations      : {tuple(observations.shape)} {observations.dtype}")
    print(f"conditions        : {tuple(conditions.shape)} {conditions.dtype}")
    print(f"all valid         : {manifest['integrity']['all_valid']}")
    print(f"unique ids        : {manifest['integrity']['unique_episode_ids']}")
    print(f"dataset_version   : {manifest['dataset_version']}")
    print(f"npz_sha256        : {npz_sha}")
    print(f"manifest          : {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
