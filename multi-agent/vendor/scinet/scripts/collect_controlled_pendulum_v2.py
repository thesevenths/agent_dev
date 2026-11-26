"""
scripts/collect_controlled_pendulum_v2.py
=========================================
P1-B1: generate a NEW 20-episode pilot under the restored spec (Q-A).

Writes into ``datasets/controlled_pendulum_v2_pilot/`` only. NEVER touches the
historical ``datasets/controlled_pendulum_pilot/``.

Outputs:
  pilot_episodes.npz  -- float64 observations [B,T+1,2], conditions [B,T,1],
                         valid_mask [B,T] bool, plus per-episode arrays
  manifest.json       -- complete resolved config, git SHA, per-episode identity
                         (initial state, physical params, action identity,
                         regime, split, episode SHA-256)

Usage:
  python scripts/collect_controlled_pendulum_v2.py \
      --config configs/dynamics/controlled_pendulum_v2_pilot.yaml
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.controlled_pendulum_v2 import (  # noqa: E402
    ValidatedPendulum, classify_regime, generate_torque_sequence_v2,
    sample_episode_specs,
)
from data.source_provenance import (  # noqa: E402
    ProvenanceError,
    assert_provenance_ok,
    capture_source_identity,
    file_sha256,
    git_head_sha,
    git_tracked_clean,
    git_worktree_isolation,
    verify_identity_stable,
)
from dynamics.contracts import DynamicsBatch  # noqa: E402
import torch  # noqa: E402


def _canonical(o) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), default=str)


def _sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    """Backward-compatible HEAD SHA helper (returns 'unavailable' on failure)."""
    sha = git_head_sha(ROOT)
    return sha if sha else "unavailable"


def resolved_config(cfg: dict) -> dict:
    """Full, explicit resolved values (no implicit defaults)."""
    sim = cfg["simulator"]
    d = cfg["distributions"]
    return {
        "integrator": sim["integrator"],
        "theta_representation": sim["theta_representation"],
        "dt": float(sim["dt"]),
        "episode_steps": int(sim["episode_steps"]),
        "torque_bounds": [float(x) for x in sim["torque_bounds"]],
        "observation_dim": int(sim["observation_dim"]),
        "action_dim": int(sim["action_dim"]),
        "per_step_finite_guard": bool(sim["per_step_finite_guard"]),
        "parameter_validation": bool(sim["parameter_validation"]),
        "gravity_g": float(d["gravity_g"]),
        "theta0_range": [float(x) for x in d["theta0_range"]],
        "omega0_range": [float(x) for x in d["omega0_range"]],
        "mass_range": [float(x) for x in d["mass_range"]],
        "length_range": [float(x) for x in d["length_range"]],
        "damping_rate_range": [float(x) for x in d["damping_rate_range"]],
        "action_profiles": [dict(p) for p in cfg["action_profiles"]],
        "action_hold_steps_options": [int(x) for x in cfg["action_hold_steps_options"]],
        "collapse_thresholds": dict(cfg["collapse_thresholds"]),
        "parameter_ood_groups": dict(cfg["parameter_ood_groups"]),
        "regime_criteria": dict(cfg["regime_criteria"]),
        "storage_dtype": cfg["pilot"]["storage_dtype"],
        "seed_derivation": "sha256_first4_bytes",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=ROOT / "configs/dynamics/controlled_pendulum_v2_pilot.yaml")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing pilot (v2 dir only).")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="Do NOT use: only for broken CI envs; records a "
                         "provenance risk instead of blocking on a dirty tree.")
    ap.add_argument("--isolated", action="store_true",
                    help="Run generation inside a throwaway git worktree at HEAD "
                         "so mid-run source edits cannot affect provenance.")
    ap.add_argument("--no-isolate", action="store_true",
                    help="Internal: used by --isolated to prevent recursion.")
    args = ap.parse_args(argv)

    # --isolated: run the real collection inside a committed worktree snapshot --
    if args.isolated and not args.no_isolate:
        sha = git_head_sha(ROOT)
        if not sha:
            print("ERROR: --isolated requires a valid git HEAD SHA.", file=sys.stderr)
            return 2
        if not git_tracked_clean(ROOT):
            print("ERROR: --isolated requires a clean tracked tree first.",
                  file=sys.stderr)
            return 2
        real_out = (args.output_dir
                    or (ROOT / yaml.safe_load(
                        args.config.resolve().read_text(encoding="utf-8"))
                        ["experiment"]["output_root"]))
        try:
            rel_cfg = str(args.config.resolve().relative_to(ROOT))
        except ValueError:
            rel_cfg = str(args.config.resolve())
        # SciNet++ may be a subdirectory of a larger repo; the worktree is created
        # at the repo root, so the collector/config live under ROOT's relative
        # path within that worktree.
        repo_root = git_repo_root(ROOT)
        repo_rel = ROOT.relative_to(repo_root) if repo_root else Path(".")
        print(f"[collect-v2] isolated worktree mode -> real output {real_out}")

        def _run_in_worktree(wt: Path) -> int:
            inner_cwd = Path(wt) / repo_rel
            inner_cfg = inner_cwd / rel_cfg
            inner = subprocess.run(
                [sys.executable,
                 str(inner_cwd / "scripts" / "collect_controlled_pendulum_v2.py"),
                 "--config", str(inner_cfg),
                 "--output-dir", str(real_out.resolve()),
                 "--no-isolate"],
                cwd=str(inner_cwd), capture_output=False, text=True)
            return inner.returncode

        return git_worktree_isolation(ROOT, _run_in_worktree)

    args.config = args.config.resolve()
    try:
        config_rel = str(args.config.relative_to(ROOT))
    except ValueError:
        config_rel = str(args.config)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out = args.output_dir or (ROOT / cfg["experiment"]["output_root"])
    npz_path = out / "pilot_episodes.npz"
    man_path = out / "manifest.json"
    if npz_path.exists() and not args.force:
        print(f"ERROR: {npz_path} exists. Use --force (v2 dir only).", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)

    # ---- P6: source-provenance precondition (fail-closed) ------------------- #
    simulator_path = ROOT / "data" / "controlled_pendulum_v2.py"
    git_commit_required = bool(cfg["experiment"].get("git_commit_required", False))
    try:
        before = capture_source_identity(
            collector_path=Path(__file__).resolve(),
            simulator_path=simulator_path,
            config_path=args.config,
            cwd=ROOT,
        )
        before["_tracked_clean"] = git_tracked_clean(ROOT)
        assert_provenance_ok(before, git_commit_required,
                             allow_dirty=args.allow_dirty)
    except ProvenanceError as exc:
        print(f"ERROR [provenance]: {exc}", file=sys.stderr)
        return 2
    print(f"[collect-v2] provenance OK: git={before['git_commit']} "
          f"clean={before['_tracked_clean']} "
          f"collector_sha={before['collector_sha256'][:12]} "
          f"sim_sha={before['simulator_sha256'][:12]} "
          f"config_sha={before['config_sha256'][:12]}")

    sim = cfg["simulator"]
    T = int(sim["episode_steps"])
    dt = float(sim["dt"])
    tb = [float(x) for x in sim["torque_bounds"]]
    master_seed = int(cfg["experiment"]["seed"])
    n_expected = int(cfg["pilot"]["episodes"])
    specs = sample_episode_specs(cfg)
    if cfg["pilot"].get("require_exactly") and len(specs) != n_expected:
        print(f"ERROR: planned {len(specs)} != required {n_expected}", file=sys.stderr)
        return 2

    print(f"[collect-v2] dt={dt} T={T} torque_bounds={tb} episodes={len(specs)}")
    print(f"[collect-v2] output={out}")

    obs_list, cond_list, init_list, scen_idx, prof_idx, holds = [], [], [], [], [], []
    ep_records = []
    for i, sp in enumerate(specs):
        tor = generate_torque_sequence_v2(
            sp.profile, T, dt, sp.seed, hold_steps=sp.hold_steps,
            torque_bounds=tb)
        pen = ValidatedPendulum(g=sp.scenario.g, L=sp.scenario.L,
                                m=sp.scenario.m, b=sp.scenario.b, dt=dt,
                                torque_bounds=tb)
        obs = pen.simulate(tor, sp.scenario.theta0, sp.scenario.omega0)
        cond = tor.reshape(-1, 1).astype(np.float64)
        reg = classify_regime(obs, sp.scenario)
        sp.regime = reg["regime"]

        obs_list.append(obs)
        cond_list.append(cond)
        init_list.append([sp.scenario.theta0, sp.scenario.omega0])
        prof_idx.append(divmod(i, len(cfg["action_profiles"]))[1])
        scen_idx.append(0)
        holds.append(sp.hold_steps)

        ep_bytes = obs.tobytes() + cond.tobytes()
        ep_records.append({
            "episode_id": sp.episode_id,
            "seed": sp.seed,
            "initial_state": [sp.scenario.theta0, sp.scenario.omega0],
            "physical_parameters": {
                "g": sp.scenario.g, "L": sp.scenario.L, "m": sp.scenario.m,
                "b": sp.scenario.b,
                "I_kg_m2": sp.scenario.m * sp.scenario.L ** 2,
                "normalized_damping_rate_1_per_s": sp.scenario.b,
            },
            "action_profile": sp.profile.as_dict(),
            "action_hold_steps": sp.hold_steps,
            "action_stats": {
                "min": float(tor.min()), "max": float(tor.max()),
                "mean": float(tor.mean()), "std": float(tor.std()),
                "abs_max": float(np.abs(tor).max()),
                "unique_values": int(len(np.unique(np.round(tor, 9)))),
                "zero_fraction": float(np.mean(np.abs(tor) < 1e-12)),
                "in_bounds": bool(np.all(tor >= tb[0] - 1e-9)
                                  and np.all(tor <= tb[1] + 1e-9)),
            },
            "regime": reg["regime"],          # string, used for stratification
            "regime_detail": reg,             # full preregistered metrics
            "episode_sha256": hashlib.sha256(ep_bytes).hexdigest(),
            "final_theta_rad": float(obs[-1, 0]),
            "max_abs_omega_rad_s": float(np.max(np.abs(obs[:, 1]))),
        })
        print(f"  + {sp.episode_id:20s} hold={sp.hold_steps:2d} "
              f"theta0={sp.scenario.theta0:+.3f} omega0={sp.scenario.omega0:+.3f} "
              f"m={sp.scenario.m:.3f} L={sp.scenario.L:.3f} "
              f"regime={reg['regime']:20s} theta_f={obs[-1, 0]:+.3f}")

    # Strict episode-level 70/15/15 split on the WHOLE set (14/3/3).
    # Stratification info is RECORDED per split (profile + regime coverage) but
    # the assignment is global: with 20 episodes, per-group splitting would
    # produce empty/tiny test sets (the exact v1 defect).
    rng = np.random.RandomState(master_seed % (2 ** 31))
    n = len(ep_records)
    n_test = max(1, int(round(0.15 * n)))
    n_val = max(1, int(round(0.15 * n)))
    n_train = n - n_val - n_test
    order = rng.permutation(n)
    split_map: dict[str, str] = {}
    for j, idx in enumerate(order):
        split_map[ep_records[idx]["episode_id"]] = (
            "train" if j < n_train else
            "validation" if j < n_train + n_val else "test")
    for r in ep_records:
        r["split"] = split_map[r["episode_id"]]

    observations = np.stack(obs_list).astype(np.float64)
    conditions = np.stack(cond_list).astype(np.float64)
    initial_state = np.asarray(init_list, dtype=np.float64)
    valid_mask = np.ones((observations.shape[0], T), dtype=bool)
    episode_ids = [r["episode_id"] for r in ep_records]

    meta = {
        "episode_ids": episode_ids,
        "dt": dt, "T": T, "obs_dim": observations.shape[-1],
        "cond_dim": conditions.shape[-1], "master_seed": master_seed,
        "torque_bounds": tb, "storage_dtype": "float64",
        "config_file": config_rel,
    }
    np.savez(npz_path,
             observations=observations, conditions=conditions,
             valid_mask=valid_mask, initial_state=initial_state,
             scenario_index=np.asarray(scen_idx, dtype=np.int64),
             profile_index=np.asarray(prof_idx, dtype=np.int64),
             hold_steps=np.asarray(holds, dtype=np.int64),
             meta_str=np.array(json.dumps(meta, sort_keys=True)))

    # DynamicsBatch contract validation (float64, reuses existing contract)
    batch = DynamicsBatch(
        torch.as_tensor(observations, dtype=torch.float64),
        torch.as_tensor(conditions, dtype=torch.float64),
        torch.as_tensor(valid_mask, dtype=torch.bool), episode_ids)
    contract = {
        "validated": True, "batch_size": batch.batch_size,
        "num_steps": batch.num_steps, "obs_dim": batch.obs_dim,
        "cond_dim": batch.cond_dim, "is_conditioned": batch.is_conditioned,
        "n_valid_transitions": batch.num_valid_transitions(),
    }

    # ---- P6: post-collection source-identity re-verification (fail-closed) --- #
    after = capture_source_identity(
        collector_path=Path(__file__).resolve(),
        simulator_path=simulator_path,
        config_path=args.config,
        cwd=ROOT,
    )
    identity_stable = verify_identity_stable(before, after)
    if not identity_stable:
        # Source identity changed mid-run: the dataset is NOT reproducible from
        # the recorded provenance, so we MUST NOT publish it. Delete outputs.
        try:
            if npz_path.exists():
                npz_path.unlink()
            if man_path.exists():
                man_path.unlink()
        except OSError:
            pass
        print("ERROR [provenance]: source identity changed DURING collection "
              "(HEAD/version/source-file hash differ before vs after). "
              "Outputs deleted; collection aborted to avoid unverifiable data.",
              file=sys.stderr)
        return 3

    res = resolved_config(cfg)
    man = {
        "dataset_version": f"cpv2-{_sha_text(_canonical(res))[:12]}",
        "config_hash": _sha_text(_canonical(res)),
        "config_file": config_rel,
        "resolved_config": res,
        "git_commit": before["git_commit"],
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "n_episodes": int(observations.shape[0]),
        "T": T, "dt": dt,
        "obs_dim": int(observations.shape[-1]),
        "cond_dim": int(conditions.shape[-1]),
        "storage_dtype": "float64",
        "npz_file": npz_path.name,
        "npz_sha256": sha256_file(npz_path),
        "episodes": ep_records,
        "split_counts": {s: sum(1 for r in ep_records if r["split"] == s)
                         for s in ("train", "validation", "test")},
        "regime_counts": {g: sum(1 for r in ep_records if r["regime"] == g)
                          for g in ("bounded_oscillation", "transitional",
                                    "continuous_rotation")},
        "dynamics_batch_contract": contract,
        "historical_pilot_preserved": "datasets/controlled_pendulum_pilot",
        "source_provenance": {
            "git_commit_required": git_commit_required,
            "git_commit_required_enforced": True,
            "tracked_tree_clean_before": bool(before["_tracked_clean"]),
            "tracked_tree_clean_allowed_dirty": bool(args.allow_dirty),
            "identity_stable_during_run": bool(identity_stable),
            "before": {k: before[k] for k in (
                "git_commit", "collector_sha256", "simulator_sha256",
                "config_sha256", "dependency_versions", "captured_at")},
            "after": {k: after[k] for k in (
                "git_commit", "collector_sha256", "simulator_sha256",
                "config_sha256", "dependency_versions", "captured_at")},
        },
    }
    man_path.write_text(json.dumps(man, indent=2, sort_keys=True), encoding="utf-8")

    print("\n=== v2 collection summary ===")
    print(f"episodes       : {man['n_episodes']}")
    print(f"observations   : {tuple(observations.shape)} {observations.dtype}")
    print(f"conditions     : {tuple(conditions.shape)} {conditions.dtype}")
    print(f"split_counts   : {man['split_counts']}")
    print(f"regime_counts  : {man['regime_counts']}")
    print(f"dataset_version: {man['dataset_version']}")
    print(f"npz_sha256     : {man['npz_sha256']}")
    print(f"git_commit     : {man['git_commit']}")
    print(f"manifest       : {man_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
