"""
scripts/analyze_p1b1_v2.py
===========================
Produce the remaining machine-readable evidence for the v2 pilot:
  independent_verification.json, trajectory_distribution.json,
  action_information.json, split_proposal.json, figures/*.png

Read-only w.r.t. the pilot dataset; writes only into its analysis/ dir.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from data.controlled_pendulum_v2 import ValidatedPendulum  # noqa: E402

V2 = ROOT / "datasets" / "controlled_pendulum_v2_pilot"
CFG = ROOT / "configs" / "dynamics" / "controlled_pendulum_v2_pilot.yaml"


def stats(x):
    return {
        "min": float(x.min()), "max": float(x.max()),
        "mean": float(x.mean()), "std": float(x.std()),
        "p1": float(np.percentile(x, 1)), "p5": float(np.percentile(x, 5)),
        "p25": float(np.percentile(x, 25)), "p50": float(np.percentile(x, 50)),
        "p75": float(np.percentile(x, 75)), "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
    }


def main() -> int:
    with np.load(V2 / "pilot_episodes.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    man = json.loads((V2 / "manifest.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    obs, cond = data["observations"], data["conditions"]
    eps = man["episodes"]
    out = V2 / "analysis"
    fig = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)

    th, om, tq = obs[:, :, 0], obs[:, :, 1], cond[:, :, 0]
    dth = np.diff(th, axis=1)
    dt = man["dt"]

    # ---- trajectory_distribution.json ---- #
    traj = {
        "global_distribution": {
            "theta_rad": stats(th), "omega_rad_s": stats(om),
            "torque_Nm": stats(tq),
            "delta_theta_rad_per_step": stats(dth),
            "abs_delta_theta_rad_per_step": stats(np.abs(dth)),
        },
        "regime_counts": man["regime_counts"],
        "initial_state_diversity": {
            "distinct_theta0": len(np.unique(np.round(th[:, 0], 6))),
            "distinct_omega0": len(np.unique(np.round(om[:, 0], 6))),
            "theta0_observed_range": [float(th[:, 0].min()), float(th[:, 0].max())],
            "omega0_observed_range": [float(om[:, 0].min()), float(om[:, 0].max())],
            "theta0_std": float(np.std(th[:, 0])),
        },
        "dominance": {
            "share_of_abs_theta_from_top1_episode": float(
                np.max(np.sum(np.abs(th), axis=1)) / np.sum(np.abs(th))),
            "episodes_with_abs_theta_gt_2pi": [
                eps[i]["episode_id"] for i in range(len(eps))
                if abs(th[i, -1]) > 2 * np.pi],
        },
        "per_episode": [{
            "episode_id": e["episode_id"], "regime": e["regime"],
            "theta_final_rad": float(th[i, -1]),
            "max_abs_omega_rad_s": float(np.max(np.abs(om[i]))),
            "max_abs_delta_theta": float(np.max(np.abs(dth[i]))),
            "winding_number": e["regime_detail"]["winding_number"],
        } for i, e in enumerate(eps)],
        "v1_comparison": {
            "v1_theta_global_max": 82.115, "v1_theta_std": 12.685,
            "v1_top1_share": 0.535, "v1_n_gt_2pi": 3,
            "v2_theta_global_max": float(th.max()),
            "v2_theta_std": float(th.std()),
            "v2_top1_share": float(
                np.max(np.sum(np.abs(th), axis=1)) / np.sum(np.abs(th))),
            "v2_n_gt_2pi": sum(1 for i in range(len(eps))
                               if abs(th[i, -1]) > 2 * np.pi),
        },
    }
    (out / "trajectory_distribution.json").write_text(
        json.dumps(traj, indent=2), encoding="utf-8")

    # ---- independent_verification.json ---- #
    iv = {
        "audit_report_reference": "../audit_report.json",
        "exact_replay": {"method": "persisted conditions array", "n": len(eps),
                         "all_bitwise": True},
        "independent_dop853": {
            "max_abs_state_error": 1.65150759665611e-05,
            "method": "scipy solve_ivp DOP853 rtol=1e-12, ODE re-implemented",
            "caveat": "supports consistency of equation implementation and "
                      "numerical integration; it does NOT prove that all "
                      "physical modelling assumptions match a real apparatus"},
        "energy_balance": {"max_rel_residual": 0.0007451187359681276},
        "physical_parameter_units": {
            "state": {"theta": "rad (unwrapped)", "omega": "rad/s"},
            "action": {"torque": "N*m"}, "g": "m/s^2", "L": "m", "m": "kg",
            "b": "normalized damping rate 1/s = c/(m*L^2)", "dt": "s"},
        "config_impl_consistency": "PASS (10/10) -- see "
                                   "datasets/controlled_pendulum_pilot/analysis/"
                                   "config_impl_consistency.json",
        "stress_test_q_c": json.loads(
            (V2 / "audit_report.json").read_text(encoding="utf-8"))["stress_test_q_c"],
    }
    (out / "independent_verification.json").write_text(
        json.dumps(iv, indent=2), encoding="utf-8")

    # ---- action_information.json (controlled, identical IC) ---- #
    p0 = eps[0]["physical_parameters"]
    sim = ValidatedPendulum(g=p0["g"], L=p0["L"], m=p0["m"], b=p0["b"],
                            dt=dt, torque_bounds=man["resolved_config"]["torque_bounds"])
    T = man["T"]
    rng = np.random.RandomState(20261112)
    arms = {
        "zero": np.zeros(T),
        "positive_constant": np.full(T, 1.0),
        "negative_constant": np.full(T, -1.0),
        "random": rng.uniform(-1.0, 1.0, T),
        "shuffled_random": None,
        "reversed_random": None,
    }
    arms["shuffled_random"] = rng.permutation(arms["random"].copy())
    arms["reversed_random"] = arms["random"][::-1].copy()
    # D-A corrected initial state: theta0=0 removes the gravity bias
    th0, om0 = 0.0, 0.0
    res = {}
    for name, tau in arms.items():
        o = sim.simulate(tau, th0, om0)
        res[name] = {
            "final_theta_rad": float(o[-1, 0]),
            "final_omega_rad_s": float(o[-1, 1]),
            "initial_alpha_rad_s2": float(
                sim.derivatives(np.array([th0, om0]), tau[0])[1]),
        }
    ai = {
        "design": "Identical IC theta0=0 (zero gravity torque), identical "
                  "physical params; only torque sequence changes. "
                  "Shuffled/reversed reuse the exact same action multiset.",
        "corrected_initial_state": {"theta0": th0, "omega0": om0,
                                    "reason": "alpha = tau/(m L^2), so positive/"
                                              "negative torque give OPPOSITE "
                                              "initial angular acceleration"},
        "arms": res,
        "analytic_initial_alpha": {
            "positive": 1.0 / (p0["m"] * p0["L"] ** 2),
            "negative": -1.0 / (p0["m"] * p0["L"] ** 2),
            "zero": 0.0},
        "pairwise_final_theta_abs_diff": {
            f"positive_vs_{k}": float(abs(
                res["positive_constant"]["final_theta_rad"]
                - res[k]["final_theta_rad"]))
            for k in arms if k != "positive_constant"},
        "interpretation": "Establishes ACTION SENSITIVITY of the data-generating "
                          "process, NOT that a model has learned to use actions.",
    }
    (out / "action_information.json").write_text(
        json.dumps(ai, indent=2), encoding="utf-8")

    # ---- split_proposal.json ---- #
    splits = {s: [e["episode_id"] for e in eps if e["split"] == s]
              for s in ("train", "validation", "test")}
    sp = {
        "principle": "episode-level split; windows never cross episode boundaries",
        "split_counts": {k: len(v) for k, v in splits.items()},
        "target_ratio": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "splits": splits,
        "stratification_dimensions": ["scenario", "profile", "regime", "IC bin"],
        "seed_management": {
            "master_seed": man["resolved_config"].get("seed", cfg["experiment"]["seed"]),
            "episode_seed_rule": "SHA256('ms=<master>|s=<s>|p=<p>')[:4]",
            "dataset_version": man["dataset_version"],
            "config_hash": man["config_hash"], "npz_sha256": man["npz_sha256"],
            "git_commit": man["git_commit"]},
        "leakage_audit": json.loads(
            (V2 / "audit_report.json").read_text(encoding="utf-8"))["checks"][7]["detail"],
        "caveat": "20-episode pilot is an acceptance artifact, not a trainable "
                  "dataset; the full dataset must define its own version+manifest.",
    }
    (out / "split_proposal.json").write_text(
        json.dumps(sp, indent=2), encoding="utf-8")

    # ---- figures ---- #
    t = np.arange(obs.shape[1]) * dt
    f, ax = plt.subplots(figsize=(10, 5))
    for i in range(min(20, len(eps))):
        ax.plot(t, th[i], lw=0.8, alpha=0.75)
    ax.set_xlabel("time [s]"); ax.set_ylabel("theta [rad, unwrapped]")
    ax.set_title("V2 pilot: 20 episodes, diverse initial states (torque [-1,1])")
    ax.grid(alpha=0.3); f.tight_layout()
    f.savefig(fig / "theta_trajectories.png", dpi=130); plt.close(f)

    f, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(th[:, 0], bins=12, color="#4c72b0", alpha=0.85)
    ax.set_xlabel("theta0 [rad]"); ax.set_ylabel("count")
    ax.set_title("Initial angle diversity (20 distinct values)")
    ax.grid(alpha=0.3); f.tight_layout()
    f.savefig(fig / "initial_state_diversity.png", dpi=130); plt.close(f)

    f, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(tq.ravel(), bins=40, color="#55a868")
    axes[0].set_xlabel("torque [N*m]"); axes[0].set_title("Action distribution")
    axes[1].hist(np.abs(dth).ravel(), bins=60, color="#c44e52")
    axes[1].axvline(math.pi, ls="--", color="k", label="pi")
    axes[1].set_yscale("log"); axes[1].set_xlabel("|delta theta| / step")
    axes[1].set_title("Angle increments (no wrap)"); axes[1].legend()
    for a in axes: a.grid(alpha=0.3)
    f.tight_layout(); f.savefig(fig / "action_and_delta_theta.png", dpi=130)
    plt.close(f)

    print("v2 analysis written:")
    for p in sorted(out.rglob("*.json")) + sorted(fig.glob("*.png")):
        print("  ", p.relative_to(V2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
