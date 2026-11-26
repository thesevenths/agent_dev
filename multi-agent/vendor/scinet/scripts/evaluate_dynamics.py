#!/usr/bin/env python3
"""
P2-02 Phase 4/5 -- evaluate all five raw-state dynamics baselines.

Produces (under <output_root>/evaluation/):
    metrics.json                  aggregated results + gate verdicts
    runs.json                     full per-(baseline, seed, split) records
    per_dimension_metrics.csv
    horizon_metrics.csv
    episode_metrics.csv
    bootstrap_results.json        episode-level paired bootstrap (95% CI)
    gates.json
    prediction_plots/*.png
    rollout_plots/*.png

Scientific rules enforced here:
* Frozen Test is quarantined (used only for final reporting).
* Normalisation statistics are loaded from the TRAIN-only file written by the
  training script; nothing is re-fitted on validation or test.
* Statistical comparisons are EPISODE-level paired bootstraps, never 30,000
  correlated transitions treated as independent samples.
* Results are never described as successful object pushing.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_dynamics as td  # noqa: E402  (reuse dataset loading / hashing)

from dynamics import evaluate_model, open_loop_rollout  # noqa: E402
from dynamics.metrics import describe, paired_bootstrap_ci  # noqa: E402
from dynamics.normalization import NormalizationStats  # noqa: E402
from models.state_dynamics import (  # noqa: E402
    PersistenceBaseline,
    ResidualStatePredictor,
    RidgeDynamics,
)

BASELINES = ("B0", "B1", "B2", "B3", "B4")
LABELS = {
    "B0": "Persistence",
    "B1": "Unconditioned MLP",
    "B2": "Action-conditioned MLP",
    "B3": "Linear / Ridge",
    "B4": "Shuffled-action control",
}
COLORS = {
    "B0": "#888888", "B1": "#1f77b4", "B2": "#d62728",
    "B3": "#2ca02c", "B4": "#ff7f0e",
}


# --------------------------------------------------------------------------- #
def load_model(baseline: str, out_root: Path, cfg_hash: str, seed: int,
               D: int, U: int, hidden):
    if baseline == "B0":
        return PersistenceBaseline(D)
    exp = out_root / f"{cfg_hash[:8]}_{baseline}_seed{seed}"
    ck = exp / "checkpoint.pt"
    if not ck.is_file():
        raise FileNotFoundError(f"missing trained checkpoint: {ck}")
    payload = torch.load(ck, map_location="cpu", weights_only=False)
    if baseline == "B3":
        m = RidgeDynamics(D, U, alpha=float(payload["alpha"]))
    else:
        m = ResidualStatePredictor(D, payload["condition_dim"], tuple(hidden))
    m.load_state_dict(payload["model_state"])
    m.eval()
    return m


def _mean_std(values: List[float]) -> Dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=0)),
            "median": float(np.median(a)), "n": int(a.size)}


def write_csv(path: Path, header: List[str], rows: List[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="P2-02 baseline evaluation")
    ap.add_argument("--config", default="configs/dynamics/pusher_v1_baselines.yaml")
    ap.add_argument("--output-root", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.output_root:
        cfg["experiment"]["output_root"] = args.output_root
    root = Path(cfg["dataset"]["root"])
    out_root = Path(cfg["experiment"]["output_root"])
    eval_dir = out_root / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "prediction_plots").mkdir(exist_ok=True)
    (eval_dir / "rollout_plots").mkdir(exist_ok=True)

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if cfg["dataset"].get("verify_manifest", True):
        td.verify_manifest(root, manifest)
    audit = json.loads((out_root / "data_audit.json").read_text(encoding="utf-8"))
    tman = json.loads((out_root / "training_manifest.json").read_text(encoding="utf-8"))
    cfg_hash = tman["config_hash"]
    stats = NormalizationStats.load(out_root / "normalization.json")
    if stats.source_split != "train":
        raise RuntimeError("normalisation was not fitted on train only")

    D, U = int(cfg["dataset"]["observation_dim"]), int(cfg["dataset"]["action_dim"])
    hidden = tuple(int(h) for h in cfg["model"]["hidden_dims"])
    splits = cfg["dataset"]["splits"]
    horizons = [int(h) for h in cfg["evaluation"]["horizons"]]
    feature_groups = {k: list(v) for k, v in cfg["dataset"]["feature_groups"].items()}
    distance_spec = cfg["dataset"]["distance_spec"]
    report_splits = [splits[k] for k in ("validation", "test", "heldout")]
    seeds = [int(s) for s in cfg["training"]["seeds"]]

    batches = {
        name: td.load_split_batch(root, name, manifest, D, U)
        for name in report_splits
    }
    print(f"[eval] splits: " + ", ".join(
        f"{n}={b.batch_size}ep" for n, b in batches.items()
    ))

    runs: List[dict] = []
    for baseline in BASELINES:
        for seed in seeds:
            model = load_model(baseline, out_root, cfg_hash, seed, D, U, hidden)
            for split_name, batch in batches.items():
                r = evaluate_model(
                    model, batch, stats,
                    horizons=horizons,
                    feature_groups=feature_groups,
                    distance_spec=distance_spec,
                )
                runs.append({
                    "baseline": baseline,
                    "label": LABELS[baseline],
                    "seed": seed,
                    "split": split_name,
                    **r,
                })
            print(f"[eval] {baseline} seed={seed}: "
                  f"frozen_test one-step mse="
                  f"{[x for x in runs if x['baseline']==baseline and x['seed']==seed and x['split']==splits['test']][0]['one_step']['mse']:.6f}")

    (eval_dir / "runs.json").write_text(
        json.dumps(runs, indent=1), encoding="utf-8"
    )

    # ---- aggregate across seeds ------------------------------------------- #
    summary: Dict[str, Dict[str, dict]] = {}
    for split_name in report_splits:
        summary[split_name] = {}
        rs = [r for r in runs if r["split"] == split_name]
        for baseline in BASELINES:
            br = [r for r in rs if r["baseline"] == baseline]
            one = {
                k: _mean_std([r["one_step"][k] for r in br])
                for k in ("mse", "rmse", "mae", "skill_vs_persistence")
            }
            groups = {}
            for gname in feature_groups:
                groups[gname] = {
                    "mse": _mean_std(
                        [r["feature_groups_one_step"][gname]["mse"] for r in br]
                    ),
                    "persistence_mse": float(
                        br[0]["feature_groups_one_step"][gname]["persistence_mse"]
                    ),
                    "skill_vs_persistence": (
                        None if any(
                            r["feature_groups_one_step"][gname]["skill_vs_persistence"]
                            is None for r in br
                        ) else _mean_std([
                            r["feature_groups_one_step"][gname]["skill_vs_persistence"]
                            for r in br
                        ])
                    ),
                }
            hor = {}
            for h in horizons:
                key = str(h)
                hor[key] = {
                    k: _mean_std([r["horizons"][key][k] for r in br])
                    for k in ("mse", "rmse", "skill_vs_persistence",
                              "persistence_mse", "n_windows")
                }
            per_dim = np.asarray(
                [r["one_step"]["per_dim_rmse"] for r in br], dtype=np.float64
            )
            goal = (
                {k: _mean_std([r["goal_distance"][k] for r in br])
                 for k in ("mae", "mse", "mean_true_distance")}
                if br[0]["goal_distance"] else None
            )
            summary[split_name][baseline] = {
                "one_step": one,
                "feature_groups_one_step": groups,
                "horizons": hor,
                "per_dim_rmse_mean": per_dim.mean(axis=0).tolist(),
                "per_dim_rmse_std": per_dim.std(axis=0, ddof=0).tolist(),
                "goal_distance": goal,
            }

    # ---- CSV outputs ------------------------------------------------------- #
    pd_rows = []
    for split_name in report_splits:
        for baseline in BASELINES:
            s = summary[split_name][baseline]
            for d in range(D):
                pd_rows.append([
                    split_name, baseline, LABELS[baseline], d,
                    f"{s['per_dim_rmse_mean'][d]:.8f}",
                    f"{s['per_dim_rmse_std'][d]:.8f}",
                ])
    write_csv(eval_dir / "per_dimension_metrics.csv",
              ["split", "baseline", "label", "dim", "rmse_mean", "rmse_std"],
              pd_rows)

    hz_rows = []
    for split_name in report_splits:
        for baseline in BASELINES:
            for h in horizons:
                m = summary[split_name][baseline]["horizons"][str(h)]
                hz_rows.append([
                    split_name, baseline, LABELS[baseline], h,
                    f"{m['mse']['mean']:.8f}", f"{m['mse']['std']:.8f}",
                    f"{m['rmse']['mean']:.8f}",
                    f"{m['skill_vs_persistence']['mean']:.6f}",
                    f"{m['skill_vs_persistence']['std']:.6f}",
                    f"{m['persistence_mse']['mean']:.8f}",
                    int(m["n_windows"]["mean"]),
                ])
    write_csv(eval_dir / "horizon_metrics.csv",
              ["split", "baseline", "label", "horizon", "mse_mean", "mse_std",
               "rmse_mean", "skill_mean", "skill_std", "persistence_mse_mean",
               "n_windows"], hz_rows)

    ep_rows = []
    for r in runs:
        ids = manifest["split_episode_ids"][r["split"]]
        oe = r["episode_errors_one_step"]
        for i, v in enumerate(oe):
            ep_rows.append([
                r["split"], r["baseline"], LABELS[r["baseline"]], r["seed"],
                ids[i], f"{v:.8f}",
                f"{_safe_ep(r, '5', i):.8f}",
                f"{_safe_ep(r, '10', i):.8f}",
                f"{_safe_ep(r, '20', i):.8f}",
            ])
    write_csv(eval_dir / "episode_metrics.csv",
              ["split", "baseline", "label", "seed", "episode_id",
               "one_step_mse", "h5_mse", "h10_mse", "h20_mse"], ep_rows)

    # ---- episode-level paired bootstrap ----------------------------------- #
    n_boot = int(cfg["evaluation"]["n_bootstrap"])
    alpha = float(cfg["evaluation"]["alpha"])
    bseed = int(cfg["evaluation"]["bootstrap_seed"])

    def ep_vector(split_name, baseline, field="episode_errors_one_step",
                  horizon=None):
        """Across-seed mean of per-episode errors -> one value per episode."""
        rs = [r for r in runs
              if r["split"] == split_name and r["baseline"] == baseline]
        if horizon is None:
            arrs = [r[field] for r in rs]
        else:
            arrs = [r["horizon_episode_errors"][str(horizon)] for r in rs]
        a = np.asarray(arrs, dtype=np.float64)
        return a.mean(axis=0)

    comparisons = [
        ("B2", "B0", "B2 vs Persistence"),
        ("B2", "B1", "B2 vs Unconditioned MLP"),
        ("B2", "B3", "B2 vs Linear/Ridge"),
        ("B2", "B4", "B2 vs Shuffled-Action Control"),
        ("B1", "B0", "B1 vs Persistence"),
        ("B4", "B1", "Shuffled-Action vs Unconditioned MLP"),
    ]
    bootstrap: Dict = {"config": {"n_boot": n_boot, "alpha": alpha,
                                  "bootstrap_seed": bseed,
                                  "unit": "episode (paired, across-seed mean)"},
                       "results": {}}
    for split_name in (splits["validation"], splits["test"]):
        bootstrap["results"][split_name] = {}
        for a_name, b_name, label in comparisons:
            entry = {}
            for h in (None, 5, 10):
                av = ep_vector(split_name, a_name, horizon=h)
                bv = ep_vector(split_name, b_name, horizon=h)
                ci = paired_bootstrap_ci(
                    av, bv, n_boot=n_boot, alpha=alpha, seed=bseed
                )
                key = "one_step" if h is None else f"horizon_{h}"
                entry[key] = ci
            bootstrap["results"][split_name][label] = entry
    (eval_dir / "bootstrap_results.json").write_text(
        json.dumps(bootstrap, indent=2), encoding="utf-8"
    )

    # ---- gates -------------------------------------------------------------- #
    test_split = splits["test"]
    s = summary[test_split]
    primary = cfg["dataset"]["primary_group"]
    pg = s["B2"]["feature_groups_one_step"][primary]
    g0 = bool(audit["gate"]["integrity_pass"])
    g1 = bool(pg["mse"]["mean"] < pg["persistence_mse"])
    g2 = bool(s["B2"]["one_step"]["mse"]["mean"] < s["B1"]["one_step"]["mse"]["mean"])
    g3 = bool(s["B2"]["one_step"]["mse"]["mean"] < s["B4"]["one_step"]["mse"]["mean"])
    h5 = s["B2"]["horizons"]["5"]
    h10 = s["B2"]["horizons"]["10"]
    g4 = bool(
        all(math.isfinite(v) for v in
            (h5["mse"]["mean"], h10["mse"]["mean"],
             h5["skill_vs_persistence"]["mean"],
             h10["skill_vs_persistence"]["mean"]))
        and h5["skill_vs_persistence"]["mean"] > 0
        and h10["skill_vs_persistence"]["mean"] > 0
    )
    gates = {
        "G0": {"pass": g0, "criterion": "Phase 0 data integrity + temporal alignment",
               "evidence": {"integrity_pass": g0,
                            "audit_verdict": audit["gate"]["verdict"]}},
        "G1": {"pass": g1,
               "criterion": f"B2 one-step MSE < Persistence on primary group '{primary}' (frozen_test)",
               "evidence": {"b2_mse_mean": pg["mse"]["mean"],
                            "persistence_mse": pg["persistence_mse"],
                            "skill": pg["skill_vs_persistence"]}},
        "G2": {"pass": g2, "criterion": "B2 one-step MSE < B1 (action conditioning value)",
               "evidence": {"b2_mse_mean": s["B2"]["one_step"]["mse"]["mean"],
                            "b1_mse_mean": s["B1"]["one_step"]["mse"]["mean"],
                            "bootstrap": bootstrap["results"][test_split][
                                "B2 vs Unconditioned MLP"]["one_step"]}},
        "G3": {"pass": g3, "criterion": "B2 one-step MSE < B4 shuffled-action control",
               "evidence": {"b2_mse_mean": s["B2"]["one_step"]["mse"]["mean"],
                            "b4_mse_mean": s["B4"]["one_step"]["mse"]["mean"],
                            "bootstrap": bootstrap["results"][test_split][
                                "B2 vs Shuffled-Action Control"]["one_step"]}},
        "G4": {"pass": g4,
               "criterion": "horizon 5/10 rollouts finite and skill > 0 vs Persistence",
               "evidence": {"h5": {k: h5[k]["mean"] for k in ("mse", "skill_vs_persistence")},
                            "h10": {k: h10[k]["mean"] for k in ("mse", "skill_vs_persistence")}}},
        "overall_pass": all((g0, g1, g2, g3, g4)),
        "note": "Gates use mean improvements on frozen_test; paired bootstrap CIs are "
                "reported separately as statistical support, not as a substituted claim.",
    }
    (eval_dir / "gates.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")

    metrics = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset_version": manifest["dataset_version"],
        "manifest_sha256": tman["manifest_sha256"],
        "config_hash": cfg_hash,
        "git_commit": tman["git_commit"],
        "python_version": tman["python_version"],
        "torch_version": tman["torch_version"],
        "device": tman["device"],
        "seeds": seeds,
        "horizons": horizons,
        "feature_groups": feature_groups,
        "primary_group": primary,
        "normalization": stats.to_dict(),
        "splits": {n: {"episodes": b.batch_size,
                       "transitions": b.num_valid_transitions()}
                   for n, b in batches.items()},
        "summary": summary,
        "gates": gates,
        "scientific_caveats": [
            "Pusher-v1 has 0/30000 fingertip-object contacts; no pushing claim.",
            "ood_test differs only by reset seed; NOT a verified OOD split.",
            "frozen_test was never used for training, early stopping or alpha selection.",
        ],
    }
    (eval_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    make_plots(eval_dir, runs, summary, batches, stats, manifest, cfg_hash,
               out_root, seeds, horizons, feature_groups, cfg, D, U, hidden)

    print(f"[eval] gates: " + ", ".join(
        f"{k}={'PASS' if v['pass'] else 'FAIL'}" for k, v in gates.items()
        if k.startswith("G")
    ))
    print(f"[eval] outputs -> {eval_dir}")
    return 0


def _safe_ep(r: dict, h: str, i: int) -> float:
    try:
        return float(r["horizon_episode_errors"][h][i])
    except (KeyError, IndexError):
        return float("nan")


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def make_plots(eval_dir, runs, summary, batches, stats, manifest, cfg_hash,
               out_root, seeds, horizons, feature_groups, cfg, D, U, hidden):
    test_split = cfg["dataset"]["splits"]["test"]
    pp = eval_dir / "prediction_plots"
    rp = eval_dir / "rollout_plots"
    s = summary[test_split]

    # 1) per-dimension one-step RMSE
    fig, ax = plt.subplots(figsize=(11, 4.2))
    xs = np.arange(D)
    for b in BASELINES:
        ax.plot(xs, s[b]["per_dim_rmse_mean"], marker="o", ms=3,
                label=LABELS[b], color=COLORS[b], alpha=0.9)
    for name, (a, e) in feature_groups.items():
        if name == "robot_kinematics":
            continue
        ax.axvline(a - 0.5, color="#cccccc", lw=0.7, ls=":")
    ax.set_xticks(xs)
    ax.set_xlabel("observation dimension")
    ax.set_ylabel("one-step RMSE (original units)")
    ax.set_title(f"Per-dimension one-step RMSE - {test_split} (mean of {len(seeds)} seeds)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(pp / "per_dimension_rmse.png", dpi=140)
    plt.close(fig)

    # 2) primary-group skill
    primary = cfg["dataset"]["primary_group"]
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = [s[b]["feature_groups_one_step"][primary]["mse"]["mean"] for b in BASELINES]
    errs = [s[b]["feature_groups_one_step"][primary]["mse"]["std"] for b in BASELINES]
    pmse = s["B2"]["feature_groups_one_step"][primary]["persistence_mse"]
    ax.bar([LABELS[b] for b in BASELINES], vals, yerr=errs,
           color=[COLORS[b] for b in BASELINES], alpha=0.85, capsize=4)
    ax.axhline(pmse, color="black", ls="--", lw=1.2, label="Persistence (B0)")
    ax.set_ylabel("MSE")
    ax.set_title(f"One-step MSE on '{primary}' - {test_split}")
    ax.legend(fontsize=8)
    plt.xticks(rotation=18, ha="right", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(pp / "primary_group_mse.png", dpi=140)
    plt.close(fig)

    # 3) horizon RMSE / skill
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for b in BASELINES:
        axes[0].plot(horizons,
                     [s[b]["horizons"][str(h)]["rmse"]["mean"] for h in horizons],
                     marker="o", label=LABELS[b], color=COLORS[b])
        axes[1].plot(horizons,
                     [s[b]["horizons"][str(h)]["skill_vs_persistence"]["mean"]
                      for h in horizons],
                     marker="s", label=LABELS[b], color=COLORS[b])
    axes[0].set_title("Open-loop rollout RMSE vs horizon")
    axes[0].set_ylabel("RMSE (original units)")
    axes[1].set_title("Skill vs Persistence (1 - MSE_model/MSE_B0)")
    axes[1].axhline(0, color="black", lw=0.8, ls="--")
    for ax in axes:
        ax.set_xlabel("horizon")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(rp / "horizon_rmse_and_skill.png", dpi=140)
    plt.close(fig)

    # 4) sample open-loop trajectories (true / B0 / B2)
    try:
        batch = batches[test_split]
        seed = seeds[0]
        m2 = load_model("B2", out_root, cfg_hash, seed, D, U, hidden)
        m0 = PersistenceBaseline(D)
        H = max(horizons)
        x0 = batch.observations[:1, 0]
        acts = batch.conditions[:1, :H] if batch.conditions is not None else None
        true = batch.observations[:1, 1:H + 1].numpy()[0]
        p2 = open_loop_rollout(m2, x0, acts, stats, H).numpy()[0]
        p0 = open_loop_rollout(m0, x0, acts, stats, H).numpy()[0]
        dims = [(0, "qpos[0]"), (7, "qvel[0]"), (14, "fingertip_x"), (17, "object_x")]
        fig, axes = plt.subplots(2, 2, figsize=(11, 6))
        for ax, (d, name) in zip(axes.ravel(), dims):
            ax.plot(np.arange(H), true[:, d], "k-", lw=2, label="true")
            ax.plot(np.arange(H), p2[:, d], color=COLORS["B2"], ls="-",
                    label="B2 open-loop")
            ax.plot(np.arange(H), p0[:, d], color=COLORS["B0"], ls="--",
                    label="Persistence")
            ax.set_title(name)
            ax.set_xlabel("rollout step")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.suptitle(f"Sample {H}-step open-loop rollout (frozen_test, seed {seed})")
        fig.tight_layout()
        fig.savefig(rp / "sample_open_loop_trajectory.png", dpi=140)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - plotting must not break eval
        (rp / "sample_open_loop_trajectory.ERROR.txt").write_text(
            repr(exc), encoding="utf-8"
        )

    # 5) one-step predicted vs true scatter for the most dynamic dimension
    try:
        batch = batches[test_split]
        seed = seeds[0]
        m2 = load_model("B2", out_root, cfg_hash, seed, D, U, hidden)
        from dynamics import training_tensors
        xn, un, tn, mask = training_tensors(batch, stats)
        with torch.no_grad():
            pn = m2(xn, un)
        pred = stats.denormalize_state(pn)[mask].numpy()
        true = stats.denormalize_state(tn)[mask].numpy()
        d = int(np.argmax(np.abs(true - pred).mean(axis=0)))
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(true[:, d], pred[:, d], s=2, alpha=0.25, color=COLORS["B2"])
        lo, hi = float(true[:, d].min()), float(true[:, d].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel(f"true x_next[dim {d}]")
        ax.set_ylabel(f"predicted x_next[dim {d}]")
        ax.set_title(f"B2 one-step prediction, {test_split}")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(pp / "one_step_pred_vs_true.png", dpi=140)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        (pp / "one_step_pred_vs_true.ERROR.txt").write_text(
            repr(exc), encoding="utf-8"
        )


if __name__ == "__main__":
    raise SystemExit(main())
