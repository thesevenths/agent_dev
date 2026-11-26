"""
Audit helper: is Pusher-v1's `ood_test` split genuinely out-of-distribution?

Method
------
All four splits are produced by robotics/data/splits.plan_collection, which
allocates the SAME three policies (random/guided/mixture) on the SAME Pusher-v5
env across split-local reset seeds. The only difference between `train` and
`ood_test` is the split name inside the seed hash. If that is the *only*
difference, the splits are IID draws from one process and `ood_test` is NOT
genuinely OOD.

We test this three ways:
  1. Per-split feature summaries (init / final object-goal distance, action
     magnitude, reward, contact rate). IID -> identical summaries.
  2. Two-sample Kolmogorov-Smirnov test: train vs ood_test on init_ogd.
  3. A logistic-regression "separability" probe: can a classifier tell
     `train` episodes from `ood_test` episodes using only initial-condition
     features? Accuracy ~ 0.5 = indistinguishable = not OOD.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from robotics.data.dataset import load_split, split_episode_ids  # noqa: E402

OUTPUT_DIR = ROOT / "datasets" / "pusher_v1"
SPLITS = ("train", "validation", "frozen_test", "ood_test")


def episode_features(traj):
    """Return summary features for one trajectory (numpy arrays)."""
    obs = np.stack([t.obs for t in traj.transitions])  # (T, 23)
    acts = np.stack([t.action for t in traj.transitions])  # (T, 7)
    rews = np.array([t.reward for t in traj.transitions])
    contacts = [t.contact_state for t in traj.transitions if t.contact_state is not None]
    init_ogd = float(np.linalg.norm(obs[0, 17:20] - obs[0, 20:23]))
    final_ogd = float(np.linalg.norm(obs[-1, 17:20] - obs[-1, 20:23]))
    return {
        "init_ogd": init_ogd,
        "final_ogd": final_ogd,
        "mean_abs_action": float(np.abs(acts).mean()),
        "mean_reward": float(rews.mean()),
        "contact_rate": float(np.mean(contacts)) if contacts else float("nan"),
    }


def main():
    assert OUTPUT_DIR.is_dir(), f"dataset missing: {OUTPUT_DIR}"
    ids = split_episode_ids(OUTPUT_DIR)
    feats = {s: [] for s in SPLITS}
    for s in SPLITS:
        for traj in load_split(OUTPUT_DIR, s):
            feats[s].append(episode_features(traj))

    keys = ["init_ogd", "final_ogd", "mean_abs_action", "mean_reward", "contact_rate"]
    print("=== per-split feature summary (mean [std]) ===")
    print(f"{'split':12s}" + "".join(f"{k:>16s}" for k in keys))
    stats = {}
    for s in SPLITS:
        arr = {k: np.array([f[k] for f in feats[s]]) for k in keys}
        stats[s] = arr
        row = f"{s:12s}"
        for k in keys:
            row += f"{arr[k].mean():14.4f}[{arr[k].std():.3f}]"
        print(row)

    # 2) KS test train vs ood_test on init_ogd
    try:
        from scipy.stats import ks_2samp
        ks = ks_2samp(stats["train"]["init_ogd"], stats["ood_test"]["init_ogd"])
        print(f"\nKS(train vs ood_test, init_ogd): D={ks.statistic:.4f} p={ks.pvalue:.4f}")
        ks2 = ks_2samp(stats["train"]["final_ogd"], stats["ood_test"]["final_ogd"])
        print(f"KS(train vs ood_test, final_ogd): D={ks2.statistic:.4f} p={ks2.pvalue:.4f}")
    except Exception as e:  # pragma: no cover
        print(f"\nscipy unavailable for KS test: {e}")

    # 3) logistic separability probe train vs ood_test
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        Xtr = np.stack([stats["train"]["init_ogd"], stats["train"]["final_ogd"],
                        stats["train"]["mean_abs_action"]]).T
        Xood = np.stack([stats["ood_test"]["init_ogd"], stats["ood_test"]["final_ogd"],
                         stats["ood_test"]["mean_abs_action"]]).T
        X = np.vstack([Xtr, Xood])
        y = np.array([0] * len(Xtr) + [1] * len(Xood))
        clf = LogisticRegression(max_iter=1000)
        acc = cross_val_score(clf, X, y, cv=5).mean()
        print(f"\nSeparability probe (train vs ood_test, 5-fold CV accuracy): "
              f"{acc:.3f}  (0.5 = chance => not OOD)")
    except Exception as e:  # pragma: no cover
        print(f"\nsklearn unavailable for separability probe: {e}")

    # Also: do the splits even contain different *policy mixes*?
    print("\n=== split policy mix ===")
    for s in SPLITS:
        mix = {}
        for traj in load_split(OUTPUT_DIR, s):
            mix[traj.policy] = mix.get(traj.policy, 0) + 1
        print(f"{s:12s} {mix}")


if __name__ == "__main__":
    main()
