#!/usr/bin/env python3
"""
P2-02 Phase 3 -- train the raw-state dynamics baselines on Pusher-v1.

    x_hat(t+1) = x(t) + g_theta(x(t), u(t))

Reads the frozen dataset through the A1 ``DynamicsBatch`` contract, fits
normalisation on the TRAIN split only, and trains B1/B2/B3/B4 with a common
protocol (B0 needs no training). Validation is used for early stopping / ridge
alpha selection; frozen_test is NEVER touched here.

Nothing in this script modifies the Pusher-v1 dataset, its manifest, legacy
models, the JEPA pipeline, or historical checkpoints.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynamics import (  # noqa: E402
    dynamics_batch_from_episodes,
    fit_normalization,
    training_tensors,
)
from dynamics.metrics import masked_mse_loss  # noqa: E402
from models.state_dynamics import (  # noqa: E402
    ResidualStatePredictor,
    RidgeDynamics,
    shuffle_conditions,
)

BASELINES = ("B1", "B2", "B3", "B4")
BASELINE_DESC = {
    "B0": "Persistence (x_next = x), no training",
    "B1": "Unconditioned MLP (x + f(x))",
    "B2": "Action-conditioned MLP (x + g(x,u))",
    "B3": "Linear / Ridge dynamics (closed form)",
    "B4": "Shuffled-action negative control (B2 architecture)",
}


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_phase0_gate(root: Path, manifest: dict, output_root: Path) -> dict:
    """Refuse to train unless the Phase 0 audit passed for THIS manifest."""
    audit_path = output_root / "data_audit.json"
    if not audit_path.is_file():
        raise RuntimeError(
            "Phase 0 audit missing: run scripts/audit_pusher_dynamics.py first"
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("gate", {}).get("integrity_pass"):
        raise RuntimeError(
            f"Phase 0 audit FAILED: {audit.get('gate', {}).get('failed_checks')}. "
            "STOP - refusing to train."
        )
    declared = audit["dataset"]["manifest_sha256"]
    actual = sha256_file(root / "manifest.json")
    if declared != actual:
        raise RuntimeError(
            "manifest.json changed since the Phase 0 audit; re-run the audit"
        )
    return audit


def verify_manifest(root: Path, manifest: dict) -> None:
    n = 0
    for split, hashes in manifest["split_file_hashes"].items():
        for epid, expected in hashes.items():
            f = root / split / f"{epid}.npz"
            if not f.is_file() or sha256_file(f) != expected:
                raise RuntimeError(f"dataset integrity FAILED for {f}")
            n += 1
    print(f"[train] manifest verified: {n} episode files, SHA-256 all match")


def load_split_batch(root: Path, split: str, manifest: dict, D: int, U: int):
    """Build ONE DynamicsBatch for a split from frozen npz files.

    obs_seq[t] = obs[t] and obs_seq[T] = next_obs[T-1]. Phase 0 proved
    obs[t+1] == next_obs[t] bitwise for all 30,000 transitions, so the last
    stored next_obs is the final state and no transition crosses an episode.
    """
    ids = manifest["split_episode_ids"][split]
    obs_seqs: List[np.ndarray] = []
    conds: List[np.ndarray] = []
    for epid in ids:
        f = root / split / f"{epid}.npz"
        with np.load(f, allow_pickle=False) as z:
            obs = z["obs"]
            nxt = z["next_obs"]
            act = z["action"]
        if obs.shape[1] != D or act.shape[1] != U:
            raise ValueError(f"dimension mismatch in {f}")
        seq = np.concatenate([obs, nxt[-1:]], axis=0)  # [T+1, D]
        obs_seqs.append(seq)
        conds.append(act)                              # [T, U]
    return dynamics_batch_from_episodes(obs_seqs, conds, ids)


def prepare(batch, stats):
    """Return normalised 3-D tensors plus flattened valid pairs."""
    x, u, target, mask = training_tensors(batch, stats)
    flat = lambda a: None if a is None else a.reshape(-1, a.shape[-1])
    xf, uf, tf = flat(x), flat(u), flat(target)
    mf = mask.reshape(-1)
    idx = torch.nonzero(mf).squeeze(-1)
    return {
        "x": x, "u": u, "target": target, "mask": mask,
        "xf": xf[idx],
        "uf": None if uf is None else uf[idx],
        "tf": tf[idx],
        "n": int(idx.numel()),
    }


# --------------------------------------------------------------------------- #
# models / training
# --------------------------------------------------------------------------- #
def build_mlp_model(baseline: str, D: int, U: int, hidden) -> ResidualStatePredictor:
    cond_dim = None if baseline == "B1" else U
    return ResidualStatePredictor(D, cond_dim, hidden)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one(
    baseline: str,
    seed: int,
    cfg: dict,
    train_p: dict,
    val_p: dict,
    stats,
    out_dir: Path,
    metadata: dict,
) -> dict:
    """Train one (baseline, seed) and write checkpoint/history/repro."""
    t0 = time.time()
    set_seed(seed)
    D, U = int(cfg["dataset"]["observation_dim"]), int(cfg["dataset"]["action_dim"])
    hidden = tuple(int(h) for h in cfg["model"]["hidden_dims"])
    tr = cfg["training"]

    cond_dim = None if baseline == "B1" else U
    # B1 is an unconditioned ablation: it must NEVER be handed an action.
    # B4 destroys the action<->state correspondence in the TRAIN split only;
    # its validation / evaluation always use correctly aligned actions.
    tr_u_3d = train_p["u"] if cond_dim is not None else None
    if baseline == "B4":
        tr_u_3d = shuffle_conditions(
            train_p["u"], train_p["mask"],
            seed=int(seed + int(tr["shuffle_seed_offset"])),
        )
        flat_u = tr_u_3d.reshape(-1, tr_u_3d.shape[-1])
        mf = train_p["mask"].reshape(-1)
        idx = torch.nonzero(mf).squeeze(-1)
        train_flat_u = flat_u[idx]
    elif cond_dim is None:
        train_flat_u = None
    else:
        train_flat_u = train_p["uf"]

    val_u_3d = val_p["u"] if cond_dim is not None else None
    val_u_flat = val_p["uf"] if cond_dim is not None else None

    result: Dict = {
        "baseline": baseline,
        "description": BASELINE_DESC[baseline],
        "seed": int(seed),
        "state_dim": D,
        "condition_dim": cond_dim,
        "hidden_dims": list(hidden),
    }

    if baseline == "B3":
        best = {"alpha": None, "val_mse": float("inf"), "state": None}
        for a in tr["ridge_alphas"]:
            model = RidgeDynamics(D, U, alpha=float(a))
            model.fit(
                train_p["x"],
                train_p["target"] - train_p["x"],
                tr_u_3d,
                train_p["mask"],
                alpha=float(a),
            )
            with torch.no_grad():
                vpred = model(val_p["x"], val_u_3d)
            vloss = float(
                ((vpred - val_p["target"]) ** 2).mean(dim=-1)[val_p["mask"]]
                .mean().item()
            )
            if vloss < best["val_mse"]:
                best = {"alpha": float(a), "val_mse": vloss,
                        "state": {k: v.clone() for k, v in model.state_dict().items()}}
        model = RidgeDynamics(D, U, alpha=best["alpha"])
        model.load_state_dict(best["state"])
        result.update(
            protocol="closed-form ridge; alpha selected on validation",
            alpha=best["alpha"], best_val_mse=best["val_mse"],
            best_epoch=0, epochs_run=0, history=[],
        )
    else:
        model = build_mlp_model(baseline, D, U, hidden)
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(tr["learning_rate"]),
            weight_decay=float(tr["weight_decay"]),
        )
        g = torch.Generator().manual_seed(int(seed))
        bs = int(tr["batch_size"])
        patience = int(tr["early_stopping_patience"])
        epochs = int(tr["epochs"])
        best_val, best_state, best_epoch, wait = float("inf"), None, -1, 0
        history: List[dict] = []
        N = train_p["n"]
        for ep in range(1, epochs + 1):
            model.train()
            perm = torch.randperm(N, generator=g)
            running = 0.0
            for i in range(0, N, bs):
                sel = perm[i:i + bs]
                xb = train_p["xf"][sel]
                ub = train_flat_u[sel] if train_flat_u is not None else None
                tb = train_p["tf"][sel]
                pred = model(xb, ub)
                loss = ((pred - tb) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                running += float(loss.item()) * sel.numel()
            train_loss = running / N
            model.eval()
            with torch.no_grad():
                vpred = model(val_p["xf"], val_u_flat)
                vloss = float(((vpred - val_p["tf"]) ** 2).mean().item())
            history.append({"epoch": ep, "train_mse": train_loss,
                            "val_mse": vloss})
            if vloss < best_val - 1e-12:
                best_val = vloss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_epoch = ep
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break
        if best_state is None:
            raise RuntimeError(f"{baseline} seed={seed}: no best checkpoint")
        model.load_state_dict(best_state)
        result.update(
            protocol="AdamW + masked MSE + validation early stopping",
            best_val_mse=best_val, best_epoch=best_epoch,
            epochs_run=len(history), history=history,
            alpha=None,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "baseline": baseline,
            "seed": int(seed),
            "state_dim": D,
            "condition_dim": result["condition_dim"],
            "hidden_dims": list(hidden),
            "alpha": result.get("alpha"),
            "best_epoch": result["best_epoch"],
            "best_val_mse": result["best_val_mse"],
            "config_hash": metadata["config_hash"],
            "dataset_version": metadata["dataset_version"],
        },
        out_dir / "checkpoint.pt",
    )
    (out_dir / "history.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (out_dir / "repro.json").write_text(
        json.dumps({**metadata, **{
            "baseline": baseline,
            "seed": int(seed),
            "duration_seconds": round(time.time() - t0, 3),
            "best_epoch": result["best_epoch"],
            "best_val_mse": result["best_val_mse"],
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }}, indent=2),
        encoding="utf-8",
    )
    print(
        f"[train] {baseline} seed={seed}: epochs={result['epochs_run']} "
        f"best_epoch={result['best_epoch']} val_mse={result['best_val_mse']:.6f} "
        f"({time.time() - t0:.1f}s) -> {out_dir.name}"
    )
    return result


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="P2-02 baseline dynamics training")
    ap.add_argument("--config", default="configs/dynamics/pusher_v1_baselines.yaml")
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seeds", default=None, help="comma-separated seeds")
    ap.add_argument("--baselines", default=None, help="comma-separated, e.g. B1,B2")
    ap.add_argument("--force", action="store_true", help="allow overwriting an experiment")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if args.output_root:
        cfg["experiment"]["output_root"] = args.output_root
    if args.epochs is not None:
        cfg["training"]["epochs"] = int(args.epochs)
    if args.seeds:
        cfg["training"]["seeds"] = [int(s) for s in args.seeds.split(",")]
    if args.baselines:
        cfg["training"]["baselines"] = [s.strip() for s in args.baselines.split(",")]

    root = Path(cfg["dataset"]["root"])
    out_root = Path(cfg["experiment"]["output_root"])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if cfg["dataset"].get("verify_manifest", True):
        verify_manifest(root, manifest)
    if cfg["dataset"].get("require_phase0_audit", True):
        audit = check_phase0_gate(root, manifest, out_root)
        print(f"[train] Phase 0 gate: PASS ({audit['dataset']['dataset_version']})")

    D, U = int(cfg["dataset"]["observation_dim"]), int(cfg["dataset"]["action_dim"])
    split_names = cfg["dataset"]["splits"]
    train_b = load_split_batch(root, split_names["train"], manifest, D, U)
    val_b = load_split_batch(root, split_names["validation"], manifest, D, U)
    print(f"[train] train {train_b.batch_size} ep / {train_b.num_valid_transitions()} tr; "
          f"validation {val_b.batch_size} ep / {val_b.num_valid_transitions()} tr")

    stats = fit_normalization(train_b)
    stats.save(out_root / "normalization.json")
    train_p, val_p = prepare(train_b, stats), prepare(val_b, stats)

    cfg_canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    cfg_hash = hashlib.sha256(cfg_canonical.encode("utf-8")).hexdigest()
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        git_commit = "unavailable"
    metadata = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "numpy_version": np.__version__,
        "git_commit": git_commit,
        "git_branch": cfg["experiment"]["name"],
        "dataset_version": manifest["dataset_version"],
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "config_path": str(cfg_path),
        "config_hash": cfg_hash,
        "experiment_name": cfg["experiment"]["name"],
        # Record the ACTUALLY used device (small MLP -> CPU, fully deterministic).
        # We never silently claim CUDA when tensors stayed on CPU.
        "device": str(train_b.observations.device),
        "device_config": cfg["experiment"]["device"],
        "training_config": cfg["training"],
        "model_config": cfg["model"],
        "normalization": stats.to_dict(),
    }
    (out_root / "training_manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    seeds = [int(s) for s in cfg["training"]["seeds"]]
    baselines = [str(b) for b in cfg["training"]["baselines"]]
    for b in baselines:
        if b not in BASELINES:
            raise ValueError(f"unknown baseline {b}; expected one of {BASELINES}")
    tag = cfg_hash[:8]
    planned = []
    for baseline in baselines:
        for seed in seeds:
            exp_id = f"{tag}_{baseline}_seed{seed}"
            d = out_root / exp_id
            if (d / "checkpoint.pt").is_file() and not args.force:
                raise FileExistsError(
                    f"experiment already exists: {d}. Use --force to overwrite "
                    "(or choose a different config/output root)."
                )
            planned.append((baseline, seed, exp_id, d))

    print(f"[train] config_hash={cfg_hash[:16]}... experiments={len(planned)} "
          f"seeds={seeds} baselines={baselines}")
    for baseline, seed, exp_id, d in planned:
        train_one(baseline, seed, cfg, train_p, val_p, stats, d, metadata)

    print(f"[train] DONE. {len(planned)} experiments under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
