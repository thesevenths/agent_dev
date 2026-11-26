"""
evaluate.py
===========
SciNet++ (2026) -- Stage 4: V-JEPA multi-step latent evaluation.

Loads a trained world model and rolls the *same* latent predictor used in
training across every configured horizon, comparing predictions against the EMA
target encoder's representations. Because training and evaluation share the
identical multi-step rollout, these numbers are meaningful (the previous version
trained a single-step predictor but evaluated it with an iterated multi-step
rollout, which measured nothing).

Reports, per horizon k:
    latent MSE   -- lower is better
    cosine sim   -- higher (->1) is better; robust to latent scale

Outputs:
    <output_dir>/horizon_mse.png, <output_dir>/horizon_cosine.png

Usage:
    python evaluate.py --config configs/pendulum.yaml
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import SPLITS, build_dataset, state_dim, validate_config
from models.world_model import WorldModel
from utils.config import ensure_output_dir, load_config, resolve_device, make_run_id, setup_logging
from utils.metrics import cosine_similarity
from utils.plotting import horizon_metric_plot
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V-JEPA multi-step evaluation")
    parser.add_argument("--config", default="configs/pendulum.yaml")
    parser.add_argument(
        "--run-id",
        default=None,
        help="reuse a specific run sub-directory (timestamp); auto-generated if omitted",
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="force a fresh timestamped run sub-directory even if one exists",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=list(SPLITS),
        help="split to evaluate on. Use 'validation' for candidate selection; "
             "reserve 'test' for the final independent evaluation "
             "(default: test, preserving historical behaviour).",
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate_rollout(model, loader, horizons, device):
    """Aggregate per-horizon latent MSE and cosine similarity over the dataset.

    For each test batch:
      1. Encode context block (block 0) with online encoder -> z_context
      2. Roll predictor k steps: z_pred_k = f^k(z_context)
      3. Encode block k with EMA target encoder -> z_target_k (no gradient)
      4. Compute MSE(z_pred_k, z_target_k) and cosine(z_pred_k, z_target_k)

    Why both MSE and cosine?
      - MSE measures absolute error but depends on latent scale.
      - Cosine measures directional alignment, scale-invariant.
      - A model can have low MSE but wrong direction (scale mismatch),
        or high cosine but large magnitude error. Both together give the full picture.
    """
    model.eval()
    mse_sum = {k: 0.0 for k in horizons}   # accumulate MSE per horizon
    cos_sum = {k: 0.0 for k in horizons}   # accumulate cosine per horizon
    n_batches = 0

    for batch in loader:
        blocks = batch["blocks"].to(device)
        # Forward pass: returns predicted latents, EMA target latents, and context latent
        preds, targets, _ = model(blocks, horizons, context_block=0)
        for k in horizons:
            p, t = preds[k], targets[k]
            mse_sum[k] += float(torch.mean((p - t) ** 2))           # mean squared error
            cos_sum[k] += cosine_similarity(p.cpu().numpy(), t.cpu().numpy())  # directional alignment
        n_batches += 1

    n_batches = max(n_batches, 1)
    mse = {k: mse_sum[k] / n_batches for k in horizons}  # average MSE per horizon
    cos = {k: cos_sum[k] / n_batches for k in horizons}  # average cosine per horizon
    return mse, cos


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.new:
        cfg["run_id"] = make_run_id()
    elif args.run_id:
        cfg["run_id"] = args.run_id
    validate_config(cfg)

    set_seed(cfg["seed"])
    device = resolve_device(cfg)
    out_dir = ensure_output_dir(cfg)
    setup_logging(out_dir, name="evaluate")
    horizons = sorted(cfg["jepa"]["rollout_steps"])

    # Load trained weights from checkpoint saved by train.py
    model = WorldModel.from_config(cfg, state_dim(cfg)).to(device)
    ckpt = torch.load(os.path.join(out_dir, "checkpoint.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # Evaluate on the EXPLICITLY requested split. Test is reserved for the
    # final independent evaluation; candidate selection must use validation.
    eval_set = build_dataset(cfg, split=args.split)
    loader = DataLoader(eval_set, batch_size=cfg["train"]["batch_size"], shuffle=False)
    print(
        f"[data] evaluate split={eval_set.split} samples={len(eval_set)} "
        f"dataset_version={eval_set.dataset_version} frozen={eval_set.frozen_path}"
    )

    # Run multi-step rollout evaluation across all configured horizons
    mse, cos = evaluate_rollout(model, loader, horizons, device)

    print(f"=== V-JEPA multi-step latent prediction (split={eval_set.split}) ===")
    print(f"{'horizon':>8} | {'latent MSE':>12} | {'cosine sim':>12}")
    print("-" * 38)
    for k in horizons:
        print(f"{k:>8} | {mse[k]:>12.5f} | {cos[k]:>12.4f}")

    horizon_metric_plot(
        horizons, [mse[k] for k in horizons], "Latent MSE",
        output_dir=out_dir, name="horizon_mse",
    )
    horizon_metric_plot(
        horizons, [cos[k] for k in horizons], "Cosine similarity",
        output_dir=out_dir, name="horizon_cosine",
    )
    # Save numerical metrics for downstream analysis (e.g. comparing configs).
    rows = np.array([[k, mse[k], cos[k]] for k in horizons])
    # Canonical name kept for downstream stages and the worker's required
    # artifact check (horizon_metrics.npy must exist for a successful job).
    np.save(os.path.join(out_dir, "horizon_metrics.npy"), rows)
    # A split-tagged copy keeps validation and test evidence independent: a
    # later "--split test" run on the same run_id cannot overwrite the metrics
    # that a "--split validation" job was judged on.
    np.save(os.path.join(out_dir, f"horizon_metrics_{eval_set.split}.npy"), rows)
    meta = {
        "split": eval_set.split,
        "dataset_version": eval_set.dataset_version,
        "frozen_path": eval_set.frozen_path,
        "samples": int(len(eval_set)),
        "horizons": horizons,
    }
    # Canonical copy for humans / downstream tooling, plus a split-tagged copy
    # that a later evaluation on another split cannot overwrite. result.json
    # reconstruction must be able to prove which split a given job used.
    for name in ("evaluation_meta.json",
                 f"evaluation_meta_{eval_set.split}.json"):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"\nSaved horizon plots + metrics (split={eval_set.split}) to {out_dir}/")


if __name__ == "__main__":
    main()
