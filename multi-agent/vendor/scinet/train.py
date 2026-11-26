"""
train.py
========
SciNet++ (2026) -- Stage 1: V-JEPA representation learning.

Trains the world model (online encoder + latent predictor + EMA target encoder)
to predict *future latent representations* over multiple rollout horizons. The
objective combines

    loss = pred_weight * multi_step_latent_MSE
         + vicreg_var_weight * variance(z)
         + vicreg_cov_weight * covariance(z)

The EMA target encoder plus the VICReg variance/covariance terms are what keep
the latent space from collapsing -- the fatal flaw of a shared-encoder + plain
MSE JEPA. A cosine EMA-momentum warmup (BYOL-style) stabilises early training.

Outputs (into cfg['output_dir']):
    checkpoint.pt   -- online encoder + predictor + EMA target + resolved config
    TensorBoard logs (loss terms, per-horizon cosine, latent-std collapse watch)

Usage:
    python train.py --config configs/pendulum.yaml
    python train.py --config configs/pendulum.yaml --epochs 3   # quick smoke run
"""

from __future__ import annotations

import argparse
import math
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:  # TensorBoard is optional; training still runs (console logs only) without it.
    from torch.utils.tensorboard import SummaryWriter

    _TB_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    _TB_AVAILABLE = False

    class SummaryWriter:  # minimal no-op fallback
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

from data.dataset import (
    build_dataset,
    num_blocks,
    state_dim,
    validate_config,
)
from models.world_model import WorldModel
from utils.config import ensure_output_dir, load_config, resolve_device, make_run_id, setup_logging
from utils.losses import prediction_loss, vicreg_loss
from utils.metrics import cosine_similarity, latent_std_per_dim
from utils.seed import set_seed


def ema_momentum(step: int, warmup_steps: int, base_momentum: float) -> float:
    """Cosine-ramp the EMA momentum from ``base_momentum`` up to ~1.0.

    BYOL/V-JEPA trick: start with low momentum (target encoder tracks online
    closely) and ramp toward 1.0 (target becomes nearly frozen). This prevents
    early-training instability when both encoders are still random.
    Formula: m = 1 - (1 - base) * (cos(pi * progress) + 1) / 2
    """
    if warmup_steps <= 0:
        return base_momentum
    progress = min(step / warmup_steps, 1.0)
    return 1.0 - (1.0 - base_momentum) * (math.cos(math.pi * progress) + 1.0) / 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V-JEPA training for SciNet++")
    parser.add_argument("--config", default="configs/pendulum.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="override cfg epochs")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.new:
        cfg["run_id"] = make_run_id()
    elif args.run_id:
        cfg["run_id"] = args.run_id
    validate_config(cfg)

    set_seed(cfg["seed"])
    device = resolve_device(cfg)
    out_dir = ensure_output_dir(cfg)
    setup_logging(out_dir, name="train")

    horizons = cfg["jepa"]["rollout_steps"]       # e.g. [1,2,3,4] = predict 1-4 blocks ahead
    max_horizon = max(horizons)
    # context_block + max_horizon must be < num_blocks, so max valid context is:
    max_context = num_blocks(cfg) - 1 - max_horizon  # last valid context block index

    train_set = build_dataset(cfg, split="train")
    # Training-time diagnostics belong to VALIDATION only. The test split is
    # frozen evidence and must not be read during training -- not even for
    # monitoring, because an Agent that later selects checkpoints using these
    # numbers would leak test information into model selection.
    validation_set = build_dataset(cfg, split="validation")
    train_loader = DataLoader(
        train_set, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=True
    )
    validation_loader = DataLoader(
        validation_set, batch_size=cfg["train"]["batch_size"], shuffle=False
    )
    for name, ds in (("train", train_set), ("validation", validation_set)):
        print(
            f"[data] {name} split={ds.split} samples={len(ds)} "
            f"dataset_version={ds.dataset_version} frozen={ds.frozen_path}"
        )

    model = WorldModel.from_config(cfg, state_dim(cfg)).to(device)

    # Only train online encoder + predictor; target_encoder has requires_grad=False
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        trainable, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    epochs = cfg["train"]["epochs"]
    # Cosine LR schedule: decays from lr to 0 over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))
    if not _TB_AVAILABLE:
        print("[info] tensorboard not installed -- logging to console only.")
    # EMA warmup: ramp momentum over first N epochs (in steps, not epochs)
    warmup_steps = max(int(cfg["jepa"]["ema_warmup_epochs"] * len(train_loader)), 1)

    global_step = 0
    for epoch in range(epochs):
        model.train()
        run_pred, run_vic, run_total = 0.0, 0.0, 0.0  # running loss accumulators
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            # Move input blocks to GPU (data lives in CPU RAM, batched to device)
            blocks = batch["blocks"].to(device)
            # Random context block per batch: prevents the model from memorising
            # a fixed position and forces it to learn position-invariant dynamics
            context = random.randint(0, max_context) if max_context > 0 else 0

            # Forward: encode context -> predict multi-step latents -> EMA targets
            preds, targets, z_ctx = model(blocks, horizons, context_block=context)

            # Loss = prediction MSE (averaged over horizons) + VICReg anti-collapse
            pred_l, _ = prediction_loss(preds, targets, horizons)
            vic_l = vicreg_loss(
                z_ctx,
                cfg["jepa"]["vicreg_var_weight"],   # variance hinge: push std toward 1.0
                cfg["jepa"]["vicreg_cov_weight"],    # covariance penalty: decorrelate dims
            )
            loss = cfg["jepa"]["pred_weight"] * pred_l + vic_l

            optimizer.zero_grad()
            loss.backward()      # gradients flow through online encoder + predictor only
            optimizer.step()

            # Update target encoder via EMA (no gradient, just parameter copying)
            momentum = ema_momentum(global_step, warmup_steps, cfg["jepa"]["ema_momentum"])
            model.update_target_encoder(momentum)

            # Accumulate losses for epoch-level logging (.detach() to free compute graph)
            run_pred += float(pred_l.detach())
            run_vic += float(vic_l.detach())
            run_total += float(loss.detach())
            n_batches += 1
            global_step += 1

        scheduler.step()  # cosine LR decay once per epoch

        # ---- epoch diagnostics (validation + collapse watch) ---------------
        # Deliberately the validation loader: test stays untouched until the
        # final independent evaluation (evaluate.py --split test).
        val = evaluate_epoch(model, validation_loader, horizons, max_context, device)
        std_mean = float(val["latent_std"].mean())  # <<1.0 = collapse danger

        # TensorBoard scalars for monitoring training dynamics
        writer.add_scalar("loss/train_total", run_total / n_batches, epoch)
        writer.add_scalar("loss/train_pred", run_pred / n_batches, epoch)
        writer.add_scalar("loss/train_vicreg", run_vic / n_batches, epoch)
        writer.add_scalar("loss/val_pred", val["pred"], epoch)          # rising = overfitting?
        writer.add_scalar("diagnostics/latent_std_mean", std_mean, epoch)  # collapse watch
        writer.add_scalar("diagnostics/ema_momentum", momentum, epoch)
        for k, c in val["cosine"].items():
            writer.add_scalar(f"val_cosine/h{k}", c, epoch)  # per-horizon alignment

        # Console log: one line per epoch with the four key metrics
        print(
            f"[epoch {epoch:3d}] train={run_total / n_batches:.4f} "
            f"val_pred={val['pred']:.4f} latent_std={std_mean:.3f} "
            f"cos(h1)={val['cosine'].get(min(horizons), float('nan')):.3f}"
        )

    writer.close()

    # Save everything needed to rebuild + resume: weights, config, dimensions
    checkpoint_path = os.path.join(out_dir, "checkpoint.pt")
    torch.save(
        {
            "model_state": model.state_dict(),  # online encoder + predictor + target encoder
            "config": cfg,                       # full resolved config for reproducibility
            "state_dim": state_dim(cfg),          # generator state dim (1 or 4)
            "latent_dim": cfg["model"]["latent_dim"],
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")


@torch.no_grad()
def evaluate_epoch(model, loader, horizons, max_context, device) -> dict:
    """One label-free validation pass: prediction MSE, per-horizon cosine, latent std."""
    model.eval()
    context = 0
    total, n = 0.0, 0
    cos_acc = {k: 0.0 for k in horizons}
    latents = []

    for batch in loader:
        blocks = batch["blocks"].to(device)
        preds, targets, z_ctx = model(blocks, horizons, context_block=context)
        pred_l, _ = prediction_loss(preds, targets, horizons)
        total += float(pred_l)
        n += 1
        for k in horizons:
            cos_acc[k] += cosine_similarity(
                preds[k].cpu().numpy(), targets[k].cpu().numpy()
            )
        latents.append(z_ctx.cpu().numpy())

    model.train()
    all_z = np.concatenate(latents, axis=0)
    return {
        "pred": total / max(n, 1),
        "cosine": {k: cos_acc[k] / max(n, 1) for k in horizons},
        "latent_std": latent_std_per_dim(all_z),
    }


if __name__ == "__main__":
    main()
