"""
utils.losses
============
SciNet++ (2026) -- V-JEPA training objectives.

Two ingredients make a Joint-Embedding Predictive Architecture train without
collapsing to a constant latent:

1. ``prediction_loss`` -- multi-step latent prediction error between the online
   predictor's rollouts and the EMA target encoder's representations, averaged
   over all rollout horizons.
2. ``vicreg_loss`` -- the VICReg variance + covariance penalty applied to the
   online latents. The variance term forces each latent dimension to vary across
   the batch (hinge at a target std), and the covariance term decorrelates the
   dimensions. Together with the EMA target they remove the trivial
   "encode everything to the same vector" solution.
"""

from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn.functional as F


def prediction_loss(
    preds: Dict[int, torch.Tensor],
    targets: Dict[int, torch.Tensor],
    horizons: Iterable[int],
):
    """Mean latent MSE across rollout horizons.

    For each horizon k, computes MSE(online_predictor^k(z_context), target_encoder(z_{context+k})).
    The final loss is the average over all horizons, so short and long-range
    predictions contribute equally.

    Returns
    -------
    loss : torch.Tensor, scalar (differentiable)
    per_horizon : dict {horizon: detached mse} for logging
    """
    horizons = sorted(set(int(h) for h in horizons))
    total = None
    per_horizon: Dict[int, torch.Tensor] = {}
    for k in horizons:
        mse = F.mse_loss(preds[k], targets[k])  # L2 between predicted and target latent
        per_horizon[k] = mse.detach()            # detach for logging (no grad needed)
        total = mse if total is None else total + mse
    return total / len(horizons), per_horizon


def variance_loss(z: torch.Tensor, target_std: float = 1.0, eps: float = 1e-4):
    """Hinge on the per-dimension std across the batch (VICReg variance term).

    If any latent dimension has std < target_std, this loss pushes it up.
    When all dims have std >= target_std, the loss is zero (hinge).
    This prevents *collapse*: if all samples mapped to the same z, std=0 -> large loss.
    eps avoids NaN gradient when std is exactly zero.
    """
    std = torch.sqrt(z.var(dim=0) + eps)   # per-dim std across batch
    return torch.mean(F.relu(target_std - std))  # hinge: only penalise below target


def covariance_loss(z: torch.Tensor):
    """Mean squared off-diagonal covariance across the batch (VICReg cov term).

    Penalises correlations between latent dimensions: if dim_i and dim_j are
    correlated, the model is redundantly encoding the same information.
    Minimising this encourages each latent dim to capture an independent concept.
    Diagonal entries (variances) are excluded via fill_diagonal_(0).
    """
    n, d = z.shape
    if n < 2:
        return z.new_zeros(())  # can't compute covariance with <2 samples
    centered = z - z.mean(dim=0)               # zero-mean per dimension
    cov = (centered.T @ centered) / (n - 1)    # (d, d) sample covariance matrix
    off_diag = (cov ** 2).clone()              # squared covariance
    off_diag.fill_diagonal_(0.0)               # zero out diagonal (variances)
    return off_diag.sum() / d                  # mean of off-diagonal squared covariances


def vicreg_loss(z: torch.Tensor, var_weight: float, cov_weight: float):
    """Weighted VICReg variance + covariance penalty."""
    return var_weight * variance_loss(z) + cov_weight * covariance_loss(z)
