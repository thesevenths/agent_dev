"""
dynamics.metrics
================

Generic, domain-independent metrics for raw-state dynamics evaluation.

Design rules
------------
* All functions accept ``pred`` / ``target`` shaped ``[..., D]`` and a bool
  ``mask`` shaped like the leading dimensions (``[B, T]`` for sequence data).
  Padded / invalid transitions NEVER contribute.
* Statistical significance is computed on EPISODE-LEVEL aggregates. Correlated
  time steps inside one episode are never treated as independent samples.
* No robotics imports, no Pusher constants.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch


def _flat_valid(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    if not isinstance(pred, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("pred and target must be torch.Tensor")
    if pred.shape != target.shape:
        raise ValueError(
            f"pred shape {tuple(pred.shape)} != target {tuple(target.shape)}"
        )
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise TypeError("mask must be a bool torch.Tensor")
    if tuple(mask.shape) != tuple(pred.shape[:-1]):
        raise ValueError(
            f"mask shape {tuple(mask.shape)} must match leading dims "
            f"{tuple(pred.shape[:-1])}"
        )
    if int(mask.sum().item()) < 1:
        raise ValueError("no valid positions in mask")
    return pred[mask], target[mask]


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """Differentiable masked MSE (mean over feature dim, then over valid rows)."""
    p, t = _flat_valid(pred, target, mask)
    per_row = ((p - t) ** 2).mean(dim=-1)
    return per_row.mean()


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    return float(masked_mse_loss(pred, target, mask).item())


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    p, t = _flat_valid(pred, target, mask)
    return float((p - t).abs().mean().item())


def per_dimension_rmse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> List[float]:
    p, t = _flat_valid(pred, target, mask)
    return [float(v) for v in torch.sqrt(((p - t) ** 2).mean(dim=0))]


def one_step_metrics(
    pred_next: torch.Tensor,
    true_next: torch.Tensor,
    mask: torch.Tensor,
) -> Dict:
    """MSE / RMSE / MAE / per-dimension RMSE over valid transitions."""
    mse = masked_mse(pred_next, true_next, mask)
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": masked_mae(pred_next, true_next, mask),
        "per_dim_rmse": per_dimension_rmse(pred_next, true_next, mask),
        "n_valid": int(mask.sum().item()),
    }


def persistence_skill_score(model_mse: float, persistence_mse: float) -> float:
    """``1 - MSE_model / MSE_persistence``.

    > 0 better than persistence, == 0 identical, < 0 worse.
    """
    if persistence_mse <= 0:
        raise ValueError(
            "persistence MSE must be > 0 for a relative skill score; "
            f"got {persistence_mse}"
        )
    return 1.0 - float(model_mse) / float(persistence_mse)


def episode_errors(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> List[float]:
    """Per-episode (row ``b``) mean squared error over valid timesteps."""
    if pred.shape != target.shape:
        raise ValueError("pred/target shape mismatch")
    if pred.dim() != 3:
        raise ValueError(
            f"episode_errors expects [B, T, D], got {tuple(pred.shape)}"
        )
    if mask.dtype != torch.bool or tuple(mask.shape) != tuple(pred.shape[:2]):
        raise ValueError("mask must be bool [B, T]")
    out: List[float] = []
    sq = ((pred - target) ** 2).mean(dim=-1)  # [B, T]
    for b in range(pred.shape[0]):
        m = mask[b]
        if int(m.sum().item()) < 1:
            raise ValueError(f"episode {b} has no valid transitions")
        out.append(float(sq[b][m].mean().item()))
    return out


def describe(values: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(list(values), dtype=np.float64)
    if a.size == 0:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std(ddof=0)),
        "min": float(a.min()),
        "max": float(a.max()),
        "n": int(a.size),
    }


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict:
    """Paired bootstrap CI of ``mean(a - b)`` on EPISODE-level values.

    ``a`` is typically the model's per-episode error and ``b`` the baseline's.
    A negative CI (both bounds < 0) means ``a`` is significantly better.
    """
    av = np.asarray(list(a), dtype=np.float64)
    bv = np.asarray(list(b), dtype=np.float64)
    if av.shape != bv.shape or av.ndim != 1 or av.size < 2:
        raise ValueError(
            "paired_bootstrap_ci requires equal-length 1-D episode arrays "
            f"(got {av.shape} / {bv.shape})"
        )
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if int(n_boot) < 100:
        raise ValueError(f"n_boot must be >= 100, got {n_boot}")
    d = av - bv
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, d.size, size=(int(n_boot), d.size))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "n_pairs": int(d.size),
        "n_boot": int(n_boot),
        "alpha": float(alpha),
        "seed": int(seed),
        "mean_a": float(av.mean()),
        "mean_b": float(bv.mean()),
        "median_a": float(np.median(av)),
        "median_b": float(np.median(bv)),
        "mean_diff": float(d.mean()),
        "median_diff": float(np.median(d)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "significant_a_better": bool(hi < 0),
        "significant_b_better": bool(lo > 0),
    }


__all__ = [
    "masked_mse",
    "masked_mse_loss",
    "masked_mae",
    "per_dimension_rmse",
    "one_step_metrics",
    "persistence_skill_score",
    "episode_errors",
    "describe",
    "paired_bootstrap_ci",
]
