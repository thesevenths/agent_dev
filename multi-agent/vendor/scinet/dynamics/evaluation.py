"""
dynamics.evaluation
===================

Generic evaluation of residual raw-state dynamics models.

Key contract
------------
* One-step prediction is teacher-forced: every transition uses the recorded
  ``x_t`` and the recorded ``u_t``.
* Multi-step prediction is **open loop**: ``x0`` plus the recorded action
  sequence only. Every predicted state is fed back as the next input; ground
  truth states are NEVER re-injected.
* All metrics are computed in the ORIGINAL (denormalised) physical space.
* Semantic feature groups and distance specifications are supplied by the
  caller (the experiment config), never hardcoded here -- this module is
  domain-independent.

No robotics imports, no Pusher constants.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .metrics import (
    episode_errors,
    one_step_metrics,
    persistence_skill_score,
)
from .normalization import NormalizationStats

GroupSpec = Dict[str, Sequence[int]]  # name -> [start, end)


def model_takes_condition(model) -> bool:
    return getattr(model, "condition_dim", None) is not None


def training_tensors(batch, stats: NormalizationStats):
    """Return ``(x_norm, u_norm, x_next_norm_target, mask)``.

    ``x_next_norm_target = normalize_state(x_next)`` equals
    ``x_norm + normalize_delta(x_next - x_t)`` by construction, so a residual
    model trained with masked MSE on ``x_next_norm_target`` learns the increment.
    """
    x_t, u_t, x_next, mask = batch.to_training_pairs()
    x_norm = stats.normalize_state(x_t)
    u_norm = stats.normalize_condition(u_t) if u_t is not None else None
    x_next_norm = stats.normalize_state(x_next)
    return x_norm, u_norm, x_next_norm, mask


def predict_next_state(model, batch, stats: NormalizationStats) -> torch.Tensor:
    """One-step next-state prediction in the ORIGINAL space, shape [B, T, D]."""
    x_norm, u_norm, _target, _mask = training_tensors(batch, stats)
    u = u_norm if model_takes_condition(model) else None
    with torch.no_grad():
        pred_norm = model(x_norm, u)
    return stats.denormalize_state(pred_norm)


def open_loop_rollout(
    model,
    x0: torch.Tensor,
    actions: Optional[torch.Tensor],
    stats: NormalizationStats,
    horizon: int,
) -> torch.Tensor:
    """Autoregressive rollout -> ``[B, H, D]`` in the ORIGINAL space."""
    if int(horizon) < 1:
        raise ValueError("horizon must be >= 1")
    x0_norm = stats.normalize_state(x0)
    if actions is not None and model_takes_condition(model):
        u_norm = stats.normalize_condition(actions)
    else:
        u_norm = None
    with torch.no_grad():
        pred_norm = model.rollout(x0_norm, u_norm, horizon=int(horizon))
    return stats.denormalize_state(pred_norm)


def _slice_groups(
    pred: torch.Tensor,
    true: torch.Tensor,
    mask: torch.Tensor,
    groups: Optional[GroupSpec],
    persistence_pred: Optional[torch.Tensor] = None,
) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    if not groups:
        return out
    for name, bounds in groups.items():
        s, e = int(bounds[0]), int(bounds[1])
        met = one_step_metrics(pred[..., s:e], true[..., s:e], mask)
        if persistence_pred is not None:
            pmse = one_step_metrics(
                persistence_pred[..., s:e], true[..., s:e], mask
            )["mse"]
            met["persistence_mse"] = pmse
            # Constant dimensions (e.g. a fixed goal) give persistence MSE == 0:
            # a relative skill score is undefined there, not infinite.
            met["skill_vs_persistence"] = (
                persistence_skill_score(met["mse"], pmse) if pmse > 0 else None
            )
        out[name] = met
    return out


def goal_distance_metrics(
    pred: torch.Tensor,
    true: torch.Tensor,
    mask: torch.Tensor,
    object_slice: Sequence[int],
    goal_slice: Sequence[int],
) -> Dict[str, float]:
    """Error on ``||object_com - goal_com||`` (only where configured)."""
    os_, oe = int(object_slice[0]), int(object_slice[1])
    gs, ge = int(goal_slice[0]), int(goal_slice[1])
    p = pred[..., os_:oe] - pred[..., gs:ge]
    t = true[..., os_:oe] - true[..., gs:ge]
    pd = torch.linalg.norm(p, dim=-1)[mask]
    td = torch.linalg.norm(t, dim=-1)[mask]
    err = pd - td
    return {
        "mae": float(err.abs().mean().item()),
        "mse": float((err ** 2).mean().item()),
        "mean_true_distance": float(td.mean().item()),
        "mean_pred_distance": float(pd.mean().item()),
    }


def _valid_rollout_windows(mask: torch.Tensor, horizon: int):
    """List ``(episode_row, start_t)`` for fully-valid length-H windows.

    Every start position inside an episode is used, but all windows remain
    strictly inside that episode (no cross-episode stitching). Statistical
    aggregation later collapses windows back to the episode level.
    """
    B, T = mask.shape
    out = []
    for b in range(B):
        for t in range(T - horizon + 1):
            if bool(mask[b, t : t + horizon].all()):
                out.append((b, t))
    return out


def _evaluate_horizon_windows(model, pers, batch, stats, horizon: int):
    """Vectorised open-loop rollouts from all valid in-episode start positions."""
    obs = batch.observations
    cond = batch.conditions
    mask = batch.valid_mask
    wins = _valid_rollout_windows(mask, horizon)
    if not wins:
        raise ValueError(f"no fully-valid rollout windows for horizon {horizon}")

    x0 = torch.stack([obs[b, t] for b, t in wins])                    # [N, D]
    actions = (
        torch.stack([cond[b, t : t + horizon] for b, t in wins])
        if cond is not None else None
    )                                                                  # [N, H, U]
    target = torch.stack(
        [obs[b, t + 1 : t + horizon + 1] for b, t in wins]
    )                                                                  # [N, H, D]
    all_valid = torch.ones(target.shape[:2], dtype=torch.bool)

    pred = open_loop_rollout(model, x0, actions, stats, horizon)
    pers_pred = open_loop_rollout(pers, x0, actions, stats, horizon)

    # Per-window MSE (mean over H and D), then collapsed to per-episode means.
    win_mse = ((pred - target) ** 2).mean(dim=(1, 2))                  # [N]
    win_pers_mse = ((pers_pred - target) ** 2).mean(dim=(1, 2))        # [N]
    win_ep = torch.tensor([b for b, _ in wins], dtype=torch.long)
    B = mask.shape[0]
    ep_model = [float(win_mse[win_ep == b].mean()) for b in range(B)]
    ep_pers = [float(win_pers_mse[win_ep == b].mean()) for b in range(B)]

    met = one_step_metrics(pred, target, all_valid)
    pmse = one_step_metrics(pers_pred, target, all_valid)["mse"]
    met["persistence_mse"] = pmse
    met["persistence_rmse"] = float(pmse ** 0.5)
    met["skill_vs_persistence"] = persistence_skill_score(met["mse"], pmse)
    met["n_windows"] = len(wins)
    met["episode_mse_mean"] = float(sum(ep_model) / len(ep_model))
    return met, ep_model, ep_pers


def evaluate_model(
    model,
    batch,
    stats: NormalizationStats,
    *,
    horizons: Sequence[int] = (1, 2, 5, 10, 20),
    feature_groups: Optional[GroupSpec] = None,
    distance_spec: Optional[Dict[str, Sequence[int]]] = None,
) -> Dict:
    """Evaluate one model on one split.

    One-step metrics are teacher-forced over ALL valid transitions. Multi-step
    metrics use open-loop rollouts from every valid in-episode start position and
    are additionally aggregated to per-episode values for paired bootstrap
    (correlated time steps are never treated as independent samples).
    """
    from models.state_dynamics import PersistenceBaseline

    x_t, _u_t, x_next, mask = batch.to_training_pairs()
    true_next = x_next
    pred_next = predict_next_state(model, batch, stats)

    one_step = one_step_metrics(pred_next, true_next, mask)
    one_step["skill_vs_persistence"] = persistence_skill_score(
        one_step["mse"], one_step_metrics(x_t, true_next, mask)["mse"]
    )
    groups = _slice_groups(
        pred_next, true_next, mask, feature_groups, persistence_pred=x_t
    )
    dist = (
        goal_distance_metrics(
            pred_next, true_next, mask,
            distance_spec["object"], distance_spec["goal"],
        )
        if distance_spec else None
    )

    max_h = max(int(h) for h in horizons)
    if max_h > mask.shape[1]:
        raise ValueError(
            f"max horizon {max_h} exceeds available episode steps {mask.shape[1]}"
        )

    pers = PersistenceBaseline(batch.obs_dim)
    horizon_metrics: Dict[str, Dict] = {}
    horizon_episode_errors: Dict[str, List[float]] = {}
    pers_horizon_episode_errors: Dict[str, List[float]] = {}
    for h in horizons:
        h = int(h)
        met, ep_model, ep_pers = _evaluate_horizon_windows(
            model, pers, batch, stats, h
        )
        horizon_metrics[str(h)] = met
        horizon_episode_errors[str(h)] = ep_model
        pers_horizon_episode_errors[str(h)] = ep_pers

    return {
        "one_step": one_step,
        "feature_groups_one_step": groups,
        "goal_distance": dist,
        "horizons": horizon_metrics,
        "episode_errors_one_step": episode_errors(pred_next, true_next, mask),
        "persistence_episode_errors_one_step": episode_errors(x_t, true_next, mask),
        "horizon_episode_errors": horizon_episode_errors,
        "persistence_horizon_episode_errors": pers_horizon_episode_errors,
        "persistence_one_step": one_step_metrics(x_t, true_next, mask),
    }


__all__ = [
    "model_takes_condition",
    "training_tensors",
    "predict_next_state",
    "open_loop_rollout",
    "evaluate_model",
    "goal_distance_metrics",
]
