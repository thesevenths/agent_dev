"""
dynamics.validation
===================

Strict validation for :class:`dynamics.contracts.DynamicsBatch`.

The validator is intentionally *strict and non-coercing* for dtype: callers
(usually the builders) are expected to pass correctly-typed tensors, and the
validator rejects wrong dtypes so that the dtype-validation tests exercise a real
rejection path rather than a silent cast.

Design rules checked
--------------------
- ``observations`` is a 3-D float tensor ``[B, T+1, D]`` with ``T+1 >= 2``.
- ``valid_mask`` is a 2-D **bool** tensor ``[B, T]`` (``T = T_obs - 1``).
- ``episode_ids`` is a sequence of length ``B``.
- If present, ``conditions`` is a 3-D float tensor ``[B, T, U]`` with ``U >= 1``.
  ``U=0`` is rejected: ``conditions=None`` is the *only* representation of an
  unconditioned system (no second "zero-width / all-zero => unconditioned"
  convention). A genuine ``[B,T,U>=1]`` tensor of zeros is valid and means a
  control channel exists but the input is zero.
- Every sample (row) has at least one valid transition.
- No non-finite (NaN / Inf) values in ``observations`` or ``conditions``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import torch

if TYPE_CHECKING:  # avoid runtime cycle with contracts.py
    from .contracts import DynamicsBatch

_FLOAT_DTYPES = (torch.float16, torch.float32, torch.float64, torch.bfloat16)


def validate_dynamics_batch(batch: "DynamicsBatch") -> bool:
    """Validate a :class:`DynamicsBatch`. Raise ``ValueError``/``TypeError``."""
    obs = batch.observations

    if not isinstance(obs, torch.Tensor):
        raise TypeError("observations must be a torch.Tensor")
    if obs.dim() != 3:
        raise ValueError(
            f"observations must be [B, T+1, D], got shape {tuple(obs.shape)}"
        )
    if obs.dtype not in _FLOAT_DTYPES:
        raise TypeError(f"observations must be a float dtype, got {obs.dtype}")
    B, T1, _D = obs.shape
    T = T1 - 1
    if T < 1:
        raise ValueError(
            f"observations must have at least 2 timesteps (T+1>=2), got {T1}"
        )

    mask = batch.valid_mask
    if not isinstance(mask, torch.Tensor):
        raise TypeError("valid_mask must be a torch.Tensor")
    if tuple(mask.shape) != (B, T):
        raise ValueError(
            f"valid_mask must be [B, T] = [B, {T}], got {tuple(mask.shape)}"
        )
    if mask.dtype != torch.bool:
        raise TypeError(f"valid_mask must be bool, got {mask.dtype}")

    if batch.conditions is not None:
        cond = batch.conditions
        if not isinstance(cond, torch.Tensor):
            raise TypeError("conditions must be a torch.Tensor or None")
        if cond.dim() != 3 or tuple(cond.shape[:2]) != (B, T):
            raise ValueError(
                f"conditions must be [B, T, U] = [B, {T}, U], got {tuple(cond.shape)}"
            )
        if cond.dtype not in _FLOAT_DTYPES:
            raise TypeError(f"conditions must be a float dtype, got {cond.dtype}")
        if cond.shape[-1] < 1:
            raise ValueError(
                "conditions must have U >= 1; a zero-width condition tensor "
                "(U=0) is not a valid representation of unconditioned "
                "dynamics. Use conditions=None instead. A real tensor of zeros "
                "with U>=1 IS valid and means 'control channel present, zero "
                "input' -- a different data-generating process from None."
            )

    if len(batch.episode_ids) != B:
        raise ValueError(
            f"episode_ids length {len(batch.episode_ids)} != B={B}"
        )

    # Every sample must contain at least one valid transition.
    per_sample = mask.sum(dim=1)
    empty = [batch.episode_ids[i] for i in range(B) if int(per_sample[i].item()) == 0]
    if empty:
        raise ValueError(
            f"every sample must have >=1 valid transition; empty episodes: {empty}"
        )

    # Reject non-finite values (NaN / Inf).
    if not torch.isfinite(obs).all():
        raise ValueError("observations contain non-finite values (NaN/Inf)")
    if batch.conditions is not None and not torch.isfinite(batch.conditions).all():
        raise ValueError("conditions contain non-finite values (NaN/Inf)")

    return True


def summarize_batch(batch: "DynamicsBatch") -> Dict:
    """Return a compact, human-readable summary of a batch (no mutation)."""
    obs = batch.observations
    B, T1, D = obs.shape
    T = T1 - 1
    n_valid = batch.num_valid_transitions()
    out: Dict = {
        "batch_size": B,
        "num_steps": T,
        "obs_dim": D,
        "conditioned": batch.is_conditioned,
        "cond_dim": batch.cond_dim,
        "num_valid_transitions": n_valid,
        "max_transitions_per_episode": int(batch.valid_mask.sum(dim=1).max().item()),
        "observations_dtype": str(obs.dtype),
        "device": str(obs.device),
        "unique_episode_ids": len(set(batch.episode_ids)),
    }
    return out
