"""
models.conditional_predictor
============================
SciNet++ (2026) -- A2: an INDEPENDENT, domain-independent conditional latent
dynamics predictor.

Mathematical interface
----------------------
    z_next = f_theta(z, condition)

Two model tracks stay deliberately separate:

* Track A (this module, A2): ``z_{t+1} = f_theta(z_t, u_t)`` -- conditional
  *latent* dynamics. Infrastructure only; no JEPA integration, no training.
* Track B (P2-02): ``x_{t+1} = x_t + g_theta(x_t, u_t)`` -- raw-state dynamics.

This module must NOT be inserted into :class:`models.world_model.WorldModel`.
``WorldModel`` keeps its ``encoder.*`` / ``predictor.*`` / ``target_encoder.*``
state_dict keys so that real legacy checkpoints still load with ``strict=True``.

Unconditioned systems
---------------------
The existing :class:`models.predictor.Predictor` (``forward(z)``) remains the
model for unconditioned dynamics. ``ConditionalPredictor.forward`` REQUIRES an
explicit condition tensor and rejects ``condition=None`` instead of silently
converting it to zeros. This mirrors the A1 data-contract rule that
``conditions=None`` (no control variable) is a different data-generating
process from a zero-valued condition, and prevents a second implicit
"unconditioned" convention from reappearing at the model layer.

No robotics imports, no Pusher-specific constants, no hardcoded action dim.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .encoder import build_mlp

_FLOAT_DTYPES = (torch.float16, torch.float32, torch.float64, torch.bfloat16)


def _check_tensor(name: str, x, min_ndim: int = 2) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x).__name__}")
    if x.dtype not in _FLOAT_DTYPES:
        raise TypeError(f"{name} must be a float dtype, got {x.dtype}")
    if x.dim() < min_ndim:
        raise ValueError(
            f"{name} must have at least {min_ndim} dims, got shape {tuple(x.shape)}"
        )


class ConditionalPredictor(nn.Module):
    """Conditional one-step latent transition model.

    Parameters
    ----------
    latent_dim
        Dimension ``L`` of the latent state ``z``.
    condition_dim
        Dimension ``U`` of the condition / control input ``u``.
    hidden_dims
        Sizes of the hidden layers (default ``(256, 256)``). May be empty,
        yielding a single linear layer ``[L+U -> L]``.

    Forward shapes
    --------------
    Single-step: ``z [B, L]``        + ``condition [B, U]``     -> ``[B, L]``
    Sequence:    ``z [B, T, L]``     + ``condition [B, T, U]``  -> ``[B, T, L]``

    Leading dimensions must match EXACTLY. Broadcasting one condition across a
    whole time sequence is never performed silently.
    """

    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        for name, value in (("latent_dim", latent_dim), ("condition_dim", condition_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if hidden_dims is None:
            raise ValueError("hidden_dims must be a sequence (may be empty), not None")
        hidden = tuple(int(h) for h in hidden_dims)
        if any(h <= 0 for h in hidden):
            raise ValueError(f"hidden_dims entries must be positive, got {hidden}")

        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.hidden_dims = hidden
        self.net = build_mlp([latent_dim + condition_dim, *hidden, latent_dim])

    # ------------------------------------------------------------- validation --
    def _validate_pair(self, z, condition) -> None:
        if condition is None:
            raise TypeError(
                "condition must be an explicit tensor. Unconditioned dynamics "
                "are modelled by models.predictor.Predictor(z); passing None "
                "here would silently equate 'no control variable' with 'zero "
                "control input'."
            )
        _check_tensor("z", z, min_ndim=2)
        _check_tensor("condition", condition, min_ndim=2)
        if z.dim() != condition.dim():
            raise ValueError(
                f"z and condition must have the same rank, got "
                f"{z.dim()}D and {condition.dim()}D (no implicit broadcasting "
                f"of a single condition across a time sequence)"
            )
        if z.shape[-1] != self.latent_dim:
            raise ValueError(
                f"invalid latent dimension: expected {self.latent_dim}, "
                f"got {z.shape[-1]}"
            )
        if condition.shape[-1] != self.condition_dim:
            raise ValueError(
                f"invalid condition dimension: expected {self.condition_dim}, "
                f"got {condition.shape[-1]}"
            )
        if tuple(z.shape[:-1]) != tuple(condition.shape[:-1]):
            raise ValueError(
                f"leading dimensions must match: z {tuple(z.shape[:-1])} vs "
                f"condition {tuple(condition.shape[:-1])}"
            )
        if not torch.isfinite(z).all():
            raise ValueError("z contains non-finite values (NaN/Inf)")
        if not torch.isfinite(condition).all():
            raise ValueError("condition contains non-finite values (NaN/Inf)")

    # ----------------------------------------------------------------- forward --
    def forward(self, z, condition):
        """Predict ``z_next`` for one step, aligned elementwise.

        ``z[:, t]`` and ``condition[:, t]`` produce ``z_next[:, t]``. No
        temporal shifting or cross-step condition reuse is performed.
        """
        self._validate_pair(z, condition)
        return self.net(torch.cat([z, condition], dim=-1))

    # ----------------------------------------------------------------- rollout --
    def rollout(self, z0, conditions):
        """Autoregressive multi-step rollout.

        Parameters
        ----------
        z0
            ``[B, L]`` initial latent state.
        conditions
            ``[B, H, U]`` condition sequence; step ``t`` uses exactly
            ``conditions[:, t]``.

        Returns
        -------
        predictions
            ``[B, H, L]`` -- each predicted latent becomes the input of the
            next step. The full computation graph is preserved (no detach),
            so gradients flow through every rollout step.
        """
        if not isinstance(z0, torch.Tensor) or not isinstance(conditions, torch.Tensor):
            raise TypeError("z0 and conditions must be torch.Tensor")
        if z0.dim() != 2:
            raise ValueError(f"z0 must be [B, L], got shape {tuple(z0.shape)}")
        if conditions.dim() != 3:
            raise ValueError(
                f"conditions must be [B, H, U], got shape {tuple(conditions.shape)}"
            )
        if z0.shape[-1] != self.latent_dim:
            raise ValueError(
                f"invalid latent dimension: expected {self.latent_dim}, "
                f"got {z0.shape[-1]}"
            )
        if conditions.shape[-1] != self.condition_dim:
            raise ValueError(
                f"invalid condition dimension: expected {self.condition_dim}, "
                f"got {conditions.shape[-1]}"
            )
        if z0.shape[0] != conditions.shape[0]:
            raise ValueError(
                f"batch mismatch: z0 B={z0.shape[0]} vs conditions "
                f"B={conditions.shape[0]}"
            )
        if conditions.shape[1] < 1:
            raise ValueError("rollout horizon H must be >= 1")

        z = z0
        predictions = []
        for t in range(conditions.shape[1]):
            z = self(z, conditions[:, t])
            predictions.append(z)
        return torch.stack(predictions, dim=1)

    def extra_repr(self) -> str:
        return (
            f"latent_dim={self.latent_dim}, condition_dim={self.condition_dim}, "
            f"hidden_dims={self.hidden_dims}"
        )
