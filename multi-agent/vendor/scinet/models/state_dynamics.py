"""
models.state_dynamics
=====================
SciNet++ (2026) -- P2-02 Track B: RAW-STATE residual dynamics baselines.

    x_{t+1} = x_t + g_theta(x_t, u_t)

This module is deliberately independent of the other two model tracks:

* A2 :class:`models.conditional_predictor.ConditionalPredictor` predicts in
  *latent* space ``z_{t+1} = f_theta(z_t, u_t)``.
* Legacy :class:`models.predictor.Predictor` / ``WorldModel`` are unconditional
  latent block predictors. Their state_dict keys ``encoder.* / predictor.* /
  target_encoder.*`` are NOT touched here.

Nothing in this module imports ``robotics``, hardcodes an action dimension, or
depends on Pusher. All dimensions are configurable.

Provided baselines
------------------
B0  :class:`PersistenceBaseline`        ``x_next = x``                (no training)
B1  :class:`ResidualStatePredictor`     ``x + f(x)``   (condition_dim=None)
B2  :class:`ResidualStatePredictor`     ``x + g(x,u)`` (condition_dim=U)
B3  :class:`RidgeDynamics`             ``delta = W[x,u] + b`` (closed form)
B4  B2 trained with :func:`shuffle_conditions` (negative control); evaluation
    always uses correctly aligned actions, so any loss of skill is attributable
    to the destroyed action-state correspondence.

Model-layer ``condition_dim=None`` means THIS MODEL takes no condition input
(an ablation). It is not the same statement as the A1 data-contract
``conditions=None`` (the system has no control variable). Pusher-v1 data is
always conditioned; B1/B0 simply ignore that channel on purpose.

Residual formulation in a normalised space
-----------------------------------------
Training scripts normalise with train-only statistics as

    x_norm = (x - state_mean) / state_std
    u_norm = (u - cond_mean)  / cond_std
    delta_norm = (x_next - x) / state_std

so that ``x_norm_next = x_norm + delta_norm`` holds EXACTLY. Therefore the same
residual model, the Persistence baseline, the ridge baseline and all metrics
share one consistent coordinate system, and predictions are mapped back with
``x_pred = x + delta_pred_norm * state_std`` before computing physical metrics.
"""

from __future__ import annotations

import numbers
from typing import Callable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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


def _check_hidden_dims(hidden_dims) -> Tuple[int, ...]:
    """Strictly validate hidden layer widths.

    P0-fix (Phase 0.1 review item A): the previous ``int(h)`` silently
    truncated ``3.7 -> 3``, ``3.0 -> 3`` and ``True -> 1``, so a mistyped
    config produced a different architecture with no error. Only real integer
    scalars are accepted now: Python ``int`` and numpy integer scalars (both
    are ``numbers.Integral``). Floats (including whole-valued ``3.0``),
    booleans (Python ``bool`` is an ``int`` subclass; ``np.bool_`` is not
    ``Integral`` at all), strings and ``None`` are rejected. No new dependency
    is introduced -- ``numbers`` is in the standard library.
    """
    if hidden_dims is None:
        raise ValueError("hidden_dims must be a sequence (may be empty), not None")
    if isinstance(hidden_dims, (str, bytes)):
        raise TypeError("hidden_dims must be a sequence of ints, not a string")
    try:
        entries = list(hidden_dims)
    except TypeError as exc:
        raise TypeError(
            f"hidden_dims must be a sequence of positive ints, got "
            f"{type(hidden_dims).__name__}"
        ) from exc
    hidden: list[int] = []
    for i, h in enumerate(entries):
        if isinstance(h, bool) or not isinstance(h, numbers.Integral):
            raise TypeError(
                f"hidden_dims[{i}] must be an int (or numpy integer scalar), "
                f"got {h!r} of type {type(h).__name__}; floats (even 3.0), "
                f"booleans, strings and None are rejected"
            )
        h_int = int(h)
        if h_int <= 0:
            raise ValueError(f"hidden_dims[{i}] must be positive, got {h_int}")
        hidden.append(h_int)
    return tuple(hidden)


def _check_positive_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return int(value)


# --------------------------------------------------------------------------- #
# Shared validation / rollout helpers
# --------------------------------------------------------------------------- #
def _validate_pair(
    state,
    condition,
    state_dim: int,
    condition_dim: Optional[int],
) -> None:
    _check_tensor("state", state, min_ndim=2)
    if state.shape[-1] != state_dim:
        raise ValueError(
            f"invalid state dimension: expected {state_dim}, got {state.shape[-1]}"
        )
    if condition_dim is None:
        if condition is not None:
            raise ValueError(
                "this model was built with condition_dim=None (unconditioned "
                "ablation); it must be called with condition=None. Build a "
                "separate model with condition_dim=U to use actions."
            )
    else:
        if condition is None:
            raise TypeError(
                "this model requires an explicit condition tensor; use "
                "condition_dim=None for an unconditioned ablation model"
            )
        _check_tensor("condition", condition, min_ndim=2)
        if condition.shape[-1] != condition_dim:
            raise ValueError(
                f"invalid condition dimension: expected {condition_dim}, "
                f"got {condition.shape[-1]}"
            )
        if condition.dim() != state.dim():
            raise ValueError(
                f"state and condition must have equal rank, got "
                f"{state.dim()}D and {condition.dim()}D (no implicit broadcast "
                f"of one condition across a time sequence)"
            )
        if tuple(state.shape[:-1]) != tuple(condition.shape[:-1]):
            raise ValueError(
                f"leading dimensions must match: state {tuple(state.shape[:-1])} "
                f"vs condition {tuple(condition.shape[:-1])}"
            )
        if condition.device != state.device:
            raise ValueError(
                f"device mismatch: state on {state.device}, condition on "
                f"{condition.device}"
            )
    if not torch.isfinite(state).all():
        raise ValueError("state contains non-finite values (NaN/Inf)")
    if condition is not None and not torch.isfinite(condition).all():
        raise ValueError("condition contains non-finite values (NaN/Inf)")


def _rollout_impl(
    step_fn: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    state0,
    conditions,
    horizon: Optional[int],
) -> torch.Tensor:
    """Autoregressive open-loop rollout shared by all baselines.

    ``state0``  [B, D]; ``conditions`` [B, H, U] or None (then ``horizon`` is
    required). Each predicted state is fed back as the next input; the recorded
    (true) actions are used, never ground-truth states.
    """
    _check_tensor("state0", state0, min_ndim=2)
    if state0.dim() != 2:
        raise ValueError(f"state0 must be [B, D], got shape {tuple(state0.shape)}")
    if conditions is not None:
        _check_tensor("conditions", conditions, min_ndim=3)
        if conditions.dim() != 3:
            raise ValueError(
                f"conditions must be [B, H, U], got shape {tuple(conditions.shape)}"
            )
        if conditions.shape[0] != state0.shape[0]:
            raise ValueError(
                f"batch mismatch: state0 B={state0.shape[0]} vs conditions "
                f"B={conditions.shape[0]}"
            )
        H = int(conditions.shape[1])
        if H < 1:
            raise ValueError("rollout horizon H must be >= 1")
    else:
        if horizon is None or int(horizon) < 1:
            raise ValueError("horizon must be a positive int when conditions is None")
        H = int(horizon)
    if not torch.isfinite(state0).all():
        raise ValueError("state0 contains non-finite values (NaN/Inf)")
    if conditions is not None and not torch.isfinite(conditions).all():
        raise ValueError("conditions contain non-finite values (NaN/Inf)")

    x = state0
    preds = []
    for t in range(H):
        u_t = conditions[:, t] if conditions is not None else None
        x = step_fn(x, u_t)
        preds.append(x)
    return torch.stack(preds, dim=1)  # [B, H, D]


# --------------------------------------------------------------------------- #
# B1 / B2 -- nonlinear residual MLP
# --------------------------------------------------------------------------- #
class ResidualStatePredictor(nn.Module):
    """Predict a state increment and add it to the current state.

    Parameters
    ----------
    state_dim
        Dimension ``D`` of the raw state ``x``.
    condition_dim
        ``None`` -> B1 (no condition input, ``x + f(x)``);
        positive int ``U`` -> B2 (``x + g(x, u)``).
    hidden_dims
        Hidden layer sizes (default ``(256, 256)``); may be empty for a single
        linear layer.

    Forward shapes
    --------------
    Any rank >= 2 is supported as long as ``state`` and ``condition`` have
    identical leading dimensions:

        state [B, D]      + condition [B, U]     -> [B, D]
        state [B, T, D]   + condition [B, T, U]  -> [B, T, D]
    """

    def __init__(
        self,
        state_dim: int,
        condition_dim: Optional[int] = None,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.state_dim = _check_positive_int("state_dim", state_dim)
        if condition_dim is not None:
            condition_dim = _check_positive_int("condition_dim", condition_dim)
        self.condition_dim = condition_dim
        self.hidden_dims = _check_hidden_dims(hidden_dims)
        in_dim = self.state_dim + (self.condition_dim or 0)
        self.net = build_mlp([in_dim, *self.hidden_dims, self.state_dim])

    def predict_delta(self, state, condition=None) -> torch.Tensor:
        """Return only the predicted increment (no residual addition)."""
        _validate_pair(state, condition, self.state_dim, self.condition_dim)
        inp = state if condition is None else torch.cat([state, condition], dim=-1)
        return self.net(inp)

    def forward(self, state, condition=None) -> torch.Tensor:
        """``x_next = x + g(x, u)`` (or ``x + f(x)`` for B1)."""
        return state + self.predict_delta(state, condition)

    def rollout(self, state0, conditions=None, horizon: Optional[int] = None):
        """Open-loop autoregressive rollout -> ``[B, H, D]``."""
        return _rollout_impl(
            lambda s, u: self(s, u), state0, conditions, horizon
        )

    def extra_repr(self) -> str:
        return (
            f"state_dim={self.state_dim}, condition_dim={self.condition_dim}, "
            f"hidden_dims={self.hidden_dims}"
        )


# --------------------------------------------------------------------------- #
# B0 -- persistence
# --------------------------------------------------------------------------- #
class PersistenceBaseline(nn.Module):
    """``x_next = x``. The minimum baseline that any learned model must beat."""

    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.state_dim = _check_positive_int("state_dim", state_dim)
        self.condition_dim: Optional[int] = None  # accepts but ignores actions

    def forward(self, state, condition=None) -> torch.Tensor:
        _check_tensor("state", state, min_ndim=2)
        if state.shape[-1] != self.state_dim:
            raise ValueError(
                f"invalid state dimension: expected {self.state_dim}, "
                f"got {state.shape[-1]}"
            )
        if not torch.isfinite(state).all():
            raise ValueError("state contains non-finite values (NaN/Inf)")
        return state

    def rollout(self, state0, conditions=None, horizon: Optional[int] = None):
        """Persistence stays at x0 for the whole horizon."""
        if conditions is not None and horizon is None:
            horizon = int(conditions.shape[1])
        if horizon is None or int(horizon) < 1:
            raise ValueError("horizon must be a positive int")
        _check_tensor("state0", state0, min_ndim=2)
        if state0.dim() != 2:
            raise ValueError(f"state0 must be [B, D], got {tuple(state0.shape)}")
        return state0.unsqueeze(1).expand(-1, int(horizon), -1)

    def extra_repr(self) -> str:
        return f"state_dim={self.state_dim}"


# --------------------------------------------------------------------------- #
# B3 -- closed-form ridge (linear) dynamics
# --------------------------------------------------------------------------- #
class RidgeDynamics(nn.Module):
    """Linear residual dynamics ``delta = W [x, u] + b`` fitted in closed form.

    ``W``/``b`` are registered buffers so ``state_dict()`` / ``torch.save`` work
    exactly like the MLP baselines. ``alpha`` is selected on the validation
    split by the training script (equivalent protocol to early stopping).
    """

    def __init__(
        self,
        state_dim: int,
        condition_dim: Optional[int] = None,
        alpha: float = 1e-3,
    ) -> None:
        super().__init__()
        self.state_dim = _check_positive_int("state_dim", state_dim)
        if condition_dim is not None:
            condition_dim = _check_positive_int("condition_dim", condition_dim)
        self.condition_dim = condition_dim
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or alpha < 0:
            raise ValueError(f"alpha must be a non-negative number, got {alpha!r}")
        self.alpha = float(alpha)
        in_dim = self.state_dim + (self.condition_dim or 0)
        self.register_buffer("W", torch.zeros(in_dim, self.state_dim))
        self.register_buffer("b", torch.zeros(self.state_dim))
        self.register_buffer("fitted", torch.tensor(False))

    def fit(self, state, target_delta, condition=None, mask=None, alpha=None):
        """Closed-form ridge fit on masked transitions.

        state / target_delta : [B, T, D]
        condition            : [B, T, U] or None
        mask                 : [B, T] bool (only True rows are fitted)
        """
        _validate_pair(state, condition, self.state_dim, self.condition_dim)
        _check_tensor("target_delta", target_delta, min_ndim=2)
        if target_delta.shape != state.shape:
            raise ValueError(
                f"target_delta shape {tuple(target_delta.shape)} != state "
                f"{tuple(state.shape)}"
            )
        if not torch.isfinite(target_delta).all():
            raise ValueError("target_delta contains non-finite values (NaN/Inf)")
        if mask is None:
            mask = torch.ones(state.shape[:-1], dtype=torch.bool, device=state.device)
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise TypeError("mask must be a bool torch.Tensor")
        if tuple(mask.shape) != tuple(state.shape[:-1]):
            raise ValueError(
                f"mask shape {tuple(mask.shape)} != {tuple(state.shape[:-1])}"
            )

        a = self.alpha if alpha is None else float(alpha)
        if a < 0:
            raise ValueError(f"alpha must be non-negative, got {a}")

        xv = state[mask]
        yv = target_delta[mask]
        uv = condition[mask] if condition is not None else None
        if xv.shape[0] == 0:
            raise ValueError("no valid transitions to fit")
        feats = torch.cat([xv, uv], dim=-1) if uv is not None else xv
        ones = torch.ones(feats.shape[0], 1, dtype=feats.dtype, device=feats.device)
        Fmat = torch.cat([feats, ones], dim=-1)
        A = Fmat.T @ Fmat
        reg = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
        reg[-1, -1] = 0.0  # never regularise the bias
        A = A + a * reg
        B = Fmat.T @ yv
        try:
            sol = torch.linalg.solve(A, B)
        except RuntimeError:
            sol = torch.linalg.lstsq(A, B).solution
        self.W.copy_(sol[:-1])
        self.b.copy_(sol[-1])
        self.fitted.fill_(True)
        self.alpha = a
        return self

    def predict_delta(self, state, condition=None) -> torch.Tensor:
        if not bool(self.fitted.item()):
            raise RuntimeError("RidgeDynamics.fit() must be called before prediction")
        _validate_pair(state, condition, self.state_dim, self.condition_dim)
        feats = (
            torch.cat([state, condition], dim=-1)
            if condition is not None
            else state
        )
        return F.linear(feats, self.W.T, self.b)

    def forward(self, state, condition=None) -> torch.Tensor:
        return state + self.predict_delta(state, condition)

    def rollout(self, state0, conditions=None, horizon: Optional[int] = None):
        return _rollout_impl(lambda s, u: self(s, u), state0, conditions, horizon)

    def extra_repr(self) -> str:
        return (
            f"state_dim={self.state_dim}, condition_dim={self.condition_dim}, "
            f"alpha={self.alpha}, fitted={bool(self.fitted.item())}"
        )


# --------------------------------------------------------------------------- #
# B4 -- shuffled-action negative control
# --------------------------------------------------------------------------- #
def shuffle_conditions(
    conditions: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    seed: int = 0,
) -> torch.Tensor:
    """Destroy the action<->state time correspondence for a negative control.

    Only VALID transitions participate in the permutation, and the result is
    written back at the same valid positions, so padding / masks are untouched.
    A fixed ``seed`` makes the shuffle reproducible. The shuffle is over the
    flattened set of valid (state, action) pairs across the whole split.

    B4 is trained with this tensor but EVALUATED with correctly aligned actions;
    any drop in skill therefore measures the value of the true action signal.
    """
    _check_tensor("conditions", conditions, min_ndim=2)
    if not torch.isfinite(conditions).all():
        raise ValueError("conditions contain non-finite values (NaN/Inf)")
    out = conditions.clone()
    if mask is None:
        mask = torch.ones(
            conditions.shape[:-1], dtype=torch.bool, device=conditions.device
        )
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise TypeError("mask must be a bool torch.Tensor")
    if tuple(mask.shape) != tuple(conditions.shape[:-1]):
        raise ValueError("mask must match the leading dims of conditions")
    n_valid = int(mask.sum().item())
    if n_valid < 2:
        raise ValueError("need at least 2 valid transitions to shuffle")
    vals = conditions[mask]
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    perm = torch.randperm(n_valid, generator=g)
    # Move the permutation index tensor to the data's device.
    perm = perm.to(vals.device)
    out[mask] = vals[perm]
    return out


__all__ = [
    "ResidualStatePredictor",
    "PersistenceBaseline",
    "RidgeDynamics",
    "shuffle_conditions",
]
