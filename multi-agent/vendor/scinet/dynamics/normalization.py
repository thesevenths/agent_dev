"""
dynamics.normalization
======================

Train-only normalisation for raw-state dynamics experiments.

The normalisation is chosen so that the residual identity holds EXACTLY in the
normalised coordinate system:

    x_norm       = (x       - state_mean) / state_std
    u_norm       = (u       - cond_mean)  / cond_std
    delta_norm   = (x_next  - x)         / state_std      <-- NOTE: state_std
    =>  x_norm_next = x_norm + delta_norm

Using ``state_std`` (rather than a separate delta std) for the increment is what
makes ``x + delta`` and all baselines (including Persistence) consistent. All
statistics are computed from the TRAIN split only and persisted to disk; the
evaluation script reloads them and applies the identical transform to
validation / frozen_test. There is deliberately no ``fit`` method that accepts
a test split.
"""

from __future__ import annotations

import json
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from pathlib import Path
from typing import Optional

import torch

_FLOAT_DTYPES = (torch.float16, torch.float32, torch.float64, torch.bfloat16)
DEFAULT_EPS = 1e-8

# Required by the P0-fix schema: every persisted dict produced by ``to_dict``
# contains all seven keys; ``load()`` reads that exact format. A missing key is
# therefore a corrupted/hand-edited file, not a legitimate partial config.
_REQUIRED_FIELDS = (
    "state_mean",
    "state_std",
    "cond_mean",
    "cond_std",
    "eps",
    "source_split",
    "n_valid_transitions",
)
# Train-only normalisation contract: no other split may be the source of stats.
_ALLOWED_SOURCE_SPLITS = frozenset({"train"})


def _validate_vector(value, name: str, *, strictly_positive: bool = False):
    """Coerce a JSON-style 1-D numeric sequence to a finite float32 tensor.

    Rejects nested sequences, booleans, strings, empty vectors, NaN/Inf and
    (optionally) zero/negative entries. ``torch.as_tensor`` silently casts
    ``True -> 1.0``, so sequence elements are checked BEFORE coercion.
    """
    if value is None or isinstance(value, (str, bytes, bool)):
        raise TypeError(
            f"{name} must be a 1-D numeric sequence, got "
            f"{type(value).__name__}"
        )
    # Boolean / object / string arrays are rejected explicitly. ``torch.as_tensor``
    # silently casts ``True/False -> 1.0/0.0`` and would otherwise turn a corrupt
    # boolean vector into a plausible-looking statistic, so this guard runs BEFORE
    # any float32 coercion. Numeric int/float tensors and 1-D numeric ndarrays are
    # still accepted and converted normally.
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.bool:
            raise TypeError(f"{name} must not be a boolean tensor")
    elif isinstance(value, np.ndarray):
        if value.dtype.kind in ("b", "O", "U", "S"):
            raise TypeError(
                f"{name} must be a 1-D numeric array, got dtype {value.dtype}"
            )
        if value.ndim != 1:
            raise ValueError(f"{name} must be 1-D, got shape {value.shape}")
    if isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            if isinstance(v, bool) or not isinstance(v, numbers.Real):
                raise TypeError(
                    f"{name}[{i}] must be a real number, got {v!r} "
                    f"of type {type(v).__name__}"
                )
            if not math.isfinite(float(v)):
                raise ValueError(f"{name}[{i}] is non-finite (NaN/Inf): {v!r}")
    try:
        t = torch.as_tensor(value, dtype=torch.float32)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(f"{name} must be a 1-D numeric sequence: {exc}") from exc
    if t.dim() != 1:
        raise ValueError(
            f"{name} must be one-dimensional, got shape {tuple(t.shape)}"
        )
    if t.numel() == 0:
        raise ValueError(f"{name} must be a non-empty vector")
    if not bool(torch.isfinite(t).all().item()):
        raise ValueError(f"{name} contains non-finite values (NaN/Inf)")
    if strictly_positive and not bool((t > 0).all().item()):
        raise ValueError(
            f"{name} must be strictly positive (zero/negative entries would "
            f"cause division by zero during normalisation)"
        )
    return t


@dataclass
class NormalizationStats:
    """Per-dimension train statistics (all plain CPU float32 tensors)."""

    state_mean: torch.Tensor          # [D]
    state_std: torch.Tensor           # [D], clamped to >= eps
    cond_mean: Optional[torch.Tensor]  # [U] or None
    cond_std: Optional[torch.Tensor]   # [U] or None
    eps: float = DEFAULT_EPS
    source_split: str = "train"
    n_valid_transitions: int = 0

    # -- transforms --------------------------------------------------------- #
    def normalize_state(self, x: torch.Tensor) -> torch.Tensor:
        m = self.state_mean.to(x.device, x.dtype)
        s = self.state_std.to(x.device, x.dtype)
        return (x - m) / s

    def denormalize_state(self, x_norm: torch.Tensor) -> torch.Tensor:
        m = self.state_mean.to(x_norm.device, x_norm.dtype)
        s = self.state_std.to(x_norm.device, x_norm.dtype)
        return x_norm * s + m

    def normalize_delta(self, delta: torch.Tensor) -> torch.Tensor:
        """Map a raw-space increment into the normalised residual space."""
        s = self.state_std.to(delta.device, delta.dtype)
        return delta / s

    def denormalize_delta(self, delta_norm: torch.Tensor) -> torch.Tensor:
        s = self.state_std.to(delta_norm.device, delta_norm.dtype)
        return delta_norm * s

    def normalize_condition(self, u: torch.Tensor) -> torch.Tensor:
        if self.cond_mean is None or self.cond_std is None:
            raise ValueError(
                "this NormalizationStats has no condition statistics; the "
                "training split was unconditioned"
            )
        m = self.cond_mean.to(u.device, u.dtype)
        s = self.cond_std.to(u.device, u.dtype)
        return (u - m) / s

    # -- persistence -------------------------------------------------------- #
    def to_dict(self) -> dict:
        return {
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "cond_mean": None if self.cond_mean is None else self.cond_mean.tolist(),
            "cond_std": None if self.cond_std is None else self.cond_std.tolist(),
            "eps": float(self.eps),
            "source_split": self.source_split,
            "n_valid_transitions": int(self.n_valid_transitions),
        }

    def save(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: Mapping) -> "NormalizationStats":
        """Rebuild stats from a persisted dict with strict schema validation.

        P0-fix (Phase 0.1 review item B). The previous implementation had no
        validation: a hand-edited or corrupted ``normalization.json`` could
        inject mismatched lengths, zero/negative std (division by zero), NaN/Inf,
        a single ``cond_*`` field, bad ``eps``, or ``source_split="frozen_test"``.
        All of these are now rejected explicitly. ``to_dict()`` output, the
        already-persisted P2-02 ``normalization.json`` and the unconditioned
        (both ``cond_*`` = None) form all satisfy this schema.
        """
        if not isinstance(d, Mapping):
            raise TypeError(f"expected a mapping, got {type(d).__name__}")
        missing = [k for k in _REQUIRED_FIELDS if k not in d]
        if missing:
            raise ValueError(f"normalization dict missing required fields: {missing}")

        state_mean = _validate_vector(d["state_mean"], "state_mean")
        state_std = _validate_vector(
            d["state_std"], "state_std", strictly_positive=True
        )
        if state_mean.shape != state_std.shape:
            raise ValueError(
                f"state_mean/state_std length mismatch: "
                f"{tuple(state_mean.shape)} vs {tuple(state_std.shape)}"
            )

        cond_mean_raw, cond_std_raw = d["cond_mean"], d["cond_std"]
        if (cond_mean_raw is None) != (cond_std_raw is None):
            raise ValueError(
                "cond_mean and cond_std must both be None or both be present"
            )
        if cond_mean_raw is None:
            cond_mean = cond_std = None
        else:
            cond_mean = _validate_vector(cond_mean_raw, "cond_mean")
            cond_std = _validate_vector(
                cond_std_raw, "cond_std", strictly_positive=True
            )
            if cond_mean.shape != cond_std.shape:
                raise ValueError(
                    f"cond_mean/cond_std length mismatch: "
                    f"{tuple(cond_mean.shape)} vs {tuple(cond_std.shape)}"
                )

        eps = d["eps"]
        if (
            isinstance(eps, bool)
            or not isinstance(eps, numbers.Real)
            or not math.isfinite(float(eps))
            or eps <= 0
        ):
            raise ValueError(f"eps must be a positive finite number, got {eps!r}")
        eps = float(eps)

        source_split = d["source_split"]
        if not isinstance(source_split, str) or source_split not in _ALLOWED_SOURCE_SPLITS:
            raise ValueError(
                f"source_split must be one of {sorted(_ALLOWED_SOURCE_SPLITS)} "
                f"(train-only normalisation contract), got {source_split!r}"
            )

        n = d["n_valid_transitions"]
        if isinstance(n, bool) or not isinstance(n, numbers.Integral) or int(n) < 0:
            raise ValueError(
                f"n_valid_transitions must be a non-negative integer, got {n!r}"
            )

        return cls(
            state_mean=state_mean,
            state_std=state_std,
            cond_mean=cond_mean,
            cond_std=cond_std,
            eps=eps,
            source_split=source_split,
            n_valid_transitions=int(n),
        )

    @classmethod
    def load(cls, path) -> "NormalizationStats":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def fit_normalization(
    train_batch,
    eps: float = DEFAULT_EPS,
) -> NormalizationStats:
    """Fit statistics on a TRAIN :class:`~dynamics.contracts.DynamicsBatch` only.

    Only ``valid_mask == True`` transitions contribute. The function signature
    accepts a single batch to make it impossible to accidentally mix splits:
    callers must pass the train split explicitly.
    """
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or eps <= 0:
        raise ValueError(f"eps must be a positive number, got {eps!r}")
    x_t, u_t, _x_next, mask = train_batch.to_training_pairs()
    if x_t.dtype not in _FLOAT_DTYPES:
        raise TypeError(f"observations must be float, got {x_t.dtype}")
    if not torch.isfinite(x_t).all():
        raise ValueError("train observations contain non-finite values (NaN/Inf)")
    if mask.dtype != torch.bool:
        raise TypeError("valid_mask must be bool")
    if int(mask.sum().item()) < 1:
        raise ValueError("cannot fit normalisation: no valid train transitions")

    xv = x_t[mask].detach().cpu().to(torch.float64)
    state_mean = xv.mean(dim=0).to(torch.float32)
    state_std = xv.std(dim=0, unbiased=False).to(torch.float32).clamp_min(float(eps))

    if u_t is not None:
        if u_t.dtype not in _FLOAT_DTYPES:
            raise TypeError(f"conditions must be float, got {u_t.dtype}")
        if not torch.isfinite(u_t).all():
            raise ValueError("train conditions contain non-finite values (NaN/Inf)")
        uv = u_t[mask].detach().cpu().to(torch.float64)
        cond_mean = uv.mean(dim=0).to(torch.float32)
        cond_std = uv.std(dim=0, unbiased=False).to(torch.float32).clamp_min(float(eps))
    else:
        cond_mean = None
        cond_std = None

    return NormalizationStats(
        state_mean=state_mean,
        state_std=state_std,
        cond_mean=cond_mean,
        cond_std=cond_std,
        eps=float(eps),
        source_split="train",
        n_valid_transitions=int(mask.sum().item()),
    )


__all__ = ["NormalizationStats", "fit_normalization", "DEFAULT_EPS"]
