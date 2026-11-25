"""
dynamics.contracts
===================

The :class:`DynamicsBatch` data contract and its builders.

Contract (fixed temporal relationship)::

    (x_t, u_t) -> x_{t+1}

Tensors
-------
observations : [B, T+1, D]   float   -- x_0, x_1, ..., x_T
conditions   : [B, T,   U]   float   -- u_0, ..., u_{T-1}   (optional)
valid_mask   : [B, T]        bool    -- True iff transition (x_t, u_t) -> x_{t+1} is real
episode_ids  : [B]           str     -- source-episode identifier per row

For *unconditioned* systems, ``conditions = None``. This is **not** the same as
a zero-valued condition: it means the dynamical system has no external control
input at all (e.g. an uncontrolled pendulum). The distinction matters for the
semantics of the model that consumes the batch.

``conditions = None`` (unconditioned)  : Pendulum, Double-Pendulum, Newton.
``conditions != None`` (conditioned)   : Pusher-v5 (U = action_dim = 7).

Episode isolation
-----------------
A single :class:`DynamicsBatch` row is *always* a contiguous slice of exactly one
episode; the builder never stitches two episodes into one row, and the window
builder never lets a window cross an episode boundary. This prevents the
train/val leakage that arises from randomly splitting overlapping sliding windows
generated across episode boundaries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from .validation import validate_dynamics_batch


@dataclass
class DynamicsBatch:
    """Domain-independent transition-level dynamics batch.

    See module docstring for the tensor contract and semantics.
    """

    observations: torch.Tensor           # [B, T+1, D] float
    conditions: Optional[torch.Tensor]   # [B, T,   U] float -- None if unconditioned
    valid_mask: torch.Tensor             # [B, T]        bool
    episode_ids: List[str]               # length B

    def __post_init__(self) -> None:
        # Episode ids are always kept as plain strings.
        self.episode_ids = [str(e) for e in self.episode_ids]
        # Full validation (shapes, dtypes, non-finite, >=1 valid transition, ...).
        validate_dynamics_batch(self)

    # -- shape properties ---------------------------------------------------- #
    @property
    def batch_size(self) -> int:
        return self.observations.shape[0]

    @property
    def num_steps(self) -> int:
        """Number of transitions T (observations has T+1 timesteps)."""
        return self.observations.shape[1] - 1

    @property
    def obs_dim(self) -> int:
        return self.observations.shape[-1]

    @property
    def cond_dim(self) -> Optional[int]:
        return None if self.conditions is None else self.conditions.shape[-1]

    @property
    def is_conditioned(self) -> bool:
        return self.conditions is not None

    @property
    def device(self) -> torch.device:
        return self.observations.device

    @property
    def dtype(self) -> torch.dtype:
        return self.observations.dtype

    # -- temporal alignment helpers ----------------------------------------- #
    def states(self) -> torch.Tensor:
        """x_t for t = 0..T-1  ->  [B, T, D]."""
        return self.observations[:, :-1]

    def next_states(self) -> torch.Tensor:
        """x_{t+1} for t = 0..T-1  ->  [B, T, D]."""
        return self.observations[:, 1:]

    def num_valid_transitions(self) -> int:
        return int(self.valid_mask.sum().item())

    # -- views --------------------------------------------------------------- #
    def to_training_pairs(
        self,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Return ``(x_t, u_t, x_{t+1}, mask)`` aligned on the T axis.

        ``u_t`` is ``None`` for unconditioned batches. ``mask`` is the bool
        ``valid_mask``; training code should only backprop through masked-True
        transitions.
        """
        return self.states(), self.conditions, self.next_states(), self.valid_mask

    def slice_rows(self, rows: Sequence[int]) -> "DynamicsBatch":
        """Return a new batch containing only the selected row indices."""
        rows = list(rows)
        obs = self.observations[rows]
        cond = self.conditions[rows] if self.conditions is not None else None
        mask = self.valid_mask[rows]
        ids = [self.episode_ids[i] for i in rows]
        return DynamicsBatch(obs, cond, mask, ids)

    # -- serialization ------------------------------------------------------- #
    def to_dict(self) -> dict:
        d = {
            "observations": self.observations.detach().cpu().numpy().astype(np.float32),
            "valid_mask": self.valid_mask.detach().cpu().numpy().astype(np.uint8),
            "meta_str": np.array(
                json.dumps(
                    {
                        "episode_ids": list(self.episode_ids),
                        "has_conditions": self.conditions is not None,
                    },
                    sort_keys=True,
                )
            ),
        }
        if self.conditions is not None:
            d["conditions"] = self.conditions.detach().cpu().numpy().astype(np.float32)
        return d

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "DynamicsBatch":
        meta = json.loads(str(d["meta_str"]))
        obs = torch.as_tensor(d["observations"], dtype=torch.float32)
        mask = torch.as_tensor(
            np.asarray(d["valid_mask"]).astype(bool), dtype=torch.bool
        )
        cond = (
            torch.as_tensor(d["conditions"], dtype=torch.float32)
            if meta.get("has_conditions") and d.get("conditions") is not None
            else None
        )
        return cls(obs, cond, mask, list(meta["episode_ids"]))

    @classmethod
    def load(cls, path) -> "DynamicsBatch":
        with np.load(path, allow_pickle=False) as z:
            return cls.from_dict({k: z[k] for k in z.files})


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _as_float_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(torch.float32)
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def dynamics_batch_from_episodes(
    observations: Sequence,
    conditions: Optional[Sequence] = None,
    episode_ids: Optional[Sequence[str]] = None,
    pad_value: float = 0.0,
) -> DynamicsBatch:
    """Build a :class:`DynamicsBatch` from a list of whole episodes.

    Each episode is represented by one row. Variable-length episodes are padded
    to the longest episode with ``pad_value`` for observations and ``0`` for
    conditions; ``valid_mask`` is True only for real (non-padded) transitions.

    Parameters
    ----------
    observations
        Sequence of arrays, each shaped ``[T_i + 1, D]`` (float).
    conditions
        Optional sequence of arrays, each shaped ``[T_i, U]`` (float). If
        ``None``, the batch is unconditioned.
    episode_ids
        Optional sequence of B identifiers. Defaults to ``ep-000000``, ...
    pad_value
        Fill value for padded observation steps.
    """
    if len(observations) == 0:
        raise ValueError("observations must contain at least one episode")
    if conditions is not None and len(conditions) != len(observations):
        raise ValueError("conditions and observations must have equal length")

    B = len(observations)
    if episode_ids is None:
        episode_ids = [f"ep-{i:06d}" for i in range(B)]

    D = _obs_dim(observations[0])
    T_max = max(_seq_len(o) - 1 for o in observations)  # max number of transitions

    obs_t = torch.full((B, T_max + 1, D), float(pad_value), dtype=torch.float32)
    mask = torch.zeros((B, T_max), dtype=torch.bool)
    cond_t = None
    if conditions is not None:
        U = _cond_dim(conditions[0])
        cond_t = torch.zeros((B, T_max, U), dtype=torch.float32)

    for i, o in enumerate(observations):
        T_i = _seq_len(o) - 1
        obs_t[i, : T_i + 1] = _as_float_tensor(o)
        mask[i, :T_i] = True
        if conditions is not None:
            ci = conditions[i]
            cond_t[i, : _seq_len(ci)] = _as_float_tensor(ci)

    return DynamicsBatch(
        obs_t,
        cond_t if conditions is not None else None,
        mask,
        list(episode_ids),
    )


def dynamics_batch_from_windows(
    observations: Sequence,
    conditions: Optional[Sequence] = None,
    episode_ids: Optional[Sequence[str]] = None,
    window_len: int = 11,
    stride: int = 1,
) -> DynamicsBatch:
    """Build a :class:`DynamicsBatch` from sliding windows *within* episodes.

    For each episode, generate every valid window of ``window_len``
    observations (stride ``stride``) such that the window never crosses the
    episode boundary. Every window is tagged with its source ``episode_id`` so
    that windows belonging to the same episode remain identifiable, but no
    transition ever connects two different episodes.

    Parameters
    ----------
    observations
        Sequence of arrays, each shaped ``[L_i, D]`` (float). ``L_i`` is the
        number of observations in episode i (so it has ``L_i - 1`` transitions).
    conditions
        Optional sequence of arrays, each shaped ``[L_i - 1, U]`` (float).
    episode_ids
        Optional sequence of B identifiers.
    window_len
        Number of observations per window (>= 2, i.e. >= 1 transition).
    stride
        Step between consecutive window starts within an episode.
    """
    if window_len < 2:
        raise ValueError("window_len must be >= 2 (at least one transition)")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if len(observations) == 0:
        raise ValueError("observations must contain at least one episode")
    if conditions is not None and len(conditions) != len(observations):
        raise ValueError("conditions and observations must have equal length")

    B_in = len(observations)
    if episode_ids is None:
        episode_ids = [f"ep-{i:06d}" for i in range(B_in)]

    rows_obs: List[torch.Tensor] = []
    rows_cond: List[torch.Tensor] = []
    rows_ids: List[str] = []

    for i, o in enumerate(observations):
        L = _seq_len(o)
        n_win = (L - window_len) // stride + 1
        if n_win <= 0:
            continue  # episode shorter than window_len -> skip silently
        base = _as_float_tensor(o)
        for w in range(n_win):
            s = w * stride
            e = s + window_len
            # s:e is strictly within [0, L); never crosses the episode boundary.
            rows_obs.append(base[s:e])
            rows_ids.append(str(episode_ids[i]))
            if conditions is not None:
                ci = _as_float_tensor(conditions[i])
                rows_cond.append(ci[s : e - 1])

    if not rows_obs:
        raise ValueError(
            "no valid windows generated; all episodes shorter than window_len"
        )

    obs_t = torch.stack(rows_obs, dim=0)
    T = window_len - 1
    mask = torch.ones((obs_t.shape[0], T), dtype=torch.bool)
    cond_t = torch.stack(rows_cond, dim=0) if conditions is not None else None
    return DynamicsBatch(obs_t, cond_t, mask, rows_ids)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _seq_len(x) -> int:
    if isinstance(x, torch.Tensor):
        return x.shape[0]
    return int(np.asarray(x).shape[0])


def _obs_dim(x) -> int:
    if isinstance(x, torch.Tensor):
        return x.shape[-1]
    return int(np.asarray(x).shape[-1])


def _cond_dim(x) -> int:
    if isinstance(x, torch.Tensor):
        return x.shape[-1]
    return int(np.asarray(x).shape[-1])


def training_pairs(
    batch: DynamicsBatch,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Convenience wrapper around :meth:`DynamicsBatch.to_training_pairs`."""
    return batch.to_training_pairs()


def slice_rows(batch: DynamicsBatch, rows: Sequence[int]) -> DynamicsBatch:
    """Convenience wrapper around :meth:`DynamicsBatch.slice_rows`."""
    return batch.slice_rows(rows)
