"""
data.stratified_split
=====================

SciNet++ P1-B1.1 / P7: genuine stratified episode assignment.

Background
----------
The previous V2 collector assigned splits with a single global
``rng.permutation(n)`` shuffle. That is random sampling, NOT stratification:
with 20 episodes it produced a validation set composed entirely of one action
profile (alternating), so any validation score would be confounded by action
distribution. Episode isolation and window isolation were correct; statistical
*distribution* stratification was absent.

This module implements a deterministic, episode-level stratified allocator over
the dimensions the reviewer required:

  * action profile
  * physical-parameter bins (mass / length / damping)
  * initial-condition bins (theta0 / omega0)
  * regime (bounded_oscillation / transitional / continuous_rotation)

Key guarantees (all machine-verifiable)
----------------------------------------
1. Deterministic: same inputs + seed -> identical assignment.
2. Episode-level: each episode gets exactly one split; no window crosses a split.
3. No normalization leakage: stratification uses only metadata, never the
   observed trajectory, so split assignment cannot leak signal into training.
4. Honest coverage: when a stratum cannot be represented in every split (too
   few episodes), the allocator reports it as INCONCLUSIVE / BLOCKED and refuses
   to fabricate balance. It does NOT silently claim stratification.
5. Verifiable distribution: actual per-split counts are compared against the
   requested ratios and the deviation is reported.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_REQUESTED_RATIOS = {"train": 0.7, "validation": 0.15, "test": 0.15}
SPLIT_ORDER = ("train", "validation", "test")


def bin_index(value: float, edges: Sequence[float]) -> int:
    """Return the bin index of ``value`` for sorted ``edges`` (len >= 2).

    Bins are [edges[i], edges[i+1]); values below the first edge map to 0 and
    values above the last edge map to the last bin. Empty/short edges -> 0.
    """
    edges = sorted(edges)
    if len(edges) < 2:
        return 0
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return i
    return len(edges) - 2 if value >= edges[-1] else 0


@dataclass
class StratificationConfig:
    requested_ratios: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_REQUESTED_RATIOS))
    profiles: Sequence[str] = ("random", "step", "sinusoid",
                               "alternating", "zero")
    regimes: Sequence[str] = ("bounded_oscillation", "transitional",
                              "continuous_rotation")
    # name -> sorted bin edges
    param_bins: Dict[str, Sequence[float]] = field(default_factory=dict)
    init_bins: Dict[str, Sequence[float]] = field(default_factory=dict)
    min_per_cell: int = 1
    global_ratio_tolerance: float = 0.05
    random_seed: int = 0

    @classmethod
    def default(cls) -> "StratificationConfig":
        return cls(
            requested_ratios=dict(DEFAULT_REQUESTED_RATIOS),
            profiles=("random", "step", "sinusoid", "alternating", "zero"),
            regimes=("bounded_oscillation", "transitional",
                     "continuous_rotation"),
            param_bins={
                "mass": [0.8, 1.0, 1.2],
                "length": [0.8, 1.0, 1.2],
                "damping": [0.05, 0.10, 0.15],
            },
            init_bins={
                "theta0": [-3.141592653589793, -1.0, 1.0, 3.141592653589793],
                "omega0": [-1.0, 0.0, 1.0],
            },
            min_per_cell=1,
            global_ratio_tolerance=0.05,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "StratificationConfig":
        req = d.get("requested_ratios", dict(DEFAULT_REQUESTED_RATIOS))
        # normalize ratio keys
        req = {k: float(v) for k, v in req.items()}
        return cls(
            requested_ratios=req,
            profiles=tuple(d.get("profiles", cls.default().profiles)),
            regimes=tuple(d.get("regimes", cls.default().regimes)),
            param_bins={k: list(v) for k, v in d.get("param_bins", {}).items()},
            init_bins={k: list(v) for k, v in d.get("init_bins", {}).items()},
            min_per_cell=int(d.get("min_per_cell", 1)),
            global_ratio_tolerance=float(d.get("global_ratio_tolerance", 0.05)),
            random_seed=int(d.get("random_seed", 0)),
        )


def _episode_stratum_key(ep: Dict[str, Any], cfg: StratificationConfig) -> Tuple:
    """Compute the (profile, param-bins, init-bins, regime) stratum key."""
    profile = str(ep.get("profile", ep.get("action_profile", "unknown")))
    pp = ep.get("physical_parameters", {}) or {}
    init = ep.get("initial_state", ep.get("initial_state_arr", [0.0, 0.0])) or [0.0, 0.0]
    regime = str(ep.get("regime", "unknown"))
    key = [profile]
    for name in ("mass", "length", "damping"):
        edges = cfg.param_bins.get(name)
        val = float(pp.get(name, pp.get(name[0], float("nan"))))
        key.append(bin_index(val, edges) if edges else -1)
    for name, idx in (("theta0", 0), ("omega0", 1)):
        edges = cfg.init_bins.get(name)
        val = float(init[idx]) if idx < len(init) else float("nan")
        key.append(bin_index(val, edges) if edges else -1)
    key.append(regime)
    return tuple(key)


def _largest_remainder(n: int, ratios: Sequence[float]) -> List[int]:
    """Split integer ``n`` into len(ratios) parts by largest remainder."""
    total = sum(ratios)
    if total <= 0:
        raise ValueError("ratios must sum to a positive number")
    raw = [n * r / total for r in ratios]
    floor = [int(x) for x in raw]
    rem = n - sum(floor)
    # distribute remainder to the largest fractional parts
    order = sorted(range(len(raw)), key=lambda i: raw[i] - floor[i], reverse=True)
    for i in range(rem):
        floor[order[i % len(order)]] += 1
    return floor


@dataclass
class SplitResult:
    assignments: Dict[str, str]               # episode_id -> split
    coverage: Dict[str, Any]                 # per-stratum coverage detail
    status: str                              # PASS / INCONCLUSIVE / BLOCKED
    unachievable_strata: List[list]          # strata with zero episodes
    inconclusive_strata: List[list]          # present in 1-2 splits only
    blocked_strata: List[list]               # cannot span all splits (n<#splits)
    global_counts: Dict[str, int]
    requested_counts: Dict[str, int]
    ratio_errors: Dict[str, float]
    n_episodes: int = 0


class StratifiedSplitter:
    """Deterministic episode-level stratified split allocator."""

    def __init__(self, config: Optional[StratificationConfig] = None):
        self.cfg = config or StratificationConfig.default()

    # ------------------------------------------------------------------ #
    def allocate(self, episodes: Sequence[Dict[str, Any]],
                 master_seed: Optional[int] = None) -> SplitResult:
        cfg = self.cfg
        splits = list(SPLIT_ORDER)
        n_splits = len(splits)
        ratios = [cfg.requested_ratios.get(s, 0.0) for s in splits]
        seed = int(master_seed if master_seed is not None else cfg.random_seed)

        # 1. group by stratum key
        groups: Dict[Tuple, List[Dict[str, Any]]] = {}
        for ep in episodes:
            k = _episode_stratum_key(ep, cfg)
            groups.setdefault(k, []).append(ep)

        assignments: Dict[str, str] = {}
        coverage: Dict[str, Any] = {}
        blocked, inconclusive, unachievable = [], [], []

        for key, members in groups.items():
            n = len(members)
            key_str = "::".join(str(x) for x in key)
            # deterministic ordering within stratum
            ordered = sorted(members, key=lambda e: str(e.get("episode_id", "")))
            if n == 0:
                unachievable.append(list(key))
                continue
            # target counts per split via largest remainder
            tgt = _largest_remainder(n, ratios)
            # enforce minimum per cell if possible
            if n < n_splits * cfg.min_per_cell:
                # Cannot meet per-cell minimum across all splits.
                blocked.append(list(key))
                # Place ALL in train; do not fabricate balance.
                for e in ordered:
                    assignments[str(e["episode_id"])] = "train"
                coverage[key_str] = {
                    "n": n, "counts": {"train": n, "validation": 0, "test": 0},
                    "status": "blocked",
                    "reason": f"n={n} < {n_splits}*min_per_cell={n_splits * cfg.min_per_cell}",
                }
                continue
            # If n is large enough, first reserve min_per_cell in each split,
            # then distribute the remainder proportionally.
            if n >= n_splits * cfg.min_per_cell:
                reserved = [cfg.min_per_cell] * n_splits
                remainder = n - sum(reserved)
                extra = _largest_remainder(remainder, ratios) if remainder > 0 else [0] * n_splits
                tgt = [reserved[i] + extra[i] for i in range(n_splits)]
                # fix any rounding drift
                diff = n - sum(tgt)
                order = sorted(range(n_splits), key=lambda i: tgt[i], reverse=True)
                for i in range(abs(diff)):
                    tgt[order[i % n_splits]] += 1 if diff > 0 else -1
            # assign deterministically
            cursor = 0
            counts = {s: 0 for s in splits}
            for si, s in enumerate(splits):
                for _ in range(tgt[si]):
                    e = ordered[cursor]
                    cursor += 1
                    assignments[str(e["episode_id"])] = s
                    counts[s] += 1
            # coverage classification
            present = [s for s in splits if counts[s] > 0]
            if len(present) < n_splits:
                blocked.append(list(key))
                cstat = "blocked"
            elif any(counts[s] < cfg.min_per_cell for s in splits):
                inconclusive.append(list(key))
                cstat = "partial"
            else:
                cstat = "covered"
            coverage[key_str] = {
                "n": n, "counts": counts, "status": cstat,
                "key": list(key),
            }

        global_counts = {s: sum(1 for v in assignments.values() if v == s)
                         for s in splits}
        N = len(assignments)
        requested_counts = {s: _largest_remainder(N, ratios)[i]
                            for i, s in enumerate(splits)}
        ratio_errors = {}
        for s in splits:
            actual = global_counts[s] / N if N else 0.0
            req = cfg.requested_ratios.get(s, 0.0)
            ratio_errors[s] = abs(actual - req)

        if blocked:
            status = "BLOCKED"
        elif inconclusive or any(e > cfg.global_ratio_tolerance
                                for e in ratio_errors.values()):
            status = "INCONCLUSIVE"
        else:
            status = "PASS"

        return SplitResult(
            assignments=assignments,
            coverage=coverage,
            status=status,
            unachievable_strata=[list(x) for x in unachievable],
            inconclusive_strata=[list(x) for x in inconclusive],
            blocked_strata=[list(x) for x in blocked],
            global_counts=global_counts,
            requested_counts=requested_counts,
            ratio_errors=ratio_errors,
            n_episodes=N,
        )

    # ------------------------------------------------------------------ #
    def verify_distribution(self, result: SplitResult) -> Dict[str, Any]:
        """Compare actual split distribution to requested ratios."""
        N = result.n_episodes
        out = {}
        for s in SPLIT_ORDER:
            actual = result.global_counts.get(s, 0)
            req = self.cfg.requested_ratios.get(s, 0.0)
            out[s] = {
                "actual": actual,
                "requested": req,
                "actual_ratio": actual / N if N else 0.0,
                "requested_ratio": req,
                "abs_error": result.ratio_errors.get(s, 0.0),
                "within_tolerance": result.ratio_errors.get(s, 0.0)
                <= self.cfg.global_ratio_tolerance,
            }
        return out


def build_episode_metadata(manifest: dict) -> List[Dict[str, Any]]:
    """Convert a v2 pilot manifest into allocator-ready episode metadata."""
    out = []
    for e in manifest["episodes"]:
        prof = e.get("action_profile", {})
        profile_name = prof.get("name") if isinstance(prof, dict) else str(prof)
        out.append({
            "episode_id": e["episode_id"],
            "profile": profile_name,
            "physical_parameters": e.get("physical_parameters", {}),
            "initial_state": e.get("initial_state", [0.0, 0.0]),
            "regime": e.get("regime", "unknown"),
        })
    return out
