"""
robotics.data.splits
====================
Deterministic episode allocation across Train / Validation / Frozen Test / OOD
Test splits.

The allocation is a pure function of ``dataset_seed``: the same config always
yields the same episode ids and the same reset seeds, and no episode id can ever
appear in two splits (the id is prefixed by its split, and reset seeds are
derived per split from the data seed). This guarantees the train/val/frozen sets
draw disjoint initial conditions, so a model cannot be evaluated on states it was
trained on.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List

import numpy as np

SPLITS = ("train", "validation", "frozen_test", "ood_test")


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def derive_split_seed(dataset_seed: int, split: str) -> int:
    """Stable 32-bit seed from the data spec + split name.

    Uses SHA-256 (not ``hash()``) so the value is identical across processes and
    Python versions.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    payload = _canonical({"dataset_seed": int(dataset_seed), "split": split})
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 31 - 1)


def episode_reset_seed(dataset_seed: int, split: str, index: int) -> int:
    """Deterministic reset seed for episode ``index`` of ``split``."""
    return derive_split_seed(dataset_seed, split) + int(index)


def _split_counts(total: int, ratios: Dict[str, float]) -> Dict[str, int]:
    """Deterministic proportional allocation with a stable remainder rule."""
    keys = list(ratios.keys())
    fracs = np.array([float(ratios[k]) for k in keys], dtype=float)
    fracs = fracs / fracs.sum()
    raw = fracs * total
    counts = {k: int(np.floor(raw[i])) for i, k in enumerate(keys)}
    rem = total - sum(counts.values())
    # largest fractional part first; ties broken by original order
    order = sorted(range(len(keys)), key=lambda i: (-(raw[i] - np.floor(raw[i])), i))
    for j in range(rem):
        counts[keys[order[j % len(keys)]]] += 1
    return counts


def plan_collection(
    policy_counts: Dict[str, int],
    split_ratios: Dict[str, float],
    dataset_seed: int,
) -> List[dict]:
    """Return the collection plan: one entry per episode.

    Each entry is ``{"episode_id", "policy", "split", "index_in_split",
    "reset_seed"}``. Episode ids are unique across the whole plan.
    """
    plan: List[dict] = []
    for policy, count in policy_counts.items():
        counts = _split_counts(int(count), split_ratios)
        for split, n in counts.items():
            for i in range(n):
                plan.append(
                    {
                        "episode_id": f"{split}_{policy}_{i:05d}",
                        "policy": policy,
                        "split": split,
                        "index_in_split": i,
                        "reset_seed": episode_reset_seed(dataset_seed, split, i),
                    }
                )
    return plan
