"""
data.dataset
============
SciNet++ (2026) -- block-sequence dataset and generator factory.

Modern JEPA models predict *latent representations of future context* rather
than reconstructing raw inputs. To make that concrete for 1D/short time series,
each trajectory of length ``L`` is split into ``num_blocks = L // block_size``
consecutive, non-overlapping blocks. A block is the atomic unit the encoder maps
to a latent vector, and the predictor rolls forward block-by-block:

    z_b --f--> z_{b+1} --f--> ... --f--> z_{b+k}     (k = rollout horizon)

This is what makes the *same* predictor usable for multi-step training (train.py)
and multi-step evaluation (evaluate.py), fixing the old train/eval mismatch.

The factory :func:`build_dataset` selects the generator from the config so the
identical pipeline runs on both the 1D pendulum and the chaotic double pendulum.

Reproducibility (research_runtime P0)
-------------------------------------
Historically every generator drew from the **global** NumPy RNG. ``train.py``
therefore produced its test split *after* consuming 50k training draws, while
``evaluate.py`` re-seeded and produced a test split from a fresh stream -- two
different test sets. Baselines and candidates were never compared on identical
samples.

Two complementary fixes, both opt-in at the config level but deterministic by
default:

1. Each split owns an independent :class:`numpy.random.RandomState`, derived
   from ``seed`` + split name + a canonical data spec. No global RNG is touched,
   so ``build_dataset(cfg, "test")`` is identical in train.py, evaluate.py,
   tta.py and any number of separate processes.
2. ``cfg['data']['frozen_dir']`` points at immutable ``.npz`` files plus a
   ``manifest.json`` holding SHA-256 hashes. Loading verifies the hash before
   use, so a Frozen Test set cannot silently drift.

Four splits are supported (``train`` / ``validation`` / ``adapt`` / ``test``).
Only ``train`` and ``test`` are used by the existing stage scripts; the extra
two exist so validation and test-time adaptation never share the test set.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .pendulum import PendulumGenerator
from .double_pendulum import DoublePendulumGenerator
from .newton import NewtonGenerator

_GENERATORS = {
    "pendulum": PendulumGenerator,
    "double_pendulum": DoublePendulumGenerator,
    "newton": NewtonGenerator,
}

#: All supported splits. Order is part of the seed derivation contract.
SPLITS = ("train", "validation", "adapt", "test")


# --------------------------------------------------------------------------- #
# Reproducible data identity
# --------------------------------------------------------------------------- #
def _canonical(obj) -> str:
    """Stable JSON encoding: sorted keys, no whitespace, deterministic str()."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | os.PathLike) -> str:
    """Stream a file through SHA-256 (files can be hundreds of MB)."""
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def samples_for_split(cfg: dict, split: str) -> int:
    """Sample count for ``split``; validation/adapt default to test_samples."""
    d = cfg["data"]
    if split == "train":
        return int(d["train_samples"])
    key = f"{split}_samples"
    return int(d.get(key, d["test_samples"]))


def dataset_spec(cfg: dict) -> dict:
    """Minimal, canonical snapshot of everything that shapes the raw data.

    Deliberately excludes training hyper-parameters (lr / epochs / model dims)
    and the MODEL seed (``cfg['seed']``): a paired multi-seed experiment must
    vary model initialisation while keeping byte-identical data. The data seed
    lives in ``data.dataset_seed`` and falls back to the top-level seed for
    backward compatibility -- that fallback keeps every historical
    ``dataset_version`` unchanged. Excludes ``frozen_dir`` (an output location).
    """
    d = cfg["data"]
    spec = {
        "generator": d["generator"],
        "sequence_length": int(d["sequence_length"]),
        "block_size": int(d["block_size"]),
        "t_max": float(d.get("t_max", 10.0)),
        "seed": int(d.get("dataset_seed", cfg.get("seed", 42))),
        "samples": {s: samples_for_split(cfg, s) for s in SPLITS},
    }
    if d["generator"] == "newton":
        spec["newton"] = dict(d.get("newton", {}))
    return spec


def dataset_version(cfg: dict) -> str:
    """Content-addressed dataset id, e.g. ``pendulum-3f8a1c0d92e7``.

    Identical config -> identical version; any data-affecting change -> new
    version. Recorded in ExperimentSpec so two experiments can be proven to
    have (or not have) used the same data.
    """
    spec = dataset_spec(cfg)
    return f"{spec['generator']}-{_sha256_text(_canonical(spec))[:12]}"


def derive_split_seed(cfg: dict, split: str) -> int:
    """Derive a stable 32-bit seed from the data spec + split name.

    Uses SHA-256 rather than Python's ``hash()`` so the value is stable across
    processes, Python versions and PYTHONHASHSEED settings.
    """
    payload = _canonical({"spec": dataset_spec(cfg), "split": split})
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 32 - 1)


def split_rng(cfg: dict, split: str):
    """An independent legacy-algorithm RNG for one split.

    ``RandomState`` (MT19937) is used instead of ``default_rng`` (PCG64) on
    purpose: it produces the same stream as the original ``np.random.uniform``
    calls, so fixing reproducibility does not silently change the numeric
    distribution of every historical experiment.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    return np.random.RandomState(derive_split_seed(cfg, split))


def frozen_manifest_path(cfg: dict) -> Path:
    return Path(cfg["data"]["frozen_dir"]) / "manifest.json"


def load_frozen_split(cfg: dict, split: str) -> "BlockSequenceDataset":
    """Load an immutable split, verifying its SHA-256 before returning it."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    base = Path(cfg["data"]["frozen_dir"])
    manifest_file = base / "manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(
            f"frozen_dir={base} has no manifest.json; run "
            f"python -m data.frozen --config <cfg> --out {base} first."
        )
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    # SHA-256 alone proves the FILES were not tampered with; it cannot prove
    # they still correspond to the CURRENT config. Without this check, editing
    # dataset_seed / sample counts / generator would silently keep loading an
    # older dataset and report a stale dataset_version as if it were current.
    expected = dataset_version(cfg)
    declared = manifest.get("dataset_version")
    if declared != expected:
        raise RuntimeError(
            f"Frozen dataset / config mismatch: config resolves to "
            f"dataset_version={expected!r}, but {manifest_file} declares "
            f"{declared!r}. The data spec (generator / sequence_length / "
            f"block_size / t_max / dataset_seed / split samples) changed since "
            f"this dataset was frozen. Re-freeze it before evaluating: "
            f"python -m data.frozen --config <cfg> --out {base} "
            f"--splits train validation adapt test"
        )
    entry = manifest["splits"].get(split)
    if not entry:
        raise KeyError(
            f"split {split!r} absent from {manifest_file}; re-freeze with "
            f"--splits including {split!r}"
        )
    path = base / entry["file"]
    if not path.is_file():
        raise FileNotFoundError(f"frozen split file missing: {path}")

    actual = sha256_file(path)
    if actual != entry["sha256"]:
        raise RuntimeError(
            f"Frozen dataset integrity check FAILED for {path}: "
            f"manifest={entry['sha256']} actual={actual}. "
            "Refusing to evaluate on a modified test set."
        )

    with np.load(path, allow_pickle=False) as npz:
        X = np.asarray(npz["X"])
        labels = np.asarray(npz["labels"])

    dataset = BlockSequenceDataset(X, labels, int(cfg["data"]["block_size"]))
    dataset.label_names = tuple(entry["label_names"])
    dataset.dataset_version = manifest["dataset_version"]
    dataset.split = split
    dataset.frozen_path = str(path.resolve())
    return dataset


# --------------------------------------------------------------------------- #
# Generator / dataset factory
# --------------------------------------------------------------------------- #
def build_generator(cfg: dict, rng=None):
    """Instantiate the generator named by ``cfg['data']['generator']``.

    ``rng`` is optional and injected into the generator. Leaving it ``None``
    preserves the historical global-RNG behaviour for direct callers such as
    ``PendulumGenerator().sample(5)``; ``build_dataset`` always injects a
    per-split RNG.
    """
    name = cfg["data"]["generator"]
    if name not in _GENERATORS:
        raise ValueError(
            f"Unknown generator '{name}'. Expected one of {list(_GENERATORS)}."
        )
    length = cfg["data"]["sequence_length"]
    t_max = cfg["data"].get("t_max", None)
    if name == "pendulum":
        return PendulumGenerator(length=length, t_max=t_max if t_max else 10.0, rng=rng)
    if name == "newton":
        # allow per-experiment tuning of force/mass ranges from the config
        kwargs = dict(cfg["data"].get("newton", {}))
        return NewtonGenerator(length=length, t_max=t_max if t_max else 10.0, rng=rng, **kwargs)
    # double pendulum: the analogous knob is the integration duration
    return DoublePendulumGenerator(length=length, duration=t_max if t_max else 10.0, rng=rng)


def encoder_input_dim(cfg: dict) -> int:
    """Dimensionality of a single encoded block = block_size * state_dim."""
    state_dim = _GENERATORS[cfg["data"]["generator"]].state_dim
    return int(cfg["data"]["block_size"]) * int(state_dim)


def state_dim(cfg: dict) -> int:
    """Per-timestep feature count of the selected generator (1 or 4)."""
    return int(_GENERATORS[cfg["data"]["generator"]].state_dim)


def num_blocks(cfg: dict) -> int:
    """Number of blocks per trajectory."""
    return int(cfg["data"]["sequence_length"]) // int(cfg["data"]["block_size"])


def validate_config(cfg: dict) -> None:
    """Fail fast on inconsistent configs (blocks vs. rollout horizons)."""
    nb = num_blocks(cfg)
    max_horizon = max(cfg["jepa"]["rollout_steps"])
    if nb - 1 < max_horizon:
        raise ValueError(
            f"sequence_length/block_size yields {nb} blocks, but rollout_steps "
            f"needs up to {max_horizon} (require num_blocks-1 >= max horizon). "
            f"Increase data.sequence_length or shrink jepa.rollout_steps."
        )
    if cfg["data"]["sequence_length"] % cfg["data"]["block_size"] != 0:
        raise ValueError("data.sequence_length must be divisible by data.block_size.")


class BlockSequenceDataset(Dataset):
    """Wrap raw trajectories into per-block tensors plus concept labels."""

    def __init__(self, X: np.ndarray, labels: np.ndarray, block_size: int) -> None:
        n, length, state_dim = X.shape
        nb = length // block_size
        trimmed = X[:, : nb * block_size, :]
        # (n, num_blocks, block_size * state_dim)
        blocks = trimmed.reshape(n, nb, block_size * state_dim)

        self.blocks = np.ascontiguousarray(blocks, dtype=np.float32)
        self.labels = np.ascontiguousarray(labels, dtype=np.float32)
        self.num_blocks = nb
        self.block_dim = block_size * state_dim
        self.label_dim = self.labels.shape[1]

    def __len__(self) -> int:
        return self.blocks.shape[0]

    def __getitem__(self, idx: int):
        return {
            "blocks": torch.from_numpy(self.blocks[idx]),   # (num_blocks, block_dim)
            "labels": torch.from_numpy(self.labels[idx]),   # (label_dim,)
        }


def build_dataset(cfg: dict, split: str = "train", rng=None) -> BlockSequenceDataset:
    """Generate (or load) a dataset for ``split``.

    Resolution order:
      1. ``cfg['data']['frozen_dir']`` set  -> load immutable .npz + verify hash;
      2. ``rng`` explicitly supplied       -> use it (advanced/testing);
      3. otherwise                         -> per-split deterministic RandomState.

    Every returned dataset carries ``split``, ``dataset_version`` and
    ``label_names`` so downstream stages can record what they actually used.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    if cfg["data"].get("frozen_dir"):
        return load_frozen_split(cfg, split)

    effective_rng = rng if rng is not None else split_rng(cfg, split)
    generator = build_generator(cfg, rng=effective_rng)
    n = samples_for_split(cfg, split)

    X, labels = generator.sample(n)
    dataset = BlockSequenceDataset(X, labels, cfg["data"]["block_size"])
    # expose concept names for downstream stages (probe / symbolic regression)
    dataset.label_names = generator.label_names
    dataset.dataset_version = dataset_version(cfg)
    dataset.split = split
    dataset.frozen_path = None
    return dataset
