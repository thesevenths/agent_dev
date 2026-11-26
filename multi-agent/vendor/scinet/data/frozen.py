"""
data.frozen
===========
Build (and verify) immutable, content-addressed datasets for reproducible
experiments.

Why this exists
---------------
An autonomous research loop may submit dozens of candidate experiments. If the
test set is regenerated on every run from a process-global RNG, an apparent
improvement can be pure sampling noise. Freezing the data once and recording
SHA-256 hashes makes every experiment provably comparable, and gives
``ExperimentSpec.dataset_version`` something concrete to point at.

Layout produced::

    <out_dir>/
    ├── manifest.json
    ├── train.npz        # X, labels
    ├── validation.npz   # optional
    ├── adapt.npz        # optional
    └── test.npz

To consume it, add to the YAML config::

    data:
      frozen_dir: datasets/pendulum_v1

``build_dataset`` then loads from these files and verifies the hash on every
call (see :func:`data.dataset.load_frozen_split`).

CLI::

    python -m data.frozen --config configs/pendulum.yaml --out datasets/pendulum_v1
    python -m data.frozen --config configs/pendulum.yaml --out datasets/pendulum_v1 \
        --splits train validation adapt test
    python -m data.frozen --verify --out datasets/pendulum_v1
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np

from .dataset import (
    SPLITS,
    _canonical,
    _sha256_text,
    build_generator,
    dataset_spec,
    dataset_version,
    derive_split_seed,
    samples_for_split,
    sha256_file,
    split_rng,
)

MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1


def build_frozen_dataset(cfg: dict, out_dir: str | Path,
                         splits: tuple[str, ...] = ("train", "test"),
                         config_path: str | None = None) -> dict:
    """Materialise each split as an immutable ``.npz`` and write manifest.json.

    Returns the manifest dict. Re-running with an identical config rewrites the
    same bytes, so the hash stays stable.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split in splits:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    config_sha256 = None
    if config_path and Path(config_path).is_file():
        config_sha256 = sha256_file(config_path)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": dataset_version(cfg),
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "config_path": config_path,
        "config_sha256": config_sha256,
        "data_spec": dataset_spec(cfg),
        "generator": cfg["data"]["generator"],
        "splits": {},
    }

    for split in splits:
        n = samples_for_split(cfg, split)
        rng = split_rng(cfg, split)
        generator = build_generator(cfg, rng=rng)
        X, labels = generator.sample(n)

        filename = f"{split}.npz"
        path = out / filename
        np.savez_compressed(path, X=np.asarray(X, dtype=np.float32),
                            labels=np.asarray(labels, dtype=np.float32))

        manifest["splits"][split] = {
            "file": filename,
            "sha256": sha256_file(path),
            "samples": int(n),
            "split_seed": derive_split_seed(cfg, split),
            "split_seed_alg": "sha256(spec|split)->RandomState(MT19937)",
            "X_shape": list(X.shape),
            "labels_shape": list(labels.shape),
            "label_names": list(generator.label_names),
        }

    (out / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def verify_frozen_dataset(out_dir: str | Path) -> dict:
    """Re-hash every split file against the manifest. Raises on any mismatch."""
    out = Path(out_dir)
    manifest = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
    report = {"dataset_version": manifest["dataset_version"], "splits": {}}
    for split, entry in manifest["splits"].items():
        path = out / entry["file"]
        if not path.is_file():
            raise FileNotFoundError(f"missing split file: {path}")
        actual = sha256_file(path)
        ok = actual == entry["sha256"]
        report["splits"][split] = {"ok": ok, "expected": entry["sha256"], "actual": actual}
        if not ok:
            raise RuntimeError(
                f"integrity check FAILED for {split}: expected {entry['sha256']}, "
                f"got {actual}"
            )
    # Guard against a manifest whose declared spec no longer matches its id.
    recomputed = _sha256_text(_canonical(manifest["data_spec"]))[:12]
    declared = manifest["dataset_version"].rsplit("-", 1)[-1]
    if recomputed != declared:
        raise RuntimeError(
            f"dataset_version {declared} does not match data_spec hash {recomputed}"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/verify frozen SciNet++ datasets")
    parser.add_argument("--config", default="configs/pendulum.yaml")
    parser.add_argument("--out", default="datasets/pendulum_v1")
    parser.add_argument("--splits", nargs="+", default=["train", "test"],
                        choices=list(SPLITS))
    parser.add_argument("--verify", action="store_true",
                        help="verify an existing frozen dataset instead of building")
    args = parser.parse_args()

    if args.verify:
        report = verify_frozen_dataset(args.out)
        print(json.dumps(report, indent=2))
        print("Frozen dataset verified: OK")
        return

    from utils.config import load_config
    cfg = load_config(args.config)
    manifest = build_frozen_dataset(
        cfg, args.out, splits=tuple(args.splits), config_path=args.config
    )
    print(json.dumps({
        "dataset_version": manifest["dataset_version"],
        "out_dir": str(Path(args.out).resolve()),
        "splits": {
            k: {"file": v["file"], "sha256": v["sha256"], "samples": v["samples"]}
            for k, v in manifest["splits"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
