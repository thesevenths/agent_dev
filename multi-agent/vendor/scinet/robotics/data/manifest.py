"""
robotics.data.manifest
======================
Content-addressed dataset manifest with SHA-256 integrity.

Two distinct failure modes are separated, exactly like ``data.frozen`` in the
Pendulum pipeline:

* **config mismatch** -- the manifest declares a ``dataset_version`` that no
  longer matches the current config (seed / policies / splits changed). Caught by
  :func:`verify_config_match`.
* **tampered files** -- a split file on disk no longer hashes to the value
  recorded at freeze time. Caught by :func:`verify_manifest`.

Both must hold before any downstream stage is allowed to consume the data.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .splits import SPLITS, _canonical, _sha_text
from .trajectory import DEFAULT_DTYPE


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_version(config: dict) -> str:
    """Content-addressed id, e.g. ``pusher-3f8a1c0d92e7``."""
    spec = {
        "env_id": config["env"]["id"],
        "obs_dim": int(config["env"]["obs_dim"]),
        "action_dim": int(config["env"]["action_dim"]),
        "max_episode_steps": int(config["env"]["max_episode_steps"]),
        "dataset_seed": int(config["dataset_seed"]),
        "policies": dict(config["collection"]["policies"]),
        "splits": dict(config["collection"]["splits"]),
        "record_contact": bool(config["collection"].get("record_contact", False)),
    }
    return f"pusher-{_sha_text(_canonical(spec))[:12]}"


def build_manifest(
    config: dict,
    output_dir,
    *,
    gym_version: str,
    mujoco_version: str,
    policy_versions: Dict[str, str],
    created_at: Optional[str] = None,
) -> dict:
    output_dir = Path(output_dir)
    version = dataset_version(config)
    split_episode_ids: Dict[str, list] = {}
    split_file_hashes: Dict[str, Dict[str, str]] = {}
    for split in SPLITS:
        d = output_dir / split
        files = sorted(d.glob("*.npz")) if d.is_dir() else []
        split_episode_ids[split] = [p.stem for p in files]
        split_file_hashes[split] = {p.stem: sha256_file(p) for p in files}

    collection_config_hash = _sha_text(
        _canonical(
            {
                "dataset_seed": int(config["dataset_seed"]),
                "policies": config["collection"]["policies"],
                "splits": config["collection"]["splits"],
                "record_contact": bool(config["collection"].get("record_contact", False)),
                "max_episode_steps": int(config["env"]["max_episode_steps"]),
            }
        )
    )

    return {
        "dataset_version": version,
        "environment_id": config["env"]["id"],
        "gymnasium_version": gym_version,
        "mujoco_version": mujoco_version,
        "python_version": platform.python_version(),
        "observation_dim": int(config["env"]["obs_dim"]),
        "action_dim": int(config["env"]["action_dim"]),
        "action_bounds": [
            float(config["env"]["action_low"]),
            float(config["env"]["action_high"]),
        ],
        "obs_dtype": str(np.dtype(DEFAULT_DTYPE)),
        "collector_version": "1.0.0",
        "policy_versions": dict(policy_versions),
        "dataset_seed": int(config["dataset_seed"]),
        "collection_config_hash": collection_config_hash,
        "split_episode_ids": split_episode_ids,
        "split_file_hashes": split_file_hashes,
        "created_at": created_at or datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def write_manifest(path, manifest: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def read_manifest(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_manifest(manifest: dict, output_dir) -> None:
    """Raise RuntimeError if any recorded file is missing or altered."""
    output_dir = Path(output_dir)
    for split, hashes in manifest["split_file_hashes"].items():
        for epid, expected in hashes.items():
            f = output_dir / split / f"{epid}.npz"
            if not f.is_file():
                raise RuntimeError(f"manifest references missing file: {f}")
            actual = sha256_file(f)
            if actual != expected:
                raise RuntimeError(
                    f"dataset integrity FAILED for {f}: "
                    f"manifest={expected} actual={actual}. "
                    "Refusing to use a modified dataset."
                )
    for split, ids in manifest["split_episode_ids"].items():
        for epid in ids:
            if not (output_dir / split / f"{epid}.npz").is_file():
                raise RuntimeError(f"manifest lists missing episode: {output_dir / split / epid}")


def verify_config_match(manifest: dict, expected_version: str) -> None:
    if manifest.get("dataset_version") != expected_version:
        raise RuntimeError(
            f"dataset/config mismatch: config resolves to dataset_version="
            f"{expected_version!r}, but manifest declares "
            f"{manifest.get('dataset_version')!r}. The collection config "
            "(dataset_seed / policies / splits / max_episode_steps) changed "
            "since this dataset was frozen. Re-collect before using."
        )
