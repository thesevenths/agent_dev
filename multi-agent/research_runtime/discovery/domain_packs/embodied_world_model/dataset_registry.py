"""M7 Domain Pack (embodied): frozen dataset and INDEPENDENT family registry.

A family is one physical episode, not a seed. Train / validation / test / OOD
episode sets are disjoint and their SHAs are re-read from disk on every call.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from ....harness_rsi.identity import sha256_file, sha256_json
from ....domains.embodied import BASELINE_CONFIG, PROJECT_DIR
from ....domains.local_engine import read_frozen_manifest

GENERATOR_VERSION = "m7_embodied_family_registry_v1"


def baseline_config() -> dict[str, Any]:
    return yaml.safe_load((PROJECT_DIR / BASELINE_CONFIG).read_text(
        encoding="utf-8"))


def dataset_root() -> Path:
    return PROJECT_DIR / baseline_config()["data"]["frozen_dir"]


def dataset_identity() -> dict[str, Any]:
    config = baseline_config()
    manifest = read_frozen_manifest(PROJECT_DIR, config)
    manifest_path = dataset_root() / "manifest.json"
    result = {
        "version": GENERATOR_VERSION,
        "project_dir": str(PROJECT_DIR),
        "dataset_version": manifest["dataset_version"],
        "dataset_seed": manifest["dataset_seed"],
        "dataset_manifest_sha": sha256_file(manifest_path),
        "splits": {name: {"file": info["file"], "rows": info["rows"],
                          "sha256": info["sha256"]}
                   for name, info in manifest["splits"].items()},
    }
    result["dataset_identity_sha"] = sha256_json(result)
    return result


def family_manifest() -> dict[str, Any]:
    """Build a family manifest from the ACTUAL frozen CSV episode ids."""
    ident = dataset_identity()
    splits: dict[str, Any] = {}
    all_families: list[str] = []
    for split, info in ident["splits"].items():
        path = dataset_root() / info["file"]
        with path.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        episodes = sorted({str(r["episode"]) for r in rows}, key=int)
        families = [f"{split}:ep{e}" for e in episodes]
        digest = hashlib.sha256(json.dumps(families, sort_keys=True).encode()
                                ).hexdigest()
        splits[split] = {"families": families, "n_families": len(families),
                         "family_set_sha": digest,
                         "dataset_split_sha": info["sha256"]}
        all_families.extend(families)
    overlaps = sorted({f for f in all_families
                       if sum(f in s["families"] for s in splits.values()) > 1})
    obj = {"version": GENERATOR_VERSION, "unit": "episode",
           "split_manifest_sha": ident["dataset_manifest_sha"],
           "splits": splits, "overlapping_families": overlaps,
           "seed_is_not_family": True}
    obj["family_manifest_sha"] = sha256_json(obj)
    return obj


def validate_family_isolation() -> dict[str, Any]:
    manifest = family_manifest()
    problems: list[str] = []
    if manifest["overlapping_families"]:
        problems.append("family_overlap_between_splits")
    splits = manifest["splits"]
    for split in ("train", "validation", "test", "ood"):
        if not splits.get(split, {}).get("n_families"):
            problems.append(f"missing_{split}_families")
    return {"ok": not problems, "problems": problems,
            "family_manifest_sha": manifest["family_manifest_sha"],
            "n_families": {k: v["n_families"] for k, v in splits.items()},
            "seed_is_not_family": True}


__all__ = ["GENERATOR_VERSION", "baseline_config", "dataset_identity",
           "family_manifest", "validate_family_isolation"]
