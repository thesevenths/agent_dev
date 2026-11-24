"""ARR-07 core: identity, hashing, provenance for autonomous research loop."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

CAMPAIGN_ID = "ARR-07-2026-001"
BRANCH = "feature/arr-research-capability-07"
CAMPAIGN_ROOT = Path("experiments/campaigns") / CAMPAIGN_ID
BENCHMARK_VERSION = "arr07-benchmark-v1"


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def git_head(repo: str | Path = "..") -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""

# Leak keys: anything that reveals the hidden cause to the LLM.
LEAK_KEYS = frozenset({"ground_truth", "cause", "truth", "root_cause",
    "hidden", "model.hidden", "model.lr", "lr", "data.n_episodes", "n_episodes",
    "action_mode", "action_scale", "failure_mode", "target_diagnosis"})


class IntegrityError(RuntimeError):
    pass


def assert_no_leak(obj: Any, path: str = "$") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k) in LEAK_KEYS:
                raise IntegrityError(f"leak key {k!r} at {path}")
            assert_no_leak(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_no_leak(v, f"{path}[{i}]")


__all__ = ["CAMPAIGN_ID", "BRANCH", "CAMPAIGN_ROOT", "BENCHMARK_VERSION",
           "sha256_json", "now_iso", "git_head", "LEAK_KEYS",
           "IntegrityError", "assert_no_leak"]
