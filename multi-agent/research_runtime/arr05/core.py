"""ARR-05 core: campaign identity, hashing, provenance, sealed ground truth."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

CAMPAIGN_ID = "ARR-05-2026-001"
EXPERIMENT_ID = "ARR_RESEARCH_CAPABILITY_05"
BRANCH = "feature/arr-research-capability-05"
ARR04_CAMPAIGN_ID = "ARR-04-2026-001"
CAMPAIGN_ROOT = Path("experiments/campaigns") / CAMPAIGN_ID

BENCHMARK_VERSION = "arr05-benchmark-v1"


def sha256_json(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def git_head(repo: str | Path = "..") -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


LEAK_KEYS = frozenset({
    "ground_truth", "ground_truth_root_cause", "ground_truth_answer",
    "correct_intervention", "oracle_answer", "hidden_test_result",
    "hidden_test", "failure_id", "failure_mode", "root_cause",
    "sealed_truth", "injection_id", "dataset_identity", "benchmark_family",
    "cause", "truth", "task_truth", "gt",
})


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


class SealedStore:
    def __init__(self, seal_dir: str | Path):
        self.dir = Path(seal_dir).resolve()
        self.dir.mkdir(parents=True, exist_ok=True)

    def seal(self, name: str, obj: Any) -> dict[str, Any]:
        d = sha256_json(obj)
        p = self.dir / f"{name}.sealed.json"
        p.write_text(json.dumps({"sha256": d, "body": obj}, indent=2,
                                sort_keys=True, ensure_ascii=False, default=str),
                     encoding="utf-8")
        return {"name": name, "sha256": d, "path": str(p)}

    def load(self, name: str, expected: str) -> Any:
        payload = json.loads((self.dir / f"{name}.sealed.json").read_text(encoding="utf-8"))
        if payload["sha256"] != expected or sha256_json(payload["body"]) != expected:
            raise IntegrityError(f"seal {name} hash mismatch")
        return payload["body"]


__all__ = ["CAMPAIGN_ID", "EXPERIMENT_ID", "BRANCH", "ARR04_CAMPAIGN_ID",
           "CAMPAIGN_ROOT", "BENCHMARK_VERSION", "sha256_json", "sha256_file",
           "now_iso", "git_head", "LEAK_KEYS", "IntegrityError",
           "assert_no_leak", "SealedStore"]
