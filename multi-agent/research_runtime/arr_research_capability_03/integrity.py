"""ARR-03 integrity primitives: stable hashes and ground-truth isolation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# Any key carrying a hidden answer must never appear in an agent-facing object.
LEAK_KEYS = frozenset({
    "ground_truth", "ground_truth_root_cause", "ground_truth_answer",
    "correct_intervention", "oracle_answer", "hidden_test_result",
    "hidden_test", "failure_id", "failure_mode", "root_cause",
    "sealed_truth", "injection_id", "dataset_identity", "benchmark_family",
})


class IntegrityError(RuntimeError):
    pass


def sha256_json(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def assert_no_ground_truth(obj: Any, path: str = "$") -> None:
    """Recursively reject any hidden-answer key in an agent-facing payload."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key) in LEAK_KEYS:
                raise IntegrityError(f"ground-truth key {key!r} at {path}")
            assert_no_ground_truth(value, f"{path}.{key}")
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            assert_no_ground_truth(value, f"{path}[{i}]")


class SealedGroundTruthStore:
    """Ground truth sealed OUTSIDE any agent-visible run directory."""

    def __init__(self, seal_dir: str | Path):
        self.dir = Path(seal_dir).resolve()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "ground_truth.sealed.json"

    def seal(self, truth: dict[str, Any]) -> dict[str, Any]:
        digest = sha256_json(truth)
        payload = {"ground_truth_sha256": digest, "truth": truth}
        self.path.write_text(json.dumps(
            payload, indent=2, sort_keys=True, ensure_ascii=False, default=str),
            encoding="utf-8")
        return {"ground_truth_sha256": digest, "seal_path": str(self.path)}

    def load(self, expected_sha256: str) -> dict[str, Any]:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload["ground_truth_sha256"] != expected_sha256:
            raise IntegrityError("ground-truth seal hash mismatch")
        if sha256_json(payload["truth"]) != expected_sha256:
            raise IntegrityError("ground-truth body tampered")
        return payload["truth"]


__all__ = ["IntegrityError", "LEAK_KEYS", "assert_no_ground_truth",
           "sha256_json", "SealedGroundTruthStore"]
