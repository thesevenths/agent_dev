"""ARR_EMB_RSI_02 integrity helpers (V1 kept immutable)."""
from __future__ import annotations

import hashlib
import json
from typing import Any

LEAK_KEYS = frozenset({
    "ground_truth", "correct_intervention", "oracle_answer",
    "hidden_test_result", "failure_id", "failure_mode", "injection",
    "ground_truth_root_cause",
})


class IntegrityError(RuntimeError):
    pass


def sha256_json(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def assert_no_ground_truth(obj: Any, path: str = "$") -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key) in LEAK_KEYS:
                raise IntegrityError(f"ground truth leak at {path}.{key}")
            assert_no_ground_truth(value, f"{path}.{key}")
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            assert_no_ground_truth(value, f"{path}[{i}]")


__all__ = ["IntegrityError", "sha256_json", "assert_no_ground_truth"]
