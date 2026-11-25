"""Turn SciNet++ stage artefacts into one machine-readable ``result.json``.

We deliberately do NOT parse console text. Existing scripts already persist
numeric results (``evaluate.py`` -> ``horizon_metrics.npy``) and a checkpoint
(``train.py`` -> ``checkpoint.pt``); we read those and nothing else. This keeps
the Agent/Runtime contract stable even if log wording changes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

RESULT_SCHEMA_VERSION = 1


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_evaluation_split(out_dir: str | Path,
                          requested_split: str | None = None) -> str | None:
    """The split the stage ACTUALLY evaluated, as written by evaluate.py.

    Prefer the split-tagged meta when the job's own command names a split:
    the canonical ``evaluation_meta.json`` is shared by every evaluation on a
    run_id and gets overwritten by later jobs. The tagged copy is immutable
    evidence for that specific split.
    """
    root = Path(out_dir)
    candidates = []
    if requested_split:
        candidates.append(root / f"evaluation_meta_{requested_split}.json")
    candidates.append(root / "evaluation_meta.json")
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        split = payload.get("split")
        return str(split) if split else None
    return None


def parse_horizon_metrics(out_dir: str | Path,
                          split: str | None = None) -> dict[str, float]:
    """``horizon_metrics[_<split>].npy`` rows are ``[horizon, mse, cosine]``.

    When a split is known, read ONLY its split-tagged file. Falling back to the
    canonical ``horizon_metrics.npy`` would be dangerous: on a shared run_id a
    later ``--split test`` job overwrites that file, so a validation job could
    silently inherit test numbers. Missing split-tagged evidence yields empty
    metrics (and a diagnosable failure), never a wrong split's numbers.
    """
    import numpy as np

    root = Path(out_dir)
    if split:
        path = root / f"horizon_metrics_{split}.npy"
    else:
        path = root / "horizon_metrics.npy"
    if not path.is_file():
        return {}
    arr = np.asarray(np.load(path, allow_pickle=False))
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"unexpected {path.name} shape: {arr.shape}")
    metrics: dict[str, float] = {}
    for row in arr:
        k = int(row[0])
        metrics[f"latent_mse@{k}"] = float(row[1])
        metrics[f"cosine@{k}"] = float(row[2])
    return metrics


def describe_checkpoint(out_dir: str | Path) -> dict | None:
    path = Path(out_dir) / "checkpoint.pt"
    if not path.is_file():
        return None
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def collect_artifacts(out_dir: str | Path) -> list[str]:
    """Top-level files produced by the stage (relative names, sorted)."""
    root = Path(out_dir)
    if not root.is_dir():
        return []
    return sorted(
        p.name for p in root.iterdir()
        if p.is_file() and not (
            p.name == "result.json"
            or (p.name.startswith("result_") and p.suffix == ".json")
        )
    )


def log_tail(log_path: str | Path | None, max_chars: int = 3000) -> str:
    if not log_path or not Path(log_path).is_file():
        return ""
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def build_result(*, job: dict, status: str, exit_code: int | None,
                 out_dir: str | Path, engine: str = "scinet++",
                 started_at: str | None = None, finished_at: str | None = None,
                 error: str | None = None, error_type: str | None = None,
                 dataset_version: str | None = None,
                 split: str | None = None) -> dict[str, Any]:
    """Single shape for success AND failure -- the Runtime never sees a partial dict."""
    out = Path(out_dir)
    is_eval = job.get("stage") == "evaluate"
    # Actual split wins over the requested flag (they must agree; recording
    # both makes a mismatch visible instead of silently trusting either one).
    actual_split = read_evaluation_split(out, split) if is_eval else None
    metrics = (parse_horizon_metrics(out, split=actual_split or split)
               if status == "succeeded" and is_eval else {})
    checkpoint = describe_checkpoint(out) if status == "succeeded" else None
    spec = job.get("spec") or {}
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "experiment_id": job.get("experiment_id"),
        "job_id": job.get("job_id"),
        "stage": job.get("stage"),
        "status": status,
        "engine": engine,
        "run_id": job.get("run_id"),
        "model_version": spec.get("model_version"),
        "dataset_version": dataset_version or spec.get("dataset_version"),
        "requested_split": split if is_eval else None,
        "evaluation_split": actual_split or (split if is_eval else None),
        "metrics": metrics,
        "checkpoint": checkpoint,
        "artifacts": collect_artifacts(out),
        "log_path": job.get("log_path"),
        "result_path": job.get("result_path"),
        "exit_code": exit_code,
        "error_type": error_type,
        "error": error,
        "started_at": started_at or job.get("started_at"),
        "finished_at": finished_at,
    }


def write_result(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                              sort_keys=True), encoding="utf-8")
    tmp.replace(p)  # atomic: readers never see a half-written result.json


def read_result(path: str | Path | None) -> dict | None:
    if not path or not Path(path).is_file():
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
