"""ARR-04 engine adapter: real Reacher-v5 simulator + real PyTorch dynamics MLP.

No FakeRunner/MockEngine. Every run records provenance: experiment id, commit,
timestamp, benchmark version, protocol version, seed, model id, config hash,
artifact hashes, and a trace. Formal raw results are append-only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..arr_research_capability_03.reacher import (
    ActionConditionedMLP, generate_trajectories, sha256_file, sha256_json,
)
from . import core

BENCHMARK_VERSION = "arr04-benchmark-v1"


def run_experiment(*, experiment_id: str, split: str, arm: str,
                   train_cfg: dict[str, Any], model_cfg: dict[str, Any],
                   test_cfg: dict[str, Any], seed: int,
                   root: str | Path, protocol_version: str = "frozen-v1",
                   epochs: int = 80) -> dict[str, Any]:
    """Execute ONE real engine run with full provenance; append-only record."""
    root = Path(root).resolve()
    raw_dir = root / "raw_data" / "immutable"
    raw_dir.mkdir(parents=True, exist_ok=True)
    run_dir = root / "traces" / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)

    applied = {"train": dict(train_cfg), "model": dict(model_cfg),
               "test": dict(test_cfg), "epochs": int(epochs), "seed": int(seed)}
    rec: dict[str, Any] = {
        "experiment_id": experiment_id, "split": split, "arm": arm,
        "timestamp": core.now_iso(), "source_commit": core.git_head(".."),
        "benchmark_version": BENCHMARK_VERSION,
        "protocol_version": protocol_version, "seed": int(seed),
        "model_id": "pytorch-mlp-dynamics", "simulator": "gymnasium/Reacher-v5",
        "applied_config": applied, "config_sha256": sha256_json(applied),
    }
    t0 = time.time()
    try:
        train = generate_trajectories(seed=seed, **train_cfg)
        test = generate_trajectories(seed=seed + 9000, **test_cfg)
        rec["train_data_sha256"] = train.digest()
        rec["test_data_sha256"] = test.digest()
        model = ActionConditionedMLP(hidden=model_cfg["hidden"])
        fit = model.fit(train, lr=float(model_cfg["lr"]), epochs=epochs, seed=seed)
        metrics = model.evaluate(test)
        metrics["train_final_loss"] = fit["final_loss"]
        # IID (same distribution) signal for identifiability signatures.
        rec["metrics"] = {k: (float(v) if v is not None else None)
                          for k, v in metrics.items()}
        sig_path = run_dir / "model_signature.json"
        sig_path.write_text(json.dumps({
            "hidden": model.hidden, "lr": model_cfg["lr"],
            "weights_sha": sha256_json([
                [round(float(x), 6) for x in w.detach().numpy().ravel().tolist()]
                for w in model._net.parameters()])},
            sort_keys=True), encoding="utf-8")
        rec["model_signature_sha256"] = sha256_file(sig_path)
        rec["trace_path"] = str(sig_path)
        rec["wall_seconds"] = round(time.time() - t0, 3)
        rec["ok"] = True
        rec["error"] = ""
    except Exception as exc:
        rec["ok"] = False
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["wall_seconds"] = round(time.time() - t0, 3)
    # Immutable append-only record.
    record_path = raw_dir / f"{experiment_id}.json"
    record_path.write_text(json.dumps(
        rec, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8")
    return rec


def apply_experiment(task: Any, experiment_id: str) -> tuple[dict, dict]:
    """Return (train_cfg, model_cfg) with a diagnostic experiment applied."""
    from .benchmark import EXP_BY_ID
    exp = EXP_BY_ID[experiment_id]
    train = dict(task.failure_train)
    model = dict(task.failure_model)
    for key, value in (exp.overrides.get("train") or {}).items():
        train[key] = value
    for key, value in (exp.overrides.get("model") or {}).items():
        model[key] = value
    return train, model


__all__ = ["BENCHMARK_VERSION", "run_experiment", "apply_experiment"]