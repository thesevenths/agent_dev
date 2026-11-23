"""ARR-05 engine adapter: real Reacher-v5 + PyTorch MLP, full provenance,
append-only immutable records, deterministic replay support."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..arr_research_capability_03.reacher import (
    ActionConditionedMLP, generate_trajectories, sha256_file, sha256_json,
)
from . import core


def run_experiment(*, experiment_id: str, track: str, strategy: str, task_id: str,
                   seed: int, train_cfg: dict[str, Any], model_cfg: dict[str, Any],
                   test_cfg: dict[str, Any], root: str | Path,
                   protocol_version: str = "arr05-frozen-v1",
                   epochs: int = 60) -> dict[str, Any]:
    root = Path(root).resolve()
    raw_dir = root / "raw_data" / "immutable"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / f"{experiment_id}.json"
    if out.is_file():
        return json.loads(out.read_text(encoding="utf-8"))

    rec: dict[str, Any] = {
        "experiment_id": experiment_id, "track": track, "strategy": strategy,
        "task_id": task_id, "seed": int(seed), "timestamp": core.now_iso(),
        "source_commit": core.git_head(".."),
        "benchmark_version": core.BENCHMARK_VERSION,
        "protocol_version": protocol_version, "model_id": "pytorch-mlp-dynamics",
        "simulator": "gymnasium/Reacher-v5",
        "applied_config": {"train": dict(train_cfg), "model": dict(model_cfg),
                           "test": dict(test_cfg), "epochs": int(epochs)},
    }
    rec["config_sha256"] = sha256_json(rec["applied_config"])
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
        rec["metrics"] = {k: float(v) for k, v in metrics.items() if v is not None}
        trace_dir = root / "raw_data" / "traces" / experiment_id
        trace_dir.mkdir(parents=True, exist_ok=True)
        sig = {"hidden": model.hidden, "lr": model_cfg["lr"],
               "weights_sha": sha256_json([
                   [round(float(x), 6) for x in w.detach().numpy().ravel().tolist()]
                   for w in model._net.parameters()])}
        tp = trace_dir / "model_signature.json"
        tp.write_text(json.dumps(sig, sort_keys=True), encoding="utf-8")
        rec["trace_path"] = str(tp)
        rec["trace_hash"] = sha256_file(tp)
        rec["wall_seconds"] = round(time.time() - t0, 3)
        rec["ok"] = True
        rec["error"] = ""
    except Exception as exc:
        rec["ok"] = False
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["wall_seconds"] = round(time.time() - t0, 3)
    out.write_text(json.dumps(rec, indent=2, sort_keys=True, ensure_ascii=False,
                              default=str), encoding="utf-8")
    return rec


def apply_experiment(task: Any, experiment_id: str) -> tuple[dict, dict]:
    from .benchmark import EXP_BY_ID
    exp = EXP_BY_ID[experiment_id]
    tr, mo = dict(task.failure_train), dict(task.failure_model)
    for k, v in (exp.overrides.get("train") or {}).items():
        tr[k] = v
    for k, v in (exp.overrides.get("model") or {}).items():
        mo[k] = v
    return tr, mo


__all__ = ["run_experiment", "apply_experiment"]
