"""ARR-07 real SciNet++ backend for the autonomous loop.

Each experiment is a genuine SciNet++ train -> evaluate pair via the existing
SciNetAdapter (subprocess JSON-RPC into the conda scinet_gpu engine). No mock.
Returns a record compatible with the loop's metrics view.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import yaml

from ..adapters.scinet_adapter import SciNetAdapter
from ..schemas import ExperimentSpec
from ..settings import DEFAULT_CONFIG, DEFAULT_SCINET_DIR
from . import core

ALLOWED = {
    "train.lr": (0.0, 1.0),
    "jepa.pred_weight": (0.0, 100.0),
    "jepa.ema_momentum": (0.0, 1.0),
    "jepa.vicreg_var_weight": (0.0, 100.0),   # LeJEPA-style embedding regularisation
    "jepa.vicreg_cov_weight": (0.0, 100.0),
    "model.predictor_hidden": None,
    "model.latent_dim": None,
}
DEFAULT_METRIC = "latent_mse@1"


def _set_nested(d: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = d
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def materialise_config(base_path: str | Path, overrides: dict[str, Any],
                       out: Path, seed: int | None = None) -> str:
    bp = Path(base_path)
    if not bp.is_absolute():
        bp = DEFAULT_SCINET_DIR / bp
    base = yaml.safe_load(bp.read_text(encoding="utf-8")) or {}
    for k, v in overrides.items():
        if k not in ALLOWED:
            raise ValueError(f"unsupported SciNet knob: {k}")
        _set_nested(base, k, v)
    # CRITICAL: SciNet train.py reads cfg["seed"] from YAML, NOT the CLI/spec.
    # Without this, all "multi-seed" runs silently use the SAME seed.
    if seed is not None:
        base["seed"] = int(seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    return str(out.resolve())


def _assert_no_hidden_failure(result: Any, stage: str) -> None:
    """SciNet++ worker can exit 0 with a traceback in stdout -> AdapterError.

    Detect structured failure markers in the error field; do NOT parse raw
    stdout here (adapter already formats it into error). This is a harness-side
    guard: an engine run with a hidden crash must never count as success.
    """
    err = str(getattr(result, "error", "") or "")
    markers = ("Traceback (most recent call last)", "TypeError:", "RuntimeError:",
               "ValueError:", "AdapterError:")
    hit = [m for m in markers if m in err]
    if hit:
        raise RuntimeError(f"SciNet {stage} hidden failure markers {hit}: "
                           f"{err[:400]}")


def run_scinet_experiment(*, experiment_id: str, task_id: str, seed: int,
                          overrides: dict[str, Any], root: Path,
                          adapter: Any = None, metric: str = DEFAULT_METRIC,
                          timeout: float = 900.0,
                          protocol_hash: str = "") -> dict[str, Any]:
    root = Path(root).resolve()
    out = root / "raw_data" / "scinet" / f"{experiment_id}.json"
    if out.is_file():
        return json.loads(out.read_text(encoding="utf-8"))
    a = adapter or SciNetAdapter(strict=False)
    cfg = materialise_config(a.project_dir / DEFAULT_CONFIG, overrides,
                             root / "raw_data" / "configs" / f"{experiment_id}.yaml",
                             seed=seed)
    rec: dict[str, Any] = {
        "experiment_id": experiment_id, "task_id": task_id, "seed": seed,
        "timestamp": core.now_iso(), "source_commit": core.git_head(".."),
        "engine": "scinet++", "backend": "real_subprocess_json_rpc",
        "protocol_hash": protocol_hash, "overrides": overrides,
        "config_path": cfg, "primary_metric": metric,
    }
    t0 = time.time()
    try:
        train = ExperimentSpec(experiment_id=experiment_id, objective_id=task_id,
            hypothesis_id="auto", stage="train", config_path=cfg, seed=seed,
            split="train", budget={"epochs": 1})
        jt = a.submit(train)
        a.wait(jt, timeout=timeout)
        tr = a.collect_result(jt)
        if tr.status != "succeeded" or not tr.run_id:
            raise RuntimeError(f"SciNet train failed: {tr.status} {tr.error}")
        _assert_no_hidden_failure(tr, "train")
        ev = ExperimentSpec(experiment_id=experiment_id, objective_id=task_id,
            hypothesis_id="auto", stage="evaluate", config_path=cfg, seed=seed,
            split="test", run_id=tr.run_id)
        je = a.submit(ev)
        a.wait(je, timeout=timeout)
        er = a.collect_result(je)
        if er.status != "succeeded":
            raise RuntimeError(f"SciNet evaluate failed: {er.status} {er.error}")
        _assert_no_hidden_failure(er, "evaluate")
        rec["metrics"] = {k: float(v) for k, v in er.metrics.items()
                          if isinstance(v, (int, float))}
        rec["result_hash"] = core.sha256_json(rec["metrics"])
        rec["run_id"] = tr.run_id
        rec["train_job"] = jt
        rec["evaluate_job"] = je
        rec["artifacts"] = list(er.artifacts or [])
        rec["ok"] = True
        rec["error"] = ""
    except Exception as exc:
        rec["ok"] = False
        rec["error"] = f"{type(exc).__name__}: {exc}"
    rec["wall_seconds"] = round(time.time() - t0, 3)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2, sort_keys=True, ensure_ascii=False,
                              default=str), encoding="utf-8")
    return rec


def run_analyze(*, run_id: str, experiment_id: str, task_id: str,
                root: Path, adapter: Any = None, timeout: float = 600.0,
                protocol_hash: str = "") -> dict[str, Any]:
    """Run SciNet++ Stage 2 (physics probe / concept discovery) on a train run.

    Produces gamma/omega probe R2, participation ratio, attribution matrix —
    quantitative evidence of whether the latent space DISCOVERED physical
    concepts. Honest parse from analyze.log (no fabricated metrics).
    """
    import re
    root = Path(root).resolve()
    out = root / "raw_data" / "analyze" / f"{experiment_id}.json"
    if out.is_file():
        return json.loads(out.read_text(encoding="utf-8"))
    a = adapter or SciNetAdapter(strict=False)
    rec: dict[str, Any] = {"experiment_id": experiment_id, "task_id": task_id,
                           "engine": "scinet++", "stage": "analyze",
                           "run_id": run_id, "protocol_hash": protocol_hash}
    try:
        spec = ExperimentSpec(experiment_id=experiment_id, objective_id=task_id,
            hypothesis_id="concept_discovery", stage="analyze",
            config_path=str(a.project_dir / DEFAULT_CONFIG), seed=42,
            split="test", run_id=run_id)
        jid = a.submit(spec); a.wait(jid, timeout=timeout)
        r = a.collect_result(jid)
        if r.status != "succeeded":
            raise RuntimeError(f"analyze failed: {r.status} {str(r.error)[:200]}")
        _assert_no_hidden_failure(r, "analyze")
        log = ""
        for f in (a.project_dir / "outputs" / "pendulum_smoke" / run_id).glob("*.log"):
            if "analyze" in f.name:
                log = f.read_text(encoding="utf-8", errors="replace")
        concepts = {}
        for m in re.finditer(r"(gamma|omega)\s*:\s*R2=([-\d.]+)\s+RMSE=([\d.]+)",
                             log):
            concepts[m.group(1)] = {"probe_r2": float(m.group(2)),
                                    "rmse": float(m.group(3))}
        names = []
        ln = (a.project_dir / "outputs" / "pendulum_smoke" / run_id /
              "label_names.json")
        if ln.is_file():
            names = json.loads(ln.read_text(encoding="utf-8"))
        rec.update({"ok": True, "concepts": concepts, "concept_names": names,
                    "artifacts": list(r.artifacts or []), "log_excerpt": log[-1500:],
                    "result_hash": core.sha256_json(concepts)})
    except Exception as exc:
        rec.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2, sort_keys=True, ensure_ascii=False,
                              default=str), encoding="utf-8")
    return rec


__all__ = ["ALLOWED", "DEFAULT_METRIC", "materialise_config",
           "run_scinet_experiment", "run_analyze"]
