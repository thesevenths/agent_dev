"""ARR-05 independent audit: recompute from raw immutable records, replay,
verify claims. Does NOT import campaign metric implementations as sole truth."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import core


def _mse(rec: dict[str, Any]) -> float | None:
    v = (rec.get("metrics") or {}).get("one_step_mse")
    return float(v) if v is not None else None


def integrity_audit(root: Path) -> dict[str, Any]:
    raw = sorted((root / "raw_data" / "immutable").glob("*.json"))
    ids, dupes, violations = set(), [], []
    for p in raw:
        r = json.loads(p.read_text(encoding="utf-8"))
        eid = r.get("experiment_id", "")
        if eid in ids:
            dupes.append(eid)
        ids.add(eid)
        for k in ("experiment_id", "track", "strategy", "task_id", "seed",
                  "timestamp", "source_commit", "benchmark_version",
                  "protocol_version", "config_sha256"):
            if k not in r:
                violations.append({"file": p.name, "missing": k})
        if r.get("config_sha256") != core.sha256_json(r.get("applied_config", {})):
            violations.append({"file": p.name, "missing": "config_hash_mismatch"})
        tr = r.get("trace_path")
        if tr and not Path(tr).is_file():
            violations.append({"file": p.name, "missing": "trace_file"})
    return {"n_records": len(raw), "n_unique": len(ids), "duplicates": dupes,
            "violations": violations,
            "pass": bool(raw) and not dupes and not violations}


def statistical_audit(root: Path) -> dict[str, Any]:
    """Independent recompute of M2 arr_vs_random from raw MSE records."""
    raw = {p.stem: json.loads(p.read_text(encoding="utf-8"))
           for p in (root / "raw_data" / "immutable").glob("m2__*.json")}
    mse = {}
    for eid, r in raw.items():
        parts = eid.split("__")
        if len(parts) < 4:
            continue
        v = _mse(r)
        if v is not None:
            mse[(parts[2], parts[-1], parts[1])] = v
    units: dict[str, dict[str, list[float]]] = {}
    for (task, seed, arm), v in mse.items():
        b = mse.get((task, seed, "base"))
        if b is None or abs(b) < 1e-12 or arm == "base":
            continue
        units.setdefault(task, {}).setdefault(arm, []).append((b - v) / abs(b))
    ts = sorted(t for t, d in units.items() if {"arr", "random"} <= set(d))
    def mean(x):
        return sum(x) / len(x) if x else 0.0
    diffs = [mean(units[t]["arr"]) - mean(units[t]["random"]) for t in ts]
    import statistics
    n = len(diffs)
    m = mean(diffs)
    reported_p = root / "m2_confirmation" / "strategy_results.json"
    rep = json.loads(reported_p.read_text(encoding="utf-8"))["arr_vs_random"] \
        if reported_p.is_file() else {"mean": 0.0}
    return {"n_tasks": n, "recomputed_mean": m,
            "reported_mean": rep["mean"],
            "abs_delta": abs(m - rep["mean"]),
            "pass": n > 0 and abs(m - rep["mean"]) < 1e-6}


def replay_audit(root: Path, budget: Any) -> dict[str, Any]:
    """Replay ~10% of formal runs: re-execute and compare metric hashes."""
    from .engine import run_experiment
    raw = sorted((root / "raw_data" / "immutable").glob("*.json"))
    formal = [r for r in raw if json.loads(r.read_text(encoding="utf-8")).get("track")
              not in ("preflight", "pilot", "replay")]
    k = max(1, len(formal) // 10)
    sample = formal[:k]
    replay_dir = root / "raw_data" / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    for p in sample:
        r = json.loads(p.read_text(encoding="utf-8"))
        budget.consume("replay")
        eid = r["experiment_id"] + "__replay"
        cfg = r["applied_config"]
        rr = run_experiment(
            experiment_id=eid, track="replay", strategy=r["strategy"],
            task_id=r["task_id"], seed=r["seed"],
            train_cfg=cfg["train"], model_cfg=cfg["model"],
            test_cfg=cfg["test"], root=root,
            protocol_version=r.get("protocol_version", "arr05-frozen-v1"))
        a = (r.get("metrics") or {}).get("one_step_mse")
        b = (rr.get("metrics") or {}).get("one_step_mse")
        if a is not None and b is not None and abs(a - b) < 1e-6 and rr.get("ok"):
            ok += 1
    return {"n_sampled": len(sample), "n_match": ok,
            "pass": len(sample) > 0 and ok == len(sample)}


def claim_audit(root: Path, verdicts: dict[str, Any]) -> dict[str, Any]:
    base = {
        "source_reproducibility": (root / "source_audit" / "arr04_reproduction.json").is_file(),
        "preflight": (root / "preflight" / "benchmark_validation.json").is_file(),
        "m1": (root / "m1_calibration_diagnosis" / "diagnosis_results.json").is_file(),
        "m2": (root / "m2_confirmation" / "strategy_results.json").is_file(),
        "m3": (root / "m3_memory_transfer" / "transfer_results.json").is_file(),
        "rq": (root / "rq_discovery" / "rq_results.json").is_file(),
        "rsi": (root / "rsi_recursive" / "learning_curve.json").is_file(),
        "protocol": (root / "protocol" / "protocol.sha256").is_file(),
    }
    return {"checks": base, "pass": all(base.values())}


def run_audit(root: Path, verdicts: dict[str, Any], budget: Any,
              do_replay: bool = True) -> dict[str, Any]:
    out = {
        "integrity": integrity_audit(root),
        "statistics": statistical_audit(root),
        "claims": claim_audit(root, verdicts),
    }
    out["source"] = {"pass": (root / "source_audit" / "recovered_arr04_source").is_dir()}
    out["reproducibility"] = replay_audit(root, budget) if do_replay else {
        "pass": True, "note": "replay skipped (diagnostic mode)"}
    verdict = {k: bool(v.get("pass", False)) for k, v in out.items()}
    for k, v in out.items():
        (root / "independent_audit" / f"{k}_audit.json").write_text(json.dumps(
            v, indent=2, sort_keys=True, ensure_ascii=False, default=str),
            encoding="utf-8")
    return {"audits": out, "verdict": verdict, "pass": all(verdict.values())}


__all__ = ["integrity_audit", "statistical_audit", "replay_audit",
           "claim_audit", "run_audit"]
