"""ARR-06 independent audit: recompute from raw, replay, verify claims."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import core


def integrity_audit(root: Path) -> dict[str, Any]:
    raw = sorted((root / "raw_data" / "immutable").glob("*.json"))
    ids, dupes, bad = set(), [], []
    for p in raw:
        r = json.loads(p.read_text(encoding="utf-8"))
        eid = r.get("experiment_id", "")
        if eid in ids:
            dupes.append(eid)
        ids.add(eid)
        required = ["experiment_id", "source_commit", "config_sha256",
                    "result_hash", "seed", "timestamp"]
        # protocol_hash is only required AFTER protocol freeze (formal tracks);
        # Phase-1 determinism records intentionally predate the protocol.
        if r.get("track") == "formal":
            required.append("protocol_hash")
        for k in required:
            if not r.get(k):
                bad.append({"file": p.name, "missing": k})
        if r.get("config_sha256") != core.sha256_json(r.get("applied_config", {})):
            bad.append({"file": p.name, "missing": "config_hash_mismatch"})
        m = r.get("metrics") or {}
        if m.get("one_step_mse") is not None and r.get("result_hash") != core.sha256_json(m):
            bad.append({"file": p.name, "missing": "result_hash_mismatch"})
    return {"n_records": len(raw), "n_unique": len(ids), "duplicates": dupes,
            "violations": bad, "pass": bool(raw) and not dupes and not bad}


def statistical_audit(root: Path) -> dict[str, Any]:
    raw = {p.stem: json.loads(p.read_text(encoding="utf-8"))
           for p in (root / "raw_data" / "immutable").glob("rq2_*.json")}
    mse = {}
    for eid, r in raw.items():
        parts = eid.split("__")
        if len(parts) < 3:
            continue
        arm = parts[0].replace("rq2_", "")
        task, seed = parts[1], parts[2]
        v = (r.get("metrics") or {}).get("one_step_mse")
        if v is not None:
            mse[(task, seed, arm)] = float(v)
    base = {}
    for p in (root / "raw_data" / "immutable").glob("base__*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        parts = p.stem.split("__")
        if len(parts) >= 3:
            v = (r.get("metrics") or {}).get("one_step_mse")
            if v is not None:
                base[(parts[1], parts[2])] = float(v)
    units: dict[str, dict[str, list[float]]] = {}
    for (task, seed, arm), v in mse.items():
        b = base.get((task, seed))
        if b is None or abs(b) < 1e-12:
            continue
        units.setdefault(task, {}).setdefault(arm, []).append((b - v) / abs(b))
    ts = sorted(t for t, d in units.items() if {"arr", "random"} <= set(d))
    def mn(x): return sum(x) / len(x) if x else 0.0
    diffs = [mn(units[t]["arr"]) - mn(units[t]["random"]) for t in ts]
    recomputed = mn(diffs) if diffs else 0.0
    reported = root / "formal" / "rq2_selection" / "results.json"
    rep = json.loads(reported.read_text(encoding="utf-8"))["arr_vs_random"]["mean"] \
        if reported.is_file() else 0.0
    return {"n_tasks": len(ts), "recomputed_mean": recomputed,
            "reported_mean": rep, "abs_delta": abs(recomputed - rep),
            "pass": len(ts) > 0 and abs(recomputed - rep) < 1e-6}


def replay_audit(root: Path, budget: Any) -> dict[str, Any]:
    from .engine import run_experiment
    raw = sorted((root / "raw_data" / "immutable").glob("*.json"))
    formal = [r for r in raw if json.loads(r.read_text(encoding="utf-8")).get("track") == "formal"]
    k = max(1, len(formal) // 10)
    sample = formal[:k]
    ok = 0
    for p in sample:
        r = json.loads(p.read_text(encoding="utf-8"))
        budget.consume("replay")
        cfg = r["applied_config"]
        rr = run_experiment(experiment_id=r["experiment_id"] + "__replay",
            track="replay", strategy=r["strategy"], task_id=r["task_id"],
            seed=r["seed"], train_cfg=cfg["train"], model_cfg=cfg["model"],
            test_cfg=cfg["test"], root=root, mode=cfg["mode"],
            num_threads=cfg["num_threads"],
            deterministic_algorithms=cfg["deterministic_algorithms"],
            protocol_hash=r.get("protocol_hash", ""))
        a = (r.get("metrics") or {}).get("one_step_mse")
        b = (rr.get("metrics") or {}).get("one_step_mse")
        if a is not None and b is not None and abs(a - b) < 1e-9 and rr.get("ok"):
            ok += 1
    return {"n_sampled": len(sample), "n_match": ok,
            "pass": len(sample) > 0 and ok == len(sample)}


def claims_audit(root: Path) -> dict[str, Any]:
    checks = {
        "source": (root / "source_audit" / "source_audit.json").is_file(),
        "determinism": (root / "preflight" / "determinism" / "determinism_study.json").is_file(),
        "evidence_system": (root / "evidence_system" / "evidence_store.json").is_file(),
        "rq1": (root / "formal" / "rq1_belief_calibration" / "results.json").is_file(),
        "rq2": (root / "formal" / "rq2_selection" / "results.json").is_file(),
        "rq3": (root / "formal" / "rq3_transfer" / "results.json").is_file(),
        "rq4": (root / "formal" / "rq4_recursive" / "results.json").is_file(),
        "rq5": (root / "formal" / "rq5_discovery" / "results.json").is_file(),
    }
    return {"checks": checks, "pass": all(checks.values())}


def run_audit(root: Path, budget: Any) -> dict[str, Any]:
    out = {"integrity": integrity_audit(root), "statistics": statistical_audit(root),
           "claims": claims_audit(root), "replay": replay_audit(root, budget)}
    verdict = {k: bool(v.get("pass")) for k, v in out.items()}
    for k, v in out.items():
        (root / "independent_audit" / f"{k}_audit.json").write_text(json.dumps(
            v, indent=2, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")
    return {"audits": out, "verdict": verdict, "pass": all(verdict.values())}


__all__ = ["integrity_audit", "statistical_audit", "replay_audit",
           "claims_audit", "run_audit"]
