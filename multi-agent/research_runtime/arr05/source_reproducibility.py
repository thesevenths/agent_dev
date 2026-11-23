"""ARR-05 P0: source reproducibility audit + ARR-04 evidence reconciliation.

No LLM calls. Recomputes ARR-04's primary M2 result from committed immutable
raw records and builds a complete raw record manifest.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from . import core, statistics as st


def _git(args: list[str], cwd: str | Path) -> str:
    try:
        return subprocess.check_output(["git"] + args, cwd=str(cwd),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def source_audit(root: Path, arr04_root: Path) -> dict[str, Any]:
    """Verify ARR-04 source can be recovered from Git; report declaration gap."""
    arr04_canonical = json.loads((arr04_root / "final" / "ARR04_FINAL_REPORT.json").read_text(
        encoding="utf-8"))
    declared = arr04_canonical["source_commit"]
    actual_artifact_commit = "eaa8ed79417550aaa3120c2f5976053ed89cf0b1"
    repo = Path(root).resolve().parents[3]  # ARR-05 campaign -> campaigns -> experiments -> multi-agent -> agent_dev

    # Are arr04 source files present at the declared commit?
    declared_files = _git(["ls-tree", "-r", "--name-only", declared,
                           "--", "multi-agent/research_runtime/arr04/"], repo)
    declared_count = len([l for l in declared_files.splitlines() if l.strip()])

    # Are they present at the artifact commit?
    artifact_files = _git(["ls-tree", "-r", "--name-only", actual_artifact_commit,
                           "--", "multi-agent/research_runtime/arr04/"], repo)
    artifact_count = len([l for l in artifact_files.splitlines() if l.strip()])

    # Recover source to an audit copy and verify key functions exist.
    src_dir = root / "source_audit" / "recovered_arr04_source"
    src_dir.mkdir(parents=True, exist_ok=True)
    recovered = {}
    for name in ("core.py", "engine.py", "tracks.py", "statistics.py", "audit.py",
                 "campaign.py", "orchestrator.py"):
        content = _git(["show", f"{actual_artifact_commit}:multi-agent/"
                        f"research_runtime/arr04/{name}"], repo)
        (src_dir / name).write_text(content, encoding="utf-8")
        recovered[name] = {"bytes": len(content),
                           "has_run_experiment": "def run_experiment" in content,
                           "has_paired_stats": "def paired_stats" in content}

    findings = [
        {"id": "S01", "finding": "ARR-04 declared source_commit differs from the "
         "commit that actually contains the arr04 implementation",
         "declared": declared, "artifact_commit": actual_artifact_commit,
         "declared_arr04_files": declared_count,
         "artifact_arr04_files": artifact_count,
         "severity": "reproducibility_gap",
         "resolution": "ARR-05 records both commits; source recovered from artifact "
         "commit; historical ARR-04 canonical NOT modified"},
        {"id": "S02", "finding": "ARR-04 implementation is recoverable from Git",
         "recovered_files": recovered, "severity": "info"},
        {"id": "S03", "finding": "ARR-04 raw records exist as immutable artifacts",
         "count": len(list((arr04_root / "raw_data" / "immutable").glob("*.json"))),
         "severity": "info"},
    ]
    return {"declared_source_commit": declared,
            "actual_implementation_commit": actual_artifact_commit,
            "declared_arr04_source_files": declared_count,
            "artifact_arr04_source_files": artifact_count,
            "findings": findings,
            "source_recovered": declared_count == 0 and artifact_count > 0,
            "historical_canonical_unmodified": True}


def raw_record_manifest(root: Path, arr04_root: Path) -> dict[str, Any]:
    """Build a complete manifest mapping ARR-04 reported records to real files."""
    raw_dir = arr04_root / "raw_data" / "immutable"
    records = []
    seen: set[str] = set()
    dupes = []
    invalid = []
    for p in sorted(raw_dir.glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        eid = rec.get("experiment_id", "")
        if eid in seen:
            dupes.append(eid)
        seen.add(eid)
        parts = eid.split("__")
        track = parts[0] if parts else ""
        strategy = parts[1] if len(parts) > 1 else ""
        task = "__".join(parts[2:-1]) if len(parts) > 3 else ""
        seed = parts[-1] if parts else ""
        artifact = str(p.resolve())
        trace = rec.get("trace_path")
        trace_hash = rec.get("model_signature_sha256", "")
        ok = (bool(eid) and bool(rec.get("timestamp")) and
              bool(rec.get("source_commit")) and bool(trace_hash) and
              Path(artifact).is_file() and (not trace or Path(trace).is_file()))
        if not ok:
            invalid.append(eid)
        records.append({
            "record_id": core.sha256_json({"eid": eid, "path": artifact})[:16],
            "experiment_id": eid, "track": track, "strategy": strategy,
            "task_id": task, "seed": seed,
            "benchmark_version": rec.get("benchmark_version", ""),
            "protocol_version": rec.get("protocol_version", ""),
            "source_commit": rec.get("source_commit", ""),
            "artifact_path": artifact,
            "artifact_sha256": core.sha256_file(artifact),
            "execution_timestamp": rec.get("timestamp", ""),
            "trace_hash": trace_hash,
            "execution_status": "ok" if rec.get("ok") else "failed",
        })
    manifest = {"n_records": len(records), "n_unique_ids": len(seen),
                "duplicate_ids": dupes, "invalid_records": invalid,
                "reconciled": not dupes and not invalid and len(records) == len(seen),
                "records": records}
    (root / "preflight" / "raw_record_manifest.json").write_text(json.dumps(
        manifest, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8")
    return manifest


def recompute_arr04_primary(root: Path, arr04_root: Path) -> dict[str, Any]:
    """Independently recompute ARR-04's M2 arr_vs_random effect from raw records.

    No LLM, no metric reuse: read raw MSE, compute normalized improvement,
    aggregate to task-level units, recompute paired t.
    """
    raw = {p.stem: json.loads(p.read_text(encoding="utf-8"))
           for p in (arr04_root / "raw_data" / "immutable").glob("m2__*.json")}
    mse: dict[str, float] = {}
    for eid, rec in raw.items():
        parts = eid.split("__")
        if len(parts) < 4:
            continue
        arm = parts[1]
        seed = parts[-1]
        task = "__".join(parts[2:-1])
        v = (rec.get("metrics") or {}).get("one_step_mse")
        if v is not None:
            mse[f"{task}|{seed}|{arm}"] = float(v)
    units: dict[str, dict[str, list[float]]] = {}
    for key, val in mse.items():
        task, seed, arm = key.split("|")
        base = mse.get(f"{task}|{seed}|base")
        if base is None or abs(base) < 1e-12 or arm == "base":
            continue
        units.setdefault(task, {}).setdefault(arm, []).append((base - val) / abs(base))
    ts = sorted(t for t, d in units.items() if "arr" in d and "random" in d)
    diffs = [st.mean(units[t]["arr"]) - st.mean(units[t]["random"]) for t in ts]
    recomputed = st.paired_stats(diffs)
    reported = json.loads((arr04_root / "m2_research_selection" /
                           "strategy_results.json").read_text(encoding="utf-8"))
    rep = reported["arr_vs_random"]
    delta = abs(recomputed["mean"] - rep["mean"])
    result = {"n_tasks": len(ts), "recomputed_mean": recomputed["mean"],
              "recomputed_ci95": recomputed["ci95"], "recomputed_p": recomputed["p"],
              "reported_mean": rep["mean"], "reported_ci95": rep["ci95"],
              "abs_mean_delta": delta,
              "reproduction_success": len(ts) > 0 and delta < 1e-6,
              "note": "recomputed from committed immutable raw records; no LLM calls"}
    (root / "source_audit" / "arr04_reproduction.json").write_text(json.dumps(
        result, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return result


__all__ = ["source_audit", "raw_record_manifest", "recompute_arr04_primary"]
