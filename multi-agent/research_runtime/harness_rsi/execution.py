"""M6 -- controlled Base/Candidate paired execution.

The Base harness is frozen once, the Candidate is frozen once, and both are
executed on the SAME task families, model version, tool versions, datasets,
seeds and resource limits. Order is randomised deterministically per task
family so a causal ordering effect cannot masquerade as method improvement.

Every outcome is retained -- failed, invalid, interrupted and infrastructure
errors included. Infrastructure failures are labelled separately from research
failures and are NEVER silently dropped from the denominator; the evaluator
fails closed if any infrastructure failure is present.

This module does not know which engine it drives: the caller supplies a real
``runner(harness_artifact, task) -> dict``. The production driver passes the
existing SciNet++/Embodied/Quant adapters; engine-free tests pass a
deterministic stub. No synthetic success response is generated here.
"""
from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Callable

from .evaluation import TaskResult
from .harness import HarnessBundle, freeze_bundle
from .identity import freeze_json, sha256_json

INFRASTRUCTURE_MARKERS = (
    "timeout", "timed out", "connection", "oom", "out of memory", "crash",
    "permission denied", "not found", "missing engine", "subprocess",
    "sandbox", "safe-delete", "rate limit", "network",
)


class ExecutionError(RuntimeError):
    """Raised when the paired execution contract itself is violated."""


def classify_failure(error: str) -> str:
    """Distinguish an infrastructure fault from a research failure."""
    text = str(error).lower()
    return ("infrastructure" if any(m in text for m in INFRASTRUCTURE_MARKERS)
            else "research")


def _trajectory(record: dict[str, Any]) -> dict[str, Any]:
    """Normalise a runner record into a frozen, auditable trajectory."""
    return {
        "prompt_hashes": list(record.get("prompt_hashes", [])),
        "tool_calls": list(record.get("tool_calls", [])),
        "tokens": int(record.get("tokens", 0)),
        "wall_s": float(record.get("wall_s", 0.0)),
        "budget": dict(record.get("budget", {})),
        "errors": list(record.get("errors", [])),
        "artifacts": list(record.get("artifacts", [])),
        "output_sha256": str(record.get("output_sha256", "")),
        "harness_sha": str(record.get("harness_sha", "")),
        "task_id": str(record.get("task_id", "")),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def run_paired(*, base: HarnessBundle, candidate: HarnessBundle,
               tasks: list[Any], runner: Callable[[dict[str, Any], Any], dict],
               out_dir: str | Path, experiment_id: str, seed: int = 20261116
               ) -> dict[str, Any]:
    """Execute one paired experiment and freeze the complete trajectory log."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    base_frozen = freeze_bundle(base, root / "base_harness.json")
    cand_frozen = freeze_bundle(candidate, root / "candidate_harness.json")
    if base_frozen["candidate_sha"] == cand_frozen["candidate_sha"]:
        raise ExecutionError("base and candidate are the same harness")

    rng = random.Random(seed)
    families = sorted({getattr(t, "family_id", getattr(t, "domain", "unknown"))
                       for t in tasks})
    # Randomise WHICH arm runs first per family, not which task each arm sees.
    order = {fam: rng.choice(("base_first", "candidate_first"))
             for fam in families}

    results: list[TaskResult] = []
    trajectories: list[dict[str, Any]] = []
    for task in tasks:
        tid = getattr(task, "task_id", getattr(task, "id", ""))
        fam = getattr(task, "family_id", getattr(task, "domain", "unknown"))
        arms = ("base", "candidate") if order[fam] == "base_first" \
            else ("candidate", "base")
        for arm in arms:
            harness = base if arm == "base" else candidate
            frozen = base_frozen if arm == "base" else cand_frozen
            started = time.time()
            try:
                raw = runner({"harness": harness.to_dict(),
                              "harness_sha": frozen["candidate_sha"],
                              "artifact_path": frozen["path"],
                              "experiment_id": experiment_id,
                              "seed": getattr(task, "seed", 0)}, task)
                errors = list(raw.get("errors") or [])
                ok = bool(raw.get("ok", True)) and not errors
                kind = classify_failure(" ".join(errors)) if errors else ""
                traj = _trajectory({**raw, "task_id": tid,
                                    "harness_sha": frozen["candidate_sha"]})
                traj["wall_s"] = traj["wall_s"] or (time.time() - started)
                paths = traj["artifacts"]
                traj["artifact_sha256"] = sha256_json(paths)
                result = TaskResult(
                    task_id=tid, family_id=fam,
                    harness_sha=frozen["candidate_sha"], arm=arm,
                    metrics=dict(raw.get("metrics", {})), ok=ok,
                    failure_kind=kind,
                    artifact_sha256=traj["artifact_sha256"],
                    trajectory_sha256=sha256_json(traj),
                    tokens=traj["tokens"], wall_s=traj["wall_s"])
            except Exception as exc:  # noqa: BLE001 - retain infrastructure fault
                kind = classify_failure(f"{type(exc).__name__}: {exc}")
                traj = _trajectory({"task_id": tid, "errors": [
                    f"{type(exc).__name__}: {exc}"],
                    "harness_sha": frozen["candidate_sha"]})
                traj["wall_s"] = time.time() - started
                traj["artifact_sha256"] = ""
                result = TaskResult(
                    task_id=tid, family_id=fam,
                    harness_sha=frozen["candidate_sha"], arm=arm,
                    metrics={}, ok=False, failure_kind=kind,
                    trajectory_sha256=sha256_json(traj),
                    wall_s=traj["wall_s"])
            results.append(result)
            trajectories.append(traj)

    log = freeze_json({
        "experiment_id": experiment_id, "seed": seed,
        "base_sha": base_frozen["candidate_sha"],
        "candidate_sha": cand_frozen["candidate_sha"],
        "arm_order_by_family": order,
        "n_results": len(results),
        "results": [r.to_dict() for r in results],
        "trajectories": trajectories,
        "retention_policy": "ALL outcomes retained; no silent exclusion",
    }, root / "paired_trajectory.json")
    return {"base_sha": base_frozen["candidate_sha"],
            "candidate_sha": cand_frozen["candidate_sha"],
            "results": results, "trajectory_sha256": log["sha256"],
            "trajectory_path": log["path"],
            "arm_order_by_family": order,
            "n_results": len(results),
            "n_infra_failures": len([r for r in results
                                     if r.failure_kind == "infrastructure"])}


__all__ = ["ExecutionError", "classify_failure", "run_paired"]
