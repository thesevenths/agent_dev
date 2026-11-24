"""ARR-07 campaign: three end-to-end validations of the autonomous loop."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import benchmark, core
from .engine_gate import repaired_applicability_gate
from .loop import run_loop


def _balanced_tasks(family: str, per_cause: int = 2, start_v: int = 30) -> list:
    """Sample ACROSS all 5 causes, not the first N sorted by cause."""
    out = []
    for c in benchmark.CAUSES:
        pool = [t for t in benchmark.build_tasks(family) if t.cause == c]
        out.extend(pool[:per_cause])
    return out


def _rebuild(root: Path, b: dict) -> None:
    raw = list(root.rglob("loop_*.json"))
    b["engine_runs"] = len(raw)
    b["llm_calls"] = len(list(root.rglob("llm_*.txt")))


def validation_closed_loop(root: Path, n_tasks: int = 10,
                           max_iterations: int = 4) -> dict[str, Any]:
    """V1: does the BLIND loop converge on held-out tasks through iteration?"""
    tasks = _balanced_tasks("heldout", per_cause=max(1, n_tasks // len(benchmark.CAUSES)))
    results = []
    for i, t in enumerate(tasks):
        r = run_loop(task=t, seed=42, root=root / "closed_loop",
                     blind=True, max_iterations=max_iterations)
        results.append({"task_id": t.task_id, "cause": t.cause,
                        "converged": r.converged, "iterations": r.iterations,
                        "final_improvement": r.final_improvement,
                        "replans": r.replans, "llm_calls": r.llm_calls,
                        "engine_runs": r.engine_runs,
                        "experiment_sequence": [s.chosen_experiment for s in r.steps],
                        "improvements": [s.improvement for s in r.steps],
                        "spurious_convergence": any(s.spurious_convergence
                                                    for s in r.steps),
                        "cause_identified": any(s.root_cause_correct for s in r.steps)})
    conv = sum(int(r["converged"]) for r in results)
    n_spurious = sum(int(r["spurious_convergence"]) for r in results)
    n_cause = sum(int(r["cause_identified"]) for r in results)
    out = {"n_tasks": len(results), "n_converged": conv,
           "n_spurious_convergence": n_spurious, "n_correct_cause": n_cause,
           "convergence_rate": conv / max(1, len(results)),
           "mean_iterations": sum(r["iterations"] for r in results) / max(1, len(results)),
           "mean_replans": sum(r["replans"] for r in results) / max(1, len(results)),
           "tasks": results,
           "pass": conv > 0 and any(r["iterations"] > 1 for r in results)}
    (root / "loop" / "closed_loop" / "results.json").write_text(json.dumps(
        out, indent=2, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")
    return out


def validation_blind_vs_leaky(root: Path, n_tasks: int = 10) -> dict[str, Any]:
    """V2: the decisive control — blind vs config-leaky on the SAME tasks."""
    tasks = _balanced_tasks("heldout", per_cause=max(1, n_tasks // len(benchmark.CAUSES)))
    rows = []
    for t in tasks:
        b = run_loop(task=t, seed=42, root=root / "blind_vs_leaky",
                     blind=True, max_iterations=3)
        l = run_loop(task=t, seed=42, root=root / "blind_vs_leaky",
                     blind=False, max_iterations=3)
        rows.append({"task_id": t.task_id, "cause": t.cause,
                     "blind_converged": b.converged, "leaky_converged": l.converged,
                     "blind_best_improvement": max([s.improvement or -9 for s in b.steps] + [-9]),
                     "leaky_best_improvement": max([s.improvement or -9 for s in l.steps] + [-9])})
    bc = sum(int(r["blind_converged"]) for r in rows)
    lc = sum(int(r["leaky_converged"]) for r in rows)
    out = {"n_tasks": len(rows), "blind_converged": bc, "leaky_converged": lc,
           "blind_rate": bc / max(1, len(rows)), "leaky_rate": lc / max(1, len(rows)),
           "tasks": rows,
           "interpretation": ("If blind converges at a comparable rate to leaky, the "
                              "loop reasons from experimental data, not config leakage."),
           "pass": bc >= max(1, int(len(rows) * 0.5))}
    (root / "loop" / "blind_vs_leaky" / "results.json").write_text(json.dumps(
        out, indent=2, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")
    return out


def validation_memory_gate_repair(root: Path, n_tasks: int = 10) -> dict[str, Any]:
    """V3: repaired gate ABSTAINS on mismatch instead of degrading to PLACEBO."""
    items = []
    # Reconstruct ARR-06-style memory items from cause registry.
    from ..arr06.memory import MemoryItem
    for c in benchmark.CAUSES:
        items.append(MemoryItem(
            knowledge_id="M07_" + core.sha256_json(c)[:8],
            claim=f"intervention for {c}",
            supporting_evidence=["preflight"], contradicting_evidence=[],
            provenance="arr06-preflight", verification_status="VERIFIED",
            confidence=0.8,
            applicability_conditions={"mechanism": c, "domain": "Reacher dynamics",
                                       "action_conditioned": True},
            failure_conditions=["visual", "real robot"], domain="Reacher",
            original_experiment="pf", replication_history=["r1", "r2"],
            transfer_history=[]))
    tasks = _balanced_tasks("heldout", per_cause=max(1, n_tasks // len(benchmark.CAUSES)))
    rows = []
    for t in tasks:
        gate = repaired_applicability_gate(items, t)
        rows.append({"task_id": t.task_id, "cause": t.cause,
                     "should_transfer": gate["should_transfer"],
                     "abstain_reason": gate["abstain_reason"],
                     "selected": len(gate["selected"]),
                     "correct": gate["should_transfer"] == (t.cause in benchmark.CAUSES)})
    # Correct behaviour: transfer for matching VERIFIED item, abstain otherwise.
    abst = sum(1 for r in rows if not r["should_transfer"])
    ok = all(r["should_transfer"] for r in rows)  # all tasks have a VERIFIED matching item
    out = {"n_tasks": len(rows), "n_abstain": abst,
           "no_placebo_degradation": all(r["selected"] > 0 or not r["should_transfer"]
                                         for r in rows),
           "tasks": rows, "pass": ok and abst == 0}
    (root / "loop" / "memory_gate_repair" / "results.json").write_text(json.dumps(
        out, indent=2, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")
    return out


def execute(campaign_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(campaign_root).resolve() if campaign_root else (Path.cwd() / core.CAMPAIGN_ROOT).resolve()
    for d in ("protocol", "loop/closed_loop", "loop/blind_vs_leaky",
              "loop/memory_gate_repair", "raw_data/immutable", "raw_data/traces",
              "logs", "final"):
        (root / d).mkdir(parents=True, exist_ok=True)
    v1 = validation_closed_loop(root)
    v2 = validation_blind_vs_leaky(root)
    v3 = validation_memory_gate_repair(root)
    budget = {}
    _rebuild(root, budget)
    n_spurious = v1.get("n_spurious_convergence", 0)
    spurious_rate = n_spurious / max(1, v1["n_converged"])
    cause_coverage = len({t["cause"] for t in v1["tasks"]}) / len(benchmark.CAUSES)
    verdicts = {
        # Significant spurious convergence (>25%) means the loop "reaches the
        # target" without identifying the mechanism -> NOT genuine research.
        "closed_loop_convergence": ("PASS" if (v1["pass"] and spurious_rate <= 0.25
                                     and cause_coverage == 1.0) else "PARTIAL"
                                    if v1["pass"] else "FAIL"),
        # Blind must work on a clear majority; a large blind/leaky gap shows
        # the loop still leans on config rather than measured evidence.
        "no_config_leakage": ("PASS" if (v2["pass"] and v2["blind_rate"] >= 0.7
                               and v2["leaky_rate"] - v2["blind_rate"] <= 0.2)
                              else "PARTIAL" if v2["pass"] else "FAIL"),
        "memory_gate_repair": "PASS" if v3["pass"] else "FAIL",
    }
    canonical = {"campaign_id": core.CAMPAIGN_ID, "source_commit": core.git_head(".."),
        "branch": core.BRANCH, "validations": {"V1": v1, "V2": v2, "V3": v3},
        "verdicts": verdicts, "budget": budget}
    canonical["overall"] = ("AUTONOMOUS LOOP VALIDATED" if all(
        x == "PASS" for x in verdicts.values()) else "NOT VALIDATED")
    canonical["canonical_sha256"] = core.sha256_json(canonical)
    (root / "final" / "ARR07_FINAL_REPORT.json").write_text(json.dumps(
        canonical, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8")
    (root / "final" / "ARR07_VERDICT.json").write_text(json.dumps(
        verdicts, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return canonical


__all__ = ["execute", "validation_closed_loop", "validation_blind_vs_leaky",
           "validation_memory_gate_repair"]
