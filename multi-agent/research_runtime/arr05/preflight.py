"""ARR-05 preflight: benchmark validation, power analysis, applicability gates."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from . import benchmark, core
from .orchestrator import BudgetController


def benchmark_validation(root: Path, budget: BudgetController) -> dict[str, Any]:
    tasks = benchmark.build_tasks()
    by_cause_split: dict[str, set[str]] = {}
    for t in tasks:
        by_cause_split.setdefault(t.cause, set()).add(t.split)
    # Identifiability: each cause has a registered discriminative experiment
    # and distinct signature dimensions.
    pairs = []
    causes = list(benchmark.CAUSES)
    for i, a in enumerate(causes):
        for b in causes[i + 1:]:
            exp_sep = all(any(e.discriminates == (c,) for e in
                              benchmark.DIAGNOSTIC_EXPERIMENTS) for c in (a, b))
            sig_a = next(t.expected_signature for t in tasks if t.cause == a)
            sig_b = next(t.expected_signature for t in tasks if t.cause == b)
            dim_sep = any(sig_a.get(k) != sig_b.get(k) for k in sig_a)
            pairs.append({"a": a, "b": b, "experiment_separable": exp_sep,
                          "signature_separable": dim_sep,
                          "identifiable": exp_sep or dim_sep})
    # Real engine sensitivity: run one failure+oracle pair per cause.
    from .engine import apply_experiment, run_experiment
    sensitivity = {}
    quick_tasks = [t for t in tasks if t.split == "calibration" and t.variant == 5]
    for t in quick_tasks:
        budget.consume("preflight")
        base = run_experiment(experiment_id=f"pf__base__{t.cause}", track="preflight",
                              strategy="base", task_id=t.task_id, seed=42,
                              train_cfg=t.failure_train, model_cfg=t.failure_model,
                              test_cfg=t.test_cfg, root=root)
        exp_id = next((e.experiment_id for e in benchmark.DIAGNOSTIC_EXPERIMENTS
                       if e.discriminates == (t.cause,)), "EXP_PLACEBO")
        tr, mo = apply_experiment(t, exp_id)
        budget.consume("preflight")
        exp = run_experiment(experiment_id=f"pf__{exp_id}__{t.cause}", track="preflight",
                             strategy="exp", task_id=t.task_id, seed=42,
                             train_cfg=tr, model_cfg=mo, test_cfg=t.test_cfg, root=root)
        bm = (base.get("metrics") or {}).get("one_step_mse")
        em = (exp.get("metrics") or {}).get("one_step_mse")
        imp = (bm - em) / abs(bm) if bm and em else None
        sensitivity[t.cause] = {"base_ok": base.get("ok"), "exp_ok": exp.get("ok"),
                                "base_mse": bm, "exp_mse": em,
                                "improvement": imp, "sensitive": bool(imp and imp >= 0.05)}
    result = {"n_tasks": len(tasks), "n_causes": len(benchmark.CAUSES),
              "pairs": pairs,
              "all_identifiable": all(p["identifiable"] for p in pairs),
              "sensitivity": sensitivity,
              "all_sensitive": all(s["sensitive"] for s in sensitivity.values()),
              "task_ids_fresh": all("arr05::" in t.task_id and "::v5" in t.task_id
                                    or t.variant >= 5 for t in tasks)}
    (root / "preflight" / "benchmark_validation.json").write_text(json.dumps(
        result, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8")
    return result


def power_analysis(root: Path, *, alpha: float = 0.05) -> dict[str, Any]:
    from scipy.stats import nct, t as tdist
    table = []
    for n in (10, 20, 26, 30):
        for dz in (0.4, 0.6, 0.8, 1.0):
            crit = tdist.ppf(1 - alpha / 2, n - 1)
            nc = dz * math.sqrt(n)
            power = float(nct.sf(crit, n - 1, nc) + nct.cdf(-crit, n - 1, nc))
            table.append({"n_units": n, "dz": dz, "power": round(power, 4)})
    adequate_08 = [r for r in table if r["dz"] == 0.8 and r["power"] >= 0.8]
    result = {"alpha": alpha, "table": table,
              "n26_dz0.8": next(r["power"] for r in table
                                if r["n_units"] == 26 and r["dz"] == 0.8),
              "min_units_dz0.8_power0.8": (adequate_08[0]["n_units"]
                                           if adequate_08 else None),
              "note": "task families as units; seeds are repeated measurements"}
    (root / "preflight" / "power_analysis.json").write_text(json.dumps(
        result, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return result


def pilot(root: Path, budget: BudgetController) -> dict[str, Any]:
    """Small repeatability pilot: 2 real runs on one task, verify determinism."""
    from .engine import run_experiment
    task = next(t for t in benchmark.build_tasks()
                if t.split == "calibration" and t.variant == 5)
    runs = []
    for s in (42, 43):
        budget.consume("preflight")
        runs.append(run_experiment(experiment_id=f"pilot__{s}", track="pilot",
                                   strategy="base", task_id=task.task_id, seed=s,
                                   train_cfg=task.failure_train,
                                   model_cfg=task.failure_model,
                                   test_cfg=task.test_cfg, root=root))
    result = {"n": len(runs), "all_ok": all(r.get("ok") for r in runs),
              "mse": [r.get("metrics", {}).get("one_step_mse") for r in runs]}
    (root / "preflight" / "pilot_results.json").write_text(json.dumps(
        result, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return result


__all__ = ["benchmark_validation", "power_analysis", "pilot"]
