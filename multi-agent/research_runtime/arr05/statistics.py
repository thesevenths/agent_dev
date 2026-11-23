"""ARR-05 statistics: task-level paired inference, calibration, info gain, verdicts."""
from __future__ import annotations

import math
import random
import statistics
from typing import Any

from research_runtime.evolution.statistics import paired_p_value, student_t_crit_two_sided


def mean(v: list[float]) -> float:
    return sum(v) / len(v) if v else 0.0


def paired_stats(diffs: list[float], *, alpha: float = 0.05,
                 n_boot: int = 3000, n_perm: int = 3000,
                 seed: int = 20261123) -> dict[str, Any]:
    vals = [float(d) for d in diffs]
    n = len(vals)
    m = mean(vals)
    if n < 2:
        return {"n": n, "mean": m, "p": 1.0, "ci95": [0.0, 0.0],
                "permutation_p": 1.0, "bootstrap_ci95": [0.0, 0.0],
                "dz": None, "sd": 0.0}
    sd = statistics.stdev(vals)
    p = paired_p_value(m, sd, n)
    crit = student_t_crit_two_sided(alpha=alpha, df=n - 1)
    margin = crit * sd / math.sqrt(n)
    rng = random.Random(seed)
    boots = sorted(mean([vals[rng.randrange(n)] for _ in range(n)])
                   for _ in range(n_boot))
    lo, hi = int(alpha / 2 * n_boot), min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    obs = abs(m)
    ge = sum(1 for _ in range(n_perm)
             if abs(mean([v * (1 if rng.random() < 0.5 else -1) for v in vals])) >= obs - 1e-15)
    return {"n": n, "mean": m, "sd": sd, "p": p,
            "ci95": [m - margin, m + margin],
            "permutation_p": (ge + 1) / (n_perm + 1),
            "bootstrap_ci95": [boots[lo], boots[hi]],
            "dz": m / sd if sd > 0 else None}


def holm(p_values: list[float], alpha: float = 0.05) -> list[dict[str, Any]]:
    indexed = sorted((float(p), i) for i, p in enumerate(p_values))
    m = len(indexed)
    out = [0.0] * m
    running = 0.0
    for rank, (p, idx) in enumerate(indexed):
        running = max(running, min(1.0, (m - rank) * p))
        out[idx] = running
    return [{"p": p, "holm_p": out[i], "significant": out[i] <= alpha}
            for i, p in enumerate(p_values)]


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def calibration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confs = [float(r["confidence"]) for r in rows]
    correct = [int(bool(r["correct"])) for r in rows]
    n = len(rows)
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "ece": 0.0, "brier": 0.0,
                "log_loss": 0.0, "ace": 0.0, "calibration_slope": 0.0,
                "calibration_intercept": 0.0, "reliability": [],
                "mean_confidence": 0.0, "false_certainty_rate": 0.0}
    brier = mean([(c - o) ** 2 for c, o in zip(confs, correct)])
    eps = 1e-12
    logloss = -mean([o * math.log(max(c, eps)) + (1 - o) * math.log(max(1 - c, eps))
                     for c, o in zip(confs, correct)])
    bins = [[] for _ in range(10)]
    for c, o in zip(confs, correct):
        bins[min(9, max(0, int(c * 10)))].append((c, o))
    ece = ace = 0.0
    rel = []
    for i, b in enumerate(bins):
        if not b:
            rel.append({"bin": i, "n": 0})
            continue
        mc, emp = mean([x[0] for x in b]), mean([float(x[1]) for x in b])
        ece += len(b) / n * abs(emp - mc)
        ace += len(b) / n * abs(emp - mc)  # equal-width adaptive ECE
        rel.append({"bin": i, "n": len(b), "mean_confidence": mc, "empirical": emp})
    mc_all, emp_all = mean(confs), mean([float(x) for x in correct])
    var = sum((c - mc_all) ** 2 for c in confs) / n
    slope = (sum((c - mc_all) * (o - emp_all) for c, o in zip(confs, correct)) / n
             / var) if var > 1e-12 else 0.0
    intercept = emp_all - slope * mc_all
    fc = [(c, o) for c, o in zip(confs, correct) if not o and c >= 0.7]
    return {"n": n, "accuracy": emp_all, "ece": ece, "ace": ace, "brier": brier,
            "log_loss": logloss, "calibration_slope": slope,
            "calibration_intercept": intercept, "reliability": rel,
            "mean_confidence": mc_all,
            "false_certainty_rate": len(fc) / n}


def temperature_scale(logits_conf: list[float], labels: list[int],
                      n_grid: int = 100) -> float:
    """Fit a single temperature T>0 minimizing NLL on a CALIBRATION split only."""
    import numpy as np
    conf = np.clip(np.asarray(logits_conf, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(labels, dtype=int)

    def nll(t: float) -> float:
        p = np.clip(conf ** (1.0 / max(t, 1e-3)), 1e-6, 1 - 1e-6)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    best_t, best = 1.0, nll(1.0)
    for t in np.linspace(0.2, 5.0, n_grid):
        v = nll(float(t))
        if v < best:
            best, best_t = v, float(t)
    return best_t


def verdicts(*, engineering: dict[str, Any], integrity: dict[str, Any],
             m1: dict[str, Any], m2: dict[str, Any], m3: dict[str, Any],
             rq: dict[str, Any], rsi: dict[str, Any], audit: dict[str, Any],
             reproduction: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    def sig(x: dict[str, Any], alpha: float) -> bool:
        return bool(x.get("significant", x.get("p", 1.0) <= alpha))
    alpha = protocol["alpha"]
    g = protocol["gates"]
    v = {
        "source_reproducibility": "PASS" if reproduction.get("reproduction_success")
                                   and not reproduction.get("reconciliation_failed")
                                   else "PARTIAL",
        "engineering": "PASS" if bool(engineering.get("n_runs", 0)) and all(
            bool(x) for k, x in engineering.items() if k != "n_runs") else "FAIL",
        "benchmark_integrity": "PASS" if all(integrity.values()) else "BLOCKED",
        "m1_active_diagnosis": "FAIL",
        "m2_confirmatory_selection": "FAIL",
        "m3_memory_transfer": "FAIL",
        "rq_discovery": "FAIL",
        "rsi_recursive_methodology": "INCONCLUSIVE",
        "independent_audit": "PASS" if all(audit.values()) else "FAIL",
    }
    # M1: active improves + calibration acceptable + info gain + less false certainty.
    a, p = m1["active"], m1["passive"]
    m1_ok = (sig(m1["active_vs_passive"], alpha)
             and m1["active_vs_passive"]["mean"] >= g["m1_min_accuracy_gain"]
             and a["calibrated_ece"] <= g["m1_max_ece"]
             and m1["realized_information_gain"] >= g["m1_min_info_gain"]
             and a["false_certainty_rate"] <= p["false_certainty_rate"]
             and a["abstention_precision"] >= g["m1_min_abstention_precision"])
    v["m1_active_diagnosis"] = ("PARTIAL" if (sig(m1["active_vs_passive"], alpha)
                                  and m1["active_vs_passive"]["mean"] > 0
                                  and not m1_ok) else
                                ("PASS" if m1_ok else "FAIL"))
    # M2 confirmatory.
    m2_ok = (sig(m2["arr_vs_random"], alpha)
             and m2["arr_vs_random"]["mean"] >= g["m2_mme"]
             and sig(m2["arr_vs_rule"], alpha))
    v["m2_confirmatory_selection"] = "PASS" if m2_ok else (
        "INCONCLUSIVE" if m2["arr_vs_random"]["mean"] > 0 else "FAIL")
    # M3 transfer.
    m3_ok = (sig(m3["gated_vs_fresh"], alpha)
             and m3["gated_vs_fresh"]["mean"] >= g["m3_mme"]
             and m3["negative_transfer_rate"] <= g["m3_max_negative_transfer"]
             and m3["applicability_match"] >= g["m3_min_applicability"])
    v["m3_memory_transfer"] = ("FAIL" if sig(m3["raw_vs_fresh"], alpha)
                               and m3["raw_vs_fresh"]["mean"] < 0 else
                               ("PASS" if m3_ok else "INCONCLUSIVE"))
    # RQ discovery.
    rq_ok = (sig(rq["arr_vs_random"], alpha)
             and rq["arr_vs_random"]["mean"] >= g["rq_mme"]
             and rq["falsifiability"] >= g["rq_min_falsifiability"]
             and rq["independent_evaluation"])
    v["rq_discovery"] = "PASS" if rq_ok else "FAIL"
    # RSI recursive methodology.
    rsi_ok = (rsi["actual_methodology_changes"] > 0
              and sig(rsi["evolved_vs_frozen"], alpha)
              and sig(rsi["evolved_vs_memory_only"], alpha)
              and not rsi["ood_regression"])
    v["rsi_recursive_methodology"] = "PASS" if rsi_ok else "INCONCLUSIVE"
    critical = [v["m1_active_diagnosis"], v["m2_confirmatory_selection"],
                v["m3_memory_transfer"], v["rq_discovery"],
                v["rsi_recursive_methodology"]]
    if all(x == "PASS" for x in critical):
        v["overall"] = "VALIDATED WITHIN DEFINED BENCHMARK SCOPE"
    elif sum(x == "PASS" for x in critical) >= 2:
        v["overall"] = "PARTIALLY VALIDATED"
    else:
        v["overall"] = "AUTONOMOUS RESEARCH SYSTEM — NOT VALIDATED"
    return v


__all__ = ["mean", "paired_stats", "holm", "wilson", "calibration",
           "temperature_scale", "verdicts"]
