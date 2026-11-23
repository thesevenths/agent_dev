"""ARR-04 statistics + evaluation: task-level paired inference, calibration,
information gain, verdicts. Seeds are repeated measurements, never units."""
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
                "permutation_p": 1.0, "bootstrap_ci95": [0.0, 0.0], "dz": None}
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
    brier = mean([(c - o) ** 2 for c, o in zip(confs, correct)])
    bins = [[] for _ in range(5)]
    for c, o in zip(confs, correct):
        bins[min(4, max(0, int(c * 5)))].append((c, o))
    ece = 0.0
    rel = []
    for i, b in enumerate(bins):
        if not b:
            rel.append({"bin": i, "n": 0})
            continue
        mc, emp = mean([x[0] for x in b]), mean([float(x[1]) for x in b])
        ece += len(b) / n * abs(emp - mc)
        rel.append({"bin": i, "n": len(b), "mean_confidence": mc, "empirical": emp})
    fc = [(c, o) for c, o in zip(confs, correct) if not o and c >= 0.7]
    return {"n": n, "accuracy": mean([float(x) for x in correct]),
            "ece": ece, "brier": brier, "reliability": rel,
            "mean_confidence": mean(confs),
            "false_certainty_rate": mean([float(c) for c, _ in fc]) / max(1, n)}


def verdicts(*, engineering: dict[str, Any], integrity: dict[str, Any],
             m1: dict[str, Any], m2: dict[str, Any], m3: dict[str, Any],
             recursive: dict[str, Any], audit: dict[str, Any],
             protocol: dict[str, Any]) -> dict[str, Any]:
    eng = bool(engineering.get("n_runs", 0)) and all(
        bool(v) for k, v in engineering.items() if k != "n_runs")
    integ = all(bool(v) for v in integrity.values())
    ag = m1["active_arr"]
    m1_ok = bool(ag["accuracy"] >= protocol["diagnosis_gates"]["min_accuracy"]
                 and ag["ece"] <= protocol["diagnosis_gates"]["max_ece"]
                 and ag["information_gain"] >= protocol["diagnosis_gates"]["min_information_gain"]
                 and ag["accuracy"] > m1["passive_arr"]["accuracy"]
                 and ag["accuracy"] >= m1["rule_experiment"]["accuracy"])
    def _sig(x: dict[str, Any], alpha: float) -> bool:
        return bool(x.get("significant", x.get("p", 1.0) <= alpha))
    alpha = protocol["alpha"]
    m2_ok = bool(_sig(m2["arr_vs_random"], alpha)
                 and m2["arr_vs_random"]["mean"] >= protocol["research_selection_gates"]["mme"]
                 and m2["arr_vs_rule"]["mean"] > 0)
    m3_ok = bool(_sig(m3["transfer_vs_fresh"], alpha)
                 and m3["transfer_vs_fresh"]["mean"] >= protocol["transfer_gates"]["mme"]
                 and m3["negative_transfer_rate"] <= 0.2)
    rec_ok = bool(recursive.get("improvement_significant", False)
                  and not recursive.get("regression", True))
    audit_ok = all(bool(v) for v in audit.values())
    out = {
        "engineering": "PASS" if eng else "FAIL",
        "benchmark_integrity": "PASS" if integ else "BLOCKED",
        "m1_active_diagnosis": "PASS" if m1_ok else "FAIL",
        "m2_research_selection": "PASS" if m2_ok else "FAIL",
        "m3_knowledge_transfer": "PASS" if m3_ok else "FAIL",
        "recursive_improvement": "PASS" if rec_ok else "INCONCLUSIVE",
        "independent_audit": "PASS" if audit_ok else "FAIL",
    }
    out["overall_scientific_verdict"] = (
        "VALIDATED" if all(v == "PASS" for v in out.values()) else "NOT_VALIDATED")
    return out


__all__ = ["mean", "paired_stats", "holm", "wilson", "calibration", "verdicts"]