"""ARR_EMB_RSI_01 -- independent evaluation and five-layer verdicts."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from ..harness_rsi.evaluation import PreRegistration, analyse_paired
from ..evolution.statistics import paired_t_statistic


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def diagnosis_metrics(*, diagnoses: list[dict[str, Any]],
                      truth: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_controller: dict[str, dict[str, Any]] = {}
    for controller in sorted({d["controller"] for d in diagnoses}):
        rows = [d for d in diagnoses if d["controller"] == controller]
        acc = top2 = key_acc = 0
        precision = recall = conf = abstain = 0
        for d in rows:
            t = truth.get(d["case_id"]) or {}
            gt = str(t.get("ground_truth") or "")
            acc += int(d.get("root_cause") == gt)
            top2 += int(gt in (d.get("top2") or []))
            iv = d.get("proposed_intervention") or {}
            overrides = iv.get("overrides") or {}
            key_acc += int(bool(overrides) and overrides == (
                t.get("correct_intervention") or {}))
            cited = set(d.get("evidence") or [])
            valid = set(t.get("valid_evidence") or [])
            if valid:
                precision += len(cited & valid) / max(1, len(cited))
                recall += len(cited & valid) / max(1, len(valid))
            conf += float(d.get("confidence") or 0.0)
            abstain += int(bool(d.get("abstain")))
        n = max(1, len(rows))
        by_controller[controller] = {
            "n": len(rows), "root_cause_accuracy": acc / n,
            "top2_accuracy": top2 / n, "intervention_key_accuracy": key_acc / n,
            "evidence_precision": precision / n, "evidence_recall": recall / n,
            "mean_confidence": conf / n, "abstention_rate": abstain / n,
        }
    # ECE-style calibration in 5 bins.
    for controller, vals in by_controller.items():
        rows = [d for d in diagnoses if d["controller"] == controller]
        bins = [0] * 5
        correct = [0] * 5
        for d in rows:
            c = float(d.get("confidence") or 0.0)
            idx = min(4, max(0, int(c * 5)))
            bins[idx] += 1
            t = truth.get(d["case_id"]) or {}
            correct[idx] += int(d.get("root_cause") == t.get("ground_truth"))
        ece = sum(bins[i] / max(1, len(rows)) * abs(
            correct[i] / max(1, bins[i]) - (i + 0.5) / 5) for i in range(5))
        vals["confidence_ece"] = round(ece, 4)
    return {"by_controller": by_controller, "n_diagnoses": len(diagnoses)}


def causal_stats(*, comparisons: dict[str, list[dict[str, Any]]],
                 metric: str) -> dict[str, Any]:
    out = {}
    for name, rows in comparisons.items():
        diffs = [float(r.get(metric) or 0.0) for r in rows]
        n = len(diffs)
        mean = sum(diffs) / n if n else 0.0
        sd = (math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
              if n > 1 else 0.0)
        out[name] = {"n": n, "mean": mean, "sd": sd,
                     "ci95": [mean - 1.96 * sd / math.sqrt(n),
                              mean + 1.96 * sd / math.sqrt(n)] if n else [0, 0],
                     "t_stat": paired_t_statistic(mean, sd, n)}
    return out


def five_verdicts(*, engineering: dict[str, Any], integrity: dict[str, Any],
                  diagnosis: dict[str, Any], intervention: dict[str, Any],
                  causal: dict[str, Any],
                  prereg: dict[str, Any]) -> dict[str, str]:
    def _gate(name: str, ok: bool) -> str:
        return "PASS" if ok else "FAIL"
    eng_ok = bool(engineering.get("n_cases_executed", 0)) and all(
        bool(v) for k, v in engineering.items() if k != "n_cases_executed")
    integ_ok = all(bool(v) for v in integrity.values())
    d = diagnosis.get("by_controller", {}).get("evidence", {})
    diag_ok = (d.get("root_cause_accuracy", 0) >= prereg.get("min_diag_accuracy", 0.75)
               and d.get("intervention_key_accuracy", 0) >= prereg.get(
                   "min_key_accuracy", 0.75)
               and d.get("evidence_precision", 0) >= prereg.get(
                   "min_evidence_precision", 0.8)
               and d.get("confidence_ece", 1) <= prereg.get(
                   "max_confidence_ece", 0.15))
    intv_ok = bool(intervention.get("applied_rate") == 1.0 and
                   intervention.get("oracle_effect") is True)
    n = int(causal.get("n_units", 0))
    min_units = int(prereg.get("min_independent_units_for_causal", 6))
    ci_ok = all(causal.get("ci", {}).get(k, [1, 1])[1] < 0 for k in (
        "candidate_vs_random", "candidate_vs_shuffled"))
    causal_ok = bool(n >= min_units and ci_ok and
                     causal.get("candidate_beats_random") and
                     causal.get("candidate_beats_shuffled") and
                     causal.get("meets_mme"))
    causal_reject = bool(causal.get("oracle_effect") is False or
                         causal.get("candidate_beats_base") is False)
    return {
        "engineering": _gate("engineering", eng_ok),
        "benchmark_integrity": _gate("integrity", integ_ok),
        "diagnosis": _gate("diagnosis", diag_ok),
        "intervention": _gate("intervention", intv_ok),
        "causal_improvement": (
            "FAIL" if causal_reject else
            ("PASS" if causal_ok else "INCONCLUSIVE")),
    }


__all__ = ["diagnosis_metrics", "causal_stats", "five_verdicts"]
