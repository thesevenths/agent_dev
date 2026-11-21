"""M7.2 Scientific Judgment Benchmark runner and metrics."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from research_runtime.harness_rsi.identity import freeze_json, sha256_json

ALLOWED_NOVELTY = {"LIKELY_NOVEL", "POSSIBLE_PRIOR_ART", "NOT_NOVEL",
                   "NOVELTY_UNCERTAIN"}
ALLOWED_FALSIFIABILITY = {"FALSIFIABLE", "UNFALSIFIABLE"}
ALLOWED_EVIDENCE = {"SUFFICIENT", "INSUFFICIENT", "CONTRADICTORY"}


def _parse_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.lstrip().startswith("json"):
            raw = raw.lstrip()[4:]
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object")
    obj = json.loads(raw[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("not an object")
    return obj


def load_cases(root: str | Path) -> list[dict[str, Any]]:
    root = Path(root)
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((root / "cases").glob("*.json"))]


def load_ground_truth(root: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(root)
    return {p.stem: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((root / "ground_truth").glob("*.json"))}


def select_pilot(cases: list[dict[str, Any]], per_category: int = 2
                 ) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for c in cases:
        buckets.setdefault(c["category"], []).append(c)
    selected = []
    for cat in sorted(buckets):
        selected.extend(buckets[cat][:per_category])
    return selected


def make_prompt(case: dict[str, Any]) -> str:
    # Ground truth MUST NOT be included.
    return json.dumps({
        "task": "scientific_judgment",
        "case_id": case["case_id"],
        "question": case["question"],
        "domain": case["domain"],
        "evidence_summary": case["evidence_summary"],
        "source_references": case["source_references"],
        "required_output": {
            "novelty": sorted(ALLOWED_NOVELTY),
            "falsifiability": sorted(ALLOWED_FALSIFIABILITY),
            "evidence_sufficiency": sorted(ALLOWED_EVIDENCE),
            "abstain": "boolean",
            "research_value_category": "HIGH | MEDIUM | LOW | NOT_ASSESSED",
            "rationale": "string with evidence references"
        }
    }, ensure_ascii=False, sort_keys=True)


def run_benchmark(*, cases: list[dict[str, Any]], agent: Any, out_dir: str | Path,
                  run_id: str) -> dict[str, Any]:
    root = Path(out_dir)
    (root / "raw_results").mkdir(parents=True, exist_ok=True)
    (root / "trajectories").mkdir(parents=True, exist_ok=True)
    predictions = []
    for case in cases:
        traj = agent.call(
            prompt=make_prompt(case),
            system="You are a strict scientific judge. Return only JSON. "
                   "If evidence is insufficient, abstain.",
            toolset=[], response_format_json=True, temperature=0.0,
            max_tokens=700)
        item = {"case_id": case["case_id"], "category": case["category"],
                "domain": case["domain"], "question": case["question"],
                "trajectory_sha256": sha256_json(traj.to_dict())}
        try:
            pred = _parse_object(traj.content)
            item["prediction"] = {
                "novelty": pred.get("novelty"),
                "falsifiability": pred.get("falsifiability"),
                "evidence_sufficiency": pred.get("evidence_sufficiency"),
                "abstain": bool(pred.get("abstain")),
                "research_value_category": pred.get("research_value_category"),
                "rationale": str(pred.get("rationale", ""))[:1000]}
            item["parse_ok"] = True
        except (ValueError, TypeError) as exc:
            item["prediction"] = None
            item["parse_ok"] = False
            item["parse_error"] = str(exc)
        freeze_json(item, root / "raw_results" / f"{case['case_id']}.json")
        freeze_json(traj.to_dict(),
                    root / "trajectories" / f"{case['case_id']}.json")
        predictions.append(item)
    frozen = freeze_json({"run_id": run_id, "n_cases": len(cases),
                          "predictions": predictions},
                         root / "raw_results.json")
    return {"ok": True, "raw_sha256": frozen["sha256"],
            "n_cases": len(cases)}


def _binary_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None
          and precision + recall else None)
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def wilson_interval(successes: int, n: int, z: float = 1.96
                    ) -> list[float | None]:
    if not n:
        return [None, None]
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, center - half), 4),
            round(min(1.0, center + half), 4)]


def compute_metrics(*, results_dir: str | Path, ground_truth: dict[str, dict],
                    benchmark_manifest: dict[str, Any]) -> dict[str, Any]:
    root = Path(results_dir)
    rows = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((root / "raw_results").glob("*.json"))]
    confusion: dict[str, dict[str, int]] = {}
    counts = {"ok": 0, "parse_failed": 0}
    abst_tp = abst_fp = abst_fn = abst_tn = 0
    by_category: dict[str, dict[str, Any]] = {}
    for row in rows:
        gt = ground_truth.get(row["case_id"])
        if not gt or not row.get("prediction"):
            counts["parse_failed"] += 1
            continue
        counts["ok"] += 1
        pred = row["prediction"]
        for key, allowed in (("novelty", ALLOWED_NOVELTY),
                             ("falsifiability", ALLOWED_FALSIFIABILITY),
                             ("evidence_sufficiency", ALLOWED_EVIDENCE)):
            truth = str(gt[key])
            guess = str(pred.get(key))
            if guess not in allowed:
                guess = "INVALID"
            matrix = confusion.setdefault(key, {})
            cell = f"{truth}->{guess}"
            matrix[cell] = matrix.get(cell, 0) + 1
        expected = bool(gt["abstention_expected"])
        actual = bool(pred.get("abstain"))
        if expected and actual:
            abst_tp += 1
        elif not expected and actual:
            abst_fp += 1
        elif expected and not actual:
            abst_fn += 1
        else:
            abst_tn += 1
        cat = by_category.setdefault(row["category"],
                                     {"n": 0, "correct_novelty": 0,
                                      "correct_falsifiability": 0,
                                      "correct_evidence": 0})
        cat["n"] += 1
        cat["correct_novelty"] += int(pred.get("novelty") == gt["novelty"])
        cat["correct_falsifiability"] += int(
            pred.get("falsifiability") == gt["falsifiability"])
        cat["correct_evidence"] += int(
            pred.get("evidence_sufficiency") == gt["evidence_sufficiency"])
    n = counts["ok"]

    def _accuracy(key: str) -> float | None:
        cm = confusion.get(key, {})
        hit = sum(v for k, v in cm.items()
                  if k.split("->")[0] == k.split("->")[1])
        return hit / n if n else None

    # Novelty PRECISION: of the cases predicted genuinely novel, how many the
    # independent ground truth also marks novel. Distinct from overall accuracy.
    nov_cm = confusion.get("novelty", {})
    novel_predicted = sum(v for k, v in nov_cm.items()
                          if k.split("->")[1] == "LIKELY_NOVEL")
    novel_correct = nov_cm.get("LIKELY_NOVEL->LIKELY_NOVEL", 0)
    novelty_precision = (novel_correct / novel_predicted
                         if novel_predicted else None)

    abst = _binary_metrics(abst_tp, abst_fp, abst_fn, abst_tn)
    metrics = {
        "version": "m7_2_metrics_v2_multiple_primary",
        "n_total": len(rows), "n_scored": n, "counts": counts,
        "confusion": confusion,
        "abstention": abst,
        "abstention_accuracy_ci95": wilson_interval(abst_tp + abst_tn, n),
        "by_category": by_category,
        "novelty_accuracy": _accuracy("novelty"),
        "novelty_precision": novelty_precision,
        "falsifiability_accuracy": _accuracy("falsifiability"),
        "evidence_sufficiency_accuracy": _accuracy("evidence_sufficiency"),
        "expert_blind_review": benchmark_manifest.get(
            "expert_blind_review", "NOT_PERFORMED"),
        # Pre-registered BEFORE evaluation. Novelty is secondary only and can
        # NEVER carry a PASS by itself (M7.6 P1-B fix).
        "preregistered_outcomes": {
            "primary": ["evidence_sufficiency_accuracy",
                        "falsifiability_accuracy"],
            "co_primary": ["novelty_precision", "abstention_f1"],
            "secondary": ["novelty_accuracy"],
            "min_cases": int(benchmark_manifest.get("min_cases", 30)),
            "thresholds": {
                "evidence_sufficiency_accuracy": 0.75,
                "falsifiability_accuracy": 0.75,
                "novelty_precision": 0.70,
                "abstention_f1": 0.60,
            },
        },
    }
    po = metrics["preregistered_outcomes"]
    observed = {
        "evidence_sufficiency_accuracy": metrics["evidence_sufficiency_accuracy"],
        "falsifiability_accuracy": metrics["falsifiability_accuracy"],
        "novelty_precision": novelty_precision,
        "abstention_f1": abst.get("f1"),
    }
    # Insufficient labels / unavailable expert review => never a capability PASS.
    if (n < po["min_cases"]
            or metrics["expert_blind_review"] != "PERFORMED"
            or any(v is None for v in observed.values())):
        metrics["verdict"] = "INCONCLUSIVE"
        metrics["verdict_reason"] = "insufficient_or_unverified_evidence"
    else:
        failed = [k for k, v in observed.items() if v < po["thresholds"][k]]
        metrics["verdict"] = "PASS" if not failed else "FAIL"
        metrics["verdict_reason"] = ("all_preregistered_outcomes_met"
                                     if not failed else "below:" + ",".join(failed))
    return metrics
