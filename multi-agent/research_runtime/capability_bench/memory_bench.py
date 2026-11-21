"""M7.3 Memory benchmark runner: OFF / ON / SHUFFLED / EQUAL_TOKEN."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research_runtime.harness_rsi.identity import freeze_json, sha256_json

ARMS = ("OFF", "ON", "SHUFFLED", "EQUAL_TOKEN")


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


def _prompt(question: str, context: str) -> str:
    return json.dumps({
        "task": "research_decision_with_memory",
        "question": question,
        "context": context,
        "required_output": {
            "repeat_failure": "boolean: would this design repeat a known failure",
            "experiment_decision": "string",
            "used_memory_ids": "list",
            "rejected_memory_ids": "list",
            "reason": "string"
        }
    }, ensure_ascii=False, sort_keys=True)


def run_benchmark(*, root: str | Path, agent: Any, out_dir: str | Path,
                  run_id: str) -> dict[str, Any]:
    root = Path(root)
    out = Path(out_dir)
    (out / "raw_results").mkdir(parents=True, exist_ok=True)
    (out / "trajectories").mkdir(parents=True, exist_ok=True)
    scenarios = sorted((root / "scenarios").glob("*.json"))
    rows = []
    for sp in scenarios:
        scenario = json.loads(sp.read_text(encoding="utf-8"))
        snapshot = json.loads((root / "memory_snapshots" / sp.name).read_text(
            encoding="utf-8"))
        for arm in ARMS:
            context = snapshot.get(f"arm_{arm.lower()}", "")
            traj = agent.call(
                prompt=_prompt(scenario["question"], context),
                system="You are a research methodologist. Return only JSON. "
                       "Use contextual experience only when its conditions apply.",
                toolset=[], response_format_json=True, temperature=0.0,
                max_tokens=500)
            row = {"scenario_id": scenario["scenario_id"], "arm": arm,
                   "context_chars": len(context),
                   "context_sha256": sha256_json(context),
                   "trajectory_sha256": sha256_json(traj.to_dict())}
            try:
                pred = _parse_object(traj.content)
                row["prediction"] = {
                    "repeat_failure": bool(pred.get("repeat_failure")),
                    "experiment_decision": str(pred.get("experiment_decision", ""))[:500],
                    "used_memory_ids": list(pred.get("used_memory_ids") or []),
                    "rejected_memory_ids": list(pred.get("rejected_memory_ids") or []),
                    "reason": str(pred.get("reason", ""))[:500]}
                row["parse_ok"] = True
            except (ValueError, TypeError) as exc:
                row["prediction"] = None
                row["parse_ok"] = False
                row["parse_error"] = str(exc)
            fname = f"{scenario['scenario_id']}_{arm}"
            freeze_json(row, out / "raw_results" / f"{fname}.json")
            freeze_json(traj.to_dict(), out / "trajectories" / f"{fname}.json")
            rows.append(row)
    frozen = freeze_json({"run_id": run_id, "n_results": len(rows),
                          "arms": list(ARMS), "results": rows},
                         out / "raw_results.json")
    return {"ok": True, "n_results": len(rows),
            "raw_sha256": frozen["sha256"]}


def compute_metrics(*, root: str | Path, results_dir: str | Path,
                    manifest: dict[str, Any]) -> dict[str, Any]:
    root = Path(root)
    out = Path(results_dir)
    gt = {p.stem: json.loads(p.read_text(encoding="utf-8"))
          for p in sorted((root / "ground_truth").glob("*.json"))}
    rows = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((out / "raw_results").glob("*.json"))]
    def _truth(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        v = str(value or "").strip().upper()
        if v in ("YES", "TRUE", "1"):
            return True
        if v in ("NO", "FALSE", "0"):
            return False
        return None

    per_arm: dict[str, Any] = {}
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm and r.get("prediction")]
        predicted_repeats = 0
        tp = fp = fn = tn = 0
        used_relevant = 0
        harmful = 0
        confounders_avoided = 0
        confounder_total = 0
        for r in arm_rows:
            truth = gt.get(r["scenario_id"], {})
            truth_repeat = _truth(truth.get("expected_repeat_failure"))
            pred_repeat = bool(r["prediction"]["repeat_failure"])
            predicted_repeats += int(pred_repeat)
            # Ground-truth-based confusion (NOT the model's own prediction).
            if truth_repeat is not None:
                if pred_repeat and truth_repeat:
                    tp += 1
                elif pred_repeat and not truth_repeat:
                    fp += 1
                elif not pred_repeat and truth_repeat:
                    fn += 1
                else:
                    tn += 1
            used = set(r["prediction"]["used_memory_ids"])
            if arm in ("ON", "SHUFFLED"):
                relevant = set(truth.get("relevant_memory_ids", []))
                shuffled = set(truth.get("shuffled_memory_ids", []))
                used_relevant += int(bool(used & relevant))
                harmful += int(bool(used & shuffled))
                # confounder avoidance: a scenario with a known confounder was
                # NOT repeated and no irrelevant/shuffled memory was applied.
                known_confounders = truth.get("relevant_memory_ids", [])
                if known_confounders:
                    confounder_total += 1
                    confounders_avoided += int(
                        (not pred_repeat) and not bool(used & shuffled))
        n_arm = len(arm_rows)
        repeat_correct = tp + tn
        per_arm[arm] = {
            "n": n_arm,
            # Descriptive ONLY -- this is the model's PREDICTION rate, not proof
            # that a historical failure was actually repeated.
            "predicted_repeat_failure_rate": round(predicted_repeats / n_arm, 4)
                                             if n_arm else None,
            # Ground-truth-based metrics (M7.6 P1-C fix).
            "repeat_failure_accuracy": round(repeat_correct / n_arm, 4)
                                       if n_arm else None,
            "repeat_failure_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "confounder_avoidance_rate": round(
                confounders_avoided / confounder_total, 4)
                                         if confounder_total else None,
            "memory_usage_rate": round(used_relevant / n_arm, 4)
                                 if n_arm and arm in ("ON", "SHUFFLED") else None,
            "harmful_memory_rate": round(harmful / n_arm, 4)
                                   if n_arm and arm in ("ON", "SHUFFLED") else None,
            "mean_context_chars": round(sum(r["context_chars"]
                                            for r in arm_rows) / n_arm, 2)
                                   if n_arm else None}
    result = {"version": "m7_3_metrics_v2_grounded", "n_results": len(rows),
              "per_arm": per_arm, "expert_review": manifest.get(
                  "expert_review", "NOT_PERFORMED")}
    n_min = int(manifest.get("min_results", 40))
    seen = {r["scenario_id"] for r in rows}
    if len(rows) < n_min or result["expert_review"] != "PERFORMED":
        result["verdict"] = "INCONCLUSIVE"
        result["verdict_reason"] = "insufficient_or_unreviewed_evidence"
    else:
        on = per_arm["ON"]
        off = per_arm["OFF"]
        shuf = per_arm["SHUFFLED"]
        # Memory is useful only if GROUNDED accuracy improves AND it beats the
        # shuffled control; harmful usage must not exceed the shuffled arm.
        improved = ((on["repeat_failure_accuracy"] or 0)
                    > (off["repeat_failure_accuracy"] or 0))
        beats_shuffle = ((on["repeat_failure_accuracy"] or 0)
                         >= (shuf["repeat_failure_accuracy"] or 0))
        low_harm = (on["harmful_memory_rate"] or 0) <= (shuf["harmful_memory_rate"] or 1)
        result["verdict"] = ("PASS" if improved and beats_shuffle and low_harm
                             else "FAIL")
        result["verdict_reason"] = {
            "grounded_accuracy_improved": improved,
            "beats_shuffled_control": beats_shuffle,
            "harm_not_increased": low_harm}
    return result
