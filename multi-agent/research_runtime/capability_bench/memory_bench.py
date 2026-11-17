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
    per_arm: dict[str, Any] = {}
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm and r.get("prediction")]
        repeated = sum(1 for r in arm_rows
                       if r["prediction"]["repeat_failure"])
        used_relevant = 0
        harmful = 0
        for r in arm_rows:
            truth = gt.get(r["scenario_id"], {})
            used = set(r["prediction"]["used_memory_ids"])
            if arm in ("ON", "SHUFFLED"):
                relevant = set(truth.get("relevant_memory_ids", []))
                shuffled = set(truth.get("shuffled_memory_ids", []))
                used_relevant += int(bool(used & relevant))
                harmful += int(bool(used & shuffled))
        per_arm[arm] = {
            "n": len(arm_rows),
            "repeated_failure_rate": round(repeated / len(arm_rows), 4)
                                     if arm_rows else None,
            "memory_usage_rate": round(used_relevant / len(arm_rows), 4)
                                 if arm_rows and arm in ("ON", "SHUFFLED") else None,
            "harmful_memory_rate": round(harmful / len(arm_rows), 4)
                                   if arm_rows and arm in ("ON", "SHUFFLED") else None,
            "mean_context_chars": round(sum(r["context_chars"]
                                            for r in arm_rows) / len(arm_rows), 2)
                                   if arm_rows else None}
    result = {"version": "m7_3_metrics_v1", "n_results": len(rows),
              "per_arm": per_arm, "expert_review": manifest.get(
                  "expert_review", "NOT_PERFORMED")}
    result["verdict"] = ("INCONCLUSIVE" if len(rows) < 40 or
                         result["expert_review"] != "PERFORMED" else
                         "PASS" if per_arm["ON"]["repeated_failure_rate"] <
                         per_arm["OFF"]["repeated_failure_rate"] else "FAIL")
    return result
