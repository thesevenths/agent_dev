"""M7.5 integrated capability benchmark: A/B/C/D groups on the same tasks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research_runtime.harness_rsi.identity import freeze_json, sha256_json

GROUPS = {
    "A_BASE": {"memory": False, "rsi": False},
    "B_MEMORY": {"memory": True, "rsi": False},
    "C_RSI": {"memory": False, "rsi": True},
    "D_FULL": {"memory": True, "rsi": True},
}


def _parse(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object")
    obj = json.loads(raw[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("not an object")
    return obj


def select_tasks(cases: list[dict[str, Any]], n: int = 8
                 ) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict]] = {}
    for c in cases:
        buckets.setdefault(c["category"], []).append(c)
    out = []
    for cat in sorted(buckets):
        out.extend(buckets[cat][:max(1, n // len(buckets))])
    return out[:n]


def _prompt(case: dict, memory_context: str, rsi: bool) -> tuple[str, str]:
    system = ("You answer scientific judgment questions. Return only JSON."
              if not rsi else
              "You are a strict scientific judge. Return only JSON. Actively "
              "seek counterexamples and alternative explanations; abstain when "
              "evidence is insufficient.")
    payload = {"case_id": case["case_id"], "question": case["question"],
               "evidence_summary": case["evidence_summary"],
               "memory_context": memory_context,
               "required_output": {
                   "novelty": "LIKELY_NOVEL|POSSIBLE_PRIOR_ART|NOT_NOVEL|NOVELTY_UNCERTAIN",
                   "falsifiability": "FALSIFIABLE|UNFALSIFIABLE",
                   "evidence_sufficiency": "SUFFICIENT|INSUFFICIENT|CONTRADICTORY",
                   "abstain": "boolean",
                   "used_memory_ids": "list",
                   "reason": "string"}}
    return system, json.dumps(payload, ensure_ascii=False, sort_keys=True)


def run(*, cases: list[dict], ground_truth: dict[str, dict],
        memories: list[dict], agent: Any, out_dir: str | Path,
        run_id: str) -> dict[str, Any]:
    out = Path(out_dir)
    (out / "raw_results").mkdir(parents=True, exist_ok=True)
    (out / "trajectories").mkdir(parents=True, exist_ok=True)
    rows = []
    memory_by_domain = {}
    for m in memories:
        for domain in m.get("applicability", []):
            memory_by_domain.setdefault(domain, []).append(m)
    for case in cases:
        relevant = (memory_by_domain.get(case["domain"], [])
                    + memory_by_domain.get("scientific_judgment", []))[:2]
        memory_context = json.dumps(relevant, sort_keys=True) if relevant else ""
        for group, cfg in GROUPS.items():
            ctx = memory_context if cfg["memory"] else ""
            system, prompt = _prompt(case, ctx, cfg["rsi"])
            traj = agent.call(prompt=prompt, system=system, toolset=[],
                              response_format_json=True, temperature=0.0,
                              max_tokens=600)
            try:
                pred = _parse(traj.content)
                ok = True
            except (ValueError, TypeError):
                pred, ok = None, False
            gt = ground_truth[case["case_id"]]
            scores = {"evidence": int(bool(pred) and pred.get(
                "evidence_sufficiency") == gt["evidence_sufficiency"]),
                "novelty": int(bool(pred) and pred.get("novelty")
                               == gt["novelty"]),
                "falsifiability": int(bool(pred) and pred.get(
                    "falsifiability") == gt["falsifiability"])}
            row = {"case_id": case["case_id"], "category": case["category"],
                   "group": group, "parse_ok": ok, "prediction": pred,
                   "scores": scores,
                   "prompt_tokens": traj.prompt_tokens,
                   "completion_tokens": traj.completion_tokens,
                   "context_chars": len(ctx),
                   "trajectory_sha256": sha256_json(traj.to_dict())}
            rows.append(row)
            name = f"{case['case_id']}_{group}"
            freeze_json(row, out / "raw_results" / f"{name}.json")
            freeze_json(traj.to_dict(), out / "trajectories" / f"{name}.json")
    frozen = freeze_json({"run_id": run_id, "n_results": len(rows),
                          "groups": list(GROUPS), "results": rows},
                         out / "raw_results.json")
    return {"n_results": len(rows), "raw_sha256": frozen["sha256"]}


def metrics(*, results_dir: str | Path, mme: float, min_families: int
            ) -> dict[str, Any]:
    rows = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((Path(results_dir) / "raw_results").glob("*.json"))]
    per_group: dict[str, Any] = {}
    for group in GROUPS:
        gr = [r for r in rows if r["group"] == group and r["parse_ok"]]
        per_group[group] = {
            "n": len(gr),
            "evidence_accuracy": sum(r["scores"]["evidence"] for r in gr)
                                 / len(gr) if gr else None,
            "novelty_accuracy": sum(r["scores"]["novelty"] for r in gr)
                                / len(gr) if gr else None,
            "falsifiability_accuracy": sum(
                r["scores"]["falsifiability"] for r in gr) / len(gr) if gr else None,
            "total_tokens": sum(r["prompt_tokens"] + r["completion_tokens"]
                                for r in gr),
            "mean_context_chars": sum(r["context_chars"] for r in gr) / len(gr)
                                  if gr else None}
    base = per_group["A_BASE"]["evidence_accuracy"] or 0.0
    deltas = {g: (v["evidence_accuracy"] or 0.0) - base
              for g, v in per_group.items() if g != "A_BASE"}
    sufficient = all(d >= mme for d in deltas.values()) \
        and per_group["A_BASE"]["n"] >= min_families
    return {"version": "m7_5_metrics_v1", "n_results": len(rows),
            "per_group": per_group, "deltas_vs_base": deltas,
            "mme": mme, "min_families": min_families,
            "verdict": "INCONCLUSIVE" if not sufficient else "PASS"}
