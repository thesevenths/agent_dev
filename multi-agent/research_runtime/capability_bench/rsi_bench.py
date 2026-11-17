"""M7.4 Recursive Improvement pilot harness.

Each round changes ONE method component. The paired comparison uses the same
cases, model, budget and score function. Holdout is selected before results.
No Candidate is adopted without holdout evidence and human approval.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research_runtime.harness_rsi.identity import freeze_json, sha256_json

METHODS = {
    "base": {
        "component": "scientific_judgment_instruction",
        "system": "You answer scientific judgment questions. Return only JSON.",
    },
    "candidate_evidence_gate": {
        "component": "scientific_judgment_instruction",
        "system": "You are a strict scientific judge. Return only JSON. "
                  "Score sufficiency from controls, replication and holdout. "
                  "If evidence is insufficient, abstain.",
    },
    "candidate_adversarial": {
        "component": "scientific_judgment_instruction",
        "system": "You are a strict scientific judge. Return only JSON. "
                  "Actively look for the strongest counterexample and alternative "
                  "explanation; abstain when evidence is insufficient.",
    },
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


def split_tasks(cases: list[dict[str, Any]]) -> dict[str, list[dict]]:
    dev_cats = {"KNOWN", "PRIOR_ART", "UNFALSIFIABLE"}
    hold_cats = {"OPEN", "COMPETING"}
    ood_cats = {"CONTRADICTORY", "INSUFFICIENT"}
    out = {"development": [], "holdout": [], "ood": []}
    for c in cases:
        if c["category"] in dev_cats and len(out["development"]) < 6:
            out["development"].append(c)
        elif c["category"] in hold_cats and len(out["holdout"]) < 4:
            out["holdout"].append(c)
        elif c["category"] in ood_cats and len(out["ood"]) < 4:
            out["ood"].append(c)
    return out


def _case_prompt(case: dict[str, Any]) -> str:
    return json.dumps({"case_id": case["case_id"], "question": case["question"],
                       "evidence_summary": case["evidence_summary"],
                       "source_references": case["source_references"],
                       "required_output": {
                           "novelty": "LIKELY_NOVEL|POSSIBLE_PRIOR_ART|NOT_NOVEL|NOVELTY_UNCERTAIN",
                           "falsifiability": "FALSIFIABLE|UNFALSIFIABLE",
                           "evidence_sufficiency": "SUFFICIENT|INSUFFICIENT|CONTRADICTORY",
                           "abstain": "boolean",
                           "reason": "string"}}, ensure_ascii=False,
                      sort_keys=True)


def _score(pred: dict[str, Any] | None, gt: dict[str, Any]) -> dict[str, int]:
    if not pred:
        return {"primary": 0, "novelty": 0, "falsifiability": 0, "evidence": 0}
    return {
        "primary": int(pred.get("evidence_sufficiency")
                       == gt["evidence_sufficiency"]),
        "novelty": int(pred.get("novelty") == gt["novelty"]),
        "falsifiability": int(pred.get("falsifiability")
                              == gt["falsifiability"]),
        "evidence": int(pred.get("evidence_sufficiency")
                        == gt["evidence_sufficiency"]),
    }


def run_round(*, round_no: int, candidate_method: str, tasks: list[dict],
              ground_truth: dict[str, dict], agent: Any, out_dir: Path
              ) -> dict[str, Any]:
    rows = []
    for task in tasks:
        for method in ("base", candidate_method):
            traj = agent.call(
                prompt=_case_prompt(task),
                system=METHODS[method]["system"], toolset=[],
                response_format_json=True, temperature=0.0, max_tokens=500)
            try:
                pred = _parse(traj.content)
                parse_ok = True
            except (ValueError, TypeError):
                pred, parse_ok = None, False
            scores = _score(pred, ground_truth[task["case_id"]])
            rows.append({"case_id": task["case_id"], "category": task["category"],
                         "method": method, "parse_ok": parse_ok,
                         "prediction": pred, "scores": scores,
                         "trajectory_sha256": sha256_json(traj.to_dict())})
            freeze_json(traj.to_dict(), out_dir / f"r{round_no}_{method}_{task['case_id']}_traj.json")
    base = [r for r in rows if r["method"] == "base"]
    cand = [r for r in rows if r["method"] == candidate_method]
    paired = [b["scores"]["primary"] - c["scores"]["primary"]
              for b, c in zip(base, cand)]
    # Positive paired delta means base scored higher; candidate delta = -paired.
    result = {
        "round": round_no,
        "candidate_method": candidate_method,
        "changed_component": METHODS[candidate_method]["component"],
        "n_pairs": len(paired),
        "base_primary_mean": sum(r["scores"]["primary"] for r in base) / len(base),
        "candidate_primary_mean": sum(r["scores"]["primary"] for r in cand) / len(cand),
        "candidate_minus_base": -sum(paired) / len(paired),
        "family_wins": {"base": sum(1 for d in paired if d > 0),
                        "candidate": sum(1 for d in paired if d < 0),
                        "tie": sum(1 for d in paired if d == 0)},
        "rows": rows}
    freeze_json(result, out_dir / f"round_{round_no}.json")
    return result


def summarise(*, rounds: list[dict[str, Any]], mme: float,
              min_families: int) -> dict[str, Any]:
    all_rounds = []
    for r in rounds:
        all_rounds.append({
            "round": r["round"], "candidate": r["candidate_method"],
            "delta": r["candidate_minus_base"], "n_pairs": r["n_pairs"],
            "component": r["changed_component"]})
    best = max(all_rounds, key=lambda r: r["delta"]) if all_rounds else None
    evidence_sufficient = bool(best and best["delta"] >= mme
                               and best["n_pairs"] >= min_families)
    return {"version": "m7_4_rsi_pilot_v1", "rounds": all_rounds,
            "mme": mme, "min_families": min_families,
            "best_candidate": best,
            "promotion_decision": ("INCONCLUSIVE" if not evidence_sufficient
                                   else "HOLDOUT_REQUIRED"),
            "human_approval": "NOT_PERFORMED",
            "adoption": "NOT_ADOPTED",
            "post_adoption": "NOT_RUN",
            "rollback": "NOT_APPLICABLE_NO_ADOPTION"}
