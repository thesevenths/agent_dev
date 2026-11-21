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


def select_tasks(cases: list[dict[str, Any]], n: int = 8, *,
                 seed: int = 20261117) -> list[dict[str, Any]]:
    """Deterministically select EXACTLY n distinct cases, or fail explicitly.

    M7.6 P1-D: the old quota rule could return fewer than n when the category
    count did not divide n (e.g. 8 from 7 categories -> 7). This version keeps
    category balance first, then tops up deterministically from the remaining
    eligible cases, never duplicating a case and preserving the recorded seed.
    """
    import random
    buckets: dict[str, list[dict]] = {}
    for c in cases:
        buckets.setdefault(str(c.get("category", "unknown")), []).append(c)
    # Stable order within each bucket by case_id for reproducibility.
    for cat in buckets:
        buckets[cat] = sorted(buckets[cat], key=lambda c: str(c.get("case_id", "")))
    n = min(int(n), len(cases))
    if n < 0:
        return []
    out: list[dict] = []
    seen: set[str] = set()

    def _add(case):
        cid = str(case.get("case_id", id(case)))
        if cid not in seen:
            seen.add(cid)
            out.append(case)

    cats = sorted(buckets)
    # Round-robin across categories for balance until n is reached.
    idx = {cat: 0 for cat in cats}
    while len(out) < n and cats:
        progressed = False
        for cat in cats:
            if len(out) >= n:
                break
            if idx[cat] < len(buckets[cat]):
                _add(buckets[cat][idx[cat]])
                idx[cat] += 1
                progressed = True
        if not progressed:
            break
    # Deterministic top-up from any remainder (should be unnecessary when
    # len(cases) >= n, but keeps the EXACT-n guarantee explicit).
    if len(out) < n:
        rng = random.Random(seed)
        rest = [c for c in cases if str(c.get("case_id", id(c))) not in seen]
        rng.shuffle(rest)
        for c in rest:
            if len(out) >= n:
                break
            _add(c)
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
                   # M7.6 P1-D: family/domain/split/contamination are FIRST-CLASS
                   # identity fields. Statistical independence is measured at
                   # family level, never at raw-case level.
                   "family_id": str(case.get("family_id")
                                    or case.get("category")
                                    or "unknown"),
                   "domain_id": str(case.get("domain", "unknown")),
                   "split": str(case.get("split", "development")),
                   "contamination_group": str(
                       case.get("contamination_group", "unknown")),
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


def metrics(*, results_dir: str | Path, mme: float, min_families: int,
            seed: int = 20261117) -> dict[str, Any]:
    """Family-level paired analysis with cluster-aware CIs (M7.6 P1-D).

    Cases from one research family are CORRELATED and are never treated as
    independent samples. Each candidate arm is compared to A_BASE on the same
    case_id; per-case paired differences are aggregated within family, and the
    CI is computed by resampling WHOLE families.
    """
    from research_runtime.harness_rsi.evaluation import _clustered_bootstrap

    rows = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((Path(results_dir) / "raw_results").glob("*.json"))]
    by_key = {(r["case_id"], r["group"]): r for r in rows if r.get("parse_ok")}
    base_rows = {r["case_id"]: r for r in rows
                 if r["group"] == "A_BASE" and r.get("parse_ok")}
    unique_families = sorted({r.get("family_id", r.get("category", "unknown"))
                              for r in rows})

    per_group: dict[str, Any] = {}
    for group in GROUPS:
        gr = [r for r in rows if r["group"] == group and r.get("parse_ok")]
        fams = sorted({r.get("family_id", r.get("category", "unknown"))
                       for r in gr})
        per_group[group] = {
            "n_cases": len(gr),
            "n_families": len(fams),
            "families": fams,
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

    # Paired per-case deltas vs A_BASE; robust to missing pairs.
    paired: dict[str, Any] = {}
    for group in GROUPS:
        if group == "A_BASE":
            continue
        diffs: list[float] = []
        fams: list[str] = []
        per_family: dict[str, list[float]] = {}
        for cid, base_r in base_rows.items():
            cand = by_key.get((cid, group))
            if not cand:
                continue
            d = float(cand["scores"]["evidence"]) - float(base_r["scores"]["evidence"])
            fam = str(base_r.get("family_id", base_r.get("category", "unknown")))
            diffs.append(d)
            fams.append(fam)
            per_family.setdefault(fam, []).append(d)
        family_means = {f: sum(v) / len(v) for f, v in per_family.items()}
        mean_delta = sum(diffs) / len(diffs) if diffs else None
        ci = _clustered_bootstrap(diffs, fams, seed=seed) if diffs else [None, None]
        paired[group] = {
            "n_pairs": len(diffs), "n_families": len(family_means),
            "per_family_mean_delta": family_means,
            "mean_paired_delta": mean_delta,
            "cluster_bootstrap_ci95": ci,
            "ci_lower": ci[0], "ci_upper": ci[1],
            "met_mme": (mean_delta is not None and ci[0] is not None
                        and ci[0] >= mme),
        }

    n_families_ok = len(unique_families) >= min_families
    all_met = n_families_ok and all(p["met_mme"] for p in paired.values()) \
        and bool(paired)
    result = {
        "version": "m7_5_metrics_v2_family_clustered", "n_results": len(rows),
        "n_unique_families": len(unique_families),
        "families": unique_families,
        "per_group": per_group, "paired_vs_base": paired,
        "mme": mme, "min_families": min_families,
        "seed": seed,
        "n_families_ok": n_families_ok,
    }
    if not paired or any(p["mean_paired_delta"] is None for p in paired.values()):
        result["verdict"] = "INCONCLUSIVE"
        result["verdict_reason"] = "missing_paired_data"
    elif all_met:
        result["verdict"] = "PASS"
        result["verdict_reason"] = "all_arms_ci_lower_above_mme"
    elif n_families_ok:
        result["verdict"] = "FAIL"
        result["verdict_reason"] = "some_arms_below_mme"
    else:
        result["verdict"] = "INCONCLUSIVE"
        result["verdict_reason"] = "too_few_independent_families"
    return result
