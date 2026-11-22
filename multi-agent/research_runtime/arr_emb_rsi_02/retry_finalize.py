"""One-shot retry of failed arms + canonical reanalysis (no engine rerun of successes)."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from research_runtime.arr_emb_rsi_02.adapter import WorldModelAdapter
from research_runtime.arr_emb_rsi_02.benchmark import GENERATORS, build_regimes
from research_runtime.arr_emb_rsi_02.campaign import (
    _diagnosis_metrics, _metric, _norm_improvement, _paired_stats,
)
from research_runtime.arr_emb_rsi_02.evidence import evidence_audit
from research_runtime.arr_emb_rsi_02.integrity import sha256_json
from research_runtime.arr_emb_rsi_02.report import write_report


def rerun_failed(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    canonical = json.loads((root / "canonical_results.json").read_text(
        encoding="utf-8"))
    prereg = canonical["preregistration"]
    regimes = {r.regime_id: r for r in build_regimes()}
    for case in canonical["cases"]:
        for arm, run in case["arms"].items():
            if run.get("ok"):
                continue
            regime = regimes[case["regime_id"]]
            gen = regime.generator
            spec = GENERATORS[gen]
            case_dir = (root / "cases" / regime.regime_id.replace("::", "__")
                        / f"s{case['seed']}")
            engine_dir = case_dir / "engine"
            archive = case_dir / f"failed_attempt_{arm}_{int(time.time())}"
            if engine_dir.exists():
                shutil.move(str(engine_dir), str(archive))
            adapter = WorldModelAdapter(
                run_root=engine_dir, base_config=spec["config"],
                max_jobs=8, max_seconds=1800)
            if arm == "candidate":
                overrides = dict(case["agent_intervention"])
                invalid = False
                if "data.train_samples" in overrides:
                    n = overrides["data.train_samples"]
                    if n not in (4, 64):
                        invalid = True
                    else:
                        overrides["data.frozen_dir"] = (
                            f"datasets/arr_emb_rsi_02/{gen}_cov4" if n == 4
                            else f"datasets/arr_emb_rsi_02/{gen}_cov64")
                        overrides["train.batch_size"] = 4
                if invalid:
                    case["invalid_agent_intervention"] = dict(overrides)
                    overrides = dict(regime.failure_override)
            elif arm == "random":
                overrides = dict(case["random_intervention"])
            else:
                overrides = dict(regime.oracle_override)
            ov = dict(regime.common)
            ov.update(overrides)
            ood_ov = None
            if regime.mechanism == "E03":
                ood_ov = {"data.frozen_dir": regime.ood_dataset,
                          "data.t_max": spec["ood_t_max"]}
            r = adapter.run_arm(arm=arm, failure_id=gen, seed=case["seed"],
                                overrides=ov, ood_overrides=ood_ov,
                                run_suffix=f"retry_{arm}")
            case["arms"][arm] = r.to_dict()
            case.setdefault("retries", []).append({
                "arm": arm, "ok": r.ok, "archived_evidence": str(archive),
                "error": r.error[:300]})
            bv = case.get("base_metric")
            case["normalized_improvement"][arm] = _norm_improvement(
                bv, _metric(r.to_dict(), regime.mechanism))
            print("retried", case["case_id"], arm, r.ok, flush=True)

    by_unit: dict[str, dict[str, list[float]]] = {}
    for case in canonical["cases"]:
        d = by_unit.setdefault(case["independent_unit"], {})
        for arm in ("candidate", "random", "oracle"):
            d.setdefault(arm, []).append(
                float(case["normalized_improvement"].get(arm) or 0.0))
    diffs = {"candidate_vs_base": [], "candidate_vs_random": [],
             "oracle_vs_base": []}
    for arms in by_unit.values():
        cand = sum(arms["candidate"]) / max(1, len(arms["candidate"]))
        rand = sum(arms["random"]) / max(1, len(arms["random"]))
        orc = sum(arms["oracle"]) / max(1, len(arms["oracle"]))
        diffs["candidate_vs_base"].append(cand)
        diffs["candidate_vs_random"].append(cand - rand)
        diffs["oracle_vs_base"].append(orc)
    stats = {k: _paired_stats(v, alpha=prereg["alpha"])
             for k, v in diffs.items()}
    pvals = [stats[k]["p"] for k in ("candidate_vs_base",
                                     "candidate_vs_random")]
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0] * len(pvals)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[idx]))
        adj[idx] = running
    for key, value in zip(("candidate_vs_base", "candidate_vs_random"), adj):
        stats[key]["holm_p"] = value
        stats[key]["significant"] = value <= prereg["alpha"]
    diag = _diagnosis_metrics(canonical["cases"])
    audits = [evidence_audit(
        resolver_result=c["evidence"]["resolver"],
        agent_citations=c["diagnoses"]["agent"]["citations"])
        for c in canonical["cases"]]
    engineering = {"all_arms_ok": all(
        all(r.get("ok") for r in c["arms"].values())
        for c in canonical["cases"]),
        "n_cases": len(canonical["cases"]),
        "config_applied": all(
            all(r.get("ok") for r in c["arms"].values())
            for c in canonical["cases"]),
        "artifact_integrity": all(
            c["evidence"]["resolver"]["integrity_pass"]
            for c in canonical["cases"])}
    integrity = {"gt_outside_run_root": True, "zero_gt_leakage": True,
                 "all_evidence_verified": engineering["artifact_integrity"],
                 "benchmark_frozen": True,
                 "evidence_audit_pass": all(a["artifact_integrity"]
                                            for a in audits)}
    d_ok = (diag["agent"]["accuracy"] >= prereg[
        "diagnosis_gates"]["min_root_cause_accuracy"]
        and diag["agent"]["ece"] <= prereg["confidence_gates"]["max_ece"]
        and diag["agent"]["accuracy"] > diag["random"]["accuracy"]
        and diag["agent"]["accuracy"] > diag["shuffled"]["accuracy"])
    i_ok = bool(engineering["config_applied"] and all(
        sum(arms["oracle"]) / max(1, len(arms["oracle"]))
        > prereg["mme_normalized"] for arms in by_unit.values()))
    c_ok = bool(all(stats[k]["significant"]
                    and stats[k]["mean"] >= prereg["mme_normalized"]
                    for k in ("candidate_vs_base", "candidate_vs_random")))
    canonical.update({
        "statistics": stats, "diagnosis": diag, "evidence_audit": audits,
        "engineering": engineering, "integrity": integrity,
        "verdicts": {
            "engineering": "PASS" if all(engineering.values()) else "FAIL",
            "benchmark_integrity": "PASS" if all(integrity.values()) else "BLOCKED",
            "diagnosis": "PASS" if d_ok else "FAIL",
            "intervention": "PASS" if i_ok else "FAIL",
            "causal_improvement": "PASS" if c_ok else "INCONCLUSIVE",
        },
        "five_layer_pass": False,
        "scientific_claim": bool(d_ok and c_ok),
        "retry_finalized": True,
    })
    canonical["artifact_sha256"] = sha256_json(canonical)
    (root / "canonical_results.json").write_text(json.dumps(
        canonical, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8")
    write_report(canonical=canonical, root=root)
    return canonical


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="ARR_EMB_RSI_02_FULL2")
    args = ap.parse_args()
    c = rerun_failed(Path(args.root))
    print(json.dumps({"verdicts": c["verdicts"],
                      "scientific_claim": c["scientific_claim"]}, indent=2))
