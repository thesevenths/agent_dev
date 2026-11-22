"""ARR-03 evaluation: diagnosis metrics, evidence audit and verdict gates."""
from __future__ import annotations

from typing import Any

from .statistics import mean, wilson


def _brier_ece(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confs = [float(r.get("confidence", 0.0)) for r in rows]
    correct = [int(r.get("correct", False)) for r in rows]
    n = len(rows)
    brier = mean([(c - o) ** 2 for c, o in zip(confs, correct)])
    bins: list[list[tuple[float, int]]] = [[] for _ in range(5)]
    for c, o in zip(confs, correct):
        idx = min(4, max(0, int(c * 5)))
        bins[idx].append((c, o))
    ece = 0.0
    reliability = []
    for i, b in enumerate(bins):
        if not b:
            reliability.append({"bin": i, "n": 0})
            continue
        mc = mean([x[0] for x in b])
        emp = mean([float(x[1]) for x in b])
        ece += len(b) / n * abs(emp - mc)
        reliability.append({"bin": i, "n": len(b),
                            "mean_confidence": mc, "empirical": emp})
    return {"brier": brier, "ece": ece, "reliability": reliability,
            "mean_confidence": mean(confs)}


def diagnosis_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    strategies = ("arr", "rule", "random", "shuffled", "oracle")
    for strat in strategies:
        strat_rows = []
        for c in cases:
            d = c["diagnoses"][strat]
            truth = c["truth"]["cause"]
            primary = d.get("primary", "UNKNOWN")
            strat_rows.append({
                "confidence": d.get("confidence", 0.0),
                "correct": primary == truth,
                "candidates": d.get("candidates", [primary]),
            })
        n = len(strat_rows)
        acc_n = sum(r["correct"] for r in strat_rows)
        top2 = sum(c["truth"]["cause"] in r["candidates"][:2]
                   for r, c in zip(strat_rows, cases))
        ce = _brier_ece(strat_rows)
        abst = sum(c["diagnoses"][strat].get("abstain", False) for c in cases)
        out[strat] = {
            "n": n, "accuracy": acc_n / max(1, n),
            "accuracy_ci95": wilson(acc_n, n),
            "top2_accuracy": top2 / max(1, n),
            "abstention_rate": abst / max(1, n),
            **ce,
        }
    return out


def evidence_quality(cases: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for c in cases:
        ev = c.get("evidence", {})
        # M2 (V2 evidence) stores the resolver under "resolver"; M3 stores it
        # under "artifact_resolver". Accept both.
        art = ev.get("artifact_resolver") or ev.get("resolver") or {}
        cites = c.get("diagnoses", {}).get("arr", {}).get("evidence_for", [])
        verified = {r["claim_id"] for r in art.get("claims", []) if r.get("ok")}
        # A citation is valid if it is a verified artifact claim OR it grounds
        # itself in a concrete observed metric/config token present in the
        # case evidence (claims are FREE TEXT by design in the diagnosis schema).
        ev_metrics = ev.get("metrics", {}) or ev.get("payload", {}).get("metrics", {})
        metric_tokens = {str(k) for k in ev_metrics}
        words = set()
        for k, value in ev_metrics.items():
            words.add(str(k).lower())
        relevant = [x for x in cites if any(
            tok in str(x).lower() for tok in ("mse", "cosine", "loss", "hidden",
                                              "episode", "lr", "action", "artifact",
                                              "gap", "horizon", "cos"))]
        valid = [x for x in cites
                 if x in verified or str(x).startswith("artifact:")
                 or any(tok in str(x).lower() for tok in metric_tokens)
                 or any(w in str(x).lower() for w in words if w)]
        rows.append({
            "artifact_integrity": bool(art.get("integrity_pass", False)),
            "citation_validity": (len(valid) / len(cites)) if cites else 0.0,
            "evidence_relevance": (len(relevant) / len(cites)) if cites else 0.0,
            "evidence_sufficiency": len(cites) >= 2,
        })
    return {
        "artifact_integrity_pass": all(r["artifact_integrity"] for r in rows),
        "citation_validity": mean([r["citation_validity"] for r in rows]),
        "evidence_relevance": mean([r["evidence_relevance"] for r in rows]),
        "evidence_sufficiency_rate": mean(
            [float(r["evidence_sufficiency"]) for r in rows]),
        "n_cases": len(rows),
    }


def verdicts(*, engineering: dict[str, Any], integrity: dict[str, Any],
             diag: dict[str, Any], causal: dict[str, Any],
             transfer: dict[str, Any] | None, prereg: dict[str, Any]) -> dict[str, Any]:
    eng = bool(engineering.get("n_cases", 0)) and all(
        bool(v) for k, v in engineering.items() if k != "n_cases")
    integ = all(bool(v) for v in integrity.values())
    arr = diag["arr"]
    d_gates = prereg["diagnosis_gates"]
    diag_ok = bool(
        arr["accuracy"] >= d_gates["min_accuracy"]
        and arr["accuracy"] > diag["random"]["accuracy"]
        and arr["accuracy"] > diag["shuffled"]["accuracy"]
        and arr["accuracy"] >= diag["rule"]["accuracy"]
        and arr["ece"] <= d_gates["max_ece"]
        and arr["brier"] <= diag["random"]["brier"]
        and arr["top2_accuracy"] >= d_gates["min_top2_accuracy"])
    c_gates = prereg["causal_gates"]
    causal_ok = bool(
        causal["candidate_vs_base"]["significant"]
        and causal["candidate_vs_base"]["mean"] >= c_gates["mme"]
        and causal["candidate_vs_random"]["significant"]
        and causal["candidate_vs_random"]["mean"] >= c_gates["mme"]
        and causal["oracle_vs_base"]["mean"] >= c_gates["mme"])
    causal_reject = bool(
        causal["oracle_vs_base"]["mean"] < c_gates["mme"]
        or causal.get("n_units", 0) < c_gates["min_units"])
    causal_verdict = ("FAIL" if causal_reject else
                      ("PASS" if causal_ok else "INCONCLUSIVE"))
    out = {
        "engineering": "PASS" if eng else "FAIL",
        "benchmark_integrity": "PASS" if integ else "BLOCKED",
        "m1_diagnosis": "PASS" if diag_ok else "FAIL",
        "m2_causal_improvement": causal_verdict,
    }
    if transfer is not None:
        t_gates = prereg["transfer_gates"]
        t_ok = bool(
            transfer["arr_transfer_vs_fresh"]["significant"]
            and transfer["arr_transfer_vs_fresh"]["mean"] >= t_gates["mme"])
        t_reject = bool(transfer["oracle_vs_base"]["mean"] < t_gates["mme"])
        out["m3_embodied_transfer"] = (
            "FAIL" if t_reject else ("PASS" if t_ok else "INCONCLUSIVE"))
        out["overall_research_capability"] = (
            "PASS" if all(v == "PASS" for k, v in out.items())
            else ("BLOCKED" if out["benchmark_integrity"] == "BLOCKED" else
                  "FAIL" if (causal_reject or t_reject) else "INCONCLUSIVE"))
    else:
        out["overall_research_capability"] = (
            "PASS" if all(v == "PASS" for v in out.values()) else "INCONCLUSIVE")
    return out


__all__ = ["diagnosis_metrics", "evidence_quality", "verdicts"]
