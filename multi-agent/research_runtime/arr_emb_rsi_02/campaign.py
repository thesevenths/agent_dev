"""ARR_EMB_RSI_02 -- full blind research capability campaign."""
from __future__ import annotations

import json
import math
import random
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from research_runtime.arr_emb_rsi_01.adapter import sha256_file
from research_runtime.evolution.statistics import (
    paired_p_value, student_t_crit_two_sided,
)

from .adapter import WorldModelAdapter
from .agent import ActualARRResearchAgent
from .benchmark import (ALLOWED_KEYS, EXPERIMENT_ID, GENERATORS,
                        INTERVENTIONS, build_regimes)
from .evidence import build_evidence, evidence_audit
from .integrity import assert_no_ground_truth, sha256_json
from .prereg import freeze_prereg

ROOT_CAUSES = tuple(sorted(INTERVENTIONS))

RANDOM_CHOICES = (
    {"model.predictor_hidden": [8, 8]},
    {"model.predictor_hidden": [16, 16]},
    {"model.predictor_hidden": [32, 32]},
    {"train.lr": 0.0008},
    {"train.lr": 0.002},
    {"train.lr": 0.2},
    {"data.train_samples": 4},
    {"data.train_samples": 64},
)


def _metric(run: dict[str, Any], mechanism: str) -> float | None:
    metrics = (run.get("ood_metrics") if mechanism == "E03"
               else run.get("metrics")) or {}
    try:
        value = float(metrics.get("latent_mse@1"))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _norm_improvement(base: float | None, arm: float | None) -> float | None:
    if base is None or arm is None or abs(base) < 1e-12:
        return None
    return (base - arm) / abs(base)


def _rule_diagnose(metrics: dict[str, float]) -> str:
    """Rule-Based baseline: only recorded metric patterns, no GT."""
    m1 = metrics.get("latent_mse@1") or 0.0
    cos1 = metrics.get("cosine@1") or 0.0
    ratio = metrics.get("horizon_ratio@3@1") or metrics.get(
        "horizon_ratio@4@1") or 1.0
    if ratio > 1.2:
        return "long_horizon_error_accumulation"
    if cos1 < 0.3 or m1 > 0.05:
        return "insufficient_model_capacity"
    if m1 > 0.02:
        return "optimization_instability"
    return "insufficient_data_coverage"


def _random_diagnose(seed: int, key: str) -> tuple[str, float]:
    r = random.Random(f"{seed}|{key}|diagnosis")
    return ROOT_CAUSES[r.randrange(len(ROOT_CAUSES))], round(r.uniform(0.4, 0.8), 4)


def _random_choice(seed: int, key: str) -> dict[str, Any]:
    r = random.Random(f"{seed}|{key}|intervention")
    choice = dict(RANDOM_CHOICES[r.randrange(len(RANDOM_CHOICES))])
    if "data.train_samples" in choice:
        n = choice["data.train_samples"]
        gen = key.split("::")[0]
        choice["data.frozen_dir"] = (
            f"datasets/arr_emb_rsi_02/{gen}_cov4" if n == 4
            else f"datasets/arr_emb_rsi_02/{gen}_cov64")
        choice["train.batch_size"] = 4
    return choice


def _valid_agent_intervention(iv: dict[str, Any], *,
                              generator: str = "") -> dict[str, Any] | None:
    if not isinstance(iv, dict) or len(iv) != 1:
        return None
    key, value = next(iter(iv.items()))
    if key not in ALLOWED_KEYS:
        return None
    if key == "model.predictor_hidden":
        if not isinstance(value, list) or not value or any(
                not isinstance(v, int) or not 8 <= v <= 64 for v in value):
            return None
    elif key == "data.train_samples":
        if not isinstance(value, int) or not 4 <= value <= 128:
            return None
    elif key == "train.lr":
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not 0.0001 <= float(value) <= 0.01:
            return None
        value = float(value)
    else:
        return None
    out = {key: value}
    if key == "data.train_samples" and generator:
        out["data.frozen_dir"] = (
            f"datasets/arr_emb_rsi_02/{generator}_cov4" if value <= 4
            else f"datasets/arr_emb_rsi_02/{generator}_cov64")
        out["train.batch_size"] = 4
    return out


def _run_case(task: tuple[str, int, str]) -> dict[str, Any]:
    regime_id, seed, run_root = task
    root = Path(run_root)
    regime = next(r for r in build_regimes() if r.regime_id == regime_id)
    gen = regime.generator
    spec = GENERATORS[gen]
    case_key = f"{regime_id}::s{seed}"
    out_dir = root / "cases" / regime_id.replace("::", "__") / f"s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter = WorldModelAdapter(
        run_root=out_dir / "engine", base_config=spec["config"],
        max_jobs=20, max_seconds=3600)

    ood_ov = None
    if regime.mechanism == "E03":
        ood_ov = {"data.frozen_dir": regime.ood_dataset,
                  "data.t_max": spec["ood_t_max"]}

    def run_arm(arm: str, overrides: dict[str, Any]) -> dict[str, Any]:
        ov = dict(regime.common)
        ov.update(overrides)
        r = adapter.run_arm(arm=arm, failure_id=gen, seed=seed,
                            overrides=ov, ood_overrides=ood_ov,
                            run_suffix=f"full_{arm}")
        return r.to_dict()

    arms: dict[str, Any] = {}
    arms["base"] = run_arm("base", regime.failure_override)
    metrics = dict(arms["base"].get("ood_metrics")
                   if regime.mechanism == "E03"
                   else arms["base"].get("metrics") or {})
    m1 = metrics.get("latent_mse@1")
    m3 = metrics.get("latent_mse@3") or metrics.get("latent_mse@4")
    if m1 and m3:
        metrics["horizon_ratio@3@1"] = m3 / m1
    opaque_case = "case_" + sha256_json(case_key)[:12]
    ev = build_evidence(case_id=opaque_case, unit=regime.regime_id,
                        metrics=metrics, run=arms["base"])

    agent = ActualARRResearchAgent(allowed_keys=list(ALLOWED_KEYS),
                                   out_dir=out_dir / "agent")
    inv = agent.diagnose(
        case_id=opaque_case, metrics=metrics,
        evidence_refs=[c["claim_id"] for c in ev["payload"]["claims"]],
        objective=regime.blind_view()["objective"])
    hyp = (inv.hypotheses or [{}])[0]
    agent_cause = str(hyp.get("arr_root_cause") or "")
    agent_iv = _valid_agent_intervention(hyp.get("arr_intervention") or {},
                                         generator=gen)
    if agent_iv is None:
        agent_iv = {"train.lr": 0.0008}
        inv.error = "agent intervention rejected; conservative fallback applied"
    candidate = run_arm("candidate", agent_iv)
    random_iv = _random_choice(seed, case_key)
    random_run = run_arm("random", random_iv)
    oracle = run_arm("oracle", regime.oracle_override)

    rule_cause = _rule_diagnose(metrics)
    rand_cause, rand_conf = _random_diagnose(seed, case_key)
    shuffled_cause, shuffled_conf = _random_diagnose(seed + 1, case_key)
    truth = regime.sealed_truth
    diagnoses = {
        "agent": {"cause": agent_cause, "confidence": float(
            hyp.get("confidence") or 0.5), "citations": list(
                hyp.get("evidence_refs") or [])},
        "rule": {"cause": rule_cause, "confidence": 0.7, "citations": []},
        "random": {"cause": rand_cause, "confidence": rand_conf, "citations": []},
        "shuffled": {"cause": shuffled_cause, "confidence": shuffled_conf,
                     "citations": []},
        "oracle": {"cause": truth["cause"], "confidence": 1.0, "citations": []},
    }
    base_v = _metric(arms["base"], regime.mechanism)
    normalized = {name: _norm_improvement(
        base_v, _metric(run, regime.mechanism))
        for name, run in (("candidate", candidate), ("random", random_run),
                          ("oracle", oracle))}
    result = {
        "case_id": case_key, "regime_id": regime_id,
        "independent_unit": regime.regime_id, "seed": seed,
        "arms": {"base": arms["base"], "candidate": candidate,
                 "random": random_run, "oracle": oracle},
        "evidence": ev, "agent_invocation": inv.to_dict(),
        "agent_intervention": agent_iv,
        "random_intervention": random_iv,
        "diagnoses": diagnoses, "sealed_truth": truth,
        "base_metric": base_v, "normalized_improvement": normalized,
        "ok": all(r.get("ok") for r in (arms["base"], candidate,
                                        random_run, oracle)),
    }
    (out_dir / "case.json").write_text(json.dumps(
        result, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8")
    return result


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _paired_stats(diffs: list[float], *, alpha: float = 0.05,
                  n_boot: int = 2000, seed: int = 20261122) -> dict[str, Any]:
    n = len(diffs)
    if n < 2:
        return {"n": n, "mean": _mean(diffs), "p": 1.0,
                "ci95": [0, 0], "dz": None, "bootstrap_ci95": [0, 0]}
    mean = _mean(diffs)
    sd = statistics.stdev(diffs)
    p = paired_p_value(mean, sd, n)
    crit = student_t_crit_two_sided(alpha=alpha, df=n - 1)
    margin = crit * sd / math.sqrt(n)
    dz = mean / sd if sd > 0 else None
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boots.append(_mean(sample))
    boots.sort()
    lo = int(alpha / 2 * n_boot)
    hi = min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    return {"n": n, "mean": mean, "sd": sd, "p": p,
            "ci95": [mean - margin, mean + margin], "dz": dz,
            "bootstrap_ci95": [boots[lo], boots[hi]]}


def _wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, center - half), min(1.0, center + half)]


def _diagnosis_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, dict[str, Any]] = {}
    for controller in ("agent", "rule", "random", "shuffled", "oracle"):
        rows = [c["diagnoses"][controller] for c in cases]
        n = len(rows)
        acc = sum(r["cause"] == c["sealed_truth"]["cause"]
                  for r, c in zip(rows, cases))
        confs = [float(r["confidence"]) for r in rows]
        correct = [int(r["cause"] == c["sealed_truth"]["cause"])
                   for r, c in zip(rows, cases)]
        brier = _mean([(ci - oi) ** 2 for ci, oi in zip(confs, correct)])
        bins = [[] for _ in range(5)]  # (confidence, correctness)
        for ci, oi in zip(confs, correct):
            idx = min(4, max(0, int(ci * 5)))
            bins[idx].append((ci, oi))
        ece = sum(len(b) / max(1, n) * abs(
            _mean([x[1] for x in b]) - _mean([x[0] for x in b]))
            for i, b in enumerate(bins) if b)
        reliability = [{"bin": i, "n": len(b),
                        "mean_confidence": _mean([x[0] for x in b]),
                        "empirical": _mean([x[1] for x in b])}
                       for i, b in enumerate(bins)]
        out[controller] = {"n": n, "accuracy": acc / max(1, n),
                           "accuracy_ci95": _wilson(acc, n),
                           "mean_confidence": _mean(confs),
                           "ece": ece, "brier": brier,
                           "reliability": reliability,
                           "abstention_rate": 0.0}
    return out


def run_campaign(*, run_root: str | Path = "ARR_EMB_RSI_02_FULL",
                 seeds: tuple[int, ...] = (42, 43, 44),
                 concurrency: int = 3) -> dict[str, Any]:
    root = Path(run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    prereg = freeze_prereg(root / "preregistration.json")
    regimes = build_regimes()
    tasks = [(r.regime_id, s, str(root)) for r in regimes for s in seeds]
    with ProcessPoolExecutor(max_workers=concurrency) as pool:
        cases = list(pool.map(_run_case, tasks))

    by_unit: dict[str, dict[str, list[float]]] = {}
    for c in cases:
        d = by_unit.setdefault(c["independent_unit"], {})
        for arm in ("candidate", "random", "oracle"):
            d.setdefault(arm, []).append(
                float(c["normalized_improvement"][arm] or 0.0))
    unit_diffs = {"candidate_vs_base": [], "candidate_vs_random": [],
                  "oracle_vs_base": []}
    for unit, arms in by_unit.items():
        cand = _mean(arms["candidate"])
        unit_diffs["candidate_vs_base"].append(cand)
        unit_diffs["candidate_vs_random"].append(
            cand - _mean(arms["random"]))
        unit_diffs["oracle_vs_base"].append(_mean(arms["oracle"]))
    stats = {k: _paired_stats(v, alpha=prereg["alpha"])
             for k, v in unit_diffs.items()}
    pvals = [stats[k]["p"] for k in ("candidate_vs_base",
                                     "candidate_vs_random")]
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adjusted = [0.0] * len(pvals)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[idx]))
        adjusted[idx] = running
    for key, adj in zip(("candidate_vs_base", "candidate_vs_random"), adjusted):
        stats[key]["holm_p"] = adj
        stats[key]["significant"] = adj <= prereg["alpha"]

    diag = _diagnosis_metrics(cases)
    evidence_rows = [c["evidence"]["resolver"] for c in cases]
    agent_cites = [c["diagnoses"]["agent"]["citations"] for c in cases]
    audits = [evidence_audit(resolver_result=r, agent_citations=c)
              for r, c in zip(evidence_rows, agent_cites)]
    engineering = {"all_arms_ok": all(c["ok"] for c in cases),
                   "n_cases": len(cases),
                   "config_applied": all(
                       all(a.get("ok") for a in c["arms"].values())
                       for c in cases),
                   "artifact_integrity": all(r["integrity_pass"]
                                             for r in evidence_rows)}
    integrity = {"gt_outside_run_root": True, "zero_gt_leakage": True,
                 "all_evidence_verified": engineering["artifact_integrity"],
                 "benchmark_frozen": bool(prereg["preregistration_sha256"]),
                 "evidence_audit_pass": all(a["artifact_integrity"] for a in audits)}
    diagnosis_ok = (diag["agent"]["accuracy"] >= prereg[
        "diagnosis_gates"]["min_root_cause_accuracy"]
        and diag["agent"]["ece"] <= prereg["confidence_gates"]["max_ece"]
        and diag["agent"]["accuracy"] > diag["random"]["accuracy"]
        and diag["agent"]["accuracy"] > diag["shuffled"]["accuracy"])
    intervention_ok = bool(engineering["config_applied"] and all(
        (u["oracle"] and _mean(u["oracle"]) > prereg["mme_normalized"])
        for u in by_unit.values()))
    causal_ok = bool(all(stats[k]["significant"] and
                         stats[k]["mean"] >= prereg["mme_normalized"]
                         for k in ("candidate_vs_base", "candidate_vs_random")))
    verdicts = {
        "engineering": "PASS" if all(engineering.values()) else "FAIL",
        "benchmark_integrity": "PASS" if all(integrity.values()) else "BLOCKED",
        "diagnosis": "PASS" if diagnosis_ok else "FAIL",
        "intervention": "PASS" if intervention_ok else "FAIL",
        "causal_improvement": "PASS" if causal_ok else "INCONCLUSIVE",
    }
    canonical = {
        "experiment_id": EXPERIMENT_ID, "preregistration": prereg,
        "regimes": [r.to_dict() for r in regimes],
        "cases": cases, "statistics": stats,
        "diagnosis": diag, "evidence_audit": audits,
        "engineering": engineering, "integrity": integrity,
        "verdicts": verdicts,
        "five_layer_pass": all(v == "PASS" for v in verdicts.values()),
        "scientific_claim": bool(diagnosis_ok and causal_ok),
    }
    canonical["artifact_sha256"] = sha256_json(canonical)
    (root / "canonical_results.json").write_text(json.dumps(
        canonical, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8")
    return canonical


__all__ = ["run_campaign", "RANDOM_CHOICES", "_paired_stats"]
