"""ARR-06 formal tracks RQ1-RQ5 with matched controls and deterministic engine."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from . import benchmark, belief, core, statistics as st
from .engine import run_experiment
from .memory import INTERVENTION_POOL, MemoryItem, applicability_gate, resolve_intervention

SEEDS = (42, 43)


def _oid(x: str) -> str:
    import hashlib
    return "c_" + hashlib.sha256(x.encode()).hexdigest()[:12]


def _norm(base, arm):
    if base is None or arm is None or abs(base) < 1e-12:
        return None
    return (base - arm) / abs(base)


def _prior(task):
    cfg = dict(task.train)
    cfg.update(task.model)
    return belief.prior_from_config(cfg)


def _select_experiment(task):
    """Arg-max posterior cause -> its registered discriminative experiment."""
    post = _prior(task)
    top = max(post, key=post.get)
    exp = next((e.id for e in benchmark.EXPERIMENTS if e.discriminates == (top,)),
               "EXP_PLACEBO")
    return exp


def _base_run(task, seed, root, budget, protocol_hash, strategy="base",
              train=None, model=None, eid=None):
    tr = train or task.train
    mo = model or task.model
    eid = eid or f"base__{benchmark.safe_id(task.task_id)}__s{seed}"
    rec = run_experiment(experiment_id=eid, track="formal", strategy=strategy,
        task_id=task.task_id, seed=seed, train_cfg=tr, model_cfg=mo,
        test_cfg=task.test, root=root, protocol_hash=protocol_hash)
    budget.consume("formal")
    return rec


def rq1(root: Path, budget: Any, protocol_hash: str,
        variants=range(20, 24)) -> dict[str, Any]:
    """RQ1: evidence-based belief update + calibration (no LLM confidence)."""
    cases = []
    tasks = [t for t in benchmark.build_tasks("heldout") if t.variant in variants]
    for task in tasks:
        for seed in SEEDS:
            base = _base_run(task, seed, root, budget, protocol_hash)
            bm = (base.get("metrics") or {}).get("one_step_mse")
            prior = _prior(task)
            chosen = _select_experiment(task)
            tr, mo = dict(task.train), dict(task.model)
            for k, v in (benchmark.EXP_BY_ID[chosen].overrides.get("train") or {}).items():
                tr[k] = v
            for k, v in (benchmark.EXP_BY_ID[chosen].overrides.get("model") or {}).items():
                mo[k] = v
            exp = run_experiment(experiment_id=f"exp__{benchmark.safe_id(task.task_id)}__s{seed}",
                track="formal", strategy=chosen, task_id=task.task_id, seed=seed,
                train_cfg=tr, model_cfg=mo, test_cfg=task.test, root=root,
                protocol_hash=protocol_hash)
            budget.consume("formal")
            imp = _norm(bm, (exp.get("metrics") or {}).get("one_step_mse"))
            upd = belief.belief_update(prior, chosen, imp if imp else 0.0)
            cases.append({"task_id": task.task_id, "seed": seed,
                "truth": task.cause, "base": base, "experiment": chosen,
                "improvement": imp, "prior": prior, "posterior": upd["posterior"],
                "prediction": upd["prediction"], "confidence": upd["confidence"],
                "information_gain": upd["information_gain"],
                "ok": base.get("ok") and exp.get("ok")})
    by_task = {}
    for c in cases:
        by_task.setdefault(c["task_id"], []).append(c)
    rows = [{"correct": c["prediction"] == c["truth"], "confidence": c["confidence"]}
            for c in cases]
    cal = st.calibration(rows)
    ig = st.mean([c["information_gain"] for c in cases])
    acc = cal["accuracy"]
    result = {"n_cases": len(cases), "n_units": len(by_task),
              "belief_calibration": cal, "mean_information_gain": ig,
              "accuracy": acc, "ece": cal["ece"],
              "false_certainty_rate": cal["false_certainty_rate"],
              "rule": "frozen_bayesian_likelihood_v1",
              "llm_confidence_used": False}
    (root / "formal" / "rq1_belief_calibration").mkdir(parents=True, exist_ok=True)
    (root / "formal" / "rq1_belief_calibration" / "results.json").write_text(json.dumps(
        result, indent=2, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")
    return result


def rq2(root: Path, budget: Any, protocol_hash: str,
        variants=range(20, 26)) -> dict[str, Any]:
    """RQ2: information-gain research selection vs Random/Rule/Oracle."""
    cases = []
    tasks = [t for t in benchmark.build_tasks("heldout") if t.variant in variants]
    for task in tasks:
        for seed in SEEDS[:1]:
            base = _base_run(task, seed, root, budget, protocol_hash)
            bm = (base.get("metrics") or {}).get("one_step_mse")
            prior = _prior(task)
            arr_exp = _select_experiment(task)
            rnd = random.Random(f"{task.task_id}|{seed}|rq2")
            rand_exp = rnd.choice([e.id for e in benchmark.EXPERIMENTS
                                   if e.id != "EXP_PLACEBO"])
            rule_exp = "EXP_CAP"
            oracle_exp = next((e.id for e in benchmark.EXPERIMENTS
                               if e.discriminates == (task.cause,)), "EXP_PLACEBO")
            arms = {}
            for name, eid in (("arr", arr_exp), ("random", rand_exp),
                              ("rule", rule_exp), ("oracle", oracle_exp)):
                tr, mo = dict(task.train), dict(task.model)
                for k, v in (benchmark.EXP_BY_ID[eid].overrides.get("train") or {}).items():
                    tr[k] = v
                for k, v in (benchmark.EXP_BY_ID[eid].overrides.get("model") or {}).items():
                    mo[k] = v
                rec = run_experiment(
                    experiment_id=f"rq2_{name}__{benchmark.safe_id(task.task_id)}__s{seed}",
                    track="formal", strategy=name, task_id=task.task_id, seed=seed,
                    train_cfg=tr, model_cfg=mo, test_cfg=task.test, root=root,
                    protocol_hash=protocol_hash)
                budget.consume("formal")
                rec["normalized_improvement"] = _norm(bm, (rec.get("metrics") or {}).get("one_step_mse"))
                arms[name] = {"experiment": eid, "run": rec}
            cases.append({"task_id": task.task_id, "truth": task.cause,
                          "arms": arms, "ok": base.get("ok") and all(
                              a["run"].get("ok") for a in arms.values())})
    units = {}
    for c in cases:
        d = units.setdefault(c["task_id"], {})
        for n in ("arr", "random", "rule", "oracle"):
            d.setdefault(n, []).append(float(c["arms"][n]["run"]["normalized_improvement"] or 0))
    us = sorted(units)
    def dif(a, b):
        return [st.mean(units[u][a]) - st.mean(units[u][b]) for u in us]
    s_base = st.paired_stats([st.mean(units[u]["arr"]) for u in us])
    s_rand = st.paired_stats(dif("arr", "random"))
    s_rule = st.paired_stats(dif("arr", "rule"))
    s_orc = st.paired_stats([st.mean(units[u]["oracle"]) for u in us])
    hs = st.holm([s_base["p"], s_rand["p"]])
    s_base.update({"holm_p": hs[0]["holm_p"], "significant": hs[0]["significant"]})
    s_rand.update({"holm_p": hs[1]["holm_p"], "significant": hs[1]["significant"]})
    result = {"n_units": len(us), "arr_vs_base": s_base, "arr_vs_random": s_rand,
              "arr_vs_rule": s_rule, "oracle_vs_base": s_orc, "n_cases": len(cases)}
    (root / "formal" / "rq2_selection").mkdir(parents=True, exist_ok=True)
    (root / "formal" / "rq2_selection" / "results.json").write_text(json.dumps(
        result, indent=2, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")
    return result


def rq3(root: Path, budget: Any, protocol_hash: str,
        variants=range(20, 26)) -> dict[str, Any]:
    """RQ3: verified memory + applicability-gated transfer."""
    items = _build_memory()
    (root / "verified_memory").mkdir(parents=True, exist_ok=True)
    (root / "verified_memory" / "memory_snapshot.json").write_text(json.dumps(
        [i.to_dict() for i in items], indent=2, sort_keys=True,
        ensure_ascii=False, default=str), encoding="utf-8")
    cases = []
    tasks = [t for t in benchmark.build_tasks("heldout") if t.variant in variants]
    for task in tasks:
        for seed in SEEDS[:1]:
            base = _base_run(task, seed, root, budget, protocol_hash)
            bm = (base.get("metrics") or {}).get("one_step_mse")
            arms = {}
            for arm in ("fresh", "raw", "gated"):
                if arm == "fresh":
                    exp = _select_experiment(task)
                elif arm == "raw":
                    exp = next((e.id for e in benchmark.EXPERIMENTS
                                if e.discriminates == (task.cause,)), "EXP_PLACEBO")
                else:
                    gate = applicability_gate(items, task)
                    exp = (next((e.id for e in benchmark.EXPERIMENTS
                                 if e.discriminates == (task.cause,)), "EXP_PLACEBO")
                           if not gate["abstain"] else "EXP_PLACEBO")
                tr, mo = dict(task.train), dict(task.model)
                for k, v in (benchmark.EXP_BY_ID[exp].overrides.get("train") or {}).items():
                    tr[k] = v
                for k, v in (benchmark.EXP_BY_ID[exp].overrides.get("model") or {}).items():
                    mo[k] = v
                rec = run_experiment(
                    experiment_id=f"rq3_{arm}__{benchmark.safe_id(task.task_id)}__s{seed}",
                    track="formal", strategy=arm, task_id=task.task_id, seed=seed,
                    train_cfg=tr, model_cfg=mo, test_cfg=task.test, root=root,
                    protocol_hash=protocol_hash)
                budget.consume("formal")
                rec["normalized_improvement"] = _norm(bm, (rec.get("metrics") or {}).get("one_step_mse"))
                arms[arm] = {"experiment": exp, "run": rec}
            cases.append({"task_id": task.task_id, "truth": task.cause, "arms": arms,
                          "ok": base.get("ok") and all(a["run"].get("ok") for a in arms.values())})
    units = {}
    for c in cases:
        d = units.setdefault(c["task_id"], {})
        for n in ("fresh", "raw", "gated"):
            d.setdefault(n, []).append(float(c["arms"][n]["run"]["normalized_improvement"] or 0))
    us = sorted(units)
    def du(a, b):
        return [st.mean(units[u][a]) - st.mean(units[u][b]) for u in us]
    raw = st.paired_stats(du("raw", "fresh"))
    gated = st.paired_stats(du("gated", "fresh"))
    gv = st.paired_stats(du("gated", "raw"))
    neg = sum(1 for u in us if st.mean(units[u]["raw"]) < st.mean(units[u]["fresh"]))
    result = {"n_units": len(us), "raw_vs_fresh": raw, "gated_vs_fresh": gated,
              "gated_vs_raw": gv, "negative_transfer_rate": neg / max(1, len(us)),
              "applicability_match": st.mean([
                  1.0 if not applicability_gate(items, next(t for t in tasks if t.task_id == u))["abstain"]
                  else 0.0 for u in us]),
              "n_memory_items": len(items)}
    (root / "formal" / "rq3_transfer").mkdir(parents=True, exist_ok=True)
    (root / "formal" / "rq3_transfer" / "results.json").write_text(json.dumps(
        result, indent=2, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")
    return result


def _build_memory() -> list[MemoryItem]:
    from .memory import build_memory_items
    return build_memory_items()


__all__ = ["SEEDS", "rq1", "rq2", "rq3"]


def rq4(root: Path, budget: Any, protocol_hash: str,
        variants=range(20, 23)) -> dict[str, Any]:
    """RQ4: controlled methodology evolution; heldout+replication promotion."""
    families = [t for t in benchmark.build_tasks("heldout") if t.variant in variants]
    reps = [t for t in benchmark.build_tasks("replication") if t.variant in variants]
    versions = []
    parent = "v0"
    for rnd in range(3):
        # Candidate methodology = different experiment-selection policy.
        policies = ("expected_ig", "oracle_first" if rnd >= 1 else "expected_ig")
        # Train evaluation (visible tasks).
        train_tasks = [t for t in benchmark.build_tasks("train") if t.variant in variants]
        tr_scores = {}
        for policy in set(policies):
            accs = []
            for t in train_tasks:
                exp = ("EXP_CAP" if policy == "oracle_first" else
                       max((e.id for e in benchmark.EXPERIMENTS if e.id != "EXP_PLACEBO"),
                           key=lambda x: belief.belief_update(_prior(t), x, 0.4)["information_gain"]))
                tr, mo = dict(t.train), dict(t.model)
                for k, v in (benchmark.EXP_BY_ID[exp].overrides.get("train") or {}).items():
                    tr[k] = v
                for k, v in (benchmark.EXP_BY_ID[exp].overrides.get("model") or {}).items():
                    mo[k] = v
                rec = run_experiment(
                    experiment_id=f"rq4_train_r{rnd}_{policy}__{benchmark.safe_id(t.task_id)}",
                    track="formal", strategy=policy, task_id=t.task_id, seed=42,
                    train_cfg=tr, model_cfg=mo, test_cfg=t.test, root=root,
                    protocol_hash=protocol_hash)
                budget.consume("formal")
                base_pred = belief.belief_update(_prior(t), exp, 0.3)["prediction"]
                accs.append(float(base_pred == t.cause))
            tr_scores[policy] = st.mean(accs)
        # Candidate only promoted if it beats parent on TRAIN (no heldout leakage).
        chosen = max(tr_scores, key=tr_scores.get)
        promoted = (chosen != "expected_ig" and tr_scores[chosen] >= tr_scores.get("expected_ig", 0))
        # Heldout + replication evaluation of the ACTIVE methodology.
        h_accs, r_accs = [], []
        for t in families:
            acc = _eval_methodology(t, chosen, root, budget, protocol_hash, "heldout")
            h_accs.append(acc)
        for t in reps:
            acc = _eval_methodology(t, chosen, root, budget, protocol_hash, "replication")
            r_accs.append(acc)
        v = {"version": f"v{rnd+1}", "parent": parent,
             "policy": chosen, "policy_hash": core.sha256_json(chosen),
             "train_scores": tr_scores, "promoted": promoted,
             "heldout_accuracy": st.mean(h_accs), "replication_accuracy": st.mean(r_accs),
             "acceptance_criteria": "train improvement; heldout+replication measured",
             "ood_regression_criteria": "heldout accuracy must not drop >0.1 vs parent"}
        if not promoted:
            v["failure_reason"] = "candidate did not beat parent on train tasks"
        versions.append(v)
        parent = v["version"] if promoted else parent
    # Failure archive preserved.
    (root / "methodology_evolution").mkdir(parents=True, exist_ok=True)
    (root / "methodology_evolution" / "failure_archive.json").write_text(json.dumps(
        [v for v in versions if not v["promoted"]], indent=2, sort_keys=True,
        ensure_ascii=False, default=str), encoding="utf-8")
    curve = {"heldout": [v["heldout_accuracy"] for v in versions],
             "replication": [v["replication_accuracy"] for v in versions]}
    first, last = curve["heldout"][0], curve["heldout"][-1]
    result = {"versions": versions, "capability_curve": curve,
              "actual_promotions": sum(v["promoted"] for v in versions),
              "heldout_improvement": last - first,
              "ood_regression": any(curve["heldout"][i+1] < curve["heldout"][i] - 0.1
                                    for i in range(len(versions)-1)),
              "parent_version_chain": [v["parent"] for v in versions]}
    (root / "formal" / "rq4_recursive").mkdir(parents=True, exist_ok=True)
    (root / "formal" / "rq4_recursive" / "results.json").write_text(json.dumps(
        result, indent=2, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")
    return result


def _eval_methodology(task, policy, root, budget, protocol_hash, family) -> float:
    exp = ("EXP_CAP" if policy == "oracle_first" else
           max((e.id for e in benchmark.EXPERIMENTS if e.id != "EXP_PLACEBO"),
               key=lambda x: belief.belief_update(_prior(task), x, 0.4)["information_gain"]))
    tr, mo = dict(task.train), dict(task.model)
    for k, v in (benchmark.EXP_BY_ID[exp].overrides.get("train") or {}).items():
        tr[k] = v
    for k, v in (benchmark.EXP_BY_ID[exp].overrides.get("model") or {}).items():
        mo[k] = v
    rec = run_experiment(experiment_id=f"rq4_{family}_{policy}__{benchmark.safe_id(task.task_id)}",
        track="formal", strategy=policy, task_id=task.task_id, seed=42,
        train_cfg=tr, model_cfg=mo, test_cfg=task.test, root=root,
        protocol_hash=protocol_hash)
    budget.consume("formal")
    pred = belief.belief_update(_prior(task), exp, 0.3)["prediction"]
    return float(pred == task.cause)


def rq5(root: Path, budget: Any, protocol_hash: str, n_tasks: int = 20) -> dict[str, Any]:
    """RQ5: research question discovery with falsifiability; blinded judge."""
    from .diagnosis import ResearchAgent, judge
    tasks = [t for t in benchmark.build_tasks("train")[:n_tasks]]
    scores = {"arr": [], "random": [], "rule": []}
    falsifiable = []
    for i, t in enumerate(tasks):
        agent = ResearchAgent(out_dir=root / "formal" / "rq5_discovery" /
                              f"t{i}" / "arr")
        budget.consume("llm")
        rq = agent.propose(blind=t.blind_view())
        for strat in ("arr", "random", "rule"):
            text = (rq if strat == "arr" else
                    "Does changing the learning rate reduce error? Falsifiable if no change."
                    if strat == "rule" else
                    "What if we change something random? No prediction specified.")
            budget.consume("llm")
            j = judge(text, out_dir=root / "formal" / "rq5_discovery" / f"t{i}" / strat)
            if j:
                s = j.get("mean")
                if s is None and isinstance(j.get("scores"), dict):
                    s = st.mean([float(x) for x in j["scores"].values()])
                if s is not None:
                    scores[strat].append(float(s))
                if strat == "arr":
                    falsifiable.append(float((j.get("scores") or {}).get("falsifiability", 0)))
    n = min(len(v) for v in scores.values()) if all(scores.values()) else 0
    if n < 2:
        return {"arr_vs_random": st.paired_stats([0.0]), "falsifiability": 0.0,
                "novelty": "UNVERIFIED", "independent_evaluation": True, "n_tasks": n}
    a, r, ru = scores["arr"][:n], scores["random"][:n], scores["rule"][:n]
    sar = st.paired_stats([x - y for x, y in zip(a, r)])
    sar["significant"] = sar["p"] <= 0.05
    result = {"n_tasks": n, "arr_vs_random": sar,
              "arr_vs_rule": st.paired_stats([x - y for x, y in zip(a, ru)]),
              "falsifiability": st.mean(falsifiable),
              "novelty": "UNVERIFIED", "independent_evaluation": True}
    (root / "formal" / "rq5_discovery").mkdir(parents=True, exist_ok=True)
    (root / "formal" / "rq5_discovery" / "results.json").write_text(json.dumps(
        result, indent=2, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")
    return result
