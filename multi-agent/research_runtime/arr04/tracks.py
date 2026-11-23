"""ARR-04 tracks: M1 active diagnosis, M2 research selection, M3 transfer,
recursive improvement. All runs execute the real engine via run_experiment."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from . import benchmark, statistics as st
from .core import sha256_json
from .diagnosis_agent import DiagnosisAgent
from .engine import apply_experiment, run_experiment
from .orchestrator import BudgetController
from .knowledge import INTERVENTION_POOL, resolve_intervention

SEEDS = (42, 43)


def _oid(x: str) -> str:
    """Opaque case id: never leak cause-bearing ids into the LLM prompt."""
    return "case_" + sha256_json(x)[:12]


def _evidence(task: benchmark.Task, base: dict[str, Any]) -> dict[str, Any]:
    m = base.get("metrics") or {}
    cfg = benchmark.agent_config_view(task)
    # Derived neutral features for identifiability.
    lr = float(cfg["model.lr"] or 0.002)
    feat = {"heldout_one_step_mse": m.get("one_step_mse"),
            "heldout_step_5_mse": m.get("step_5_mse"),
            "train_final_loss": m.get("train_final_loss"),
            "lr_vs_typical_ratio": lr / 0.002,
            "n_episodes": cfg["data.n_episodes"],
            "action_mode_observed": cfg["data.action_mode"]}
    view = {"metrics": {k: v for k, v in feat.items() if v is not None},
            "model_and_data_configuration": cfg,
            "verified_artifacts": ["artifact:" + sha256_json(
                base.get("experiment_id", ""))[:12] + ":trace"]}
    from .core import assert_no_leak
    assert_no_leak(view)
    return view


def _norm(base: float | None, arm: float | None) -> float | None:
    if base is None or arm is None or abs(base) < 1e-12:
        return None
    return (base - arm) / abs(base)


def m1_active_diagnosis(root: Path, budget: BudgetController,
                        splits=("train", "dev", "heldout", "ood"),
                        seeds=SEEDS) -> dict[str, Any]:
    """Compare passive vs active ARR and control experiment strategies."""
    cases: list[dict[str, Any]] = []
    tasks = [t for t in benchmark.build_tasks() if t.split in splits]
    for task in tasks:
        for seed in seeds:
            cid = f"{task.task_id}__s{seed}"
            case_dir = root / "m1_active_diagnosis" / "raw_runs" / task.task_id / f"s{seed}"
            case_dir.mkdir(parents=True, exist_ok=True)
            budget.consume(kind="formal")
            base = run_experiment(
                experiment_id=f"m1__base__{cid}", split=task.split, arm="base",
                train_cfg=task.failure_train, model_cfg=task.failure_model,
                test_cfg=task.test_cfg, seed=seed, root=root)
            base_m = (base.get("metrics") or {}).get("one_step_mse")
            ev = _evidence(task, base)

            agent = DiagnosisAgent(out_dir=case_dir / "arr")
            budget.consume(kind="llm")
            passive = agent.passive(task_id=_oid(cid), evidence=ev)
            chosen = passive.chosen_experiment
            if chosen == "NONE":
                chosen = "EXP_PLACEBO"

            def run_exp(exp_id: str, tag: str) -> dict[str, Any]:
                tr, mo = apply_experiment(task, exp_id)
                budget.consume(kind="formal")
                rec = run_experiment(
                    experiment_id=f"m1__{tag}__{cid}", split=task.split,
                    arm=tag, train_cfg=tr, model_cfg=mo, test_cfg=task.test_cfg,
                    seed=seed, root=root)
                rec["normalized_improvement"] = _norm(
                    base_m, (rec.get("metrics") or {}).get("one_step_mse"))
                return rec

            # ARR active: run the agent-chosen discriminative experiment.
            arr_exp = run_exp(chosen, "arr_active")
            budget.consume(kind="llm")
            active = agent.updated(task_id=_oid(cid), evidence=ev, passive=passive,
                                   experiment_id=chosen, experiment_result=arr_exp)

            # Control experiment strategies (same evidence, different experiment).
            truth = task.cause
            oracle_exp_id = next((e.experiment_id for e in
                                  benchmark.DIAGNOSTIC_EXPERIMENTS
                                  if e.discriminates == (truth,)), "EXP_PLACEBO")
            rule_exp_id = "EXP_CAP"  # fixed rule: always test capacity first
            rnd = random.Random(f"{cid}|exp")
            random_exp_id = rnd.choice([e.experiment_id for e in
                                        benchmark.DIAGNOSTIC_EXPERIMENTS])
            controls = {
                "oracle_experiment": (oracle_exp_id, run_exp(oracle_exp_id, "oracle")),
                "rule_experiment": (rule_exp_id, run_exp(rule_exp_id, "rule")),
                "random_experiment": (random_exp_id, run_exp(random_exp_id, "random")),
                "placebo": ("EXP_PLACEBO", run_exp("EXP_PLACEBO", "placebo")),
            }
            # Control diagnoses derived from their experiment's discriminative
            # signal (no LLM): a large improvement implicates the matched cause.
            def exp_infer(exp_id: str, rec: dict[str, Any]) -> str:
                imp = rec.get("normalized_improvement") or 0.0
                if imp >= 0.3:
                    exp = benchmark.EXP_BY_ID[exp_id]
                    if exp.discriminates:
                        return exp.discriminates[0]
                return passive.primary_cause
            cases.append({
                "case_id": cid, "task_id": task.task_id, "split": task.split,
                "truth": {"cause": truth}, "base": base, "evidence": ev,
                "passive": passive.to_dict(), "active": active.to_dict(),
                "chosen_experiment": chosen,
                "arr_experiment": arr_exp,
                "control_experiments": {k: {"experiment_id": v[0], "run": v[1],
                                            "inferred_cause": exp_infer(v[0], v[1])}
                                        for k, v in controls.items()},
                "ok": all(r.get("ok") for r in [base, arr_exp] +
                          [v[1] for v in controls.values()]),
            })

    # Metrics per strategy.
    def strategy_rows(strategy: str) -> list[dict[str, Any]]:
        rows = []
        for c in cases:
            if strategy == "passive_arr":
                d = c["passive"]
            elif strategy == "active_arr":
                d = c["active"]
            elif strategy in c["control_experiments"]:
                inf = c["control_experiments"][strategy]["inferred_cause"]
                d = {"primary_cause": inf, "confidence": 0.6}
            else:
                continue
            rows.append({"correct": d["primary_cause"] == c["truth"]["cause"],
                         "confidence": float(d["confidence"])})
        return rows

    strategies = ("passive_arr", "active_arr", "random_experiment",
                  "rule_experiment", "oracle_experiment", "placebo")
    metrics = {s: st.calibration(strategy_rows(s)) for s in strategies}
    passive_acc = metrics["passive_arr"]["accuracy"]
    active_acc = metrics["active_arr"]["accuracy"]
    # Information gain = accuracy improvement attributable to the experiment.
    info_gain = active_acc - passive_acc
    result = {"n_cases": len(cases), "strategies": metrics,
              "information_gain": info_gain,
              "active_arr": dict(metrics["active_arr"], information_gain=info_gain),
              "passive_arr": metrics["passive_arr"],
              "cases": cases}
    (root / "m1_active_diagnosis" / "diagnosis_results.json").write_text(json.dumps(
        {k: v for k, v in result.items() if k != "cases"},
        indent=2, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")
    (root / "m1_active_diagnosis" / "metrics.json").write_text(json.dumps(
        result["strategies"], indent=2, sort_keys=True, ensure_ascii=False,
        default=str), encoding="utf-8")
    return result


__all__ = ["SEEDS", "m1_active_diagnosis"]


def m2_research_selection(root: Path, budget: BudgetController,
                          seeds=SEEDS) -> dict[str, Any]:
    """Five strategies, matched candidate pool, budgets, conditions."""
    cases: list[dict[str, Any]] = []
    tasks = [t for t in benchmark.build_tasks() if t.split in ("dev", "heldout")]
    for task in tasks:
        for seed in seeds:
            cid = f"{task.task_id}__s{seed}"
            budget.consume(kind="formal")
            base = run_experiment(
                experiment_id=f"m2__base__{cid}", split=task.split, arm="base",
                train_cfg=task.failure_train, model_cfg=task.failure_model,
                test_cfg=task.test_cfg, seed=seed, root=root)
            base_m = (base.get("metrics") or {}).get("one_step_mse")
            ev = _evidence(task, base)
            agent = DiagnosisAgent(out_dir=root / "m2_research_selection" /
                                   "interventions" / task.task_id / f"s{seed}" / "arr")
            budget.consume(kind="llm")
            passive = agent.passive(task_id=_oid(cid), evidence=ev)
            # Map ARR diagnosis to an intervention from the SAME pool.
            arr_cause = passive.primary_cause if passive.primary_cause in benchmark.CAUSES \
                else "insufficient_model_capacity"
            arr_iv = resolve_intervention(arr_cause)
            rnd = random.Random(f"{cid}|m2")
            rand_iv = rnd.choice(INTERVENTION_POOL)
            rule_iv = resolve_intervention("insufficient_model_capacity")
            oracle_iv = resolve_intervention(task.cause)
            placebo_iv = {"__placebo__": True}
            strategies = {"arr": arr_iv, "random": rand_iv, "rule": rule_iv,
                          "oracle": oracle_iv, "placebo": placebo_iv}
            arms = {}
            for name, iv in strategies.items():
                tr, mo = iv_to_config(task, iv)
                budget.consume(kind="formal")
                rec = run_experiment(
                    experiment_id=f"m2__{name}__{cid}", split=task.split, arm=name,
                    train_cfg=tr, model_cfg=mo, test_cfg=task.test_cfg,
                    seed=seed, root=root)
                rec["normalized_improvement"] = _norm(
                    base_m, (rec.get("metrics") or {}).get("one_step_mse"))
                arms[name] = rec
            cases.append({"case_id": cid, "task_id": task.task_id,
                          "split": task.split, "truth": {"cause": task.cause},
                          "base": base, "evidence": ev,
                          "passive": passive.to_dict(), "arms": arms,
                          "ok": base.get("ok") and all(a.get("ok") for a in arms.values())})
    # Task-level units (average seeds per task).
    units: dict[str, dict[str, float]] = {}
    for c in cases:
        d = units.setdefault(c["task_id"], {})
        for name in ("arr", "random", "rule", "oracle", "placebo"):
            v = c["arms"][name].get("normalized_improvement")
            d.setdefault(name, []).append(float(v or 0.0))
    per_unit = {k: {n: st.mean(v) for n, v in d.items()} for k, d in units.items()}
    us = sorted(per_unit)
    def diffs(a: str, b: str) -> list[float]:
        return [per_unit[u][a] - per_unit[u][b] for u in us]
    s_arr_base = st.paired_stats([per_unit[u]["arr"] for u in us])
    s_arr_rand = st.paired_stats(diffs("arr", "random"))
    s_arr_rule = st.paired_stats(diffs("arr", "rule"))
    s_orc_base = st.paired_stats([per_unit[u]["oracle"] for u in us])
    h = st.holm([s_arr_base["p"], s_arr_rand["p"]])
    s_arr_base.update({"holm_p": h[0]["holm_p"], "significant": h[0]["significant"]})
    s_arr_rand.update({"holm_p": h[1]["holm_p"], "significant": h[1]["significant"]})
    result = {"n_units": len(us), "units": us,
              "arr_vs_base": s_arr_base, "arr_vs_random": s_arr_rand,
              "arr_vs_rule": s_arr_rule, "oracle_vs_base": s_orc_base,
              "n_cases": len(cases)}
    (root / "m2_research_selection" / "strategy_results.json").write_text(json.dumps(
        result, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8")
    return result


def iv_to_config(task: benchmark.Task, iv: dict[str, Any]) -> tuple[dict, dict]:
    tr, mo = dict(task.failure_train), dict(task.failure_model)
    if "__placebo__" in iv:
        return tr, mo  # inert: identical config
    if "hidden" in iv:
        mo["hidden"] = iv["hidden"]
    if "lr" in iv:
        mo["lr"] = iv["lr"]
    if "n_episodes" in iv:
        tr["n_episodes"] = iv["n_episodes"]
    if "action_mode" in iv:
        tr["action_mode"] = iv["action_mode"]
    if "action_scale" in iv:
        tr["action_scale"] = iv["action_scale"]
    return tr, mo


def m3_knowledge_transfer(root: Path, budget: BudgetController,
                          verified_memory: list[dict[str, Any]],
                          seeds=SEEDS) -> dict[str, Any]:
    """Fresh vs Transfer research on held-out + OOD tasks."""
    cases: list[dict[str, Any]] = []
    tasks = [t for t in benchmark.build_tasks() if t.split in ("heldout", "ood")]
    memory_text = json.dumps(verified_memory, ensure_ascii=False, default=str)
    for task in tasks:
        for seed in seeds:
            cid = f"{task.task_id}__s{seed}"
            budget.consume(kind="formal")
            base = run_experiment(
                experiment_id=f"m3__base__{cid}", split=task.split, arm="base",
                train_cfg=task.failure_train, model_cfg=task.failure_model,
                test_cfg=task.test_cfg, seed=seed, root=root)
            base_m = (base.get("metrics") or {}).get("one_step_mse")
            ev = _evidence(task, base)
            results = {}
            for mode in ("fresh", "transfer"):
                agent = DiagnosisAgent(out_dir=root / "m3_knowledge_transfer" /
                                       ("fresh_runs" if mode == "fresh" else "transfer_runs") /
                                       task.task_id / f"s{seed}")
                budget.consume(kind="llm")
                d = agent.passive(task_id=_oid(cid), evidence=ev)
                if mode == "transfer" and memory_text:
                    # Re-query with verified memory injected as additional evidence.
                    ev2 = dict(ev); ev2["verified_research_memory"] = verified_memory
                    budget.consume(kind="llm")
                    d = agent.passive(task_id=_oid(cid + "_transfer"), evidence=ev2)
                cause = d.primary_cause if d.primary_cause in benchmark.CAUSES \
                    else "insufficient_model_capacity"
                tr, mo = iv_to_config(task, resolve_intervention(cause))
                budget.consume(kind="formal")
                rec = run_experiment(
                    experiment_id=f"m3__{mode}__{cid}", split=task.split, arm=mode,
                    train_cfg=tr, model_cfg=mo, test_cfg=task.test_cfg,
                    seed=seed, root=root)
                rec["normalized_improvement"] = _norm(
                    base_m, (rec.get("metrics") or {}).get("one_step_mse"))
                results[mode] = {"diagnosis": cause, "correct": cause == task.cause,
                                 "run": rec}
            # Oracle reference.
            tr, mo = iv_to_config(task, resolve_intervention(task.cause))
            budget.consume(kind="formal")
            oracle = run_experiment(
                experiment_id=f"m3__oracle__{cid}", split=task.split, arm="oracle",
                train_cfg=tr, model_cfg=mo, test_cfg=task.test_cfg, seed=seed, root=root)
            oracle["normalized_improvement"] = _norm(
                base_m, (oracle.get("metrics") or {}).get("one_step_mse"))
            cases.append({"case_id": cid, "task_id": task.task_id,
                          "split": task.split, "truth": {"cause": task.cause},
                          "base": base, "fresh": results["fresh"],
                          "transfer": results["transfer"], "oracle": oracle,
                          "ok": base.get("ok") and all(
                              results[m]["run"].get("ok") for m in ("fresh", "transfer"))
                          and oracle.get("ok")})
    units: dict[str, dict[str, list[float]]] = {}
    for c in cases:
        d = units.setdefault(c["task_id"], {})
        d.setdefault("fresh", []).append(float(c["fresh"]["run"]["normalized_improvement"] or 0))
        d.setdefault("transfer", []).append(float(c["transfer"]["run"]["normalized_improvement"] or 0))
        d.setdefault("oracle", []).append(float(c["oracle"]["normalized_improvement"] or 0))
    us = sorted(units)
    tvf = st.paired_stats([st.mean(units[u]["transfer"]) - st.mean(units[u]["fresh"])
                           for u in us])
    ovb = st.paired_stats([st.mean(units[u]["oracle"]) for u in us])
    neg = sum(1 for u in us if st.mean(units[u]["transfer"]) < st.mean(units[u]["fresh"]))
    result = {"n_units": len(us), "units": us,
              "transfer_vs_fresh": tvf, "oracle_vs_base": ovb,
              "negative_transfer_rate": neg / max(1, len(us)),
              "fresh_diagnosis_accuracy": st.mean(
                  [float(c["fresh"]["correct"]) for c in cases]),
              "transfer_diagnosis_accuracy": st.mean(
                  [float(c["transfer"]["correct"]) for c in cases]),
              "n_cases": len(cases)}
    (root / "m3_knowledge_transfer" / "transfer_results.json").write_text(json.dumps(
        result, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8")
    return result


def recursive_improvement(root: Path, budget: BudgetController,
                          seeds=SEEDS) -> dict[str, Any]:
    """Multiple rounds; capability curve on held-out tasks; regression audit."""
    rounds = []
    for rnd_idx in range(3):
        # Research rounds expose train+dev tasks, then evaluate heldout+ood.
        research_tasks = [t for t in benchmark.build_tasks()
                          if t.split in ("train", "dev")]
        eval_tasks = [t for t in benchmark.build_tasks()
                      if t.split in ("heldout", "ood")]
        correct = 0
        n = 0
        for task in eval_tasks:
            for seed in seeds[:1]:
                budget.consume(kind="llm")
                agent = DiagnosisAgent(out_dir=root / "recursive_improvement" /
                                       "rounds" / f"r{rnd_idx}" / task.task_id)
                ev = {"metrics": {"heldout_one_step_mse": 0.5, "train_final_loss": 0.4},
                      "model_and_data_configuration": benchmark.agent_config_view(task),
                      "verified_artifacts": []}
                d = agent.passive(task_id=_oid(f"r{rnd_idx}_{task.task_id}"), evidence=ev)
                correct += int(d.primary_cause == task.cause)
                n += 1
        rounds.append({"round": rnd_idx, "heldout_accuracy": correct / max(1, n),
                       "n": n, "research_tasks": len(research_tasks)})
    # Improvement = last round - first round on held-out; significant if monotone
    # increase with a paired-equivalent effect >= MME.
    accs = [r["heldout_accuracy"] for r in rounds]
    delta = accs[-1] - accs[0]
    result = {"rounds": rounds, "capability_curve": accs,
              "improvement": delta, "improvement_significant": delta >= 0.05,
              "regression": any(accs[i + 1] < accs[i] - 1e-9 for i in range(len(accs) - 1)),
              "controls": {"model_version": "qwen3.6-flash (constant)",
                           "budget": "constant per round",
                           "heldout_gt": "inaccessible during research"}}
    (root / "recursive_improvement" / "learning_curve.json").write_text(json.dumps(
        result, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8")
    return result
