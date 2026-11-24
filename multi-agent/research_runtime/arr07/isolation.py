"""ARR-07 link-level isolation validations.

V4 planner causality: (a) repeat SAME observation -> choice stability;
(b) CHANGE observation in a targeted way -> choice changes systematically.
V5 evaluator independent audit: re-derive goal judgements from raw traces with
a SEPARATE implementation, do not import loop.frozen_evaluate.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import benchmark
from .loop import ResearchBrain


def planner_determinism(root: Path, observation: dict, repeats: int = 5) -> dict[str, Any]:
    """Same observation repeated: does the planner give the same choice?"""
    brain = ResearchBrain(out_dir=root / "raw_data" / "traces" / "isolation" / "determinism")
    choices = []
    for i in range(repeats):
        _, exp = brain.plan(observation, [])
        choices.append(exp)
    from collections import Counter
    counts = Counter(choices)
    top, n = counts.most_common(1)[0]
    return {"repeats": repeats, "choices": choices, "mode": top,
            "mode_count": n, "stability": n / repeats,
            "deterministic": n == repeats}


def planner_causality(root: Path, *, repeats: int = 3) -> dict[str, Any]:
    """Change ONE evidence dimension; check if choice shifts systematically.

    Construct two observations differing only in the train-test gap signature:
    - HIGH gap (train loss tiny, heldout MSE huge): generic-coverage signal
    - LOW gap (train loss ≈ heldout MSE): no generalization signal
    A causal planner should respond differently to these.
    """
    brain = ResearchBrain(out_dir=root / "raw_data" / "traces" / "isolation" / "causality")
    base = {"task_ref": "isolation_probe", "family": "heldout",
            "available_experiments": [{"id": e.id, "name": e.name}
                                       for e in benchmark.EXPERIMENTS]}
    # Distinct MECHANISM signatures, not just different numbers:
    # - optimisation failure: huge training loss (model cannot even fit train)
    # - coverage/generalisation: tiny train loss but huge heldout error
    # - well-fit model: both losses small (no failure to diagnose)
    conditions = {
        "high_gap": dict(base, measurement={"one_step_mse": 17.0, "final_loss": 0.001}),
        "optimisation_failure": dict(base, measurement={"one_step_mse": 19.0, "final_loss": 12.0}),
        "well_fit": dict(base, measurement={"one_step_mse": 0.02, "final_loss": 0.015}),
    }
    out = {}
    for name, obs in conditions.items():
        choices = []
        for _ in range(repeats):
            reasoning, exp = brain.plan(obs, [])
            choices.append(exp)
        out[name] = {"choices": choices, "set": sorted(set(choices))}
    all_choices = {c for v in out.values() for c in v["choices"]}
    distinct_sets = {tuple(v["set"]) for v in out.values()}
    return {"conditions": out, "distinct_choice_sets": len(distinct_sets),
            "causal": len(distinct_sets) >= 2,
            "note": "causal if the planner responds systematically to changed evidence"}


def validate_v4(root: Path) -> dict[str, Any]:
    r = Path(root)
    (r / "loop" / "isolation").mkdir(parents=True, exist_ok=True)
    # Fixed observation for determinism test.
    obs = {"task_ref": "det_probe", "family": "heldout",
           "measurement": {"one_step_mse": 17.0, "final_loss": 0.001},
           "available_experiments": [{"id": e.id, "name": e.name}
                                      for e in benchmark.EXPERIMENTS]}
    det = planner_determinism(r, obs, repeats=5)
    causality = planner_causality(r, repeats=3)
    out = {"determinism": det, "causality": causality,
           "pass": det["deterministic"] and causality["causal"]}
    (r / "loop" / "isolation" / "planner_causality.json").write_text(json.dumps(
        out, indent=2, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")
    return out


def independent_evaluator_audit(root: Path, campaign_root: Path) -> dict[str, Any]:
    """V5: independently re-derive each loop's goal verdict from raw traces.

    Uses a SEPARATE implementation: read the experiment records for each
    (task, iteration), recompute improvement, compare to the EXP->cause map.
    Does NOT import loop.frozen_evaluate.
    """
    r = Path(root)
    exp_to_cause = {e.id: e.discriminates[0] for e in benchmark.EXPERIMENTS
                    if e.discriminates}
    results = json.loads((Path(campaign_root) / "loop" / "closed_loop" /
                          "results.json").read_text(encoding="utf-8"))
    rows = []
    for t in results["tasks"]:
        cause = t["cause"]
        seq = t["experiment_sequence"]
        imps = t["improvements"]
        # Independent recomputation: met = any step with imp>=0.3 AND correct mechanism.
        met_steps = []
        for exp, imp in zip(seq, imps):
            correct = exp_to_cause.get(exp) == cause
            met_steps.append(bool(imp is not None and imp >= 0.3 and correct))
        independently_met = any(met_steps)
        reported_met = bool(t["converged"])
        rows.append({"task_id": t["task_id"], "cause": cause,
                     "reported_converged": reported_met,
                     "independent_converged": independently_met,
                     "agree": reported_met == independently_met,
                     "mechanism_hits": sum(1 for e in seq if exp_to_cause.get(e) == cause)})
    agree = sum(int(x["agree"]) for x in rows)
    out = {"n_tasks": len(rows), "n_agree": agree,
           "agreement_rate": agree / max(1, len(rows)), "rows": rows,
           "pass": agree == len(rows)}
    (r / "loop" / "isolation" / "evaluator_audit.json").write_text(json.dumps(
        out, indent=2, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")
    return out


__all__ = ["validate_v4", "independent_evaluator_audit"]
