"""Warm-start causality test: does the planner change its plan after a FAILED experiment?

V4 tested cold start (history=[]) and found a fixed EXP_COV prior. That does not
answer whether the agent revises plans from evidence — a human-like researcher
typically rules out a hypothesis AFTER an experiment fails. This test supplies
the same three measurement signatures PLUS an identical failed EXP_COV history
and checks whether subsequent choices diverge.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..arr07.loop import ResearchBrain
from ..arr07 import benchmark


def run(root: str | Path, repeats: int = 2) -> dict[str, Any]:
    root = Path(root)
    brain = ResearchBrain(out_dir=root / "raw_data" / "traces" / "warm_causality")
    conditions = {
        "high_gap": {"one_step_mse": 17.0, "final_loss": 0.001},
        "optimisation_failure": {"one_step_mse": 19.0, "final_loss": 12.0},
        "well_fit": {"one_step_mse": 0.02, "final_loss": 0.015},
    }
    failed_history = [{"experiment": "EXP_COV", "measured_mse": 16.9,
                       "improvement_vs_baseline": 0.01,
                       "llm_reasoning": "coverage test performed"}]
    out = {}
    for name, meas in conditions.items():
        choices = []
        for _ in range(repeats):
            obs = {"task_ref": "warm_probe", "family": "heldout",
                   "measurement": meas,
                   "available_experiments": [{"id": e.id, "name": e.name}
                                              for e in benchmark.EXPERIMENTS]}
            _, exp = brain.plan(obs, list(failed_history))
            choices.append(exp)
        out[name] = {"choices": choices, "stable": len(set(choices)) == 1}
    distinct = {tuple(v["choices"]) for v in out.values()}
    result = {"conditions": out, "distinct_choice_sets": len(distinct),
              "causal_after_failure": len(distinct) >= 2,
              "cold_start_comparison": "all EXP_COV at history=[] (V4)",
              "note": "warm test measures plan revision after failed evidence, "
                      "not cold-start prior"}
    p = root / "warm_causality.json"
    p.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


__all__ = ["run"]
