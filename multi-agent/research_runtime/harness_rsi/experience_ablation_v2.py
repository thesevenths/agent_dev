"""M6.1 -- Experience Retrieval ablation judged on REAL research quality.

M6 only checked ``injected_experience_ids`` (context injection) and called that
"behaviour change" -- the user explicitly rejected this. Here:

  * source experiences are REAL historical records (source campaign/task/evidence
    sha, failure type, root cause, intervention), never holdout-derived;
  * the SAME real runner (production agent+engine) is called with retrieval ON
    and OFF on matched tasks/seeds/budget;
  * the grader is INDEPENDENT and scores research quality from the REAL results;
  * verdict is based on quality delta, NOT on whether context was injected:
      improvement -> SUPPORTED, no change -> INCONCLUSIVE, regression -> REJECTED.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .identity import freeze_json, sha256_json

ABLATION_VERSION = "m6_1_experience_ablation_v1"


def run_ablation_v2(*, bank: Any, problem: str, tasks: list[Any],
                    runner: Any, grader: Any, out_dir: str | Path,
                    seed: int = 20261117) -> dict[str, Any]:
    """Paired ON/OFF experience retrieval on REAL runner, graded independently."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    on_ctx = bank.retrieve(problem=problem, enabled=True)
    off_ctx = bank.retrieve(problem=problem, enabled=False)
    if off_ctx:
        raise ValueError("retrieval OFF must yield NO experience context")
    arms: dict[str, Any] = {}
    for arm, ctx in (("with_retrieval", on_ctx), ("without_retrieval", off_ctx)):
        graded: list[dict[str, Any]] = []
        for task in tasks:
            raw = runner({"experience_context": ctx, "arm": arm,
                          "seed": seed}, task)
            g = grader(raw, task)
            graded.append({"task_id": getattr(task, "task_id", ""),
                           "injected_experience_ids": [e.get("experience_id")
                                                       for e in ctx],
                           "quality": g.get("metrics", {}),
                           "result_sha": g.get("result_sha", "")})
        frozen = freeze_json({"arm": arm, "n_injected": len(ctx),
                              "graded": graded}, root / f"{arm}.json")
        mean_q = (sum(g["quality"].get("downstream_experiment_success", 0)
                      for g in graded) / len(graded)) if graded else 0.0
        arms[arm] = {"n_injected": len(ctx), "graded": graded,
                     "mean_research_quality": mean_q,
                     "result_sha": frozen["sha256"]}
    on = arms["with_retrieval"]["mean_research_quality"]
    off = arms["without_retrieval"]["mean_research_quality"]
    delta = on - off
    verdict = ("SUPPORTED" if delta > 0 else
               "REJECTED" if delta < 0 else "INCONCLUSIVE")
    result = {"version": ABLATION_VERSION, "problem": problem,
              "arms": arms, "delta": delta, "verdict": verdict,
              "note": "verdict is from REAL research-quality delta, NOT "
                      "context injection alone"}
    freeze_json(result, root / "ablation_result.json")
    return result


__all__ = ["ABLATION_VERSION", "run_ablation_v2"]
