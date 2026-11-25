"""ARR-07 autonomous research loop.

LLM does COGNITION (hypothesize, choose experiment, replan from data).
A FROZEN evaluator decides goal attainment from measured data — the LLM never
self-certifies success. Blind mode exposes only experimental measurements.

Loop: observe -> LLM plan/choose -> EXECUTE real experiment -> observe new data
      -> frozen evaluator judges -> replan if not met -> repeat until goal/budget.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import config  # noqa: F401
from research_runtime.llm_client import llm_config, make_generate

from ..arr06.engine import run_experiment
from . import benchmark, core


@dataclass
class LoopStep:
    iteration: int
    llm_plan: str
    chosen_experiment: str
    experiment_record: dict[str, Any]
    measurement: dict[str, float]
    improvement: float | None
    goal_met: bool
    evaluator_note: str
    root_cause_correct: bool = False
    spurious_convergence: bool = False


@dataclass
class LoopResult:
    task_id: str
    blind: bool
    converged: bool
    iterations: int
    final_improvement: float | None
    steps: list[LoopStep] = field(default_factory=list)
    llm_calls: int = 0
    engine_runs: int = 0
    replans: int = 0


class ResearchBrain:
    """Real configured LLM. Cognition only; never judges success."""

    def __init__(self, *, out_dir: Path, generate=None):
        cfg = llm_config()
        if not cfg.get("api_key"):
            raise RuntimeError("LLM not configured")
        self.model = cfg["model"]
        self._gen = generate or make_generate()
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    def _call(self, payload: dict) -> str:
        prompt = json.dumps(payload, ensure_ascii=False, default=str)
        try:
            raw = self._gen(prompt)
        except Exception:
            raw = ""
        if not isinstance(raw, str):
            raw = json.dumps(raw, ensure_ascii=False, default=str)
        (self.out / f"llm_{len(list(self.out.glob('llm_*.txt')))}.txt").write_text(
            prompt + "\n---RESPONSE---\n" + raw, encoding="utf-8")
        return raw

    def plan(self, observation: dict, history: list[dict]) -> tuple[str, str]:
        """Return (reasoning, chosen experiment id)."""
        prompt = {
            "role": "autonomous_researcher",
            "task": "You are studying a dynamical system whose held-out prediction "
                    "error is elevated. Based ONLY on the experimental measurements "
                    "below ( NOT any configuration ), state your hypothesis about the "
                    "failure mechanism and choose ONE experiment from the available "
                    "list that best tests it. If previous experiments failed, revise "
                    "your hypothesis using their measured outcomes.",
            "current_observation": observation,
            "previous_experiments_and_results": history,
            "output_format": "Return JSON: {\"reasoning\": str, \"experiment_id\": "
                             "one of the available experiment ids}",
        }
        raw = self._call(prompt)
        text = raw.strip()
        if text.startswith("```"):
            text = "\n".join(l for l in text.splitlines()
                             if not l.strip().startswith("```")).strip()
        try:
            d = json.loads(text)
            exp = str(d.get("experiment_id", ""))
            reasoning = str(d.get("reasoning", ""))[:800]
        except (ValueError, TypeError):
            exp, reasoning = "", raw[:800]
        if exp not in {e.id for e in benchmark.EXPERIMENTS}:
            # Fallback: deterministic round-robin when LLM output is malformed.
            exp = benchmark.EXPERIMENTS[len(history) % len(benchmark.EXPERIMENTS)].id
        return reasoning, exp


def frozen_evaluate(improvement: float | None, *, target: float = 0.3,
                    experiment_id: str = "", true_cause: str = "",
                    require_correct_cause: bool = True) -> tuple[bool, str, bool]:
    """FROZEN goal judgement. Returns (met, note, root_cause_correct).

    Goal = measured improvement >= target AND the chosen discriminative
    experiment's mechanism matches the true root cause. This prevents SPURIOUS
    convergence where any generic improvement (e.g. more data) lowers MSE but
    the mechanism was misidentified. The cause check uses the registered
    experiment->cause map, NOT LLM self-report.
    """
    from .benchmark import EXP_BY_ID
    exp = EXP_BY_ID.get(experiment_id)
    cause_ok = bool(exp and true_cause and exp.discriminates == (true_cause,))
    if improvement is None:
        return False, "no valid measurement", cause_ok
    if improvement < target:
        return False, f"improvement {improvement:.3f} < target {target}", cause_ok
    if require_correct_cause and not cause_ok:
        return False, (f"improvement {improvement:.3f} >= target BUT experiment "
                       f"{experiment_id} does not test the true mechanism "
                       f"(spurious convergence)"), cause_ok
    return True, f"improvement {improvement:.3f} >= target {target} and mechanism correct", cause_ok


def run_loop(*, task: benchmark.Task, seed: int, root: Path, blind: bool = True,
             max_iterations: int = 4, target: float = 0.3,
             protocol_hash: str = "") -> LoopResult:
    """One autonomous research loop on one task."""
    root = Path(root).resolve()
    out_dir = root / "raw_data" / "traces" / (
        "blind" if blind else "leaky") / benchmark.safe_id(task.task_id) / f"s{seed}"
    brain = ResearchBrain(out_dir=out_dir)
    result = LoopResult(task_id=task.task_id, blind=blind, converged=False,
                        iterations=0, final_improvement=None)
    history: list[dict] = []

    # Step 0: baseline experiment (measured, no config given to LLM in blind).
    base = run_experiment(experiment_id=f"loop_base_{'b' if blind else 'l'}_"
        f"{benchmark.safe_id(task.task_id)}_s{seed}", track="loop", strategy="base",
        task_id=task.task_id, seed=seed, train_cfg=task.train, model_cfg=task.model,
        test_cfg=task.test, root=root, protocol_hash=protocol_hash)
    result.engine_runs += 1
    bm = (base.get("metrics") or {}).get("one_step_mse")
    obs = task.blind_observation(base.get("metrics") or {}) if blind \
        else task.leaky_observation(base.get("metrics") or {})

    # Domain-agnostic frozen stopping rule (research_runtime/generic/stopping.py).
    from ..generic.stopping import GoalCriterion, GoalGate
    gate = GoalGate(GoalCriterion(target_improvement=target,
                                  max_iterations=max_iterations))
    result.stop_decisions = []

    for it in range(max_iterations):
        result.llm_calls += 1
        reasoning, exp_id = brain.plan(obs, history)
        tr, mo = dict(task.train), dict(task.model)
        for k, v in (benchmark.EXP_BY_ID[exp_id].overrides.get("train") or {}).items():
            tr[k] = v
        for k, v in (benchmark.EXP_BY_ID[exp_id].overrides.get("model") or {}).items():
            mo[k] = v
        rec = run_experiment(experiment_id=f"loop_it{it}_{'b' if blind else 'l'}_"
            f"{benchmark.safe_id(task.task_id)}_s{seed}", track="loop", strategy=exp_id,
            task_id=task.task_id, seed=seed, train_cfg=tr, model_cfg=mo,
            test_cfg=task.test, root=root, protocol_hash=protocol_hash)
        result.engine_runs += 1
        em = (rec.get("metrics") or {}).get("one_step_mse")
        imp = (bm - em) / abs(bm) if (bm is not None and em is not None and abs(bm) > 1e-12) else None
        met, note, cause_ok = frozen_evaluate(
            imp, target=target, experiment_id=exp_id, true_cause=task.cause)
        step = LoopStep(iteration=it, llm_plan=reasoning, chosen_experiment=exp_id,
                        experiment_record={"experiment_id": rec.get("experiment_id"),
                                           "ok": rec.get("ok")},
                        measurement={"base_mse": bm, "arm_mse": em},
                        improvement=imp, goal_met=met, evaluator_note=note,
                        root_cause_correct=cause_ok,
                        spurious_convergence=(imp is not None and imp >= target
                                              and not cause_ok))
        result.steps.append(step)
        result.iterations = it + 1
        result.final_improvement = imp
        # NEW observation = actual measured outcome (this is "看数据").
        history.append({"experiment": exp_id, "measured_mse": em,
                        "improvement_vs_baseline": imp, "llm_reasoning": reasoning})
        if it > 0 and reasoning and history[-1]["experiment"] != history[-2]["experiment"]:
            result.replans += 1
        # Frozen goal gate: stop on preregistered target OR iteration budget.
        # Uses MEASURED improvement only; mechanism correctness is a separate
        # judgement (frozen_evaluate above) and does not override stopping.
        decision = gate.observe(iteration=it, improvement=imp)
        result.stop_decisions.append(decision.to_dict())
        if decision.goal_met:
            result.converged = True
        if decision.should_stop:
            break
        # Replan next iteration with updated measurement (never re-show config in blind).
        obs = task.blind_observation({"one_step_mse": em}) if blind \
            else task.leaky_observation({"one_step_mse": em})
        obs["history"] = history
    return result


__all__ = ["ResearchBrain", "frozen_evaluate", "LoopStep", "LoopResult", "run_loop"]
