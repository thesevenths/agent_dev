"""Domain-agnostic pre-registered stopping rule.

Problem abstracted from ARR-07: after a 91% improvement, the agent kept chasing
residual error (EXP_CAP/EXP_OPT) and made things worse. A scientific loop needs a
FROZEN goal: once the preregistered criterion is met, STOP. This module decides
stop/continue from measured numbers only; no LLM self-certification, no domain
knowledge, no metric-name assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GoalCriterion:
    target_improvement: float          # normalized improvement vs baseline
    max_iterations: int
    min_consecutive: int = 1           # require N consecutive meets (stability)
    mechanism_hit_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"target_improvement": self.target_improvement,
                "max_iterations": self.max_iterations,
                "min_consecutive": self.min_consecutive,
                "mechanism_hit_required": self.mechanism_hit_required}


@dataclass
class StopDecision:
    should_stop: bool
    reason: str
    goal_met: bool
    consecutive_meets: int

    def to_dict(self) -> dict[str, Any]:
        return {"should_stop": self.should_stop, "reason": self.reason,
                "goal_met": self.goal_met,
                "consecutive_meets": self.consecutive_meets}


class GoalGate:
    """Frozen stopping rule. Call after each measured experiment."""

    def __init__(self, criterion: GoalCriterion):
        self.criterion = criterion
        self.consecutive = 0
        self.stopped = False
        self.stop_reason = ""
        self.history: list[dict[str, Any]] = []

    def observe(self, *, iteration: int, improvement: float | None,
                mechanism_correct: bool = True) -> StopDecision:
        if self.stopped:
            return StopDecision(True, self.stop_reason, False, self.consecutive)
        met = (improvement is not None
               and improvement >= self.criterion.target_improvement
               and (mechanism_correct or not self.criterion.mechanism_hit_required))
        if met:
            self.consecutive += 1
        else:
            self.consecutive = 0
        self.history.append({"iteration": iteration, "improvement": improvement,
                             "mechanism_correct": mechanism_correct, "met": met})
        goal = self.consecutive >= self.criterion.min_consecutive
        if goal:
            self.stopped = True
            self.stop_reason = "preregistered goal met; stop (no residual chasing)"
            return StopDecision(True, self.stop_reason, True, self.consecutive)
        if iteration + 1 >= self.criterion.max_iterations:
            self.stopped = True
            self.stop_reason = "iteration budget exhausted"
            return StopDecision(True, self.stop_reason, False, self.consecutive)
        return StopDecision(False, "continue", False, self.consecutive)

    def to_dict(self) -> dict[str, Any]:
        return {"criterion": self.criterion.to_dict(), "history": self.history}


__all__ = ["GoalCriterion", "StopDecision", "GoalGate"]
