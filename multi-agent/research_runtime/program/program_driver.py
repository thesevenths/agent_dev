"""ARR v1.0 WP-02: thin Research Program driver.

Reuses DiscoveryCoordinator as the single-round engine (via an injected
``round_fn``). Adds ONLY the outer multi-round orchestration required by WP-02:
bounded objective -> research directions -> round planning from prior artifacts
-> execution -> memory commit -> next-round planning -> stop.

It does NOT create a Runtime (M5 RuntimeState owns journal/reservation/budget),
does NOT change DiscoveryCoordinator's single-round semantics, and never
hardcodes "three scripts". Next-round decisions come from ResearchHistoryIndex
over REAL prior-round artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from ..harness_rsi.identity import freeze_json, sha256_json
from .history_index import ResearchHistoryIndex
from .program_state import (
    ProgramStateError, ResearchProgramState, new_round_record)

RoundFn = Callable[..., dict[str, Any]]


class ProgramBudgetManager:
    """Track program/round budgets; refuse silent overruns."""

    def __init__(self, policy: dict[str, Any]):
        self.policy = dict(policy or {})
        self._used: dict[str, int] = {}

    def check(self, stage: str, cost: dict[str, int]) -> dict[str, Any]:
        limits = self.policy.get(stage, self.policy.get("program", {}))
        over = []
        for k, v in (cost or {}).items():
            lim = limits.get(k)
            if lim is not None and self._used.get(k, 0) + int(v) > int(lim):
                over.append(f"{k}:{self._used.get(k, 0)}+{v}>{lim}")
        return {"ok": not over, "over": over}

    def consume(self, cost: dict[str, int]) -> None:
        for k, v in (cost or {}).items():
            self._used[k] = self._used.get(k, 0) + int(v)

    def snapshot(self) -> dict[str, int]:
        return dict(self._used)


class ResearchProgramDriver:
    def __init__(self, *, root: str | Path, program_id: str,
                 objective: dict[str, Any], domain_scope: str,
                 research_families: list[str],
                 resource_policy: dict[str, Any] | None = None,
                 approval_policy: dict[str, Any] | None = None,
                 round_fn: RoundFn | None = None,
                 max_rounds: int = 3,
                 diagnosis_fn: Callable[..., dict[str, Any]] | None = None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state = ResearchProgramState(
            root=self.root, program_id=program_id, objective=objective,
            domain_scope=domain_scope,
            resource_policy=resource_policy or {},
            approval_policy=approval_policy or {})
        self.families = list(research_families)
        self.max_rounds = int(max_rounds)
        self.round_fn = round_fn or self._default_round_fn
        self.diagnosis_fn = diagnosis_fn
        self.budget = ProgramBudgetManager(resource_policy or {})
        self.history = ResearchHistoryIndex(program_state=self.state)

    # ----------------------------------------------------------- planning
    def _default_round_fn(self, **kwargs):  # pragma: no cover - explicit inject
        raise ProgramStateError(
            "no round_fn injected: pass a DiscoveryCoordinator-backed callable")

    def _plan_next(self) -> dict[str, Any]:
        """Evidence-backed next-round plan (NOT a hardcoded script)."""
        digest = self.history.planning_digest(root=self.root)
        n = len(self.state.obj["rounds"])
        used = {r.get("family_id") for r in self.state.obj["rounds"]}
        remaining = [f for f in self.families if f not in used]
        if n >= self.max_rounds or not remaining:
            return {"plan": None, "digest": digest,
                    "reason": "objective_or_budget_or_families_exhausted"}
        parent = self.state.last_round()
        family = remaining[0]
        round_id = f"{self.state.program_id}_r{n + 1}"
        objective_text = str(self.state.obj["objective"].get("question", ""))
        # Memory refs from PRIOR rounds are injected only when applicable;
        # the round engine records whether they were used (WP-06 attribution).
        memory_refs = self.history.memory_refs()
        rec = new_round_record(
            program_id=self.state.program_id, round_id=round_id,
            family_id=family,
            parent_round_id=(parent or {}).get("round_id", ""),
            objective=objective_text)
        rec["input_memory_refs"] = memory_refs[:8]
        rec["planning_digest_sha"] = digest["digest_sha"]
        rec["planning_reason"] = {
            "verdicts": digest["digest"]["verdicts"],
            "untested_gaps_count": len(digest["digest"]["untested_gaps"])}
        return {"plan": rec, "digest": digest, "reason": "planned"}

    # --------------------------------------------------------------- run
    def run(self) -> dict[str, Any]:
        if self.state.status in ("PROGRAM_COMPLETED", "FAILED", "BLOCKED"):
            return {"ok": True, "resumed": True, "status": self.state.status,
                    "program_id": self.state.program_id}

        if self.state.status == "PROGRAM_CREATED":
            self.state.transition("OBJECTIVE_VALIDATED",
                                  reason="objective_in_bounds")
            self.state.transition("RESEARCH_QUEUED", reason="initial_queue")

        rounds_report = []
        while True:
            plan = self._plan_next()
            if plan["plan"] is None:
                self._finish(plan["reason"])
                break
            rec = plan["plan"]
            # Budget gate BEFORE any round work (no silent overrun).
            est_cost = {"rounds": 1,
                        "experiments": int(rec.get("estimated_experiments", 2))}
            chk = self.budget.check("round", est_cost)
            if not chk["ok"]:
                self.state.transition("BLOCKED",
                                      reason="budget_overrun:" + ",".join(chk["over"]))
                break

            if self.state.status == "RECOVERY_REQUIRED":
                self.state.transition("ROUND_RUNNING", reason="resume_round")
            else:
                self.state.transition("ROUND_RUNNING", reason="start_round")
            self.state.add_round(rec)

            result = self.round_fn(
                program_id=self.state.program_id, round=dict(rec),
                planning_digest=plan["digest"], root=str(self.root))
            # Record round outcome from REAL result; never invent a verdict.
            verdict = str(result.get("scientific_verdict", "INCONCLUSIVE"))
            exec_status = str(result.get("status", "NOT_RUN"))
            rec_updates = {
                "state": "COMPLETED" if verdict not in ("INCONCLUSIVE",) else "INCONCLUSIVE",
                "research_question_ref": result.get("question_id", ""),
                "hypothesis_refs": result.get("hypothesis_ids", []),
                "preregistration_sha": result.get("preregistration_sha", ""),
                "approval_ref": result.get("approval_id", ""),
                "method_sha": result.get("method_sha", ""),
                "harness_sha": result.get("harness_sha", ""),
                "dataset_sha": result.get("dataset_sha", ""),
                "campaign_refs": result.get("campaign_ids", []),
                "evidence_refs": result.get("evidence_refs", []),
                "claim_refs": result.get("claim_ids", []),
                "scientific_verdict": verdict,
                "execution_status": exec_status,
                "budget_used": result.get("budget_used", {}),
                "trace_ref": result.get("trace_ref", ""),
                "round_manifest_sha": sha256_json(result),
            }
            self.state.update_round(rec["round_id"], **rec_updates)
            self.budget.consume(est_cost)
            self.budget.consume(result.get("budget_used", {}))
            rounds_report.append({"round_id": rec["round_id"],
                                  "verdict": verdict,
                                  "status": exec_status})

            # Scientific INCONCLUSIVE is NOT an engineering FAILED.
            if self.state.status == "ROUND_RUNNING":
                self.state.transition("ROUND_COMPLETED", reason="round_executed")
            self.state.transition("MEMORY_COMMIT", reason="commit_verified_memory")

            # Optional bottleneck diagnosis from REAL evidence (WP-07 seam).
            if self.diagnosis_fn is not None:
                diag = self.diagnosis_fn(
                    program_id=self.state.program_id,
                    history=self.history.planning_digest(root=self.root),
                    root=str(self.root))
                freeze_json(diag, self.root / f"{rec['round_id']}_diagnosis.json")
                self.state.transition("METHOD_DIAGNOSIS", reason="evidence_driven_diagnosis")
                if diag.get("candidate"):
                    self.state.transition("CANDIDATE_EVALUATION",
                                          reason="candidate_from_real_failure")

            self.state.transition("NEXT_ROUND_PLANNING", reason="from_prior_artifacts")
            # loop back; _plan_next decides stop vs next round
            if self.state.status == "NEXT_ROUND_PLANNING":
                self.state.transition("RESEARCH_QUEUED", reason="next_round")

        report = {
            "ok": True, "program_id": self.state.program_id,
            "status": self.state.status, "rounds": rounds_report,
            "verdicts": self.history.verdict_summary(),
            "budget": self.budget.snapshot(),
            "integrity": self.state.verify_integrity(),
            "program_manifest_sha": self.state.obj.get("program_manifest_sha", "")}
        freeze_json(report, self.root / "program_report.json")
        return report

    def _finish(self, reason: str) -> None:
        verdicts = self.history.verdict_summary()
        if verdicts and all(v == "REFUTED" for v in verdicts):
            self.state.transition("INCONCLUSIVE",
                                  reason="all_rounds_refuted:" + reason)
        else:
            self.state.transition("PROGRAM_COMPLETED", reason=reason)


__all__ = ["ResearchProgramDriver", "ProgramBudgetManager"]
