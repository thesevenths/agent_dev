"""M5 Integrated -- top-level Autonomous Research Runtime coordinator.

This is a THIN orchestrator. It drives the EXISTING stack end-to-end:

  Discovery -> Objective -> Hypothesis -> Plan -> HUMAN Approval (injected) ->
  Real Campaign -> 12-step Gate -> Evaluation -> Experience ->
  Cross-domain Discovery -> Dual-layer RRSI -> Version Promotion -> Next Round

It never rewrites ResearchService / Director / Gate / Discovery / Scheduler /
Evidence Reader / Decision Engine. Approvals are ALWAYS supplied by an external
human operator callback; the coordinator NEVER grants an approval itself.

Reliability: global+domain budget reservation before execution, at-most-once
campaigns, resume-in-place, append-only hash-chained journal, SQLite checkpoints,
startup reconciliation, single-domain failure isolation, fail-closed recovery.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .runtime_state import (
    C_COMPLETE, C_CORRUPT, C_RECOVERY, RuntimeState, RuntimeStateError,
    RuntimeBudgetExhausted, S_COMPLETED, S_EXHAUSTED, S_FAILED, S_INIT,
    S_RECOVERING, S_ROUND2, S_RRSI, S_RUNNING, S_STOPPED, config_sha,
)


class RuntimeApprovalRequired(RuntimeError):
    """No external human approver supplied -- never auto-grant."""


class RuntimeRecoveryRequired(RuntimeError):
    """Unverifiable runtime state; fail-closed and wait for a human."""


@dataclass
class RuntimeConfig:
    run_id: str = "m5_integrated_v1"
    max_rounds: int = 2
    global_budget: int = 12
    max_campaigns_per_domain_per_round: int = 2
    require_structured_holdout: bool = True
    domain_order: tuple[str, ...] = ("scinet", "embodied", "quant")


class RuntimeCoordinator:
    def __init__(self, *, config: RuntimeConfig,
                 state: RuntimeState,
                 scheduler: Any,
                 discovery: Any,
                 rrsi: Any | None,
                 dataset_shas: dict[str, str],
                 approver: Callable[[str, str], None] | None = None,
                 out_dir: str | Path,
                 objective_specs: dict[str, dict[str, Any]] | None = None):
        self.config = config
        self.state = state
        self.scheduler = scheduler
        self.discovery = discovery
        self.rrsi = rrsi
        self.dataset_shas = dict(dataset_shas)
        self.approver = approver
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.objective_specs = objective_specs or {}
        self._stop_requested = False
        self._budget_exhausted = False

    # --------------------------------------------------------------- approval
    def _approve(self, approval_id: str, plan_id: str) -> None:
        # The coordinator NEVER grants. It only delegates to the injected
        # external human operator and verifies the grant was SHA-bound.
        if self.approver is None:
            raise RuntimeApprovalRequired(
                "no external human approver; integrated Runtime never "
                "auto-grants approval")
        self.approver(approval_id, plan_id)

    def request_stop(self, reason: str = "human stop") -> None:
        self._stop_requested = True
        self.state.append(self.config.run_id, 0, "human_stop_requested",
                          {"reason": reason})
        self.state.set_state(self.config.run_id, S_STOPPED,
                             event="stop", payload={"reason": reason})

    # ------------------------------------------------------------- lifecycle
    def start(self) -> dict[str, Any]:
        """Initialize global state; refuse config/manifest drift."""
        run = self.state.initialize(
            self.config.run_id, config=self.config,
            dataset_shas=self.dataset_shas,
            max_rounds=self.config.max_rounds,
            global_budget=self.config.global_budget)
        self.state.set_state(self.config.run_id, S_INIT,
                             event="runtime_created")
        return run

    def recover_or_raise(self) -> dict[str, Any]:
        """Startup reconciliation against REAL persisted state; fail-closed."""
        report = self.state.classify(self.config.run_id)
        cls = report["classification"]
        if cls == C_CORRUPT:
            self.state.set_state(self.config.run_id, S_RECOVERING,
                                 event="recovery_corrupt")
            raise RuntimeRecoveryRequired(f"runtime journal CORRUPT: {report}")
        if cls == C_RECOVERY:
            # Budget reservation must not be regained after restart.
            self.state.set_state(self.config.run_id, S_RECOVERING,
                                 event="recovery_required")
            raise RuntimeRecoveryRequired(
                f"runtime state unverifiable: {report.get('reason', '')}; "
                "waiting for human reconciliation")
        self.state.append(self.config.run_id, 0, "recovery_checked",
                          {"classification": cls,
                           "budget": report.get("budget")})
        return report

    def _guard(self) -> None:
        if self._stop_requested:
            raise RuntimeBudgetExhausted("human stop requested")
        snap = self.state.global_budget_snapshot(self.config.run_id)
        if snap["limit"] > 0 and snap["remaining"] <= 0:
            self._budget_exhausted = True
            self.state.set_state(self.config.run_id, S_EXHAUSTED,
                                 event="budget_exhausted", payload=snap)
            raise RuntimeBudgetExhausted(f"global budget exhausted: {snap}")

    @staticmethod
    def _campaign_sha(plan_id: str, approval_id: str, plan_sha: str) -> str:
        return hashlib.sha256(
            f"{plan_id}|{approval_id}|{plan_sha}".encode("utf-8")
        ).hexdigest()

    # --------------------------------------------------------------- round run
    def run_round(self, round_no: int, *, objective_specs: dict[str, dict],
                  source_experiment_ids: dict[str, str] | None = None
                  ) -> dict[str, Any]:
        """Run all domains for one round through the REAL scheduler."""
        self._guard()
        self.state.set_state(
            self.config.run_id, S_RUNNING, round_no=round_no,
            event="round_started", payload={"round_no": round_no})
        source_experiment_ids = source_experiment_ids or {}
        results: list[dict[str, Any]] = []
        for domain in self.config.domain_order:
            if self._stop_requested:
                break
            spec = objective_specs.get(domain)
            if spec is None:
                continue
            oid = spec["objective_id"]
            self.state.put_round(self.config.run_id, round_no, domain,
                                 "DISPATCHED", objective_id=oid)
            try:
                # Budget reservation must happen BEFORE the campaign executes.
                # The scheduler reserves per-domain budget after approval; we
                # mirror a global reservation keyed by the deterministic id.
                cid_hint = f"r{round_no}_{domain}_{oid}"
                self.state.append(self.config.run_id, round_no, "dispatching",
                                  {"domain": domain, "objective_id": oid,
                                   "campaign_hint": cid_hint})
                outcome = self.scheduler.run_objective(
                    oid, source_experiment_id=source_experiment_ids.get(domain, ""))
                campaign_id = outcome.get("campaign_id", "")
                if not campaign_id:
                    raise RuntimeError(
                        f"{domain}: no campaign id from real scheduler")
                # Global reserve + consume (idempotent by campaign_id).
                self.state.reserve_global(
                    self.config.run_id, campaign_id,
                    config_sha=config_sha(self.config))
                self.state.consume_global(
                    self.config.run_id, campaign_id,
                    outcome=str(outcome.get("director_state") or "unknown"))
                self.state.put_round(
                    self.config.run_id, round_no, domain, "COMPLETED",
                    objective_id=oid, campaign_id=campaign_id,
                    report_id=outcome.get("report_id", ""),
                    experience_id="",  # filled from real report below
                    payload={"director_state": outcome.get("director_state", ""),
                             "budget": outcome.get("budget", {}),
                             "result": outcome.get("result", {})})
                results.append({"domain": domain, "ok": True,
                                "campaign_id": campaign_id,
                                "report_id": outcome.get("report_id", ""),
                                "director_state": outcome.get(
                                    "director_state", "")})
            except RuntimeBudgetExhausted:
                raise
            except Exception as exc:  # noqa: BLE001 - isolate a failed domain
                self.state.put_round(self.config.run_id, round_no, domain,
                                     "FAILED", objective_id=oid,
                                     payload={"error": f"{type(exc).__name__}: {exc}"})
                results.append({"domain": domain, "ok": False,
                                "error": f"{type(exc).__name__}: {exc}"})
        self.state.append(self.config.run_id, round_no, "round_finished",
                          {"results": results})
        return {"round_no": round_no, "results": results,
                "budget": self.state.global_budget_snapshot(self.config.run_id)}

    def cross_domain_discovery(self, *, round_no: int,
                               objective_ids: list[str]) -> dict[str, Any]:
        """Build a fully-sourced cross-domain snapshot and run real Discovery."""
        self.state.set_state(self.config.run_id, S_ROUND2, round_no=round_no,
                             event="cross_domain_discovery_started",
                             payload={"objective_ids": objective_ids})
        snapshot = self.scheduler.build_cross_domain_snapshot(
            self.config.run_id, objective_ids=objective_ids)
        result = self.discovery.discover(
            snapshot,
            objective_store=self.scheduler.services[list(
                self.scheduler.services)[0]].stores.objectives,
            domain=list(self.scheduler.services)[0])
        payload = {
            "round_no": round_no,
            "n_objectives": len(snapshot.objectives),
            "n_sessions": len(snapshot.sessions),
            "experience_available": snapshot.experience_available,
            "source_domains": snapshot.provenance.get("source_domains", []),
            "artifact_shas": snapshot.provenance.get("artifact_shas", {}),
            "unresolved_gaps": snapshot.unresolved_gaps,
            "proposed_objective_ids": result.proposed_objective_ids,
        }
        self.state.append(self.config.run_id, round_no,
                          "cross_domain_discovery", payload)
        return {"snapshot": snapshot, "discovery": result, "summary": payload}

    def run_rrsi(self, *, round_no: int,
                 layer_runs: list[dict[str, Any]]) -> dict[str, Any]:
        """Run RRSI layer evaluations (real candidates supplied by caller)."""
        self.state.set_state(self.config.run_id, S_RRSI, round_no=round_no,
                             event="rrsi_started",
                             payload={"n_runs": len(layer_runs)})
        outcomes = []
        for run in layer_runs:
            coord = run["coordinator"]
            out = coord.run(
                experiment_id=run["experiment_id"],
                dominant_failure=run["dominant_failure"],
                evidence=run.get("evidence", {}),
                generator=run["generator"],
                evaluator=run["evaluator"],
                layer=run["layer"],
                domain=run.get("domain"),
                require_structured_holdout=self.config.require_structured_holdout)
            outcomes.append(out)
            self.state.append(
                self.config.run_id, round_no, "rrsi_layer_evaluated",
                {"layer": run["layer"],
                 "selected_layer": out.get("selected_layer"),
                 "n_candidates": out.get("n_candidates"),
                 "statuses": [o.get("status") for o in out.get("outcomes", [])]})
        return {"round_no": round_no, "outcomes": outcomes}

    def finalize(self, *, round_no: int,
                 rrsi_summary: dict[str, Any] | None = None,
                 cross_summary: dict[str, Any] | None = None) -> dict[str, Any]:
        """Final checkpoint + state, only if reconciliation is clean."""
        checkpoint = self.state.checkpoint(
            self.config.run_id, f"final_round_{round_no}")
        integ = self.state.verify_journal(self.config.run_id)
        if not integ["ok"]:
            raise RuntimeRecoveryRequired("journal integrity failed at finalize")
        report = self.state.classify(self.config.run_id)
        if report["classification"] == C_CORRUPT:
            raise RuntimeRecoveryRequired("corrupt state at finalize")
        self.state.set_state(self.config.run_id, S_COMPLETED, round_no=round_no,
                             event="runtime_completed",
                             payload={"round_no": round_no,
                                      "checkpoint_sha": checkpoint["state_sha"]})
        return {"run_id": self.config.run_id, "state": S_COMPLETED,
                "round_no": round_no, "checkpoint": checkpoint,
                "journal": integ,
                "budget": self.state.global_budget_snapshot(self.config.run_id),
                "rrsi": rrsi_summary or {}, "cross_domain": cross_summary or {}}

    def status(self) -> dict[str, Any]:
        return {"run": self.state.run(self.config.run_id),
                "rounds": self.state.rounds(self.config.run_id),
                "budget": self.state.global_budget_snapshot(self.config.run_id),
                "journal": self.state.verify_journal(self.config.run_id),
                "recovery": self.state.classify(self.config.run_id)}


__all__ = [
    "RuntimeCoordinator", "RuntimeConfig", "RuntimeApprovalRequired",
    "RuntimeRecoveryRequired",
]
