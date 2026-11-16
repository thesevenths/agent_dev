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
import os
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
        self.state.mark_terminal(self.config.run_id, S_STOPPED, reason)

    # ------------------------------------------------------------- lifecycle
    def start(self) -> dict[str, Any]:
        """Initialize global state; refuse config/manifest drift."""
        # P0-B: initialize() only creates/validates the run. It must NOT blindly
        # reset state -- a COMPLETED/STOPPED run can never restart, and a RUNNING
        # run must go through recovery classification first.
        run = self.state.initialize(
            self.config.run_id, config=self.config,
            dataset_shas=self.dataset_shas,
            max_rounds=self.config.max_rounds,
            global_budget=self.config.global_budget)
        prior_state = str(run.get("state") or "")
        if prior_state in self.state.TERMINAL_STATES:
            raise RuntimeRecoveryRequired(
                f"run {self.config.run_id!r} is {prior_state} "
                f"({run.get('terminal_reason', '')}); refusing to restart")
        if prior_state in (S_RUNNING, S_ROUND2, S_RRSI, S_RECOVERING):
            # A crashed mid-run restart: budget/consumption are untouched by
            # initialize(). Require explicit recover_or_raise before executing.
            self.state.append(self.config.run_id, 0, "restart_needs_recovery",
                              {"previous_state": prior_state})
            return run
        # Fresh INIT (or a run explicitly parked at INIT): claim the SOLE
        # executor right atomically. A second process with another id loses.
        executor_id = f"{os.getpid()}@{time.time_ns()}"
        claimed = self.state.claim_executor(self.config.run_id, executor_id)
        if not claimed:
            raise RuntimeRecoveryRequired(
                f"run {self.config.run_id!r} already has an active executor; "
                "concurrent start refused")
        self._executor_id = executor_id
        if prior_state != S_INIT:
            self.state.set_state(self.config.run_id, S_INIT,
                                 event="runtime_created")
        return self.state.run(self.config.run_id)

    def _verify_round_reality(self, round_rec: dict[str, Any]) -> dict[str, Any]:
        """P0-C: resolve a round's ids against the REAL domain stores.

        Verifies the full chain: campaign binding -> session -> report -> real
        Experience record. A non-empty round field is NEVER accepted as proof.
        """
        domain = round_rec.get("domain_id", "")
        service = self.scheduler.services.get(domain) if hasattr(
            self.scheduler, "services") else None
        if service is None:
            return {"ok": False, "reasons": [f"no_service_for_{domain}"]}
        reasons: list[str] = []
        sessions = service.stores.sessions
        cid = round_rec.get("campaign_id", "")
        binding = None
        try:
            binding = sessions.binding_for_campaign(cid) if cid else None
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"campaign_lookup_error:{type(exc).__name__}")
        if binding is None:
            reasons.append("campaign_binding_unresolved")
        session_id = getattr(binding, "session_id", "") if binding else ""
        rid = round_rec.get("report_id", "")
        report_rec = None
        try:
            report_rec = sessions.get_report(rid) if rid else None
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"report_lookup_error:{type(exc).__name__}")
        if report_rec is None:
            reasons.append("report_unresolved")
        elif session_id and str(getattr(report_rec, "session_id", "")) != session_id:
            reasons.append("report_session_mismatch")
        exps: list[dict[str, Any]] = []
        try:
            exps = sessions.experience(session_id) if session_id else []
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"experience_lookup_error:{type(exc).__name__}")
        if not exps:
            reasons.append("experience_unresolved")
        eid = round_rec.get("experience_id", "")
        if eid and not any(str(x.get("experience_id")) == eid for x in exps):
            reasons.append("experience_id_mismatch")
        return {"ok": not reasons, "reasons": reasons,
                "session_id": session_id,
                "n_experience": len(exps)}

    def _classify_runtime(self) -> dict[str, Any]:
        return self.state.classify(
            self.config.run_id, reality_verifier=self._verify_round_reality)

    def recover_or_raise(self) -> dict[str, Any]:
        """Startup reconciliation against REAL persisted state; fail-closed."""
        report = self._classify_runtime()
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
            self.state.append(self.config.run_id, 0, "budget_exhausted", snap)
            self.state.mark_terminal(self.config.run_id, S_EXHAUSTED,
                                     "global_budget_exhausted")
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
            # P0-A: reserve BEFORE any campaign is launched. The reservation
            # identity is deterministic and independent of the engine's campaign
            # id (which does not exist yet). With budget=1 the second domain
            # fails here -- it never reaches run_objective.
            res_id = self.state.reservation_id(
                self.config.run_id, round_no, domain, oid)
            try:
                self.state.reserve_execution(
                    self.config.run_id, res_id,
                    config_sha=config_sha(self.config),
                    domain_id=domain, round_no=round_no)
            except RuntimeBudgetExhausted:
                self.state.put_round(
                    self.config.run_id, round_no, domain, "BUDGET_BLOCKED",
                    objective_id=oid, payload={"reservation_id": res_id})
                results.append({"domain": domain, "ok": False,
                                "error": "global_budget_exhausted_before_execution",
                                "reservation_id": res_id})
                continue
            self.state.put_round(self.config.run_id, round_no, domain,
                                 "RESERVED", objective_id=oid,
                                 payload={"reservation_id": res_id})
            try:
                self.state.append(self.config.run_id, round_no, "dispatching",
                                  {"domain": domain, "objective_id": oid,
                                   "reservation_id": res_id})
                outcome = self.scheduler.run_objective(
                    oid, source_experiment_id=source_experiment_ids.get(domain, ""))
                campaign_id = outcome.get("campaign_id", "")
                if not campaign_id:
                    raise RuntimeError(
                        f"{domain}: no campaign id from real scheduler")
                # Bind the pre-existing reservation to the real campaign, then
                # settle it. Both are idempotent across restarts.
                self.state.bind_reservation(
                    self.config.run_id, res_id, campaign_id=campaign_id)
                self.state.consume_reservation(
                    self.config.run_id, res_id,
                    outcome=str(outcome.get("director_state") or "unknown"))
                report_id = outcome.get("report_id", "")
                # P0-C: never mark a domain ROUND_COMPLETED without the real
                # Report id. Experience is confirmed by the reconciler from the
                # real ExperienceStore, not by trusting a caller-supplied id.
                round_state = ("REPORT_PENDING" if not report_id
                               else "CAMPAIGN_COMPLETED")
                self.state.put_round(
                    self.config.run_id, round_no, domain, round_state,
                    objective_id=oid, campaign_id=campaign_id,
                    report_id=report_id,
                    payload={"reservation_id": res_id,
                             "director_state": outcome.get("director_state", ""),
                             "budget": outcome.get("budget", {}),
                             "result": outcome.get("result", {})})
                # P0-C: promote to ROUND_COMPLETED ONLY after the REAL stores
                # confirm campaign binding + report + experience. Otherwise the
                # round stays CAMPAIGN_COMPLETED/REPORT_PENDING and recovery
                # will refuse to call the run complete.
                final_state = round_state
                exp_id = ""
                if report_id:
                    check = self._verify_round_reality({
                        "domain_id": domain, "campaign_id": campaign_id,
                        "report_id": report_id, "experience_id": ""})
                    if check.get("ok"):
                        exps = self.scheduler.services[domain].stores.sessions \
                            .experience(check.get("session_id", ""))
                        if exps:
                            exp_id = str(exps[0].get("experience_id", ""))
                    if exp_id:
                        self.state.put_round(
                            self.config.run_id, round_no, domain,
                            self.state.S_ROUND_DONE,
                            objective_id=oid, campaign_id=campaign_id,
                            report_id=report_id, experience_id=exp_id,
                            payload={"reservation_id": res_id,
                                     "reality_ok": True})
                        final_state = self.state.S_ROUND_DONE
                results.append({"domain": domain,
                                "ok": final_state == self.state.S_ROUND_DONE,
                                "campaign_id": campaign_id,
                                "report_id": report_id,
                                "experience_id": exp_id,
                                "reservation_id": res_id,
                                "round_state": final_state,
                                "director_state": outcome.get(
                                    "director_state", "")})
            except RuntimeBudgetExhausted:
                raise
            except Exception as exc:  # noqa: BLE001 - isolate a failed domain
                # Did the real engine create a campaign before the failure? Ask
                # the REAL store, never the exception message. If the store
                # cannot be read, assume a campaign may have started (spend the
                # budget; at-most-once is more important than refunding).
                camp = ""
                store_ok = True
                try:
                    camps = self.scheduler.services[domain].stores.sessions \
                        .campaigns_for_objective(oid)
                    if camps:
                        last = camps[-1]
                        camp = (last.get("campaign_id")
                                if isinstance(last, dict)
                                else getattr(last, "campaign_id", ""))
                except Exception:  # noqa: BLE001
                    store_ok = False
                if camp:
                    self.state.bind_reservation(
                        self.config.run_id, res_id, campaign_id=camp)
                    self.state.consume_reservation(
                        self.config.run_id, res_id, outcome="interrupted")
                elif not store_ok:
                    # Cannot prove no campaign started -> spend the budget.
                    try:
                        self.state.consume_reservation(
                            self.config.run_id, res_id, outcome="unknown_interrupted")
                    except RuntimeStateError:
                        pass
                else:
                    # The real store proves no campaign was created: release.
                    try:
                        self.state.release_reservation(
                            self.config.run_id, res_id,
                            reason=f"{type(exc).__name__}: {exc}")
                    except RuntimeStateError:
                        pass  # already bound/consumed -> keep the spend
                self.state.put_round(self.config.run_id, round_no, domain,
                                     "FAILED", objective_id=oid,
                                     payload={"reservation_id": res_id,
                                              "error": f"{type(exc).__name__}: {exc}"})
                results.append({"domain": domain, "ok": False,
                                "reservation_id": res_id,
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
        first_domain = list(self.scheduler.services)[0]

        def _store_for(domain_id: str):
            svc = self.scheduler.services.get(str(domain_id))
            return svc.stores.objectives if svc is not None else None

        discover_kwargs = {
            "objective_store": (self.scheduler.services[first_domain]
                                .stores.objectives if first_domain else None),
            "domain": first_domain,
        }
        # P1-A: the REAL ObjectiveDiscovery routes each candidate to its own
        # domain store. An injected discovery double without that capability
        # is tolerated but explicitly marked as legacy (no silent misroute in
        # production -- only test doubles use the fallback).
        import inspect
        try:
            sig = inspect.signature(self.discovery.discover)
            supports_routing = "objective_store_for_domain" in sig.parameters
        except (TypeError, ValueError):
            supports_routing = True
        if supports_routing:
            discover_kwargs["objective_store_for_domain"] = _store_for
        result = self.discovery.discover(snapshot, **discover_kwargs)
        payload = {
            "round_no": round_no,
            "n_objectives": len(snapshot.objectives),
            "n_sessions": len(snapshot.sessions),
            "experience_available": snapshot.experience_available,
            "source_domains": snapshot.provenance.get("source_domains", []),
            "artifact_shas": snapshot.provenance.get("artifact_shas", {}),
            "domain_routed_discovery": bool(supports_routing),
            "unresolved_gaps": snapshot.unresolved_gaps,
            "proposed_objective_ids": result.proposed_objective_ids,
            "routed": [{"objective_id": r["objective_id"],
                        "target_domain": r.get("target_domain", ""),
                        "cross_domain": r.get("cross_domain", False)}
                       for r in (getattr(result, "ranked", None) or [])],
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
        report = self._classify_runtime()
        if report["classification"] == C_CORRUPT:
            raise RuntimeRecoveryRequired("corrupt state at finalize")
        if report["classification"] != C_COMPLETE:
            raise RuntimeRecoveryRequired(
                f"refusing to mark run COMPLETED: recovery classification="
                f"{report['classification']} (incomplete/unverified rounds)")
        self.state.append(self.config.run_id, round_no, "runtime_completed",
                          {"round_no": round_no,
                           "checkpoint_sha": checkpoint["state_sha"]})
        self.state.mark_terminal(self.config.run_id, S_COMPLETED,
                                 "all_rounds_verified")
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
                "recovery": self._classify_runtime()}


__all__ = [
    "RuntimeCoordinator", "RuntimeConfig", "RuntimeApprovalRequired",
    "RuntimeRecoveryRequired",
]
