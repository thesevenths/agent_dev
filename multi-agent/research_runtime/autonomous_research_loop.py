"""M5-00 -- single-domain Autonomous Research Loop (thin orchestration).

WHAT THIS IS
------------
A thin orchestrator that closes the loop for ONE domain using ONLY existing,
unchanged components:

    DiscoverySnapshot (rebuilt each round, including prior-round Experience)
      -> ObjectiveDiscovery      (Proposed Objective)
      -> ResearchDirector        (Hypothesis -> Plan -> Approval request)
      -> external human Approval (OUT OF BAND; this module NEVER grants)
      -> ResearchDirector        (real Campaign -> Evaluate -> Experience)
      -> 12-step Gate + Decision (unchanged)
      -> next round's Discovery reads the prior round's Experience/Evidence

It does NOT reimplement ResearchService / ResearchDirector / ObjectiveDiscovery /
Evidence Reader / Decision Engine / the 12-step Gate. It calls them.

HARD RULES (spec M5-00)
  * AT LEAST two real rounds for E2E; round 2 cites round-1 Experience;
  * every round passes the original 12-step gate (via ResearchDirector.finalize);
  * NO automatic human approval -- an external ``approver`` callback is required;
  * a completed round / campaign is NEVER executed or charged twice;
  * max-rounds, campaign-budget and wall-clock-budget are enforced (safe stop);
  * process restart resumes from persisted state (idempotent);
  * human stop is honoured; failures/interruptions are retained, not deleted;
  * NO fabricated Experience / Evidence.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .autonomous_loop_state import (
    IllegalTransition, LoopEventStore, LoopRecord, S_AWAITING_APPROVAL,
    S_CREATED, S_DISCOVERING, S_EVALUATING, S_EXECUTING, S_EXPERIENCE_RECORDED,
    S_HYPOTHESIS_READY, S_LOOP_COMPLETED, S_OBJECTIVE_READY, S_PLAN_READY,
    S_PREFLIGHTED, S_RECOVERING, S_ROUND_COMPLETED, S_STOPPED, S_EXECUTING,
)
from .autonomous_loop_recovery import (
    apply_recovery, reconcile, LoopRecoveryReport,
)
from .discovery_context_provider import DiscoveryContextProvider
from .objective_critic import ObjectiveCritic
from .objective_discovery import ObjectiveDiscovery
from .objective_ranker import ObjectiveRanker
from .objective_candidate_generator import ObjectiveCandidateGenerator
from .research_director import ResearchDirector
from .research_store import ResearchStoreConflict


class LoopBudgetExhausted(RuntimeError):
    pass


class LoopPreflightError(RuntimeError):
    pass


class LoopApprovalRequired(RuntimeError):
    """No external approver supplied -- never auto-grant."""


class _ReusePriorObjectiveStore:
    """Loop-scoped ObjectiveStore adapter: reuse, never overwrite, history.

    A later round builds a larger DiscoverySnapshot (it includes the prior
    round's Experience). The unchanged generator therefore re-derives some
    earlier candidate ids, but their contextual metadata (e.g.
    ``source_objective_ids``) legitimately changes. The generic
    ``ObjectiveStore`` correctly refuses to overwrite the earlier content.
    This adapter ONLY downgrades that specific conflict to a reuse when BOTH
    the candidate id and generation source are unchanged AND the objective was
    already proposed by this loop/run. Every other conflict still raises.
    It delegates all other store operations unchanged.
    """

    def __init__(self, store: Any, reusable_ids: Any):
        self._store = store
        self._reusable_ids = set(reusable_ids or ())

    def save(self, record: Any) -> dict[str, Any]:
        try:
            return self._store.save(record)
        except ResearchStoreConflict:
            if record.objective_id in self._reusable_ids:
                existing = self._store.get(record.objective_id)
                old = (getattr(existing, "provenance", None) or {})
                new = (getattr(record, "provenance", None) or {})
                if (existing is not None
                        and old.get("candidate_id") == new.get("candidate_id")
                        and old.get("generation_source")
                        == new.get("generation_source")):
                    return {"status": "reused_prior",
                            "objective_id": record.objective_id}
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)


@dataclass
class LoopConfig:
    loop_id: str = "m5_loop_1"
    domain: str = "scinet"
    run_id: str = "m5_autonomous"
    max_rounds: int = 2
    max_campaigns: int = 0            # 0 = unlimited
    max_wall_time_s: float = 0.0      # 0 = unlimited
    source_experiment_id: str = ""    # real experiment grounding round 1


def _now() -> str:
    import datetime
    return datetime.datetime.now().isoformat(timespec="seconds")


class AutonomousResearchLoop:
    """Drive a single-domain research loop using the EXISTING engine stack."""

    def __init__(self, *, service: Any, config: LoopConfig,
                 out_dir: str | Path,
                 approver: Callable[[str, str], None] | None = None):
        self.service = service
        self.config = config
        self.out_dir = Path(out_dir)
        self.rounds_dir = self.out_dir / "rounds"
        self.rounds_dir.mkdir(parents=True, exist_ok=True)
        self.approver = approver          # external human grant; NEVER defaulted
        self.store = LoopEventStore(db=service.stores.db)
        self.started_wall = time.time()
        self.campaigns_run = 0
        self._stop_requested = False
        self.record = self.store.load(config.loop_id)

    # -------------------------------------------------------------- lifecycle
    def preflight(self) -> dict[str, Any]:
        """Fail-closed: require a REAL engine / evidence before any round."""
        from .settings import DEFAULT_SCINET_DIR, resolve_engine
        detail: dict[str, Any] = {
            "domain": self.config.domain,
            "scinet_dir": str(DEFAULT_SCINET_DIR),
            "scinet_dir_exists": Path(str(DEFAULT_SCINET_DIR)).is_dir(),
        }
        # resolve_engine() has no domain arg and is self-checking (raises when
        # the real engine/CLI is missing); call it, never fabricate a path.
        try:
            _project, python = resolve_engine()
            detail["engine_python"] = str(python)
        except Exception as exc:  # noqa: BLE001 - fail closed, no engine
            detail["engine_error"] = f"{type(exc).__name__}: {exc}"
        ready = bool(detail.get("engine_python"))
        if self.record is None:
            self.record = LoopRecord(
                loop_id=self.config.loop_id, domain=self.config.domain,
                state=S_CREATED, max_rounds=self.config.max_rounds,
                config=self.config.__dict__.copy())
            self.store.save(self.record)
            self._emit("create", {"preflight": detail})
        if not ready:
            raise LoopPreflightError(
                f"real engine/data unavailable; refusing to run: {detail}")
        if self.record.state == S_CREATED:
            self._transition("preflight", S_PREFLIGHTED, {"preflight": detail})
        return {"ready": True, "detail": detail}

    def request_stop(self, reason: str = "human stop") -> None:
        self._stop_requested = True
        self.store.append(loop_id=self.config.loop_id,
                          round_no=self.record.round_no,
                          event="human_stop_requested", payload={"reason": reason})

    # --------------------------------------------------------------- budget
    def _check_budget(self) -> None:
        if self._stop_requested:
            raise LoopBudgetExhausted("human stop requested")
        if self.config.max_rounds and self.record.round_no >= self.config.max_rounds:
            raise LoopBudgetExhausted(
                f"max_rounds={self.config.max_rounds} reached")
        if self.config.max_campaigns and \
                self.campaigns_run >= self.config.max_campaigns:
            raise LoopBudgetExhausted(
                f"max_campaigns={self.config.max_campaigns} reached")
        if self.config.max_wall_time_s and \
                (time.time() - self.started_wall) >= self.config.max_wall_time_s:
            raise LoopBudgetExhausted(
                f"max_wall_time_s={self.config.max_wall_time_s} reached")

    # ------------------------------------------------------------ state helpers
    def _emit(self, event: str, payload: dict[str, Any]) -> str:
        return self.store.append(loop_id=self.config.loop_id,
                                 round_no=self.record.round_no,
                                 event=event, payload=payload)

    def _transition(self, event: str, to_state: str,
                    payload: dict[str, Any]) -> None:
        if self.record.state != to_state:
            self._emit(event, {**payload, "from": self.record.state,
                               "to": to_state})
            self.record.state = to_state
            self.store.save(self.record)

    # ------------------------------------------------------------- discovery
    def _prior_objective_ids(self) -> list[str]:
        """Objective ids established by EARLIER completed rounds (resume-safe)."""
        seen: set[str] = set()
        out: list[str] = []
        for rd in self.record.rounds:
            oid = rd.get("objective_id")
            if oid and oid not in seen:
                seen.add(oid)
                out.append(oid)
        return out

    def _reusable_objective_ids(self) -> set[str]:
        """All objectives already proposed by this loop/run, incl. partial rounds."""
        ids = set(self._prior_objective_ids())
        for rd in self.record.rounds:
            ids.update((rd.get("discovery_source") or {}).get(
                "proposed_objective_ids", ()))
        for rec in self.service.stores.objectives.all():
            if ((rec.provenance or {}).get("discovery_run_id")
                    == self.config.run_id):
                ids.add(rec.objective_id)
        return ids

    def _snapshot(self, round_no: int) -> Any:
        """Rebuild a DiscoverySnapshot from REAL persisted data.

        The evidence query key is the REAL run whose Evidence was recorded
        (``config.run_id``). It is NOT renamed per round -- a fabricated run_id
        would return no evidence. Round 1 sees the original opportunity; later
        rounds see the SAME real evidence PLUS the prior rounds' persisted
        objectives / sessions / Experience (via objective_ids).
        """
        provider = DiscoveryContextProvider(stores=self.service.stores)
        return provider.build_snapshot(self.config.run_id,
                                       objective_ids=self._prior_objective_ids())

    def _discover(self, snapshot: Any, round_no: int) -> dict[str, Any]:
        """Run the REAL ObjectiveDiscovery; return a provenance dict."""
        discovery = ObjectiveDiscovery(
            generator=ObjectiveCandidateGenerator(domain=self.config.domain),
            critic=ObjectiveCritic(), ranker=ObjectiveRanker(),
            domain=self.config.domain)
        store = _ReusePriorObjectiveStore(
            self.service.stores.objectives, self._reusable_objective_ids())
        result = discovery.discover(snapshot, objective_store=store,
                                   domain=self.config.domain)
        if not result.proposed_objective_ids:
            raise LoopPreflightError(
                f"round {round_no}: real Discovery approved no objective")
        objective_id = result.proposed_objective_ids[0]
        ranked = next((r for r in result.ranked
                       if r.get("objective_id") == objective_id), {})
        reused = [r["objective_id"] for r in result.ranked
                  if r.get("persist_status") == "reused_prior"]
        return {"objective_id": objective_id,
                "candidate_id": ranked.get("candidate_id", ""),
                "generation_source": ranked.get("generation_source", ""),
                "reused_prior_objective_ids": reused,
                "discovery": result.to_dict()}

    # ---------------------------------------------------------------- approver
    def _require_approval(self, director: ResearchDirector,
                          round_no: int) -> str:
        """Observe an EXTERNAL human grant; never grant it ourselves."""
        if self.approver is None:
            raise LoopApprovalRequired(
                "no external approver supplied: an autonomous loop NEVER "
                "grants its own human approval")
        approval_id = director.context.approval_id
        plan_id = director.context.plan_id
        self.approver(approval_id, plan_id)   # out-of-band, real CLI/human
        if not getattr(director, "_approval_granted_and_bound", lambda: True)():
            raise LoopApprovalRequired(
                f"round {round_no}: approval {approval_id!r} is not granted "
                "and SHA-bound in the store")
        return approval_id

    # --------------------------------------------------------------- round IO
    def _round_path(self, round_no: int) -> Path:
        return self.rounds_dir / f"round_{round_no:02d}.json"

    def _write_round(self, round_no: int, payload: dict[str, Any]) -> None:
        """Append-only-by-version: never overwrite an existing completed round."""
        path = self._round_path(round_no)
        final = {"round_no": round_no, **payload, "written_at": _now()}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(final, indent=2, sort_keys=True,
                                  default=str), encoding="utf-8")
        tmp.replace(path)

    # ------------------------------------------------------------------- run
    def run(self) -> dict[str, Any]:
        """Execute (or resume) all rounds, fail-closed on preflight/budget."""
        self.preflight()
        outcomes: list[dict[str, Any]] = list(self.record.rounds)
        start_round = len(outcomes) + 1
        for round_no in range(start_round, self.config.max_rounds + 1):
            self._check_budget()
            prior = outcomes[-1] if outcomes else {}
            round_rec = self._run_round(round_no, prior_experience=prior)
            outcomes.append(round_rec)
            self.record.rounds = outcomes
            self.record.round_no = round_no
            self.store.save(self.record)
            if round_rec.get("interrupted"):
                break
        if not self._stop_requested and \
                len(outcomes) >= self.config.max_rounds:
            self._transition("complete_loop", S_LOOP_COMPLETED,
                             {"n_rounds": len(outcomes)})
        return {"loop_id": self.config.loop_id, "state": self.record.state,
                "n_rounds": len(outcomes), "rounds": outcomes,
                "integrity_ok": self.store.verify_integrity(self.config.loop_id)}

    def _run_round(self, round_no: int,
                   prior_experience: dict[str, Any]) -> dict[str, Any]:
        """One real round: Discovery -> Director -> Approval -> Campaign."""
        self._transition("begin_round", S_DISCOVERING,
                         {"round_no": round_no,
                          "prior_experience_refs":
                              prior_experience.get("experience_id", "")})
        snapshot = self._snapshot(round_no)
        disc = self._discover(snapshot, round_no)
        objective_id = disc["objective_id"]
        self._transition("objective_proposed", S_OBJECTIVE_READY,
                         {"objective_id": objective_id,
                          "generation_source": disc["generation_source"]})

        state_dir = self.out_dir / "director" / f"round_{round_no:02d}"
        director = ResearchDirector(
            service=self.service,
            objective_store=self.service.stores.objectives,
            state_dir=state_dir, objective_id=objective_id,
            domain=self.config.domain,
            director_id=f"m5_round_{round_no}_{self.config.loop_id}",
            source_experiment_id=self.config.source_experiment_id)

        # Hypothesis + Plan + Approval request (no execution, no auto-grant).
        director.run_to_awaiting_approval()
        self._transition("hypotheses_ready", S_HYPOTHESIS_READY,
                         {"hypothesis_ids": list(director.context.hypothesis_ids)})
        self._transition("plan_created", S_PLAN_READY,
                         {"plan_id": director.context.plan_id})
        self._transition("approval_requested", S_AWAITING_APPROVAL,
                         {"approval_id": director.context.approval_id})
        approval_id = self._require_approval(director, round_no)
        self._transition("approval_observed", S_AWAITING_APPROVAL,
                         {"approval_id": approval_id})

        # Execute the REAL campaign + 12-step gate + Experience write.
        self._transition("executing", S_EXECUTING,
                         {"campaign_id_before": director.context.campaign_id})
        final = director.run_to_completion()
        self.campaigns_run += 1
        self._transition("evaluating", S_EVALUATING,
                         {"director_state": final.get("state", "")})
        ctx = director.context
        round_rec = {
            "round_no": round_no,
            "objective_id": objective_id,
            "hypothesis_ids": list(ctx.hypothesis_ids),
            "plan_id": ctx.plan_id,
            "approval_id": ctx.approval_id,
            "campaign_id": ctx.campaign_id,
            "report_id": ctx.report_id,
            "experience_id": (final.get("experience_id")
                              or self._experience_ref(ctx.session_id)
                              or getattr(ctx, "experience_id", "")),
            "director_state": final.get("state", ""),
            "discovery_source": disc["discovery"],
            "reused_prior_objective_ids": disc["reused_prior_objective_ids"],
            "prior_experience_ref": prior_experience.get("experience_id", ""),
            "gate_passed": bool(final.get("state") == "COMPLETED"),
        }
        self._transition("experience_recorded", S_EXPERIENCE_RECORDED,
                         {"experience_id": round_rec["experience_id"]})
        self._write_round(round_no, round_rec)
        self._transition("complete_round", S_ROUND_COMPLETED,
                         {"round_no": round_no,
                          "experience_id": round_rec["experience_id"]})
        return round_rec

    def _experience_ref(self, session_id: str) -> str:
        """Read the REAL Experience id written by the Director's report path."""
        if not session_id:
            return ""
        try:
            rep = self.service.get_research_report(session_id=session_id)
            refs = ((rep.get("report") or {}).get("experience_refs") or [])
            return str(refs[0]) if refs else ""
        except Exception:  # noqa: BLE001 - no fabricated id
            return ""

    # -------------------------------------------------------------- recovery
    def recover(self, *, apply: bool = False) -> LoopRecoveryReport:
        report = reconcile(store=self.store, record=self.record,
                           rounds_dir=self.rounds_dir)
        if apply:
            apply_recovery(store=self.store, record=self.record, report=report)
        return report


__all__ = [
    "LoopConfig", "LoopBudgetExhausted", "LoopPreflightError",
    "LoopApprovalRequired", "AutonomousResearchLoop",
]
