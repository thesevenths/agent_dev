"""A2/A3/A5: the SERVER-SIDE boundary behind the campaign-level Agent tools.

The LLM-facing tool signature is intentionally tiny::

    run_research_campaign(plan_id="plan_001", approval_id="approval_001")

Everything else is RE-READ here from trusted storage. In particular the
``approved_hypothesis_ids`` list that P1-04 accepted from its caller is now
DERIVED from persisted, hash-bound Critic reviews -- an LLM cannot supply it.

Gate chain (A3), all fail-closed and in this order
--------------------------------------------------
    1. Plan exists + its stored SHA-256 matches its own content
    2. Approval exists, is for THIS plan, is ``approved``
    3. Approval source is an allowed source (human by default)
    4. Approval carries a real approver and the required scope
    5. Approval's resource_sha256 == CURRENT plan hash  (no post-approval edits)
    6. Every candidate's hypothesis exists and has a PASSING Critic review
       bound to the CURRENT hypothesis hash
    7. Candidate overrides == the reviewed intervention
    8. Every override passes the injected config whitelist / value rules
    9. Dataset version is allowed AND verifies on disk
   10. Budget: candidates, seeds, expected jobs, wall time, held-out split
   11. Idempotency: (plan, approval, hash) -> exactly ONE campaign
   12. Research-session cap: at most N campaign rounds per objective

This module never proposes a scientific claim, never edits a config, never
promotes a model and never evaluates on the reserved final split.
"""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .campaign.schemas import CAMPAIGN_TERMINAL_STATUSES, CampaignRunSpec
from .evidence_store import EvidenceStore
from .failure_diagnoser import FailureDiagnoser
from .hypothesis_critic import HypothesisCritic
from .hypothesis_manager import HypothesisManager, RuleBasedProposer
from .research_policy import ResearchDomainPolicy
from .research_store import ResearchStores, plan_sha256
from .research_store.schemas import CampaignBinding, ResearchReport
from .schemas import (
    RESERVED_FINAL_SPLITS,
    ExperimentPlan,
    ExperimentResult,
    FailureReport,
    Hypothesis,
    ResearchObjective,
)


class ResearchGateError(Exception):
    """A rejected research operation with a stable machine-readable code."""

    def __init__(self, code: str, message: str, **details: Any):
        super().__init__(message)
        self.code = code
        self.details = details


def _fail(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"ok": False, "error_code": code, "error": message, **details}


def _json_list(raw: str, label: str) -> list[Any]:
    try:
        value = json.loads(raw) if raw else []
    except (ValueError, TypeError) as exc:
        raise ResearchGateError("invalid_json", f"{label} is not valid JSON: {exc}")
    if not isinstance(value, list):
        raise ResearchGateError("invalid_json", f"{label} must be a JSON array")
    return value


def _canon(value: Any) -> Any:
    """JSON-normalise a value so dict comparisons are not fooled by tuples."""
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                 default=str))


EvidenceProvider = Callable[[str], "tuple[ExperimentResult, list[FailureReport]]"]
DatasetVerifier = Callable[[str, str], "dict[str, Any]"]


def default_evidence_provider(experiment_id: str,
                              evidence: EvidenceStore | None = None
                              ) -> tuple[ExperimentResult, list[FailureReport]]:
    """Rebuild structured evidence from the append-only ledger, not from prose."""
    store = evidence or EvidenceStore()
    events = store.query(experiment_id=experiment_id, event_type="result")
    if not events:
        raise ResearchGateError(
            "evidence_not_found",
            f"no structured result event for experiment {experiment_id!r}; "
            "a hypothesis must be grounded in a recorded experiment")
    payload = (events[-1].get("payload") or {}).get("result") or {}
    try:
        result = ExperimentResult.model_validate(payload)
    except Exception as exc:
        raise ResearchGateError("evidence_invalid",
                                f"stored result is not a valid ExperimentResult: {exc}")
    return result, FailureDiagnoser().diagnose_all(result)


@dataclass
class ResearchDependencies:
    """Everything the service needs, injected so tests can use fakes."""
    stores: ResearchStores
    domain: ResearchDomainPolicy
    campaign_store: Any
    report_store: Any
    runner_factory: Callable[[], Any]
    comparator_factory: Callable[[], Any]
    hypothesis_manager_factory: Callable[[], HypothesisManager] | None = None
    evidence_provider: EvidenceProvider | None = None
    dataset_verifier: DatasetVerifier | None = None
    campaign_root: str = ""
    adapter: Any = None  # ARR 0.1 / R2: the domain's engine adapter (ModelAdapter)


BINDING_NAMESPACE = uuid.UUID("6f2a1c94-7b3d-4e18-9c62-1d8e5a0b7f43")


def make_campaign_id(plan_id: str, approval_id: str, plan_hash: str) -> str:
    """Deterministic idempotency id: the same approval can never spawn 2 runs."""
    digest = uuid.uuid5(BINDING_NAMESPACE,
                        f"{plan_id}|{approval_id}|{plan_hash}").hex[:16]
    return "campaign_" + digest


class ResearchService:
    def __init__(self, deps: ResearchDependencies):
        self.deps = deps
        self._run_lock = threading.RLock()

    # ------------------------------------------------------------- helpers
    @property
    def stores(self) -> ResearchStores:
        return self.deps.stores

    @property
    def domain(self) -> ResearchDomainPolicy:
        return self.deps.domain

    @property
    def adapter(self) -> Any:
        """The domain's engine adapter (ModelAdapter), wired from the bundle.

        ARR 0.1 / R2: the generic runtime never names an engine, but the
        selected domain's adapter is a first-class, observable part of the
        assembled service -- not an unused field on the bundle.
        """
        return self.deps.adapter

    def _evidence_resolver(self) -> Callable[[str], "Any | None"]:
        """Resolve an experiment_id to its REAL recorded result, or None.

        R1: the critic uses this to reject evidence references that point at a
        non-existent experiment or a metric that the real run never produced.
        Failures (missing/garbled evidence) yield None rather than raising, so
        the critic turns them into a deterministic rejection.
        """
        provider = self.deps.evidence_provider or default_evidence_provider

        def resolve(experiment_id: str):
            try:
                result, _ = provider(experiment_id)
                return result
            except Exception:
                return None

        return resolve

    def _manager(self) -> HypothesisManager:
        if self.deps.hypothesis_manager_factory is not None:
            return self.deps.hypothesis_manager_factory()
        return HypothesisManager(
            proposer=RuleBasedProposer(
                intervention_policy=dict(self.domain.intervention_policy)),
            critic=HypothesisCritic(evidence_resolver=self._evidence_resolver(),
                                    strict=True),
        )

    def _session(self, session_id: str, objective_id: str = ""):
        session = (self.stores.sessions.get_session(session_id)
                  if session_id else None)
        if session is None and objective_id:
            session = (self.stores.sessions.session_for_objective(objective_id)
                      or self.stores.sessions.create_session(
                          ResearchObjective(objective_id=objective_id,
                                             domain=self.domain.domain),
                          max_campaign_rounds=self.domain.max_campaign_rounds))
        if session is None and not session_id and not objective_id:
            # Only resolve an omitted session_id when there is EXACTLY one.
            session = self.stores.sessions.unique_session()
        if session is None:
            raise ResearchGateError(
                "session_not_found",
                f"unknown research session {session_id!r}; pass session_id "
                "(multiple sessions exist) or open one first")
        if objective_id and session.objective_id != objective_id:
            raise ResearchGateError(
                "objective_mismatch",
                f"session {session.session_id!r} belongs to objective "
                f"{session.objective_id!r}, not {objective_id!r}")
        return session

    def _guard(self, method: Callable[..., dict[str, Any]],
               *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Never let an exception escape into an Agent's quality gate."""
        try:
            return method(*args, **kwargs)
        except ResearchGateError as exc:
            return _fail(exc.code, str(exc), **exc.details)
        except Exception as exc:  # noqa: BLE001 - tool contract is structural
            return _fail("internal_error", f"{type(exc).__name__}: {exc}",
                         error_type=type(exc).__name__)

    # ------------------------------------------------------------ objective
    def start_research_session(self, *, objective_id: str, description: str = "",
                               success_criteria_json: str = "",
                               session_id: str = "") -> dict[str, Any]:
        return self._guard(self._start_research_session, objective_id, description,
                           success_criteria_json, session_id)

    def _start_research_session(self, objective_id: str, description: str,
                                criteria_json: str, session_id: str) -> dict[str, Any]:
        if not objective_id:
            raise ResearchGateError("invalid_objective", "objective_id is required")
        try:
            criteria = json.loads(criteria_json) if criteria_json else {}
        except (ValueError, TypeError) as exc:
            raise ResearchGateError("invalid_json",
                                    f"success_criteria_json is invalid: {exc}")
        if not isinstance(criteria, dict):
            raise ResearchGateError("invalid_json",
                                    "success_criteria_json must be a JSON object")
        existing = (self.stores.sessions.get_session(session_id)
                   if session_id else
                   self.stores.sessions.session_for_objective(objective_id))
        if existing is not None:
            return {"ok": True, "already_open": True,
                    "session_id": existing.session_id,
                    "objective_id": existing.objective_id}
        session = self.stores.sessions.create_session(
            ResearchObjective(objective_id=objective_id, domain=self.domain.domain,
                               description=description, success_criteria=criteria),
            session_id=session_id or None,
            max_campaign_rounds=self.domain.max_campaign_rounds)
        return {"ok": True, "already_open": False,
                "session_id": session.session_id,
                "objective_id": session.objective_id,
                "max_campaign_rounds": session.max_campaign_rounds}

    # ----------------------------------------------------------- hypotheses
    def propose_research_hypotheses(self, *, objective_id: str,
                                    source_experiment_id: str,
                                    session_id: str = "",
                                    hypotheses_json: str = "") -> dict[str, Any]:
        return self._guard(self._propose, objective_id, source_experiment_id,
                           session_id, hypotheses_json)

    def _propose(self, objective_id: str, source_experiment_id: str,
                 session_id: str, hypotheses_json: str) -> dict[str, Any]:
        session = self._session(session_id, objective_id)
        provider = self.deps.evidence_provider or default_evidence_provider
        result, reports = provider(source_experiment_id)
        skip_reasons: list[str] = []

        if hypotheses_json.strip():
            # An operator/LLM may SUPPLY a hypothesis, but it is never trusted:
            # ids and evidence source are overwritten, schema is validated, and
            # the deterministic Critic (next step) is the only approval gate.
            items = _json_list(hypotheses_json, "hypotheses_json")
            proposals: list[Hypothesis] = []
            for item in items:
                if not isinstance(item, dict):
                    raise ResearchGateError("invalid_hypothesis",
                                           "each hypothesis must be a JSON object")
                payload = dict(item)
                payload["objective_id"] = objective_id
                payload["source_experiment_id"] = source_experiment_id
                try:
                    proposals.append(Hypothesis.model_validate(payload))
                except Exception as exc:
                    raise ResearchGateError(
                        "invalid_hypothesis",
                        f"hypothesis rejected by schema validation: {exc}")
        else:
            manager = self._manager()
            proposals = manager.propose(result, reports, objective_id)
            proposer = getattr(manager, "proposer", None)
            skip_reasons = list(getattr(proposer, "skipped_reasons", []) or [])
            skip_reasons += [str(item) for item in (manager.rejections or [])]

        saved = [self.stores.hypotheses.save(h) for h in proposals]
        failure_refs = []
        for report in reports:
            fid = (report.get("failure_id") if isinstance(report, dict)
                   else getattr(report, "failure_id", None))
            if fid:
                failure_refs.append(f"failure:{fid}")
        session.link("hypothesis_ids", [s.hypothesis_id for s in saved])
        session.link("evidence_refs", [
            f"experiment:{source_experiment_id}:result:{result.job_id}",
            *failure_refs,
        ])
        diagnoses = [(r.get("category") if isinstance(r, dict)
                      else getattr(r, "category", "")) or "" for r in reports]
        self.stores.sessions.save(
            session, event_type="hypotheses_proposed",
            payload={"count": len(saved), "source_experiment_id": source_experiment_id,
                     "diagnoses": diagnoses})
        return {
            "ok": True, "session_id": session.session_id,
            "objective_id": objective_id,
            "source_experiment_id": source_experiment_id,
            "evidence_diagnoses": diagnoses,
            "hypotheses": [{"hypothesis_id": s.hypothesis_id,
                            "content_sha256": s.content_sha256,
                            "status": "proposed"} for s in saved],
            "skip_reasons": skip_reasons,
            "note": ("hypotheses are proposals only; review_research_hypotheses "
                     "must persist a passing Critic verdict before any plan"),
        }

    def review_research_hypotheses(self, *, hypothesis_ids_json: str,
                                   session_id: str = "") -> dict[str, Any]:
        return self._guard(self._review, hypothesis_ids_json, session_id)

    def _review(self, hypothesis_ids_json: str, session_id: str) -> dict[str, Any]:
        ids = [str(x) for x in _json_list(hypothesis_ids_json, "hypothesis_ids_json")]
        if not ids:
            raise ResearchGateError("empty_selection", "no hypothesis ids supplied")
        session = self._session(session_id)
        if not set(ids).issubset(set(session.hypothesis_ids)):
            raise ResearchGateError("hypothesis_not_in_session",
                                    "every hypothesis must belong to this session")
        critic = self._manager().critic   # deterministic; an LLM is never a gate
        reviews: list[dict[str, Any]] = []
        for hid in ids:
            stored = self.stores.hypotheses.get(hid)
            if stored is None:
                raise ResearchGateError("hypothesis_not_found",
                                        f"unknown hypothesis {hid!r}")
            hypothesis = Hypothesis.model_validate(stored.hypothesis)
            passed, errors = critic.review(hypothesis)
            record = self.stores.hypotheses.save_review(
                hid, passed=passed, errors=errors)
            reviews.append({"hypothesis_id": hid, "review_id": record.review_id,
                            "passed": record.passed, "errors": record.errors,
                            "hypothesis_sha256": record.hypothesis_sha256})
            if passed:
                session.link("review_ids", [record.review_id])
        self.stores.sessions.save(session, event_type="hypotheses_reviewed",
                                  payload={"reviewed": len(reviews)})
        return {
            "ok": True, "session_id": session.session_id, "reviews": reviews,
            "approved_ids": [r["hypothesis_id"] for r in reviews if r["passed"]],
            "rejected_ids": [r["hypothesis_id"] for r in reviews if not r["passed"]],
            "reviewer": "deterministic_critic",
        }

    # -------------------------------------------------------------- plans
    def create_research_plan(self, *, objective_id: str, hypothesis_ids_json: str,
                            session_id: str = "", plan_id: str = "") -> dict[str, Any]:
        return self._guard(self._create_plan, objective_id, hypothesis_ids_json,
                           session_id, plan_id)

    def _create_plan(self, objective_id: str, hypothesis_ids_json: str,
                     session_id: str, plan_id: str) -> dict[str, Any]:
        from .hypothesis_manager import build_experiment_plan
        ids = [str(x) for x in _json_list(hypothesis_ids_json, "hypothesis_ids_json")]
        session = self._session(session_id, objective_id)
        if not ids or len(ids) > self.domain.max_candidates:
            raise ResearchGateError("plan_size",
                                    f"a plan needs 1..{self.domain.max_candidates} hypotheses")
        if not set(ids).issubset(set(session.hypothesis_ids)):
            raise ResearchGateError("hypothesis_not_in_session",
                                    "plans may only use this session's hypotheses")
        reviewed: list[Hypothesis] = []
        for hid in ids:
            stored = self.stores.hypotheses.get(hid)
            if stored is None:
                raise ResearchGateError("hypothesis_not_found",
                                        f"unknown hypothesis {hid!r}")
            if self.stores.hypotheses.passed_review(
                    hid, stored.content_sha256) is None:
                raise ResearchGateError(
                    "critic_review_missing",
                    f"hypothesis {hid!r} has no PASSED Critic review for its "
                    "current content hash")
            # The status is derived from the persisted verdict, never from input.
            reviewed.append(Hypothesis.model_validate(
                stored.hypothesis).model_copy(update={"status": "approved"}))
        pid = plan_id or "plan_" + uuid.uuid4().hex[:12]
        plan = build_experiment_plan(
            plan_id=pid, objective_id=objective_id,
            baseline_experiment_id=f"baseline_{pid}",
            dataset_version=self.domain.dataset_version,
            hypotheses=reviewed,
            primary_metric=self.domain.primary_metric,
            metric_direction=self.domain.metric_direction,  # type: ignore[arg-type]
            paired_seeds=list(self.domain.paired_seeds),
            guardrails=dict(self.domain.guardrails),
            max_parallel=1,
        )
        stored_plan = self.stores.plans.save(plan)
        session.link("plan_ids", [pid])
        session.status = "awaiting_approval"
        self.stores.sessions.save(
            session, event_type="plan_created",
            payload={"plan_id": pid, "plan_sha256": stored_plan.plan_sha256})
        return {
            "ok": True, "session_id": session.session_id,
            "plan_id": pid, "plan_sha256": stored_plan.plan_sha256,
            "plan_version": stored_plan.version,
            "paired_seeds": list(plan.paired_seeds),
            "candidate_ids": [c.candidate_id for c in plan.candidates],
            "status": "awaiting_approval",
        }

    def request_plan_approval(self, *, plan_id: str, requested_by: str = "research_agent",
                             session_id: str = "") -> dict[str, Any]:
        return self._guard(self._request_approval, plan_id, requested_by, session_id)

    def _request_approval(self, plan_id: str, requested_by: str,
                           session_id: str) -> dict[str, Any]:
        stored = self.stores.plans.get(plan_id)
        if stored is None:
            raise ResearchGateError("plan_not_found", f"unknown plan {plan_id!r}")
        digest = self._plan_digest(stored)
        session = self._session(session_id, stored.objective_id)
        record = self.stores.approvals.request(
            resource_type="experiment_plan", resource_id=plan_id,
            resource_sha256=digest, requested_by=requested_by)
        session.link("plan_ids", [plan_id])
        session.link("approval_ids", [record.approval_id])
        self.stores.sessions.save(
            session, event_type="approval_requested",
            payload={"plan_id": plan_id, "approval_id": record.approval_id,
                     "plan_sha256": digest})
        return {
            "ok": True, "approval_id": record.approval_id,
            "status": "pending", "plan_id": plan_id, "plan_sha256": digest,
            "resource_type": record.resource_type, "scope": record.scope,
            "human_action": (
                "python -m research_runtime.research_store.approval_cli "
                f"grant {record.approval_id} --approved-by <human> --yes"),
            "note": ("a pending request cannot execute anything; only the "
                     "out-of-band human console can grant this approval"),
        }

    @staticmethod
    def _plan_digest(stored) -> str:
        digest = plan_sha256(stored.plan)
        if digest != stored.plan_sha256:
            raise ResearchGateError(
                "plan_integrity_error",
                "stored plan content no longer matches its recorded SHA-256")
        return digest

    # ------------------------------------------------------------- the gate
    def _authorize(self, plan_id: str, approval_id: str,
                   scope: str = "run_campaign") -> tuple[ExperimentPlan, Any, Any, str]:
        """Re-read ALL trust inputs from disk. Returns (plan, stored, approval, hash)."""
        stored = self.stores.plans.get(plan_id)
        if stored is None:
            raise ResearchGateError("plan_not_found", f"unknown plan {plan_id!r}")
        digest = self._plan_digest(stored)
        try:
            plan = ExperimentPlan.model_validate(stored.plan)
        except Exception as exc:
            raise ResearchGateError("plan_invalid", f"stored plan is invalid: {exc}")

        approval = self.stores.approvals.get(approval_id)
        if approval is None:
            raise ResearchGateError("approval_not_found",
                                    f"unknown approval_id {approval_id!r}")
        if (approval.resource_type != "experiment_plan"
                or approval.resource_id != plan_id):
            raise ResearchGateError(
                "approval_resource_mismatch",
                "approval is not bound to this experiment_plan id")
        if approval.status != "approved":
            raise ResearchGateError(
                "approval_not_approved",
                f"approval status is {approval.status!r}; only 'approved' may execute")
        policy = self.domain.approval_policy
        if approval.source not in policy.allowed_sources:
            raise ResearchGateError(
                "approval_source_rejected",
                f"approval source {approval.source!r} is not trusted "
                f"(allowed: {list(policy.allowed_sources)})")
        if policy.require_approved_by and not approval.approved_by.strip():
            raise ResearchGateError("approval_missing_approver",
                                    "approval has no human approver identity")
        if approval.resource_sha256 != digest:
            raise ResearchGateError(
                "plan_hash_mismatch",
                "plan content changed after approval; the approval is void. "
                "Create a new plan/version and request a fresh approval")
        if scope not in approval.scope:
            raise ResearchGateError("approval_scope_missing",
                                    f"approval does not grant scope {scope!r}")

        # ---- hypotheses: persisted review, bound to the CURRENT hash -----
        for candidate in plan.candidates:
            hid = candidate.hypothesis_id
            hyp = self.stores.hypotheses.get(hid)
            if hyp is None:
                raise ResearchGateError("hypothesis_not_found",
                                         f"candidate references unknown hypothesis {hid!r}")
            review = self.stores.hypotheses.passed_review(
                hid, hyp.content_sha256)
            if review is None:
                latest = self.stores.hypotheses.latest_review(hid)
                if latest is not None and not latest.passed:
                    raise ResearchGateError(
                        "critic_review_failed",
                        f"hypothesis {hid!r} FAILED the deterministic Critic: "
                        f"{latest.errors}")
                raise ResearchGateError(
                    "critic_review_missing",
                    f"hypothesis {hid!r} has no passing Critic review for its "
                    "current content hash")
            intervention = hyp.hypothesis.get("intervention") or {}
            if _canon(candidate.config_overrides) != _canon(intervention):
                raise ResearchGateError(
                    "intervention_mismatch",
                    f"candidate {candidate.candidate_id!r} overrides differ from "
                    "the reviewed hypothesis intervention")
            for key, value in candidate.config_overrides.items():
                error = self.domain.override_error(key, value)
                if error:
                    raise ResearchGateError("config_key_not_allowed", error)

        # ---- dataset identity + budget ----------------------------------
        if (self.domain.allowed_dataset_versions
                and plan.dataset_version not in self.domain.allowed_dataset_versions):
            raise ResearchGateError(
                "dataset_version_not_allowed",
                f"dataset_version {plan.dataset_version!r} is not a declared "
                "frozen dataset")
        if self.deps.dataset_verifier is None:
            raise ResearchGateError("dataset_verifier_missing",
                                    "no dataset verifier configured; failing closed")
        try:
            verification = self.deps.dataset_verifier(
                self.domain.baseline_config_path, plan.dataset_version)
        except Exception as exc:
            raise ResearchGateError(
                "dataset_not_verified",
                f"frozen dataset could not be verified: "
                f"{type(exc).__name__}: {exc}")
        if not (verification or {}).get("verified"):
            raise ResearchGateError("dataset_not_verified",
                                    "frozen dataset verification returned not verified")

        seeds = list(plan.paired_seeds)
        if len(plan.candidates) > self.domain.max_candidates:
            raise ResearchGateError("budget_exceeded", "too many candidates")
        if not seeds or len(seeds) > self.domain.max_seeds:
            raise ResearchGateError("budget_exceeded",
                                    f"paired_seeds exceeds max_seeds={self.domain.max_seeds}")
        if len(set(seeds)) != len(seeds):
            raise ResearchGateError("budget_exceeded", "paired_seeds must be unique")
        expected = self.domain.expected_jobs(len(plan.candidates), len(seeds))
        if expected > self.domain.max_jobs:
            raise ResearchGateError(
                "budget_exceeded",
                f"campaign needs {expected} jobs, budget allows {self.domain.max_jobs}")
        if plan.max_parallel != 1:
            raise ResearchGateError("budget_exceeded",
                                    "P1-04 execution is serial; max_parallel must be 1")
        for candidate in plan.candidates:
            if (candidate.evaluation_split != self.domain.evaluation_split
                    or candidate.evaluation_split in RESERVED_FINAL_SPLITS):
                raise ResearchGateError(
                    "final_split_forbidden",
                    "selection must use the declared held-out split; the frozen "
                    "final split is reserved for a separate confirmation phase")
        return plan, stored, approval, digest

    # ---------------------------------------------------------- execution
    def run_research_campaign(self, *, plan_id: str, approval_id: str,
                              session_id: str = "") -> dict[str, Any]:
        return self._guard(self._run_campaign, plan_id, approval_id, session_id)

    def _run_campaign(self, plan_id: str, approval_id: str,
                      session_id: str) -> dict[str, Any]:
        with self._run_lock:
            plan, stored, approval, digest = self._authorize(
                plan_id, approval_id, scope="run_campaign")
            session = self._session(session_id, plan.objective_id)

            existing = self.stores.sessions.campaign_for(
                plan_id, approval_id, digest)
            if existing is not None:
                status = self.deps.runner_factory().get_status(existing.campaign_id)
                return {"ok": True, "already_submitted": True,
                        "campaign_id": existing.campaign_id,
                        "status": status.get("status"), "summary": status.get("summary"),
                        "note": "duplicate submission ignored; no new jobs were created"}

            rounds = self.stores.sessions.campaigns_for_objective(plan.objective_id)
            if len(rounds) >= session.max_campaign_rounds:
                raise ResearchGateError(
                    "round_limit_reached",
                    f"objective {plan.objective_id!r} already has {len(rounds)} "
                    f"campaigns (cap={session.max_campaign_rounds}); propose a "
                    "new experiment only after an explicit human decision")

            campaign_id = make_campaign_id(plan_id, approval_id, digest)
            binding = CampaignBinding(
                campaign_id=campaign_id, plan_id=plan_id, approval_id=approval_id,
                plan_sha256=digest, objective_id=plan.objective_id,
                session_id=session.session_id)
            bound, created = self.stores.sessions.bind_campaign(binding)
            if not created:
                status = self.deps.runner_factory().get_status(bound.campaign_id)
                return {"ok": True, "already_submitted": True,
                        "campaign_id": bound.campaign_id,
                        "status": status.get("status"), "summary": status.get("summary")}

            # Only NOW is approved_hypothesis_ids derived -- from the persisted,
            # hash-bound Critic verdicts, never from an Agent tool argument.
            approved_ids = [c.hypothesis_id for c in plan.candidates]
            run_spec = CampaignRunSpec(
                campaign_id=bound.campaign_id,
                objective_id=plan.objective_id, plan_id=plan_id,
                baseline_config_path=self.domain.baseline_config_path,
                dataset_version=plan.dataset_version,
                evaluation_split=self.domain.evaluation_split,
                max_parallel=1,
                job_timeout_s=self.domain.job_timeout_s,
                max_jobs=self.domain.max_jobs,
                max_total_seconds=self.domain.max_total_seconds,
                approved_hypothesis_ids=approved_ids,
            )
            session.link("campaign_ids", [bound.campaign_id])
            session.status = "running"
            self.stores.sessions.save(
                session, event_type="campaign_submitted",
                payload={"campaign_id": bound.campaign_id,
                         "plan_sha256": digest, "approval_id": approval_id,
                         "approved_hypothesis_ids": approved_ids})
            record = self.deps.runner_factory().run(plan, run_spec)
            self.stores.sessions.save(
                session, event_type="campaign_finished",
                payload={"campaign_id": bound.campaign_id,
                         "status": record.status})
            return {
                "ok": True, "campaign_id": bound.campaign_id,
                "plan_id": plan_id, "approval_id": approval_id,
                "plan_sha256": digest, "status": record.status,
                "total_trials": len(record.trials),
                "succeeded_trials": record.completed_trials,
                "dataset_version": record.dataset_version,
                "evaluation_split": record.evaluation_split,
                "note": ("campaign executed by the existing P1-04 CampaignRunner; "
                         "no comparison, promotion or frozen-test evaluation here"),
            }

    def get_campaign_status(self, *, campaign_id: str) -> dict[str, Any]:
        return self._guard(self._campaign_status, campaign_id)

    def _campaign_status(self, campaign_id: str) -> dict[str, Any]:
        status = self.deps.runner_factory().get_status(campaign_id)
        binding = self.stores.sessions.binding_for_campaign(campaign_id)
        payload = {"ok": True, **status}
        if binding is not None:
            payload.update({"plan_id": binding.plan_id,
                            "approval_id": binding.approval_id,
                            "plan_sha256": binding.plan_sha256})
        return payload

    def resume_research_campaign(self, *, campaign_id: str,
                                 approval_id: str = "",
                                 session_id: str = "") -> dict[str, Any]:
        return self._guard(self._resume, campaign_id, approval_id, session_id)

    def _resume(self, campaign_id: str, approval_id: str,
                session_id: str) -> dict[str, Any]:
        with self._run_lock:
            record = self.deps.campaign_store.load_campaign(campaign_id)
            if record is None:
                raise ResearchGateError("campaign_unknown",
                                        f"unknown campaign {campaign_id!r}")
            if record.status in CAMPAIGN_TERMINAL_STATUSES:
                raise ResearchGateError(
                    "campaign_terminal",
                    f"campaign {campaign_id!r} is {record.status!r}; "
                    "a finished campaign cannot be resumed")
            if record.status != "paused":
                raise ResearchGateError(
                    "campaign_not_resumable",
                    f"campaign {campaign_id!r} is {record.status!r}; only a "
                    "'paused' campaign may be resumed (never a running one)")
            binding = self.stores.sessions.binding_for_campaign(campaign_id)
            if binding is None:
                raise ResearchGateError(
                    "campaign_unbound",
                    "campaign has no trusted plan/approval binding; refusing resume")
            effective = approval_id or binding.approval_id
            if effective != binding.approval_id:
                raise ResearchGateError("approval_mismatch",
                                        "approval_id does not match the bound approval")
            self._authorize(binding.plan_id, effective, scope="resume_campaign")
            session = self._session(session_id, binding.objective_id)
            resumed = self.deps.runner_factory().resume(campaign_id)
            self.stores.sessions.save(
                session, event_type="campaign_resumed",
                payload={"campaign_id": campaign_id, "status": resumed.status})
            return {"ok": True, "campaign_id": campaign_id,
                    "status": resumed.status,
                    "total_trials": len(resumed.trials),
                    "succeeded_trials": resumed.completed_trials}

    # ---------------------------------------------------------- comparison
    def compare_research_campaign(self, *, campaign_id: str,
                                  session_id: str = "") -> dict[str, Any]:
        return self._guard(self._compare, campaign_id, session_id)

    def _compare(self, campaign_id: str, session_id: str) -> dict[str, Any]:
        record = self.deps.campaign_store.load_campaign(campaign_id)
        if record is None:
            raise ResearchGateError("campaign_unknown",
                                    f"unknown campaign {campaign_id!r}")
        if record.status not in CAMPAIGN_TERMINAL_STATUSES:
            raise ResearchGateError(
                "campaign_not_terminal",
                f"campaign {campaign_id!r} is {record.status!r}; a comparison "
                "requires completed execution evidence")
        reports = self.deps.comparator_factory().compare(campaign_id)
        if not reports:
            raise ResearchGateError(
                "comparison_unavailable",
                "the P1-05 comparator produced no report; no result is invented")
        summaries = [{
            "comparison_id": r.comparison_id, "candidate_id": r.candidate_id,
            "decision": r.decision, "decision_rule": r.decision_rule,
            "n_pairs": r.n_pairs, "mean_delta": r.mean_delta,
            "guardrail_violations": list(r.guardrail_violations),
        } for r in reports]
        binding = self.stores.sessions.binding_for_campaign(campaign_id)
        session = self._session(session_id or (binding.session_id if binding else ""),
                                record.objective_id)
        session.link("comparison_ids", [r.comparison_id for r in reports])
        self.stores.sessions.save(
            session, event_type="comparison_completed",
            payload={"campaign_id": campaign_id,
                     "comparison_ids": [r.comparison_id for r in reports]})
        report = self._finalize_report(session, summaries)
        return {"ok": True, "campaign_id": campaign_id,
                "session_id": session.session_id,
                "comparisons": summaries,
                "research_report_id": report.report_id,
                "research_outcome": report.outcome,
                "note": ("accepted_for_confirmation is NOT promotion and does "
                         "not evaluate on the frozen final split")}

    def get_comparison_report(self, *, campaign_id: str,
                              comparison_id: str = "") -> dict[str, Any]:
        return self._guard(self._comparison_report, campaign_id, comparison_id)

    def _comparison_report(self, campaign_id: str,
                           comparison_id: str) -> dict[str, Any]:
        try:
            items = self.deps.report_store.list_reports(campaign_id)
        except Exception as exc:
            raise ResearchGateError(
                "comparison_store_error",
                f"cannot read comparison reports: {type(exc).__name__}: {exc}")
        if not items:
            raise ResearchGateError(
                "comparison_report_not_found",
                f"no comparison report exists for campaign {campaign_id!r}; "
                "run compare_research_campaign first")
        if not comparison_id:
            return {"ok": True, "campaign_id": campaign_id,
                    "reports": items, "note": "pass comparison_id to load one report"}
        match = next((i for i in items
                     if i.get("comparison_id") == comparison_id), None)
        if match is None:
            raise ResearchGateError(
                "comparison_report_not_found",
                f"comparison_id {comparison_id!r} is not persisted for "
                f"campaign {campaign_id!r}")
        report = self.deps.report_store.load(campaign_id, comparison_id)
        if report is None:
            raise ResearchGateError(
                "comparison_report_not_found",
                f"indexed comparison report {comparison_id!r} could not be loaded")
        return {"ok": True, "campaign_id": campaign_id,
                "report": report.model_dump(mode="json")}

    def get_research_report(self, *, session_id: str) -> dict[str, Any]:
        return self._guard(self._research_report, session_id)

    def _research_report(self, session_id: str) -> dict[str, Any]:
        session = self.stores.sessions.get_session(session_id)
        if session is None:
            raise ResearchGateError("session_not_found",
                                    f"unknown session {session_id!r}")
        if not session.report_id:
            raise ResearchGateError(
                "research_report_not_found",
                "no research report yet; complete a campaign comparison first")
        report = self.stores.sessions.get_report(session.report_id)
        if report is None:
            raise ResearchGateError("research_report_not_found",
                                    "indexed research report is missing")
        return {"ok": True, "report": report.model_dump(mode="json"),
                "report_path": session.report_path}

    # ------------------------------------------------------------- review
    def _finalize_report(self, session, summaries: list[dict[str, Any]]):
        """Research Reviewer -> Experience Store -> immutable ResearchReport."""
        hypotheses = []
        reviews = []
        for hid in session.hypothesis_ids:
            stored = self.stores.hypotheses.get(hid)
            if stored:
                hypotheses.append({"hypothesis_id": hid,
                                   "content_sha256": stored.content_sha256,
                                   "statement": stored.hypothesis.get("statement", "")})
            review = self.stores.hypotheses.latest_review(hid)
            if review:
                reviews.append(review.model_dump(mode="json"))
        plans = []
        approvals = []
        for pid in session.plan_ids:
            stored = self.stores.plans.get(pid)
            if stored:
                plans.append({"plan_id": pid, "plan_sha256": stored.plan_sha256,
                              "version": stored.version})
            for record in self.stores.approvals.for_resource(pid):
                approvals.append(record.model_dump(mode="json"))
        campaigns = []
        for cid in session.campaign_ids:
            record = self.deps.campaign_store.load_campaign(cid)
            campaigns.append({"campaign_id": cid,
                              "status": record.status if record else "unknown"})
        decisions = [item["decision"] for item in summaries]
        # Conservative: ONE inconclusive candidate prevents a positive overall
        # verdict. With multiple candidates, a single acceptance is not a
        # licence to declare the objective achieved.
        if not decisions:
            outcome = "not_evaluated"
        elif all(item == "accepted_for_confirmation" for item in decisions):
            outcome = "accepted_for_confirmation"
        elif all(item == "rejected" for item in decisions):
            outcome = "rejected"
        else:
            outcome = "inconclusive"
        notes = [
            "A verdict is a scientific decision, not a model promotion.",
            "The frozen final split was NOT used; confirmation is a separate "
            "human-gated phase.",
        ]
        if outcome in ("rejected", "inconclusive"):
            notes.append("Negative / inconclusive evidence is retained as "
                         "structured experience; thresholds were not adjusted "
                         "after seeing results.")
        experience_ref = self.stores.sessions.record_experience(
            session_id=session.session_id, objective_id=session.objective_id,
            kind="research_outcome",
            payload={"outcome": outcome, "decisions": decisions,
                     "comparison_ids": [i["comparison_id"] for i in summaries],
                     "campaign_ids": session.campaign_ids})

        report = ResearchReport(
            report_id="report_" + uuid.uuid4().hex[:12],
            session_id=session.session_id, objective_id=session.objective_id,
            outcome=outcome, campaign_count=len(campaigns),
            comparison_count=len(summaries), hypotheses=hypotheses,
            reviews=reviews, plans=plans, approvals=approvals,
            campaigns=campaigns, comparisons=summaries,
            evidence_refs=list(session.evidence_refs),
            experience_refs=[experience_ref],
            decision_rules=[str(i.get("decision_rule") or "") for i in summaries],
            notes=notes,
        )
        saved = self.stores.sessions.save_report(report)
        session.report_id = saved.report_id
        session.status = "completed"
        if self.domain.report_dir:
            directory = Path(self.domain.report_dir)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"research_report_{session.session_id}.json"
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(saved.model_dump(mode="json"), indent=2,
                                      ensure_ascii=False, sort_keys=True),
                           encoding="utf-8")
            tmp.replace(path)
            session.report_path = str(path)
        self.stores.sessions.save(session, event_type="research_report_published",
                                  payload={"report_id": saved.report_id,
                                           "outcome": outcome})
        return saved
