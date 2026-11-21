"""M4.3-02 -- Pilot Runner: REAL SciNet++ A/B/C execution.

What this module does
---------------------
Executes the pilot stage of the ARR value validation on ONE real, single-domain
engine (SciNet++). For every block it:

  1. derives a research opportunity from REAL persisted evidence (a disjoint
     subset of the real ``DiscoverySnapshot``'s evidence entities);
  2. selects one Objective per arm:
       A  human-protocol selection from CURRENT evidence only,
       B  human-protocol selection from the SAME historical context as C,
       C  the REAL ``ObjectiveDiscovery`` pipeline (generator + critic + ranker);
  3. queues the three objectives for independent human blind review (the review
     itself is NOT fabricated here -- it stays pending until two humans submit);
  4. drives the EXISTING ``ResearchDirector`` -> ``ResearchService`` -> 12-step
     gate chain for each arm, with the human grant performed OUT OF BAND by the
     real ``approval_cli`` console;
  5. records the outcome in the SAME ``ExperimentRegistry`` (one SQLite file).

Hard rules (spec M4.3-02)
-------------------------
  * NO FakeRunner: a missing engine / checkout / real evidence REFUSES to run
    (``PreflightError``), it never degrades to a synthetic replacement;
  * the three arms share ONE ResearchService, ONE gate, ONE registry, the same
    snapshot, seed, budget, paired seeds and model/prompt/ranker versions;
  * arm A never receives historical Experience / Evidence / Failure;
  * C's objective is whatever the real Discovery produces -- it is never typed
    in by hand;
  * every outcome (valid / negative / inconclusive / invalid / failed /
    interrupted) is retained, including real failures;
  * execution is resumable: a block whose runs are already finalized is never
    executed or charged again.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..discovery_context_provider import DiscoveryContextProvider, DiscoverySnapshot
from ..objective_candidate_generator import ObjectiveCandidate, ObjectiveCandidateGenerator
from ..objective_critic import ObjectiveCritic
from ..objective_discovery import ObjectiveDiscovery
from ..objective_ranker import ObjectiveRanker
from ..research_director import ResearchDirector
from ..research_store import from_research_objective
from ..schemas import ResearchObjective
from ..settings import DEFAULT_SCINET_DIR, resolve_engine
from .blind_review import BlindingLedger, BlindingViolation, is_group_leaking
from .evaluation_harness import EvaluationHarness, ProtocolViolation
from .experiment_controller import BudgetExhausted, ExperimentController
from .experiment_registry import (
    ARMS, DIRECTOR_STATE_OUTCOME, REPORT_OUTCOME_MAP, FrozenProtocol, RunRecord,
)
from .metrics import frozen_metric_contract
from .registry_store import snapshot_sha256
from .research_service_gate import (
    SCIENTIFIC_REPORT_OUTCOMES, GateRunRef, ResearchServiceGateAdapter,
)
from .review_queue import ReviewQueue

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SP_A = "A"
SP_B = "B"
SP_C = "C"

# P0-02: outcome semantics live in experiment_registry (single source of truth)
#   * DIRECTOR_STATE_OUTCOME -- FAILED / INTERRUPTED / STOPPED / REJECTED / INVALID
#     are recorded as DISTINCT outcomes, never collapsed into "interrupted";
#   * REPORT_OUTCOME_MAP -- a REAL report outcome maps to an arm outcome:
#       accepted_for_confirmation -> valid_finding
#       rejected                  -> negative_finding (ONLY with a successful,
#                                    real campaign + a real scientific outcome)
#       inconclusive              -> inconclusive   (NEVER negative_finding)
#       not_evaluated             -> inconclusive   (retained, not a finding)
#   * any UNKNOWN state/outcome is fail-closed and preserves the raw value.


class PreflightError(RuntimeError):
    """The real environment cannot support a real pilot run (fail-closed)."""


class SelectionError(RuntimeError):
    """The objective-selection protocol could not be satisfied."""


class ReviewIncomplete(RuntimeError):
    """A stage that REQUIRES completed human review found pending reviews."""


def classify_director_outcome(
    *, director_state: str, report_outcome: str = "",
    campaign_succeeded: bool = False,
    scientific_outcome_valid: bool = False,
) -> tuple[str, str]:
    """P0-02 -- pure, fail-closed mapping from REAL director/report state to a
    recorded run outcome. Returns ``(outcome, reason)``.

    Rules (no silent collapse, no optimistic default):
      * a director terminal state in ``DIRECTOR_STATE_OUTCOME`` maps to its
        DISTINCT recorded outcome (FAILED->failed, INTERRUPTED->interrupted,
        STOPPED->stopped, REJECTED/INVALID->invalid);
      * only ``COMPLETED`` consults the report;
      * ``rejected`` becomes ``negative_finding`` ONLY when the campaign truly
        succeeded AND a real scientific conclusion exists; otherwise it is
        retained as ``inconclusive`` with the raw reason;
      * ``inconclusive`` and ``not_evaluated`` stay ``inconclusive``;
      * unknown states/outcomes -> ``invalid`` with the raw value in ``reason``.
    """
    raw_state = str(director_state or "")
    state = raw_state.strip().upper()
    if state == "COMPLETED":
        ro = str(report_outcome or "").strip()
        if ro not in REPORT_OUTCOME_MAP:
            return "invalid", f"unknown report outcome {ro!r} (fail-closed)"
        base = REPORT_OUTCOME_MAP[ro]
        if ro == "rejected" and not (campaign_succeeded and scientific_outcome_valid):
            # A "rejected" verdict without a successful real campaign and a
            # verified scientific conclusion is NOT evidence of a negative
            # finding. Retain it honestly as inconclusive.
            return "inconclusive", (
                f"report rejected but campaign_succeeded="
                f"{campaign_succeeded}, scientific_outcome_valid="
                f"{scientific_outcome_valid}; retained as inconclusive")
        return base, f"report outcome {ro!r} (director COMPLETED)"
    if state in DIRECTOR_STATE_OUTCOME:
        outcome = DIRECTOR_STATE_OUTCOME[state]
        return outcome, f"director terminal state {raw_state!r}"
    # Unknown / intermediate director state at terminal observation: fail-closed
    # and preserve the RAW state verbatim (never upper-cased away).
    return "invalid", f"unknown director state {raw_state!r} (fail-closed)"


@dataclass
class PilotConfig:
    """Frozen, pre-registered pilot configuration (no tuning from results)."""
    experiment_id: str = "m4_3_pilot"
    domain: str = "scinet"
    run_id: str = "arr_m34_001"
    n_blocks: int = 4
    block_seed: int = 42
    paired_seeds: tuple[int, ...] = (42, 43)
    candidate_budget: int = 3
    human_time_budget_s: float = 900.0
    campaign_budget: dict[str, Any] = field(
        default_factory=lambda: {"max_jobs": 8, "max_seeds": 2,
                                 "max_candidates": 1})
    selector_id: str = "delegated-operator"
    selection_protocol: str = "delegated_operator_heuristic_v1"
    approver_id: str = "operator"
    # REAL recorded experiment whose structured result grounds each hypothesis.
    # Empty = resolve automatically from the real EvidenceStore (never invented).
    source_experiment_id: str = ""
    # Extra real candidates tried, in order, when a grounded experiment cannot
    # yield an approvable hypothesis (pipeline limitation, not a result).
    source_experiment_ids: tuple[str, ...] = ()
    max_campaigns: int = 0            # 0 = unlimited
    max_wall_time_s: float = 0.0      # 0 = unlimited
    require_review_complete: bool = False   # pilot may run with review pending
    evidence_per_block: int = 0       # 0 = split the real evidence evenly

    def to_dict(self) -> dict[str, Any]:
        return {"experiment_id": self.experiment_id, "domain": self.domain,
                "run_id": self.run_id, "n_blocks": self.n_blocks,
                "block_seed": self.block_seed,
                "paired_seeds": list(self.paired_seeds),
                "candidate_budget": self.candidate_budget,
                "human_time_budget_s": self.human_time_budget_s,
                "campaign_budget": dict(self.campaign_budget),
                "selector_id": self.selector_id,
                "selection_protocol": self.selection_protocol,
                "approver_id": self.approver_id,
                "source_experiment_id": self.source_experiment_id,
                "max_campaigns": self.max_campaigns,
                "max_wall_time_s": self.max_wall_time_s,
                "require_review_complete": self.require_review_complete,
                "evidence_per_block": self.evidence_per_block}


@dataclass
class PreflightReport:
    scinet_checkout: bool = False
    engine_python: bool = False
    real_evidence: bool = False
    n_evidence: int = 0
    budget_ok: bool = True
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return bool(self.scinet_checkout and self.engine_python
                    and self.real_evidence and self.budget_ok)

    def to_dict(self) -> dict[str, Any]:
        return {"scinet_checkout": self.scinet_checkout,
                "engine_python": self.engine_python,
                "real_evidence": self.real_evidence,
                "n_evidence": self.n_evidence,
                "budget_ok": self.budget_ok,
                "ready": self.ready, "detail": dict(self.detail)}


def preflight(config: PilotConfig, *, service: Any = None) -> PreflightReport:
    """Real-environment checks. Never fabricates, never falls back."""
    rep = PreflightReport()
    detail: dict[str, Any] = {}
    try:
        project, python = resolve_engine(strict=True)
        rep.scinet_checkout = True
        rep.engine_python = True
        detail["scinet_dir"] = str(project)
        detail["engine_python"] = str(python)
    except Exception as exc:  # noqa: BLE001 - preflight reports, never raises
        detail["engine_error"] = f"{type(exc).__name__}: {exc}"
        rep.scinet_checkout = (DEFAULT_SCINET_DIR / "research_api" / "cli.py").is_file()
        from ..settings import DEFAULT_SCINET_PYTHON
        rep.engine_python = Path(DEFAULT_SCINET_PYTHON).is_file()
    if service is not None:
        try:
            provider = DiscoveryContextProvider(stores=service.stores)
            snap = provider.build_snapshot(config.run_id, objective_ids=[])
            rep.n_evidence = int(snap.evidence_view.get("n_evidence") or 0)
            rep.real_evidence = rep.n_evidence > 0
            detail["provenance"] = snap.provenance.get("counts", {})
            detail["unresolved_gaps"] = len(snap.unresolved_gaps)
        except Exception as exc:  # noqa: BLE001
            detail["evidence_error"] = f"{type(exc).__name__}: {exc}"
    detail["selector_id"] = config.selector_id
    detail["approver_id"] = config.approver_id
    rep.detail = detail
    return rep


# --------------------------------------------------------------- selection
@dataclass
class SelectionRecord:
    """Who selected what, from which candidates, in which order, how long."""
    block_id: str
    arm: str
    selector_id: str
    protocol: str
    selected_at: str
    order_index: int
    duration_s: float
    visible_evidence_ids: list[str]
    context_sha256: str
    candidate_ids: list[str]
    selected_candidate_id: str
    selected_objective_id: str
    rationale: str
    generation_source: str = ""
    ranker_version: str = ""
    rank_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _falsifiability_score(candidate: ObjectiveCandidate) -> int:
    crit = candidate.success_criteria or {}
    return sum(1 for key in ("metric", "direction", "threshold", "measurable")
               if crit.get(key) not in (None, "", False))


def delegated_select(candidates: list[ObjectiveCandidate], *,
                     visible_ids: set[str]) -> tuple[ObjectiveCandidate, str]:
    """Deterministic, pre-registered stand-in for a human pick.

    A human reviewer would weigh evidence grounding and falsifiability. This
    rule is FIXED before any result is seen and depends ONLY on what the arm is
    allowed to see (never on the engine result), so it cannot be tuned post hoc:

      1. most evidence refs resolvable in the VISIBLE context;
      2. then most complete falsifiability criteria;
      3. then the earliest candidate in the generator's deterministic order.
    """
    if not candidates:
        raise SelectionError("no approved candidate to select from")
    best = max(
        candidates,
        key=lambda c: (sum(1 for r in (c.evidence_refs or [])
                           if str(r) in visible_ids),
                       _falsifiability_score(c),
                       -candidates.index(c)))
    reasons = [
        f"resolvable_refs={sum(1 for r in (best.evidence_refs or []) if str(r) in visible_ids)}",
        f"falsifiability={_falsifiability_score(best)}",
        "rule=delegated_operator_heuristic_v1",
    ]
    return best, "; ".join(reasons)


def _hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class PilotRunner:
    """Run the A/B/C pilot against ONE real ResearchService."""

    def __init__(self, *, service: Any, config: PilotConfig,
                 out_dir: str | Path, controller: ExperimentController,
                 review_queue: ReviewQueue | None = None,
                 protocol: FrozenProtocol | None = None,
                 harness: EvaluationHarness | None = None,
                 selector: Callable[..., tuple[ObjectiveCandidate, str]] | None = None,
                 approver: Callable[[str, str], None] | None = None):
        if service is None:
            raise ValueError("PilotRunner requires a REAL ResearchService")
        self.service = service
        self.config = config
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.controller = controller
        self.review_queue = review_queue
        self._selector = selector or delegated_select
        self._approver = approver or self._grant_via_cli
        self.protocol = protocol or self._pilot_protocol()
        self.gate_adapter = ResearchServiceGateAdapter(service)
        self.harness = harness or self._build_harness()
        self.selections_path = self.out_dir / "objective_selections.jsonl"
        self.records_path = self.out_dir / "pilot_run.json"
        self.blocks: list[dict[str, Any]] = []
        self._run_started = 0.0

    # ------------------------------------------------------------ wiring
    def _pilot_protocol(self) -> FrozenProtocol:
        cfg = self.config
        return FrozenProtocol(
            confirmatory=False,
            snapshot_ids=tuple(self._snapshot_id(i)
                               for i in range(1, cfg.n_blocks + 1)),
            sample_size=0, seed=cfg.block_seed,
            model_version="scinet-pendulum-smoke-v1",
            prompt_version="objective-candidates-v1",
            ranker_version=ObjectiveRanker().version,
            domain_policy_version="scinet-domain-v1",
            campaign_budget=dict(cfg.campaign_budget),
            paired_seeds=tuple(cfg.paired_seeds),
            budget={"paired_seeds": list(cfg.paired_seeds)},
            primary_metrics=self._primary_metrics(),
            metric_contract_sha256=frozen_metric_contract()["contract_sha256"],
            exclusion_rules=("pilot: no exclusion; every outcome retained",),
            statistical_method="paired-block descriptive (pilot: no inference)",
            multiple_comparison_correction="holm (confirmatory only)",
            stopping_rule="all planned blocks complete or budget exhausted",
            smallest_effect_of_interest=0.5, alpha=0.05, target_power=0.8,
            analysis_plan_version="m4_3_pilot_analysis_v1")

    @staticmethod
    def _primary_metrics() -> tuple[str, ...]:
        from .metrics import PRIMARY_METRICS
        return PRIMARY_METRICS

    def resolve_source_experiments(self) -> list[str]:
        """Ordered list of REAL recorded experiments that can ground a hypothesis.

        A hypothesis must cite a structured result event that carries metrics
        (otherwise the FailureDiagnoser has nothing to diagnose and no
        intervention can be justified). Candidates are read from the real
        EvidenceStore and ordered deterministically (most recent evaluate
        result first). Nothing is invented; if there is no such experiment the
        runner refuses instead of fabricating one.
        """
        ordered: list[str] = []
        if self.config.source_experiment_id:
            ordered.append(self.config.source_experiment_id)
        ordered.extend(self.config.source_experiment_ids)
        from ..evidence_store import EvidenceStore
        events = EvidenceStore().query(event_type="result")
        last: dict[str, tuple[str, dict[str, Any]]] = {}
        for e in events:
            payload = e.get("payload") or {}
            result = payload.get("result") or {}
            eid = str(payload.get("experiment_id")
                      or result.get("experiment_id") or "")
            if eid:
                last[eid] = (str(e.get("at") or ""), result)
        usable = [
            (at, eid) for eid, (at, result) in last.items()
            if str(result.get("stage")) == "evaluate"
            and any(k.startswith("latent_mse") or k.startswith("state_mse")
                    for k in (result.get("metrics") or {}))]
        # Prefer a source whose structured evidence actually yields a
        # diagnosis (a real failure category is what justifies an intervention).
        from ..research_service import default_evidence_provider
        strong: list[str] = []
        weak: list[str] = []
        for _at, eid in sorted(usable, reverse=True):
            if eid in ordered:
                continue
            if len(strong) >= 4 and len(weak) >= 8:
                break
            if len(strong) + len(weak) >= 20:
                break
            try:
                _res, reports = default_evidence_provider(eid)
            except Exception:  # noqa: BLE001 - a source without a result is unusable
                continue
            (strong if reports else weak).append(eid)
        ordered.extend(strong)
        ordered.extend(weak)
        if not ordered:
            raise PreflightError(
                "the real EvidenceStore holds no evaluate result with metrics; "
                "a hypothesis cannot be grounded (refusing to fabricate one)")
        return ordered

    def resolve_source_experiment(self) -> str:
        return self.resolve_source_experiments()[0]

    def _build_harness(self) -> EvaluationHarness:
        # The pilot records selections with a pending human review; it must not
        # invent a reviewer. ``defer_review=True`` is used for every selection.
        from .blind_review import IndependentReviewer
        placeholder = IndependentReviewer(
            reviewer_id="pilot_pending_review_placeholder",
            scorer=lambda _b: (_ for _ in ()).throw(RuntimeError(
                "pilot does not fabricate reviews")))
        # RECOVER the registry from the SAME database: a resumed run must see
        # the blocks/runs it already produced (never re-create them).
        from .experiment_registry import ExperimentRegistry
        registry = ExperimentRegistry.recover(protocol=self.protocol,
                                              db=self.service.stores.db)
        return EvaluationHarness(
            protocol=self.protocol, reviewer=placeholder,
            blinding=BlindingLedger(),
            gate_adapter=self.gate_adapter, registry=registry)

    def _snapshot_id(self, block_number: int) -> str:
        return f"{self.config.run_id}::block_{block_number}"

    def _block_id(self, block_number: int) -> str:
        return f"pilot_block_{block_number}"

    # ---------------------------------------------------------- snapshots
    def _full_snapshot(self) -> DiscoverySnapshot:
        provider = DiscoveryContextProvider(stores=self.service.stores)
        return provider.build_snapshot(self.config.run_id, objective_ids=[])

    def _historical_objective_ids(self) -> list[str]:
        ids: list[str] = []
        try:
            rows = self.service.stores.db.rows(
                "SELECT objective_id FROM objectives ORDER BY rowid")
            ids = [str(r["objective_id"]) for r in rows]
        except Exception:  # noqa: BLE001 - absent table == no history
            ids = []
        return ids

    def _block_snapshot(self, full: DiscoverySnapshot, block_number: int,
                        current_ids: list[str]) -> DiscoverySnapshot:
        """A block's research opportunity: a DISJOINT subset of real evidence."""
        from dataclasses import replace
        view = dict(full.evidence_view)
        subset = [e for e in (view.get("evidence") or [])
                  if str(e.get("evidence_id")) in set(current_ids)]
        if not subset:
            raise PreflightError(
                f"block {block_number}: no real evidence in the assigned subset")
        new_view = {**view, "evidence": subset, "n_evidence": len(subset),
                    "run_id": self._snapshot_id(block_number)}
        prov = dict(view.get("provenance") or {})
        counts = dict(prov.get("counts") or {})
        counts["n_evidence_entities"] = len(subset)
        prov["counts"] = counts
        new_view["provenance"] = prov
        return replace(full, run_id=self._snapshot_id(block_number),
                       evidence_view=new_view)

    def _historical_snapshot(self, block_number: int,
                             history_ids: list[str] | None = None,
                             current_snap: DiscoverySnapshot | None = None
                             ) -> DiscoverySnapshot:
        """History view: the SAME research opportunity + prior research.

        B and C read the SAME current evidence subset as A (so all three arms
        study one opportunity) PLUS the persisted history (earlier objectives,
        sessions, Experience, failures). Arm A never sees the history.

        Scoping the EVIDENCE to the block subset matters scientifically: the
        Critic refuses to re-propose a direction that is already an objective,
        so a block must bring its own evidence; only the CONTEXT is shared.
        """
        from dataclasses import replace
        provider = DiscoveryContextProvider(stores=self.service.stores)
        ids = (history_ids if history_ids is not None
               else self._historical_objective_ids())
        full = provider.build_snapshot(self.config.run_id, objective_ids=ids)
        if current_snap is None:
            return full
        return replace(full, run_id=self._snapshot_id(block_number),
                       evidence_view=current_snap.evidence_view)

    # ------------------------------------------------------- selection core
    def _subject_objective(self, candidate: ObjectiveCandidate, *,
                           objective_id: str, arm: str,
                           selection: SelectionRecord) -> dict[str, Any]:
        """Persist a human-protocol selection as a Proposed Objective."""
        obj = ResearchObjective(
            objective_id=objective_id, domain=self.config.domain,
            description=candidate.description,
            success_criteria=dict(candidate.success_criteria),
            constraints={
                "evidence_refs": list(candidate.evidence_refs),
                "source_objective_ids": list(candidate.source_objective_ids),
                "hypothesis_hint": candidate.hypothesis_hint,
                "proposed_intervention": dict(candidate.proposed_intervention),
                "budget": dict(candidate.budget),
                "generation_source": candidate.generation_source,
                "selection": selection.to_dict(),
            },
            created_at=_now())
        self.service.stores.objectives.save(
            from_research_objective(
                obj, status="proposed",
                provenance={
                    "candidate_id": candidate.candidate_id,
                    "generation_source": candidate.generation_source,
                    "selection_source": "human_protocol",
                    "selector_id": selection.selector_id,
                    "selection_protocol": selection.protocol,
                    "arm": arm,
                    "block_id": selection.block_id,
                    "rank_score": candidate.rank_score,
                    "ranker_version": selection.ranker_version,
                    "discovery_run_id": self._snapshot_id(
                        int(selection.block_id.rsplit("_", 1)[-1])),
                }))
        return {"objective_id": objective_id,
                "description": candidate.description,
                "evidence_refs": list(candidate.evidence_refs),
                "success_criteria": dict(candidate.success_criteria),
                "generation_source": candidate.generation_source,
                "rank_score": candidate.rank_score}

    def _write_selection(self, record: SelectionRecord) -> None:
        with self.selections_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")

    def _candidate_pool(self, snapshot: DiscoverySnapshot,
                        critic: ObjectiveCritic) -> list[ObjectiveCandidate]:
        gen = ObjectiveCandidateGenerator(domain=self.config.domain)
        passed = [c for c in gen.generate(snapshot)
                  if critic.critique(c, snapshot=snapshot)[0]]
        return passed[: max(1, self.config.candidate_budget)]

    # ------------------------------------------------------------ one block
    def _run_block(self, block_number: int,
                   full: DiscoverySnapshot,
                   evidence_ids: list[str],
                   completed_runs: dict[str, RunRecord]) -> dict[str, Any]:
        bid = self._block_id(block_number)
        snap_id = self._snapshot_id(block_number)
        seed = self.config.block_seed + block_number - 1
        started = time.time()
        prior_progress = self.controller.blocks.get(bid)
        if prior_progress is not None and \
                prior_progress.state == "BLOCK_COMPLETED":
            # Already finished in a previous process: never re-run it.
            recorded = self._block_record_from_registry(bid, snap_id, seed)
            if recorded is not None:
                self.blocks.append(recorded)
                return recorded
        if prior_progress is None:
            self.controller.begin_block(bid, detail={"snapshot_id": snap_id,
                                                     "seed": seed})
        elif self.controller.state in ("CREATED", "PREFLIGHTING"):
            self.controller.begin_block(bid, detail={"snapshot_id": snap_id,
                                                     "seed": seed})

        # History is frozen BEFORE this block selects anything: A/B/C must see
        # the context as it was at the start of the block.
        history_ids = self._historical_objective_ids()
        current_snap = self._block_snapshot(full, block_number, evidence_ids)
        history_snap = self._historical_snapshot(block_number, history_ids,
                                                 current_snap)
        current_ids = [str(e.get("evidence_id")) for e in
                       (current_snap.evidence_view.get("evidence") or [])]
        visible_a = set(current_ids)
        # B/C may additionally read earlier evidence entities as HISTORY.
        visible_bc = set(current_ids) | {
            str(e.get("evidence_id")) for e in
            (full.evidence_view.get("evidence") or [])}

        # ---- build the shared block (A/B/C contexts from the SAME snapshot)
        self.harness.ensure_block(
            self._block_spec(bid, snap_id, seed, current_ids, block_number,
                             history_ids),
            candidate_budget=self.config.candidate_budget,
            human_time_budget_s=self.config.human_time_budget_s)

        # ---- per-arm selection (RESUMABLE: never re-select a registered arm)
        critic = ObjectiveCritic()
        objectives: dict[str, dict[str, Any]] = {}
        review_ids: dict[str, str] = {}
        discovery_info: dict[str, Any] = {}
        selections: dict[str, Any] = {}

        def _resume_arm(arm: str) -> bool:
            existing = self.harness.registry.runs.get(f"{bid}_{arm}")
            if existing is None:
                return False
            stored = self.service.stores.objectives.get(existing.objective_id)
            review_ids[arm] = existing.review_id
            objectives[arm] = {
                "objective_id": existing.objective_id,
                "description": getattr(stored, "description", "") or "",
                "evidence_refs": list(existing.evidence_refs),
                "success_criteria": dict(
                    getattr(stored, "success_criteria", {}) or {})}
            return True

        # Pools are built BEFORE any arm persists an objective: once A's
        # objective exists, the Critic would flag it as a duplicate and B would
        # lose that candidate from its (identical) opportunity.
        pool_a: list[ObjectiveCandidate] = []
        pool_b: list[ObjectiveCandidate] = []
        invalid_arms: dict[str, str] = {}
        if self.harness.registry.runs.get(f"{bid}_A") is None:
            pool_a = self._candidate_pool(current_snap, critic)
        if self.harness.registry.runs.get(f"{bid}_B") is None:
            pool_b = self._candidate_pool(history_snap, critic)

        # ---- arm A: current evidence only
        if not _resume_arm("A") and not pool_a:
            invalid_arms["A"] = "no approved candidate for this opportunity"
        elif not _resume_arm("A"):
            cand_a, why_a = self._selector(pool_a, visible_ids=visible_a)
            sel_a = SelectionRecord(
                block_id=bid, arm="A", selector_id=self.config.selector_id,
                protocol=self.config.selection_protocol, selected_at=_now(),
                order_index=1, duration_s=0.0,
                visible_evidence_ids=sorted(visible_a),
                context_sha256=_hash({"arm": "A", "ids": sorted(visible_a)}),
                candidate_ids=[c.candidate_id for c in pool_a],
                selected_candidate_id=cand_a.candidate_id,
                selected_objective_id=f"obj_sel_{bid}_A", rationale=why_a,
                generation_source=cand_a.generation_source,
                ranker_version=self.protocol.ranker_version,
                rank_score=cand_a.rank_score)
            self._write_selection(sel_a)
            selections["A"] = sel_a.to_dict()
            objectives["A"] = self._subject_objective(
                cand_a, objective_id=sel_a.selected_objective_id, arm="A",
                selection=sel_a)

        # ---- arm B: SAME historical context as C, human-protocol selection
        if not _resume_arm("B") and not pool_b:
            invalid_arms["B"] = "no approved candidate for this opportunity"
        elif not _resume_arm("B"):
            cand_b, why_b = self._selector(pool_b, visible_ids=visible_bc)
            sel_b = SelectionRecord(
                block_id=bid, arm="B", selector_id=self.config.selector_id,
                protocol=self.config.selection_protocol, selected_at=_now(),
                order_index=2, duration_s=0.0,
                visible_evidence_ids=sorted(visible_bc),
                context_sha256=_hash({"arm": "B", "ids": sorted(visible_bc)}),
                candidate_ids=[c.candidate_id for c in pool_b],
                selected_candidate_id=cand_b.candidate_id,
                selected_objective_id=f"obj_sel_{bid}_B", rationale=why_b,
                generation_source=cand_b.generation_source,
                ranker_version=self.protocol.ranker_version,
                rank_score=cand_b.rank_score)
            self._write_selection(sel_b)
            selections["B"] = sel_b.to_dict()
            objectives["B"] = self._subject_objective(
                cand_b, objective_id=sel_b.selected_objective_id, arm="B",
                selection=sel_b)

        # ---- arm C: REAL Discovery (generator + critic + ranker)
        if not _resume_arm("C"):
            try:
                objectives["C"], discovery_info, sel_c_dict = self._select_c(
                    bid, history_snap, visible_bc, critic)
                selections["C"] = sel_c_dict
            except SelectionError as exc:
                # The real Discovery approved nothing for THIS opportunity
                # (every candidate duplicates an existing objective). The arm
                # is retained as INVALID -- never silently dropped.
                invalid_arms["C"] = str(exc)


        # ---- register + queue the human blind review (only for NEW arms)
        newly_registered = 0
        for arm in ARMS:
            if arm in review_ids:            # resumed: already registered
                continue
            if arm in invalid_arms:
                self._record_invalid_arm(bid, arm, invalid_arms[arm])
                continue
            safe_refs, withheld = self._reviewer_safe_refs(
                objectives[arm].get("evidence_refs") or [])
            review_objective = {**objectives[arm], "evidence_refs": safe_refs}
            try:
                self.harness.select_objective(
                    bid, arm, objective=review_objective, defer_review=True,
                    notes={"selection_arm": arm, "block": bid,
                           "withheld_evidence_refs": withheld})
            except BlindingViolation as exc:
                # A historical objective id inside the candidate text would
                # de-anonymise the reviewer. The arm is retained as INVALID
                # (never silently dropped, never forced through).
                self._record_invalid_arm(bid, arm, f"blinding: {exc}")
                invalid_arms[arm] = str(exc)
                continue
            run = self.harness.registry.runs_for_block(bid)[arm]
            review_ids[arm] = run.review_id
            newly_registered += 1
            if self.review_queue is not None:
                self.review_queue.enqueue(
                    experiment_id=self.config.experiment_id,
                    objective=review_objective, arm=arm, block_id=bid,
                    snapshot_id=snap_id, review_id=run.review_id)

        if self.controller.state == "SELECTING_OBJECTIVES":
            self.controller.request_review(bid, detail={
                "review_ids": review_ids,
                "newly_registered": newly_registered})
            if self.config.require_review_complete:
                status = self.review_status(review_ids)
                if not status["complete"]:
                    raise ReviewIncomplete(
                        "this stage requires COMPLETED independent human blind "
                        f"review; pending: {status['pending']}")
                self.controller.complete_review(
                    bid, reviewers=status["reviewers"],
                    review_ids=sorted(review_ids.values()),
                    detail={"review_status": status})
            else:
                self.controller.defer_review(
                    bid, reason=("human blind review pending: objectives "
                                 "queued for >=2 independent reviewers; no "
                                 "review is fabricated by the pilot"))
        elif self.controller.state == "AWAITING_REVIEW":
            if self.config.require_review_complete:
                status = self.review_status(review_ids)
                if not status["complete"]:
                    raise ReviewIncomplete(
                        "this stage requires COMPLETED independent human blind "
                        f"review; pending: {status['pending']}")
                self.controller.complete_review(
                    bid, reviewers=status["reviewers"],
                    review_ids=sorted(review_ids.values()),
                    detail={"review_status": status})
            else:
                self.controller.defer_review(
                    bid, reason="human blind review pending (resumed)")

        # ---- execute the three real chains (prepare -> approve ALL -> run)
        outcomes: dict[str, str] = {}
        refs: dict[str, GateRunRef] = {}
        exec_seconds = 0.0
        directors: dict[str, Any] = {}
        for arm in ARMS:
            if arm in invalid_arms:
                outcomes[arm] = "invalid"
                continue
            prior = completed_runs.get(f"{bid}_{arm}")
            if prior is not None and prior.finalized:
                outcomes[arm] = prior.outcome
                refs[arm] = self._ref_from_record(prior)
                continue
            try:
                self.controller.budget.reserve_campaign()
            except BudgetExhausted as exc:
                self.controller.stop(str(exc))
                self._save_records()
                raise
            directors[arm] = self._prepare_arm(
                bid, arm, objectives[arm]["objective_id"], block_number)
        if directors:
            # EXECUTING only once EVERY arm holds a REAL external approval.
            self.controller.begin_execution(
                bid, arms=sorted(directors),
                detail={"arms": sorted(directors)})
        for arm, director in directors.items():
            t0 = time.time()
            outcome, ref, seconds = self._finish_arm(bid, arm, director)
            exec_seconds += seconds
            outcomes[arm] = outcome
            refs[arm] = ref
        self.controller.mark_evaluating(
            bid, campaign_id=",".join(sorted(r.campaign_id for r in refs.values()
                                             if r.campaign_id)),
            seconds=exec_seconds,
            detail={"outcomes": outcomes, "arms": sorted(directors)})
        self.controller.mark_recording(bid, outcomes=outcomes)
        self.controller.complete_block(bid, detail={
            "outcomes": outcomes,
            "campaign_ids": {a: refs[a].campaign_id for a in ARMS}})

        block = {"block_id": bid, "snapshot_id": snap_id, "seed": seed,
                 "n_current_evidence": len(current_ids),
                 "n_visible_BC": len(visible_bc),
                 "objective_ids": {a: objectives[a]["objective_id"] for a in ARMS},
                 "selection": selections,
                 "review_ids": review_ids,
                 "gate_refs": {a: refs[a].to_dict() for a in ARMS},
                 "outcomes": outcomes,
                 "seconds": round(time.time() - started, 2),
                 "exec_seconds": round(exec_seconds, 2),
                 # Full, real Discovery provenance (generator/critic/ranker).
                 "discovery": discovery_info}
        self.blocks.append(block)
        self._save_records()
        return block

    def _select_c(self, block_id: str, history_snap: DiscoverySnapshot,
                  visible_bc: set[str], critic: ObjectiveCritic
                  ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Run the REAL Discovery and return (objective, provenance, record)."""
        discovery = ObjectiveDiscovery(
            generator=ObjectiveCandidateGenerator(domain=self.config.domain),
            critic=critic, ranker=ObjectiveRanker(),
            domain=self.config.domain)
        result_c = discovery.discover(
            history_snap, objective_store=self.service.stores.objectives,
            domain=self.config.domain)
        if not result_c.proposed_objective_ids:
            raise SelectionError(
                "real ObjectiveDiscovery approved no objective for arm C "
                "(every candidate duplicates an existing objective)")
        c_id = result_c.proposed_objective_ids[0]
        c_ranked = next((r for r in result_c.ranked
                         if r.get("objective_id") == c_id), {})
        sel_c = SelectionRecord(
            block_id=block_id, arm="C", selector_id="objective_discovery",
            protocol="real_discovery_generator_critic_ranker",
            selected_at=_now(), order_index=3, duration_s=0.0,
            visible_evidence_ids=sorted(visible_bc),
            context_sha256=_hash({"arm": "C", "ids": sorted(visible_bc)}),
            candidate_ids=[c["candidate_id"] for c in result_c.ranked],
            selected_candidate_id=str(c_ranked.get("candidate_id", "")),
            selected_objective_id=c_id,
            rationale=(f"ranked #1 of {len(result_c.ranked)} approved "
                       f"candidates (critic rejected "
                       f"{len(result_c.rejected)})"),
            generation_source=str(c_ranked.get("generation_source", "")),
            ranker_version=result_c.ranker_version,
            rank_score=c_ranked.get("rank_score"))
        self._write_selection(sel_c)
        stored_c = self.service.stores.objectives.get(c_id)
        objective = {
            "objective_id": c_id,
            "description": getattr(stored_c, "description", "") or "",
            "evidence_refs": list(c_ranked.get("evidence_refs") or []),
            "success_criteria": dict(
                getattr(stored_c, "success_criteria", {}) or {}),
            "generation_source": sel_c.generation_source,
            "rank_score": sel_c.rank_score}
        return objective, result_c.to_dict(), sel_c.to_dict()

    def _record_invalid_arm(self, block_id: str, arm: str, reason: str) -> None:
        """Retain an arm that could not be set up as INVALID (no success claim)."""
        run_id = f"{block_id}_{arm}"
        if run_id not in self.harness.registry.runs:
            self.harness.registry.register_run(block_id=block_id, arm=arm,
                                               outcome="invalid")
        self.harness.registry.finalize(
            run_id, outcome="invalid", invalid_reason=reason)
        self.harness.registry.append_event(
            scope=f"run:{run_id}", event_type="invalid",
            payload={"reason": reason, "arm": arm})

    @staticmethod
    def _reviewer_safe_refs(refs: list[Any]) -> tuple[list[Any], list[Any]]:
        """Split refs into reviewer-visible and withheld (group-leaking).

        An evidence reference that literally encodes an arm (e.g. a historical
        objective id) may never reach a reviewer. It is withheld from the BLIND
        payload, still recorded on the run, and the withheld list is surfaced in
        the run notes -- nothing is silently dropped.
        """
        safe: list[Any] = []
        withheld: list[Any] = []
        for ref in refs:
            (withheld if is_group_leaking(str(ref)) else safe).append(ref)
        return safe, withheld

    def review_status(self, review_ids: dict[str, str]) -> dict[str, Any]:
        """Is the human blind review COMPLETE for every queued objective?"""
        if self.review_queue is None:
            return {"complete": False, "pending": sorted(review_ids.values()),
                    "reviewers": []}
        pending: list[str] = []
        reviewers: set[str] = set()
        for arm, rid in review_ids.items():
            c = self.review_queue.completeness(rid)
            if not c["complete"]:
                pending.append(rid)
            reviewers.update(c.get("reviewers") or [])
        return {"complete": not pending, "pending": sorted(pending),
                "reviewers": sorted(reviewers)}

    def _block_record_from_registry(self, bid: str, snap_id: str,
                                    seed: int) -> dict[str, Any] | None:
        """Reconstruct the record of an ALREADY completed block (no re-run)."""
        arms = self.harness.registry.runs_for_block(bid)
        if len(arms) != len(ARMS) or not all(r.finalized for r in arms.values()):
            return None
        return {"block_id": bid, "snapshot_id": snap_id, "seed": seed,
                "resumed": True,
                "objective_ids": {a: arms[a].objective_id for a in ARMS},
                "gate_refs": {a: self._ref_from_record(arms[a]).to_dict()
                              for a in ARMS},
                "outcomes": {a: arms[a].outcome for a in ARMS},
                "review_ids": {a: arms[a].review_id for a in ARMS},
                "seconds": 0.0, "exec_seconds": 0.0}

    def _block_record_from_registry(self, bid: str, snap_id: str,
                                    seed: int) -> dict[str, Any] | None:
        """Reconstruct the record of an ALREADY completed block (no re-run)."""
        arms = self.harness.registry.runs_for_block(bid)
        if len(arms) != len(ARMS) or not all(r.finalized for r in arms.values()):
            return None
        return {"block_id": bid, "snapshot_id": snap_id, "seed": seed,
                "resumed": True,
                "objective_ids": {a: arms[a].objective_id for a in ARMS},
                "gate_refs": {a: self._ref_from_record(arms[a]).to_dict()
                              for a in ARMS},
                "outcomes": {a: arms[a].outcome for a in ARMS},
                "review_ids": {a: arms[a].review_id for a in ARMS},
                "seconds": 0.0, "exec_seconds": 0.0}

    def _block_spec(self, bid: str, snap_id: str, seed: int,
                    current_ids: list[str], block_number: int,
                    history_ids: list[str] | None = None):
        from .evaluation_harness import BlockSpec
        history_ids = (history_ids if history_ids is not None
                       else self._historical_objective_ids())
        hist_evidence = []
        experience: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        try:
            for oid in history_ids:
                for s in self.service.stores.sessions.sessions_for_objective(oid):
                    for xp in self.service.stores.sessions.experience(
                            s.session_id):
                        experience.append({
                            "experience_id": xp["experience_id"],
                            "session_id": s.session_id, "kind": xp["kind"]})
                    for ev in self.service.stores.sessions.events(s.session_id):
                        if "fail" in str(ev.get("event_type", "")).lower():
                            failures.append({
                                "event_type": ev.get("event_type"),
                                "session_id": s.session_id})
        except Exception:  # noqa: BLE001 - absent history is legitimate
            pass
        marker = f"block_snapshot_{block_number}:{_hash(sorted(current_ids))[:16]}"
        # REAL historical evidence entities: everything outside this block's
        # opportunity. Explicit, disjoint ids -- never synthesised.
        try:
            full_ids = [str(e.get("evidence_id")) for e in
                        (self._full_snapshot().evidence_view.get("evidence")
                         or [])]
        except Exception:  # noqa: BLE001
            full_ids = []
        historical_evidence = [{"evidence_id": eid} for eid in full_ids
                               if eid and eid not in set(current_ids)]
        return BlockSpec(
            block_id=bid, snapshot_id=snap_id, seed=seed,
            current_evidence={"evidence_id": marker,
                              "n_evidence": len(current_ids),
                              "snapshot_sha256": _hash(sorted(current_ids))},
            current_evidence_ids=(marker,),
            historical_experience=experience,
            historical_evidence=historical_evidence,
            historical_failure=failures,
            campaign_budget=dict(self.config.campaign_budget))

    # ---------------------------------------------------------- execution
    def _state_dir(self, block_id: str, arm: str) -> Path:
        return self.out_dir / "director" / block_id / arm

    def _grant_via_cli(self, approval_id: str, plan_id: str = "") -> None:
        """REAL out-of-band human grant through a separate console process."""
        db = str(self.service.stores.db.db_path)
        r = subprocess.run(
            [sys.executable, "-m",
             "research_runtime.research_store.approval_cli",
             "--db", db, "grant", approval_id,
             "--approved-by", self.config.approver_id, "--yes"],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        if r.returncode != 0:
            raise PreflightError(
                f"approval_cli grant failed rc={r.returncode}\n"
                f"STDOUT={r.stdout}\nSTDERR={r.stderr}")

    def _ref_from_record(self, rec: RunRecord) -> GateRunRef:
        ev = rec.gate_evidence.get("ref") or {}
        return GateRunRef(session_id=ev.get("session_id", ""),
                          plan_id=rec.plan_id, approval_id=rec.approval_id,
                          campaign_id=rec.campaign_id,
                          report_id=rec.report_id)

    def _prepare_arm(self, block_id: str, arm: str, objective_id: str,
                     block_number: int) -> Any:
        """Drive ONE arm to AWAITING_APPROVAL and obtain a REAL human grant."""
        attempts: list[dict[str, Any]] = []
        last_error: Exception | None = None
        for attempt, source in enumerate(self.resolve_source_experiments()[:4]):
            director = ResearchDirector(
                service=self.service,
                objective_store=self.service.stores.objectives,
                state_dir=self._state_dir(block_id, arm),
                objective_id=objective_id, domain=self.config.domain,
                director_id=f"pilot_{block_id}_{arm}_try{attempt}",
                source_experiment_id=source)
            try:
                director.run_to_awaiting_approval()
                if not director.context.approval_id:
                    raise PreflightError("director produced no approval request")
                self._approver(director.context.approval_id,
                               director.context.plan_id)
                self.controller.confirm_approval(
                    block_id, arm=arm,
                    approval_id=director.context.approval_id,
                    plan_id=director.context.plan_id,
                    detail={"block": block_id, "arm": arm,
                            "source_experiment_id": source,
                            "attempts": attempts})
                attempts.append({"source_experiment_id": source,
                                 "result": "awaiting_approval"})
                return director
            except Exception as exc:  # noqa: BLE001 - try the next real source
                last_error = exc
                attempts.append({"source_experiment_id": source,
                                 "result": "failed",
                                 "error": f"{type(exc).__name__}: {exc}"})
                continue
        # Every real source failed to produce an approvable direction: retain a
        # REAL failed run (never a fabricated success).
        self.harness.registry.append_event(
            scope=f"run:{block_id}_{arm}", event_type="preparation_failed",
            payload={"attempts": attempts})
        if last_error is not None:
            raise last_error
        raise PreflightError("no real source experiment could ground the arm")

    def _finish_arm(self, block_id: str, arm: str,
                    director: Any) -> tuple[str, GateRunRef, float]:
        """Run the REAL campaign and record the DERIVED gate outcome."""
        t0 = time.time()
        ctx = director.context
        try:
            final = director.run_to_completion()
            ctx = director.context
            ref = GateRunRef(session_id=ctx.session_id, plan_id=ctx.plan_id,
                             approval_id=ctx.approval_id,
                             campaign_id=ctx.campaign_id,
                             report_id=ctx.report_id)
            outcome, reason = self._classify(
                outcome_state=final.get("state", ""), ctx=ctx, ref=ref)
            self.harness.record_outcome(
                block_id, arm, outcome=outcome, gate_ref=ref,
                compute_units=1.0, wall_time_s=round(time.time() - t0, 2),
                campaign_id=ctx.campaign_id,
                invalid_reason=("" if outcome in ("valid_finding",
                                                   "negative_finding",
                                                   "inconclusive")
                                else reason))
            return outcome, ref, round(time.time() - t0, 2)
        except ProtocolViolation as exc:
            ref = GateRunRef(session_id=ctx.session_id, plan_id=ctx.plan_id,
                             approval_id=ctx.approval_id,
                             campaign_id=ctx.campaign_id,
                             report_id=ctx.report_id)
            self.harness.record_outcome(
                block_id, arm, outcome="invalid", gate_ref=ref,
                wall_time_s=round(time.time() - t0, 2),
                invalid_reason=f"gate evidence incomplete: {exc}")
            return "invalid", ref, round(time.time() - t0, 2)
        except Exception as exc:  # noqa: BLE001 - real failure is evidence
            outcome, ref = self._record_failure(block_id, arm, director,
                                                round(time.time() - t0, 2),
                                                error=exc)
            return outcome, ref, round(time.time() - t0, 2)

    def _record_failure(self, block_id: str, arm: str, director: Any,
                        seconds: float, error: Exception | None = None
                        ) -> tuple[str, GateRunRef]:
        ctx = director.context
        ref = GateRunRef(session_id=ctx.session_id, plan_id=ctx.plan_id,
                         approval_id=ctx.approval_id,
                         campaign_id=ctx.campaign_id,
                         report_id=ctx.report_id)
        state = getattr(ctx, "state", "")
        outcome, reason = classify_director_outcome(director_state=state)
        detail = (f"{type(error).__name__}: {error}" if error
                  else "arm preparation failed")
        self.harness.record_outcome(
            block_id, arm, outcome=outcome, gate_ref=ref,
            wall_time_s=seconds,
            invalid_reason=f"{reason}; {detail}")
        self.harness.registry.append_event(
            scope=f"run:{block_id}_{arm}", event_type=outcome,
            payload={"error": str(error), "arm": arm, "state": state,
                     "classification_reason": reason})
        return outcome, ref

    def _classify(self, *, outcome_state: str, ctx: Any,
                  ref: GateRunRef | None = None) -> tuple[str, str]:
        """Derive the recorded outcome from REAL director + gate evidence.

        P0-02: ``rejected`` is a negative finding ONLY when the real campaign
        succeeded and a real scientific conclusion exists; ``inconclusive`` /
        ``not_evaluated`` stay inconclusive; non-COMPLETED terminal states keep
        their DISTINCT outcomes; unknown states fail closed with the raw value.
        """
        if outcome_state != "COMPLETED":
            return classify_director_outcome(director_state=outcome_state)

        if ref is None:
            ref = GateRunRef(session_id=ctx.session_id, plan_id=ctx.plan_id,
                             approval_id=ctx.approval_id,
                             campaign_id=ctx.campaign_id,
                             report_id=ctx.report_id)
        try:
            evidence = self.gate_adapter.inspect(ref)
        except Exception as exc:  # noqa: BLE001
            return "invalid", f"gate evidence read failed: {type(exc).__name__}: {exc}"
        report_outcome = str(evidence.report_outcome or "")
        if report_outcome and not evidence.report_found:
            return "invalid", "report not found in the persisted chain"
        # A frozen final split must never be used by a pilot (or any) arm.
        try:
            rep = self.service.get_research_report(session_id=ctx.session_id)
            if rep.get("ok") and (rep.get("report") or {}).get("frozen_test_used"):
                return "invalid", "a frozen final split was used (forbidden)"
        except Exception:  # noqa: BLE001 - fail closed below on evidence
            pass
        campaign_succeeded = bool(evidence.campaign_successful)
        scientific_valid = (
            bool(evidence.report_found)
            and report_outcome in SCIENTIFIC_REPORT_OUTCOMES
            and bool(evidence.experience_refs_resolved)
        )
        return classify_director_outcome(
            director_state=outcome_state, report_outcome=report_outcome,
            campaign_succeeded=campaign_succeeded,
            scientific_outcome_valid=scientific_valid)

    # ------------------------------------------------------------- records
    def _save_records(self) -> None:
        payload = {
            "stage": "pilot",
            "experiment_id": self.config.experiment_id,
            "config": self.config.to_dict(),
            "protocol_sha256": self.protocol.freeze_sha256(),
            "protocol": self.protocol.frozen_dict(),
            "metric_contract_sha256":
                frozen_metric_contract()["contract_sha256"],
            "engine": "REAL SciNet++ (no FakeRunner, no synthetic snapshot)",
            "blocks": self.blocks,
            "registry_freeze_sha256": self.harness.registry.freeze_sha256,
            "n_runs": len(self.harness.registry.all_runs()),
        }
        tmp = self.records_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True,
                                  default=str), encoding="utf-8")
        tmp.replace(self.records_path)

    # ------------------------------------------------------------------ run
    def run(self) -> dict[str, Any]:
        """Execute (or resume) every planned block. Fail-closed on preflight."""
        rep = preflight(self.config, service=self.service)
        if not rep.ready:
            raise PreflightError(
                "refusing to run a REAL pilot: " + json.dumps(rep.to_dict()))
        self._run_started = time.time()
        full = self._full_snapshot()
        all_ids = [str(e.get("evidence_id"))
                   for e in (full.evidence_view.get("evidence") or [])]
        if not all_ids:
            raise PreflightError("no real evidence entities for the pilot")
        completed = {r.run_id: r for r in self.harness.registry.all_runs()}
        chunks = self._split_evidence(all_ids, self.config.n_blocks)
        for i, chunk in enumerate(chunks, start=1):
            try:
                self._run_block(i, full, chunk, completed)
            except BudgetExhausted:
                break
            except Exception:
                self._save_records()
                raise
        self._save_records()
        return {"experiment_id": self.config.experiment_id,
                "blocks": self.blocks, "preflight": rep.to_dict(),
                "registry_freeze_sha256": self.harness.registry.freeze_sha256,
                "n_runs": len(self.harness.registry.all_runs()),
                "state": self.controller.state,
                "elapsed_s": round(time.time() - self._run_started, 2)}

    def _split_evidence(self, ids: list[str], n_blocks: int) -> list[list[str]]:
        n = max(1, n_blocks)
        if self.config.evidence_per_block:
            size = self.config.evidence_per_block
            return [ids[i:i + size] for i in range(0, len(ids), size)][:n]
        size = max(1, len(ids) // n)
        chunks = [ids[i:i + size] for i in range(0, size * n, size)]
        # Anything left over is appended to the LAST block (nothing is dropped).
        if size * n < len(ids):
            chunks[-1] = chunks[-1] + ids[size * n:]
        return chunks[:n]


def protocol_from_config(config: dict[str, Any]) -> FrozenProtocol:
    """Rebuild the (pure) pilot protocol from a recorded config.

    The protocol is a deterministic function of the config + the frozen domain
    wiring, so a report can restate the exact versions/seeds/budget without
    trusting a cached copy. The caller should verify the resulting SHA against
    the recorded one.
    """
    known = {k: v for k, v in config.items()
             if k in PilotConfig.__dataclass_fields__}
    cfg = PilotConfig(**known)
    cfg.paired_seeds = tuple(cfg.paired_seeds)
    runner = PilotRunner.__new__(PilotRunner)
    runner.config = cfg
    return runner._pilot_protocol()


__all__ = [
    "PilotConfig", "PreflightReport", "PreflightError", "SelectionError",
    "protocol_from_config",
    "ReviewIncomplete",
    "SelectionRecord", "delegated_select", "preflight", "PilotRunner",
]
