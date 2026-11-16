"""M4.3-01 / M4.3-01.1 -- A/B/C Evaluation Harness (NO ResearchService modification).

This is an INDEPENDENT experiment protocol/executor module. It does NOT create
a second ResearchService / Planner / Decision Engine, and it NEVER re-implements
or relaxes the 12-step fail-closed gate. All three arms go through the SAME
existing gate: ``ResearchServiceGateAdapter`` is a READ-ONLY observer bound to
ONE real ResearchService, and every recorded gate verdict is DERIVED from that
service's persisted Approval / Campaign / Report / Experience records (P0-01).
A caller can no longer *claim* ``gate_steps_passed=12`` or
``external_approval=True``: those values come from ``GateEvidence`` only.

Groups (spec M4.3-01):
  A  Manual                  -- human picks an Objective from CURRENT evidence,
                               with NO historical Experience context.
  B  Memory-assisted Manual  -- human reads the SAME historical Evidence,
                               Experience and Failure context, then picks.
  C  Autonomous Discovery    -- ObjectiveCandidateGenerator + Critic + Ranker.

Control rules enforced here:
  1. same research-opportunity Snapshot for A/B/C (blocked pairing);
  2. A reads NO historical Experience, B/C read the SAME historical context;
  3. fixed model / prompt / Ranker version / DomainPolicy;
  4. fixed candidate budget and human selection time budget per block;
  5. SAME 12-step gate for all arms (recorded, never re-implemented);
  6. external Approval for all arms;
  7. same Campaign Budget;
  8. same Paired Seeds for applicable training;
  9. ALL outcomes retained (failed / invalid / INCONCLUSIVE / interrupted);
 10. primary metric is frozen and cannot be changed after seeing results.

The harness NEVER draws a significance conclusion. Pilot (M4.3-02) and
Confirmatory (M4.3-03) are separate stages; this module only provides the
recorded data and the metric computation they consume. It does NOT start
Pilot or Confirmatory runs by itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .experiment_registry import (
    ARMS, ARM_DESCRIPTIONS, RUN_OUTCOMES, ExperimentRegistry, FrozenProtocol,
    RunRecord,
)
from .blind_review import (
    FORBIDDEN_BLIND_FIELDS, REVIEW_KIND_HUMAN_BLIND, BlindingLedger,
    BlindingViolation, IndependentReviewer, ReviewPanel,
)
from .evidence_boundary import (
    BlockBudget, EvidenceBoundaryViolation, EvidencePartition,
    validate_block_budget, validate_evidence_partition,
)
from .metrics import (
    ALL_METRICS, METRIC_CATEGORIES, METRIC_FORMULA_VERSION,
    METRIC_DEFINITIONS, PRIMARY_METRICS, SAFETY_METRICS, SECONDARY_METRICS,
    compute_metric_set, frozen_metric_contract, metric_definitions,
)
from .research_service_gate import (
    FORGED_GATE_KEYS, GATED_OUTCOMES, GateAdapter, GateEvidence, GateRunRef,
)

# Fixed, frozen arm-context rules. A must NOT see Experience; B and C see the
# SAME historical context. These are NOT tunable from outcomes.
ARM_CONTEXT_RULES: dict[str, dict[str, bool]] = {
    "A": {"read_historical_experience": False,
          "read_historical_evidence": False,
          "read_historical_failure": False,
          "use_autonomous_discovery": False},
    "B": {"read_historical_experience": True,
          "read_historical_evidence": True,
          "read_historical_failure": True,
          "use_autonomous_discovery": False},
    "C": {"read_historical_experience": True,
          "read_historical_evidence": True,
          "read_historical_failure": True,
          "use_autonomous_discovery": True},
}


class ProtocolViolation(Exception):
    """Raised when a control rule is broken (fail-closed, no silent fix)."""


@dataclass(frozen=True)
class ArmContext:
    """The research context an arm is ALLOWED to see."""
    arm: str
    snapshot_id: str
    seed: int
    current_evidence: dict[str, Any] = field(default_factory=dict)
    historical_experience: tuple[dict[str, Any], ...] = ()
    historical_evidence: tuple[dict[str, Any], ...] = ()
    historical_failure: tuple[dict[str, Any], ...] = ()
    candidate_budget: int = 1
    human_time_budget_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm, "snapshot_id": self.snapshot_id, "seed": self.seed,
            "n_current_evidence": len(self.current_evidence),
            "n_historical_experience": len(self.historical_experience),
            "n_historical_evidence": len(self.historical_evidence),
            "n_historical_failure": len(self.historical_failure),
            "candidate_budget": self.candidate_budget,
            "human_time_budget_s": self.human_time_budget_s,
        }


def build_arm_context(*, arm: str, snapshot_id: str, seed: int,
                      current_evidence: dict[str, Any],
                      historical_experience: list[dict[str, Any]],
                      historical_evidence: list[dict[str, Any]],
                      historical_failure: list[dict[str, Any]],
                      candidate_budget: int,
                      human_time_budget_s: float) -> ArmContext:
    """Enforce rule #2: A sees NO history; B/C see the SAME history."""
    if arm not in ARMS:
        raise ProtocolViolation(f"unknown arm {arm!r}")
    rules = ARM_CONTEXT_RULES[arm]
    if not rules["read_historical_experience"]:
        if historical_experience or historical_evidence or historical_failure:
            raise ProtocolViolation(
                "arm A must not receive historical Experience/Evidence/Failure")
        return ArmContext(
            arm=arm, snapshot_id=snapshot_id, seed=seed,
            current_evidence=dict(current_evidence),
            candidate_budget=candidate_budget,
            human_time_budget_s=human_time_budget_s)
    return ArmContext(
        arm=arm, snapshot_id=snapshot_id, seed=seed,
        current_evidence=dict(current_evidence),
        historical_experience=tuple(historical_experience),
        historical_evidence=tuple(historical_evidence),
        historical_failure=tuple(historical_failure),
        candidate_budget=candidate_budget,
        human_time_budget_s=human_time_budget_s)


@dataclass
class BlockSpec:
    """ONE research opportunity with the SAME context for all three arms.

    P1-02: current and historical evidence are separated by EXPLICIT,
    DISJOINT ID sets (``current_evidence_ids`` / ``historical_ids``). When left
    empty the harness DERIVES them from the records' explicit IDs (or a stable
    content hash) and still enforces disjointness. No arm may read a frozen
    final split. ``campaign_budget`` is fixed per block.
    """
    block_id: str
    snapshot_id: str
    seed: int
    current_evidence: dict[str, Any] = field(default_factory=dict)
    historical_experience: list[dict[str, Any]] = field(default_factory=list)
    historical_evidence: list[dict[str, Any]] = field(default_factory=list)
    historical_failure: list[dict[str, Any]] = field(default_factory=list)
    # explicit ID sets (P1-02)
    current_evidence_ids: tuple[str, ...] = ()
    historical_ids: tuple[str, ...] = ()
    campaign_budget: dict[str, Any] = field(default_factory=dict)


class EvaluationHarness:
    """Orchestrates A/B/C recording + metrics, WITHOUT running the engine.

    ``objective_selector`` is an injected callback receiving an ``ArmContext``
    and returning a candidate objective dict (for A/B a human decision is
    represented by this callback; for C the operator wires the real
    ``ObjectiveDiscovery``). ``reviewer`` is an ``IndependentReviewer``.

    ``gate_adapter`` (P0-01) MUST be a real ``GateAdapter`` bound to ONE
    ResearchService. It is not optional decoration: ``record_outcome`` calls
    ``inspect()`` for EVERY outcome and derives the gate verdict, step count and
    external-approval flag from the persisted records. There is no optimistic
    default and no caller-supplied verdict. A harness built without an adapter
    is FAIL-CLOSED: no completed outcome (``valid_finding`` etc.) can be
    recorded, because no real gate evidence can be produced.
    """

    def __init__(self, *, protocol: FrozenProtocol,
                 reviewer: IndependentReviewer | None = None,
                 review_panel: ReviewPanel | None = None,
                 blinding: BlindingLedger | None = None,
                 objective_selector: Callable[[ArmContext], dict[str, Any]] | None = None,
                 gate_adapter: GateAdapter | None = None,
                 default_gate_ref: GateRunRef | None = None,
                 db_path: str | Path | None = None, db: Any = None,
                 store: Any = None, registry: ExperimentRegistry | None = None):
        # P0-02: a Pilot defaults to >= 2 independent blind human reviewers.
        # A single ``reviewer`` is accepted for engine-free contract tests, but
        # ``assert_pilot_review_ready()`` refuses to start without >= 2.
        if review_panel is None:
            if reviewer is None:
                raise ProtocolViolation("at least one reviewer is required")
            review_panel = ReviewPanel([reviewer])
        self.reviewer = reviewer
        self.review_panel = review_panel
        self.blinding = blinding or BlindingLedger()
        # P0-03: optional durable registry (ARR SQLite). ``registry`` lets a
        # recovered registry be injected directly.
        self.registry = registry or ExperimentRegistry(
            protocol=protocol, store=store, db_path=db_path, db=db)
        self.protocol = protocol
        self.objective_selector = objective_selector
        # ONE adapter shared by A/B/C -> the SAME ResearchService entry point.
        self.gate_adapter = gate_adapter
        self.default_gate_ref = default_gate_ref
        self._primary_frozen = tuple(protocol.primary_metrics)
        self._contexts: dict[str, dict[str, ArmContext]] = {}
        self._blocks_finalized: set[str] = set()
        self._partitions: dict[str, EvidencePartition] = {}
        self._budgets: dict[str, BlockBudget] = {}

    def evidence_partition(self, block_id: str) -> EvidencePartition:
        return self._partitions[block_id]

    def block_budget(self, block_id: str) -> BlockBudget:
        return self._budgets[block_id]

    @classmethod
    def recover(cls, *, protocol: FrozenProtocol,
                db_path: str | Path | None = None, db: Any = None,
                reviewer: IndependentReviewer | None = None,
                gate_adapter: GateAdapter | None = None,
                default_gate_ref: GateRunRef | None = None
                ) -> EvaluationHarness:
        """P0-03: rebuild the harness after a process restart."""
        reg = ExperimentRegistry.recover(
            protocol=protocol, db_path=db_path, db=db)
        return cls(protocol=protocol, reviewer=reviewer or IndependentReviewer(
            reviewer_id="recovered_reviewer",
            scorer=lambda _b: {d: 0 for d in
                               frozen_metric_contract()["rubric_dimensions"]}
            | {"reason": "recovered; no new review"}),
            registry=reg, gate_adapter=gate_adapter,
            default_gate_ref=default_gate_ref)

    def export(self) -> dict[str, Any]:
        return self.registry.export()

    def verify_integrity(self) -> bool:
        return self.registry.verify_integrity()

    # ------------------------------------------------- P0-02 pilot review gate
    def assert_pilot_review_ready(self) -> None:
        """Pilot default: >= 2 DISTINCT independent, blind human reviewers."""
        if not self.review_panel.is_independent:
            raise ProtocolViolation(
                "Pilot requires at least two independent blind human reviewers "
                "(distinct reviewer_id, review_kind=human_blind)")

    def unblind(self) -> None:
        """Allowed ONLY after the review phase is complete (P0-02 #6)."""
        self.blinding.unblind()

    def arm_mapping(self, review_id: str) -> dict[str, Any]:
        """SEALED until unblind(); reviewers must never see the arm."""
        return self.blinding.arm_mapping(review_id)

    # ------------------------------------------------------------- arm setup
    def add_block(self, spec: BlockSpec, *, candidate_budget: int,
                  human_time_budget_s: float) -> dict[str, ArmContext]:
        if spec.block_id in self._contexts:
            raise ProtocolViolation(f"block {spec.block_id!r} already added")
        # (P1-02) Validate BOTH boundaries BEFORE persisting anything:
        # explicit disjoint current/history ID sets + no frozen final split,
        # and fixed per-block budgets. Fail closed, no silent fixes.
        partition = validate_evidence_partition(
            snapshot_id=spec.snapshot_id,
            current_evidence=spec.current_evidence,
            historical_experience=spec.historical_experience,
            historical_evidence=spec.historical_evidence,
            historical_failure=spec.historical_failure,
            explicit_current_ids=(tuple(spec.current_evidence_ids)
                                  if spec.current_evidence_ids else None),
            explicit_historical_ids=(tuple(spec.historical_ids)
                                     if spec.historical_ids else None))
        budget = validate_block_budget(
            candidate_budget=candidate_budget,
            human_time_budget_s=human_time_budget_s,
            campaign_budget=(spec.campaign_budget
                             or self.protocol.campaign_budget
                             or self.protocol.budget),
            protocol_campaign_budget=(self.protocol.campaign_budget or None))
        self.registry.create_block(
            block_id=spec.block_id, snapshot_id=spec.snapshot_id,
            seed=spec.seed,
            snapshot={"block_id": spec.block_id,
                      "snapshot_id": spec.snapshot_id, "seed": spec.seed,
                      "current_evidence": spec.current_evidence,
                      # P1-02: history is separated by EXPLICIT ID sets; the
                      # snapshot SHA covers the shared opportunity only.
                      "historical_experience": spec.historical_experience,
                      "historical_evidence": spec.historical_evidence,
                      "historical_failure": spec.historical_failure,
                      "evidence_partition": partition.to_dict(),
                      "block_budget": budget.to_dict()})
        self._partitions[spec.block_id] = partition
        self._budgets[spec.block_id] = budget
        contexts: dict[str, ArmContext] = {}
        # Rule #1/#3/#4: identical snapshot, seed, budgets for every arm.
        # Rule #2: a BlockSpec describes ONE shared research opportunity, but
        # arm A is NOT permitted to READ its historical context. The harness
        # therefore MASKS history for A while handing B and C the SAME history.
        # ``build_arm_context`` still fails closed if anyone tries to pass
        # history to A directly (defense in depth).
        for arm in ARMS:
            allow_history = ARM_CONTEXT_RULES[arm]["read_historical_experience"]
            contexts[arm] = build_arm_context(
                arm=arm, snapshot_id=spec.snapshot_id, seed=spec.seed,
                current_evidence=spec.current_evidence,
                historical_experience=(list(spec.historical_experience)
                                       if allow_history else []),
                historical_evidence=(list(spec.historical_evidence)
                                     if allow_history else []),
                historical_failure=(list(spec.historical_failure)
                                    if allow_history else []),
                candidate_budget=candidate_budget,
                human_time_budget_s=human_time_budget_s)
        # A sees NO history; B and C see EXACTLY the same history.
        assert not contexts["A"].historical_experience
        assert contexts["B"].historical_experience == \
            contexts["C"].historical_experience
        assert contexts["B"].historical_evidence == \
            contexts["C"].historical_evidence
        assert contexts["B"].historical_failure == \
            contexts["C"].historical_failure
        self._contexts[spec.block_id] = contexts
        return contexts

    # ------------------------------------------------------------- selection
    def select_objective(self, block_id: str, arm: str,
                         objective: dict[str, Any] | None = None) -> dict[str, Any]:
        """Record the selected objective for one arm.

        For A/B an operator supplies the human-selected objective. For C, if
        none is supplied, ``objective_selector`` is invoked with the SAME
        ArmContext (wired to the real Discovery by the operator). The Ranker
        NEVER participates in quality scoring here.
        """
        ctx = self._contexts[block_id][arm]
        if objective is None:
            if self.objective_selector is None:
                raise ProtocolViolation("no objective_selector configured")
            objective = self.objective_selector(ctx)
        if not objective or not objective.get("objective_id"):
            raise ProtocolViolation("selected objective has no objective_id")
        # P0-02: strip EVERY group/Ranker/identity field before review. A C-arm
        # selector may legitimately return a ranked candidate, but the reviewer
        # must never see those signals. ``objective_id`` is kept ONLY for the
        # sealed mapping (it never reaches the blinded payload).
        forbidden = FORBIDDEN_BLIND_FIELDS - {"objective_id"}
        if forbidden.intersection(objective):
            objective = {k: v for k, v in objective.items()
                         if k not in forbidden}
        # Seal the opaque review_id -> (objective_id, arm, ...) mapping apart
        # from the reviewer-visible payload.
        blinded = self.blinding.seal(
            objective, arm=arm, block_id=block_id,
            snapshot_id=ctx.snapshot_id)
        self.registry.register_run(
            block_id=block_id, arm=arm, objective_id=objective["objective_id"],
            evidence_refs=list(objective.get("evidence_refs") or []),
            review_id=blinded.review_id,
            notes={"selection_context": ctx.to_dict(),
                   "review_id": blinded.review_id})
        # Independent, blind, fixed-rubric review by the WHOLE panel.
        reviews = self.review_panel.review(blinded)
        human = [r for r in reviews if r.is_human_blind]
        automated = [r for r in reviews if not r.is_human_blind]
        for r in reviews:
            self.blinding.record_review(r)   # append-only, never overwritten
            self.registry.record_review(r.to_dict())   # P0-03 durable
        if not human:
            raise ProtocolViolation(
                "at least one independent human blind review is required")
        self.registry.update(
            f"{block_id}_{arm}",
            quality_review=human[0].to_dict(),   # PRIMARY human-blind review
            independent_reviews=[r.to_dict() for r in human],
            automated_scores=[r.to_dict() for r in automated])
        return objective

    # ------------------------------------------------------------- outcomes
    def record_outcome(self, block_id: str, arm: str, *, outcome: str,
                       gate_ref: GateRunRef | None = None,
                       paired_seeds: list[int] | None = None,
                       compute_units: float = 0.0,
                       wall_time_s: float = 0.0,
                       campaign_id: str = "",
                       **flags: Any) -> RunRecord:
        """Retain EVERY outcome; the gate verdict is DERIVED, never claimed.

        P0-01 contract (no optimistic defaults, no caller-forged verdicts):
          * ``gate_steps_passed`` / ``external_approval`` are NOT parameters.
            They are computed from the persisted Approval / Campaign / Report /
            Experience chain via the READ-ONLY ``gate_adapter`` (one instance,
            shared by A/B/C).
          * a caller attempting to pass ANY gate-verdict keyword is rejected
            outright (fail-closed), not silently ignored.
          * a completed outcome (valid_finding / negative_finding /
            inconclusive) requires the COMPLETE real gate chain; otherwise it
            is refused and nothing is written as a success.
          * no real human Approval -> ``approval_bypass`` is recorded (and a
            completed outcome is refused, i.e. fail-closed).
          * no real Campaign / Report / Experience -> no ``valid_finding``.
          * failed / interrupted / invalid runs are ALWAYS retained (real
            failures and interruptions are evidence too), with whatever gate
            evidence actually exists.
        """
        if outcome not in RUN_OUTCOMES:
            raise ProtocolViolation(f"unknown outcome {outcome!r}")

        # (P0-01 #4) Reject ANY caller-supplied gate verdict / step count.
        forged = FORGED_GATE_KEYS.intersection(flags)
        if forged:
            raise ProtocolViolation(
                "caller-supplied gate verdict is forbidden: "
                f"{sorted(forged)}; gate status is derived from real "
                "persisted records via the read-only gate_adapter only")

        ref = gate_ref or self.default_gate_ref or GateRunRef()
        if self.gate_adapter is None:
            # (P0-01 #6) No adapter -> NO gate evidence at all. Fail closed.
            evidence = GateEvidence(
                ref=ref, errors=("no_real_gate_adapter_configured",))
        else:
            evidence = self.gate_adapter.inspect(ref)

        # (P0-01 #6/#7) Completed outcomes need the full real gate chain.
        if outcome in GATED_OUTCOMES and not evidence.gate_passed:
            missing = ", ".join(evidence.missing_artefacts) or "none"
            detail = "; ".join(evidence.errors + evidence.scope_conflicts)
            raise ProtocolViolation(
                f"cannot record {outcome!r}: real gate evidence incomplete "
                f"(missing: {missing}"
                + (f"; read errors: {detail}" if detail else "") +
                "). Gate status comes from persisted Approval/Campaign/"
                "Report/Experience records, not a caller claim.")

        # (M4.3-02 P0) A valid_finding needs MORE than a terminated campaign:
        # the campaign must have SUCCEEDED and the report must carry a real
        # scientific outcome. A failed/cancelled campaign, a campaign whose
        # only trials failed, or a comparison that never produced an outcome
        # can never support a valid finding. ``negative_finding`` stays legal
        # because a genuinely completed experiment that found nothing IS a
        # result -- it just needs ``gate_passed`` (checked above).
        if outcome == "valid_finding" and not evidence.valid_finding_evidence:
            raise ProtocolViolation(
                "cannot record 'valid_finding': the persisted chain is not "
                "evidence of a successful experiment "
                f"(campaign_status={evidence.campaign_status!r}, "
                f"successful={evidence.campaign_successful}, "
                f"trial_evidence={evidence.trial_evidence!r}, "
                f"report_outcome={evidence.report_outcome!r}). "
                "A terminal-but-failed/cancelled campaign, a campaign with no "
                "successful trial, or a report without a real scientific "
                "outcome cannot produce a valid finding.")

        # Derived ONLY. Never from the caller.
        derived_steps = evidence.gate_steps_passed
        external = evidence.external_approval
        if not external:
            # (P0-01 #6) No REAL approved human Approval -> bypass recorded.
            flags["approval_bypass"] = True

        # (P0-03 #6) Persist ALL trusted IDs from the real gate evidence, plus
        # the frozen Protocol / Snapshot SHA, so an exported registry is
        # self-contained. Experience ID comes from the persisted Experience
        # refs, never from a caller claim.
        exp_refs = evidence.report_experience_refs
        rec = self.registry.finalize(
            f"{block_id}_{arm}", outcome=outcome,
            gate_steps_passed=derived_steps,
            external_approval=external,
            paired_seeds=list(paired_seeds or self._paired_seeds(block_id)),
            compute_units=compute_units, wall_time_s=wall_time_s,
            campaign_id=campaign_id or ref.campaign_id,
            plan_id=ref.plan_id, approval_id=ref.approval_id,
            report_id=ref.report_id,
            experience_id=(exp_refs[0] if exp_refs else ""),
            gate_evidence=evidence.to_dict(),
            **flags)
        if outcome in ("failed", "interrupted", "invalid"):
            # (P0-03 #9) REAL failure / interruption is retained as an event.
            self.registry.append_event(
                scope=f"run:{rec.run_id}", event_type=outcome,
                payload={"outcome": outcome, "block_id": block_id, "arm": arm,
                         "campaign_id": rec.campaign_id})
        return rec

    def _paired_seeds(self, block_id: str) -> list[int]:
        # Rule #8: same paired seeds across arms (from frozen protocol budget).
        # P0-04: ``paired_seeds`` is now a first-class frozen field; ``budget``
        # remains a backwards-compatible alias.
        seeds = self.protocol.paired_seeds or \
            self.protocol.budget.get("paired_seeds")
        return list(seeds) if seeds else []

    # ------------------------------------------------------------- metrics
    def compute_metrics(self) -> dict[str, Any]:
        """ALL 16 metrics across ALL runs (overall, never the only view)."""
        return compute_metric_set(self.registry.all_runs())

    def metrics_by_arm(self) -> dict[str, dict[str, Any]]:
        """P1-01: ALL 16 metrics for EACH of A/B/C -- not just a merged mean.

        Uses the SAME frozen ``compute_metric_set`` implementation as
        ``compute_metrics`` so overall and per-arm can never diverge.
        """
        out: dict[str, dict[str, Any]] = {}
        all_runs = self.registry.all_runs()
        for arm in ARMS:
            runs = [r for r in all_runs if r.arm == arm]
            values = compute_metric_set(runs)
            out[arm] = {
                "n_runs": len(runs),
                "n_valid_findings": sum(1 for r in runs
                                        if r.is_valid_finding()),
                # all 16 metrics by name
                **values,
                # P1-01 #7: explicit category split
                "by_category": {
                    cat: {name: values[name] for name in names}
                    for cat, names in METRIC_CATEGORIES.items()},
                # P1-01 #2: numerator/denominator/missing/failure rules
                "metric_definitions": metric_definitions(),
            }
        return out

    def paired_metrics(self) -> dict[str, Any]:
        """Primary metrics PAIRED BY BLOCK (Snapshot / Block / Seed hierarchy).

        P1-01 #3/#4/#5/#6:
          * only COMPLETE blocks (all three arms registered AND finalized) are
            eligible for the full paired analysis;
          * every arm of a block MUST use the SAME paired seeds (fail-closed);
          * the Snapshot -> Block -> Seed hierarchy is preserved explicitly;
          * NO significance conclusion is computed (M4.3-03 only).
        """
        complete = set(self.registry.complete_blocks())
        finalized = set(self.registry.finalized_blocks())
        eligible = sorted(complete & finalized)
        blocks: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        for bid in sorted(self.registry.blocks):
            b = self.registry.blocks[bid]
            paired = self.registry.runs_for_block(bid)
            reasons = []
            if bid not in complete:
                reasons.append("incomplete block (not all three arms registered)")
            if bid not in finalized:
                reasons.append("not all arms finalized (interrupted run)")
            seed_sets = {arm: tuple(r.paired_seeds)
                         for arm, r in paired.items()}
            if seed_sets and len(set(seed_sets.values())) > 1:
                reasons.append("arms do not share the same Paired Seeds")
            if reasons:
                excluded.append({"block_id": bid, "snapshot_id": b.snapshot_id,
                                 "seed": b.seed, "reasons": reasons,
                                 "seed_sets": {k: list(v) for k, v in
                                               seed_sets.items()}})
                continue
            arm_metrics = {arm: {
                name: compute_metric_set([paired[arm]])[name]
                for name in PRIMARY_METRICS}
                for arm in ARMS}
            blocks.append({
                "snapshot_id": b.snapshot_id,   # level 1: Snapshot
                "block_id": bid,                # level 2: Block
                "seed": b.seed,                 # level 3: Seed
                "paired_seeds": list(next(iter(seed_sets.values())))
                if seed_sets else [],
                "arms": arm_metrics,
            })
        return {
            "hierarchy": ("snapshot", "block", "seed"),
            "paired_primary_metrics": list(PRIMARY_METRICS),
            "n_eligible_blocks": len(blocks),
            "eligible_blocks": blocks,
            "excluded_blocks": excluded,
            "n_excluded_blocks": len(excluded),
            "significance_conclusion": None,   # P1-01 #10: NOT computed here
        }

    def assert_paired_seeds_consistent(self) -> None:
        """P1-01 #5: fail closed if any complete block has unequal seeds."""
        for bid in self.registry.complete_blocks():
            paired = self.registry.runs_for_block(bid)
            seed_sets = {tuple(r.paired_seeds) for r in paired.values()}
            if len(seed_sets) > 1:
                raise ProtocolViolation(
                    f"block {bid!r} arms do not share the same Paired Seeds: "
                    f"{sorted({tuple(r.paired_seeds) for r in paired.values()})}")

    def assert_no_posthoc_metric_change(self) -> None:
        """Rule #10: primary metrics must equal the frozen set."""
        if tuple(self.protocol.primary_metrics) != self._primary_frozen or \
                tuple(self.protocol.primary_metrics) != PRIMARY_METRICS:
            raise ProtocolViolation(
                "primary metrics changed after the protocol was frozen")

    def snapshot(self) -> dict[str, Any]:
        self.assert_no_posthoc_metric_change()
        return {
            "protocol": self.protocol.frozen_dict(),
            "freeze_sha256": self.registry.freeze_sha256,
            "metric_contract": frozen_metric_contract(),
            "arm_context_rules": ARM_CONTEXT_RULES,
            "arm_descriptions": ARM_DESCRIPTIONS,
            "metric_names": {"primary": list(PRIMARY_METRICS),
                             "secondary": list(SECONDARY_METRICS),
                             "safety": list(SAFETY_METRICS)},
            "metric_formula_version": METRIC_FORMULA_VERSION,
            "metric_categories": {k: list(v)
                                  for k, v in METRIC_CATEGORIES.items()},
            "complete_blocks": self.registry.complete_blocks(),
            "registry": self.registry.to_dict(),
            "metrics": self.compute_metrics(),
            "metrics_by_arm": self.metrics_by_arm(),
            "paired_metrics": self.paired_metrics(),
        }


__all__ = [
    "ARM_CONTEXT_RULES", "ProtocolViolation", "ArmContext", "BlockSpec",
    "build_arm_context", "EvaluationHarness",
]
