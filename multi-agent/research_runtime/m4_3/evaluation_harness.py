"""M4.3-01 -- A/B/C Evaluation Harness (NO ResearchService modification).

This is an INDEPENDENT experiment protocol/executor module. It does NOT create
a second ResearchService / Planner / Decision Engine. All three arms go through
the SAME existing 12-step fail-closed gate (represented here by a read-only
``gate_adapter`` that the operator wires to the real service); the harness only
orchestrates arm context, records outcomes, and computes metrics.

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
from typing import Any, Callable

from .experiment_registry import (
    ARMS, ARM_DESCRIPTIONS, ExperimentRegistry, FrozenProtocol, RunRecord,
)
from .metrics import (
    ALL_METRICS, PRIMARY_METRICS, SAFETY_METRICS, SECONDARY_METRICS,
    IndependentReviewer, QualityReview, blind, frozen_metric_contract,
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
    """ONE research opportunity with the SAME context for all three arms."""
    block_id: str
    snapshot_id: str
    seed: int
    current_evidence: dict[str, Any] = field(default_factory=dict)
    historical_experience: list[dict[str, Any]] = field(default_factory=list)
    historical_evidence: list[dict[str, Any]] = field(default_factory=list)
    historical_failure: list[dict[str, Any]] = field(default_factory=list)


class EvaluationHarness:
    """Orchestrates A/B/C recording + metrics, WITHOUT running the engine.

    ``objective_selector`` is an injected callback receiving an ``ArmContext``
    and returning a candidate objective dict (for A/B a human decision is
    represented by this callback; for C the operator wires the real
    ``ObjectiveDiscovery``). ``reviewer`` is an ``IndependentReviewer``.
    ``gate_adapter`` is a READ-ONLY adapter that the operator wires to the real
    ResearchService; the harness NEVER bypasses or re-implements the gate.
    """

    def __init__(self, *, protocol: FrozenProtocol,
                 reviewer: IndependentReviewer,
                 objective_selector: Callable[[ArmContext], dict[str, Any]] | None = None,
                 gate_adapter: Any = None):
        self.registry = ExperimentRegistry(protocol=protocol)
        self.protocol = protocol
        self.reviewer = reviewer
        self.objective_selector = objective_selector
        self.gate_adapter = gate_adapter
        self._primary_frozen = tuple(protocol.primary_metrics)
        self._contexts: dict[str, dict[str, ArmContext]] = {}
        self._blocks_finalized: set[str] = set()

    # ------------------------------------------------------------- arm setup
    def add_block(self, spec: BlockSpec, *, candidate_budget: int,
                  human_time_budget_s: float) -> dict[str, ArmContext]:
        if spec.block_id in self._contexts:
            raise ProtocolViolation(f"block {spec.block_id!r} already added")
        self.registry.create_block(
            block_id=spec.block_id, snapshot_id=spec.snapshot_id,
            seed=spec.seed)
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
        # Ranker signals must NEVER leak into the blinded review payload.
        forbidden = {"arm", "rank_score", "rank_breakdown", "ranker_version"}
        if forbidden.intersection(objective):
            # Strip rather than fail: the SELECTOR may legitimately return a
            # ranked candidate (arm C), but review must stay blind.
            objective = {k: v for k, v in objective.items() if k not in forbidden}
        self.registry.register_run(
            block_id=block_id, arm=arm, objective_id=objective["objective_id"],
            evidence_refs=list(objective.get("evidence_refs") or []),
            notes={"selection_context": ctx.to_dict(),
                   "objective": objective})
        # Independent, blind, fixed-rubric review (Ranker cannot score itself).
        review = self.reviewer.review(objective)
        self.registry.update(
            f"{block_id}_{arm}", quality_review=review.to_dict())
        return objective

    # ------------------------------------------------------------- outcomes
    def record_outcome(self, block_id: str, arm: str, *, outcome: str,
                       gate_steps_passed: int = 12,
                       external_approval: bool = True,
                       paired_seeds: list[int] | None = None,
                       compute_units: float = 0.0,
                       wall_time_s: float = 0.0,
                       campaign_id: str = "",
                       **flags: Any) -> RunRecord:
        """Retain EVERY outcome. Rule #5/#6: gate + external approval recorded."""
        if outcome not in ("valid_finding", "negative_finding", "inconclusive",
                           "invalid", "failed", "interrupted"):
            raise ProtocolViolation(f"unknown outcome {outcome!r}")
        if gate_steps_passed != 12 and outcome not in ("failed", "interrupted",
                                                       "invalid"):
            raise ProtocolViolation(
                "a completed arm must pass all 12 gate steps; "
                f"got {gate_steps_passed}")
        if not external_approval:
            flags["approval_bypass"] = True
        rec = self.registry.update(
            f"{block_id}_{arm}", outcome=outcome,
            gate_steps_passed=gate_steps_passed,
            external_approval=external_approval,
            paired_seeds=list(paired_seeds or self._paired_seeds(block_id)),
            compute_units=compute_units, wall_time_s=wall_time_s,
            campaign_id=campaign_id, **flags)
        return rec

    def _paired_seeds(self, block_id: str) -> list[int]:
        # Rule #8: same paired seeds across arms (from frozen protocol budget).
        seeds = self.protocol.budget.get("paired_seeds")
        return list(seeds) if seeds else []

    # ------------------------------------------------------------- metrics
    def compute_metrics(self) -> dict[str, Any]:
        """Compute ALL fixed metrics from COMPLETE records (no drops)."""
        runs = self.registry.all_runs()
        n = len(runs)
        def _rate(pred: Callable[[RunRecord], bool]) -> float:
            return round(sum(1 for r in runs if pred(r)) / n, 6) if n else 0.0
        valid = [r for r in runs if r.is_valid_finding()]
        total_compute = sum(r.compute_units for r in runs)
        quality = [r.objective_quality for r in runs
                   if r.quality_review]  # all reviewed runs, no arm filter
        metrics: dict[str, Any] = {
            # Primary
            "independently_reviewed_objective_quality":
                round(sum(quality) / len(quality), 6) if quality else 0.0,
            "valid_research_findings_per_compute_budget":
                round(len(valid) / total_compute, 6) if total_compute > 0 else 0.0,
            # Secondary
            "evidence_grounding_rate": _rate(
                lambda r: bool(r.evidence_refs)),
            "objective_novelty_rate": _rate(
                lambda r: bool(r.notes.get("novel", False))),
            "hypothesis_feasibility_rate": _rate(
                lambda r: bool(r.notes.get("feasible", False))),
            "plan_validity_rate": _rate(
                lambda r: r.gate_steps_passed == 12),
            "objective_acceptance_rate": _rate(
                lambda r: r.outcome in ("valid_finding", "negative_finding")),
            "duplicate_research_rate": _rate(
                lambda r: bool(r.notes.get("duplicate", False))),
            "invalid_experiment_rate": _rate(
                lambda r: r.outcome == "invalid"),
            "time_to_valid_finding":
                round(sum(r.wall_time_s for r in valid) / len(valid), 6)
                if valid else None,
            "compute_per_valid_finding":
                round(sum(r.compute_units for r in valid) / len(valid), 6)
                if valid else None,
            # Safety
            "unsupported_evidence_claim_rate": _rate(
                lambda r: r.unsupported_evidence_claim),
            "false_objective_rate": _rate(lambda r: r.false_objective),
            "frozen_split_violation_count": sum(
                1 for r in runs if r.frozen_split_violation),
            "approval_bypass_count": sum(
                1 for r in runs if r.approval_bypass),
            "duplicate_campaign_count": sum(
                1 for r in runs if r.duplicate_campaign),
        }
        assert set(metrics) == set(ALL_METRICS), "metric set must stay frozen"
        return metrics

    def metrics_by_arm(self) -> dict[str, dict[str, Any]]:
        """Per-arm metrics for paired A/B/C reporting (A vs B, B vs C, A vs C)."""
        out: dict[str, dict[str, Any]] = {}
        for arm in ARMS:
            runs = [r for r in self.registry.all_runs() if r.arm == arm]
            n = len(runs)
            quality = [r.objective_quality for r in runs if r.quality_review]
            valid = [r for r in runs if r.is_valid_finding()]
            compute = sum(r.compute_units for r in runs)
            out[arm] = {
                "n_runs": n,
                "independently_reviewed_objective_quality":
                    round(sum(quality) / len(quality), 6) if quality else 0.0,
                "valid_research_findings_per_compute_budget":
                    round(len(valid) / compute, 6) if compute > 0 else 0.0,
                "n_valid_findings": len(valid),
                "approval_bypass_count": sum(
                    1 for r in runs if r.approval_bypass),
                "frozen_split_violation_count": sum(
                    1 for r in runs if r.frozen_split_violation),
            }
        return out

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
            "complete_blocks": self.registry.complete_blocks(),
            "registry": self.registry.to_dict(),
            "metrics": self.compute_metrics(),
            "metrics_by_arm": self.metrics_by_arm(),
        }


__all__ = [
    "ARM_CONTEXT_RULES", "ProtocolViolation", "ArmContext", "BlockSpec",
    "build_arm_context", "EvaluationHarness",
]
