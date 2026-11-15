"""M4.3-01 -- Experiment registry: blocks, pairing, and COMPLETE run records.

Independent of ResearchService. It records WHAT was run, under which frozen
configuration, and keeps EVERY outcome including failed / invalid /
INCONCLUSIVE / interrupted runs (they are never dropped from the denominator).

Design rules (spec M4.3-01):
  * A/B/C use the SAME research-opportunity Snapshot (blocked pairing).
  * Snapshot / Seed hierarchy is preserved: each block has one snapshot_id and
    one seed; the three arms of that block share them.
  * All arms share budget, paired seeds, model/prompt/Ranker version and the
    SAME 12-step gate (recorded as invariants, not re-implemented here).
  * Frozen fields are hashed; a post-hoc change is detectable and rejected.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .metrics import ALL_METRICS, PRIMARY_METRICS, frozen_metric_contract

ARMS: tuple[str, ...] = ("A", "B", "C")
ARM_DESCRIPTIONS: dict[str, str] = {
    "A": ("Manual: human selects an Objective from current research evidence, "
          "WITHOUT historical Experience assistance."),
    "B": ("Memory-assisted Manual: human reads the SAME historical Evidence, "
          "Experience and Failure context, then selects an Objective."),
    "C": ("Autonomous Discovery: ObjectiveCandidateGenerator + Critic + Ranker "
          "select the Objective automatically."),
}

# A run's terminal classification. Every one of these is a RECORDED outcome;
# none is silently discarded.
RUN_OUTCOMES: tuple[str, ...] = (
    "valid_finding", "negative_finding", "inconclusive", "invalid",
    "failed", "interrupted",
)


def _hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class FrozenProtocol:
    """Fields that must be frozen BEFORE the confirmatory run (M4.3-03).

    Frozen here for the harness so any post-hoc change is detectable. Pilot
    (M4.3-02) may leave ``confirmatory`` False and sample_size 0; the
    confirmatory experiment MUST set every field.
    """
    snapshot_ids: tuple[str, ...] = ()
    primary_metrics: tuple[str, ...] = PRIMARY_METRICS
    sample_size: int = 0
    seed: int = 0
    model_version: str = ""
    prompt_version: str = ""
    ranker_version: str = ""
    budget: dict[str, Any] = field(default_factory=dict)
    exclusion_rules: tuple[str, ...] = ()
    statistical_method: str = ""
    stopping_rule: str = ""
    confirmatory: bool = False

    def frozen_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["snapshot_ids"] = list(self.snapshot_ids)
        d["primary_metrics"] = list(self.primary_metrics)
        d["exclusion_rules"] = list(self.exclusion_rules)
        d["metric_contract_sha256"] = frozen_metric_contract()[
            "contract_sha256"]
        return d

    def freeze_sha256(self) -> str:
        return _hash(self.frozen_dict())

    def assert_confirmatory_ready(self) -> None:
        """M4.3-03 gate: refuse to start if the protocol was not frozen."""
        missing = []
        if not self.confirmatory:
            missing.append("confirmatory flag")
        if not self.snapshot_ids:
            missing.append("snapshot set")
        if self.sample_size <= 0:
            missing.append("sample size")
        if not self.statistical_method:
            missing.append("statistical method")
        if not self.stopping_rule:
            missing.append("stopping condition")
        if not self.exclusion_rules:
            missing.append("exclusion rules")
        if missing:
            raise ValueError(
                "confirmatory experiment requires a frozen protocol; missing: "
                + ", ".join(missing))


@dataclass
class Block:
    """ONE research opportunity: 3 paired arms sharing snapshot + seed."""
    block_id: str
    snapshot_id: str
    seed: int
    arms: dict[str, str] = field(default_factory=dict)  # arm -> run_id


@dataclass
class RunRecord:
    """A COMPLETE record of one arm's run. Nothing is dropped."""
    run_id: str
    block_id: str
    snapshot_id: str
    seed: int
    arm: str
    outcome: str = "interrupted"          # RUN_OUTCOMES
    objective_id: str = ""
    # objective-quality evidence (P0-02 independent BLIND review)
    evidence_refs: list[str] = field(default_factory=list)
    # opaque review_id only; the raw objective_id -> arm mapping is SEALED in
    # the BlindingLedger and never stored on the reviewer-visible payload.
    review_id: str = ""
    quality_review: dict[str, Any] = field(default_factory=dict)  # PRIMARY human-blind review
    independent_reviews: list[dict[str, Any]] = field(default_factory=list)
    automated_scores: list[dict[str, Any]] = field(default_factory=list)
    # execution evidence (P0-01: DERIVED from the real read-only gate adapter,
    # never caller-supplied). ``gate_evidence`` is the full persisted evidence
    # chain that produced ``gate_steps_passed`` / ``external_approval``.
    gate_steps_passed: int = 0
    external_approval: bool = False
    gate_evidence: dict[str, Any] = field(default_factory=dict)
    campaign_id: str = ""
    compute_units: float = 0.0
    wall_time_s: float = 0.0
    paired_seeds: list[int] = field(default_factory=list)
    # audit flags
    frozen_split_violation: bool = False
    approval_bypass: bool = False
    duplicate_campaign: bool = False
    unsupported_evidence_claim: bool = False
    false_objective: bool = False
    invalid_reason: str = ""
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def is_valid_finding(self) -> bool:
        return self.outcome == "valid_finding"

    @property
    def objective_quality(self) -> float:
        return float(self.quality_review.get("total") or 0.0)


class ExperimentRegistry:
    """Append-only registry of blocked, paired A/B/C runs."""

    def __init__(self, *, protocol: FrozenProtocol):
        self.protocol = protocol
        self.blocks: dict[str, Block] = {}
        self.runs: dict[str, RunRecord] = {}
        self._frozen_hash = protocol.freeze_sha256()

    @property
    def freeze_sha256(self) -> str:
        return self._frozen_hash

    # ------------------------------------------------------------- blocks
    def create_block(self, *, block_id: str, snapshot_id: str, seed: int) -> Block:
        if block_id in self.blocks:
            raise ValueError(f"block {block_id!r} already exists")
        if snapshot_id not in self.protocol.snapshot_ids:
            # Pilot may run before the snapshot set is frozen; confirmatory must
            # be frozen. Enforce strictly when confirmatory.
            if self.protocol.confirmatory:
                raise ValueError(
                    f"snapshot {snapshot_id!r} is not in the frozen snapshot set")
        block = Block(block_id=block_id, snapshot_id=snapshot_id, seed=seed)
        self.blocks[block_id] = block
        return block

    # --------------------------------------------------------------- runs
    def register_run(self, *, block_id: str, arm: str, **kwargs: Any) -> RunRecord:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
        block = self.blocks.get(block_id)
        if block is None:
            raise ValueError(f"unknown block {block_id!r}")
        if arm in block.arms:
            raise ValueError(f"arm {arm} already registered in block {block_id}")
        run_id = kwargs.pop("run_id", "") or f"{block_id}_{arm}"
        if run_id in self.runs:
            raise ValueError(f"duplicate run_id {run_id!r}")
        outcome = kwargs.get("outcome", "interrupted")
        if outcome not in RUN_OUTCOMES:
            raise ValueError(f"unknown outcome {outcome!r}")
        rec = RunRecord(
            run_id=run_id, block_id=block_id,
            snapshot_id=block.snapshot_id, seed=block.seed, arm=arm, **kwargs)
        self.runs[run_id] = rec
        block.arms[arm] = run_id
        return rec

    def update(self, run_id: str, **changes: Any) -> RunRecord:
        """Complete / amend a run. Outcomes are retained, never deleted."""
        rec = self.runs.get(run_id)
        if rec is None:
            raise ValueError(f"unknown run {run_id!r}")
        if "outcome" in changes and changes["outcome"] not in RUN_OUTCOMES:
            raise ValueError(f"unknown outcome {changes['outcome']!r}")
        for key, value in changes.items():
            if not hasattr(rec, key):
                raise ValueError(f"unknown RunRecord field {key!r}")
            setattr(rec, key, value)
        return rec

    # ------------------------------------------------------------ queries
    def runs_for_block(self, block_id: str) -> dict[str, RunRecord]:
        block = self.blocks[block_id]
        return {arm: self.runs[rid] for arm, rid in block.arms.items()}

    def complete_blocks(self) -> list[str]:
        """Blocks where ALL THREE arms are recorded (for paired comparison)."""
        return sorted(bid for bid, b in self.blocks.items()
                      if all(a in b.arms for a in ARMS))

    def all_runs(self) -> list[RunRecord]:
        return list(self.runs.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol.frozen_dict(),
            "freeze_sha256": self.freeze_sha256,
            "metric_names": list(ALL_METRICS),
            "blocks": {k: asdict(v) for k, v in self.blocks.items()},
            "runs": {k: v.to_dict() for k, v in self.runs.items()},
        }


__all__ = [
    "ARMS", "ARM_DESCRIPTIONS", "RUN_OUTCOMES", "FrozenProtocol", "Block",
    "RunRecord", "ExperimentRegistry",
]
